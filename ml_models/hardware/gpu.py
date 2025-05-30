#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
GPU Acceleration Module

This module provides GPU acceleration for machine learning models, optimized for
CUDA-enabled devices like the RTX 3050 with 8GB VRAM. It includes dynamic memory
management, precision optimization, and CUDA kernel optimization for trading-specific
operations.
"""

import os
import logging
import numpy as np
import contextlib
from typing import Dict, List, Union, Tuple, Optional, Any, Callable

# GPU-related imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.cuda as cuda
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    
# Use unified GPU utility imports
from feature_service.processor_utils import cudf, cp, HAS_GPU
HAS_CUPY = HAS_GPU

try:
    import pycuda.driver as drv
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

# Local imports
from common.logger import get_logger
from common.exceptions import HardwareError, ResourceError
from common.metrics import MetricsCollector
from ml_models.hardware.base import HardwareAccelerator

logger = get_logger(__name__)
metrics = MetricsCollector.get_instance()


class GPUManager:
    """Manages GPU resources and handles device selection, memory optimization,
    and advanced CUDA configurations for trading-specific operations."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = get_logger(f"{__name__}.GPUManager")
        self.metrics = MetricsCollector.get_instance()
        
        # Default values
        self.has_gpu = False
        self.device_count = 0
        self.devices = []
        self.selected_device = None
        self.vram_size = 0
        self.vram_available = 0
        self.compute_capability = (0, 0)
        self.tensor_cores_available = False
        self.half_precision_supported = False
        self.device_properties = {}
        self.memory_pool = None
        self.stream_pool = []
        self.kernel_cache = {}
        
        # Initialize available GPU frameworks
        self.frameworks = {
            'torch': HAS_TORCH,
            'cupy': HAS_GPU,
            'pycuda': HAS_PYCUDA
        }
        
        # Configuration
        self.use_tensor_cores = True
        self.use_half_precision = True
        self.reserved_memory = 0.1  # Reserve 10% of VRAM
        self.max_batch_size = 512
        self.adaptive_batch_sizing = True
        self.prefetch_enabled = True
        self.async_execution = True
        
        # Initialize GPU environment
        self._detect_gpus()
        if self.has_gpu:
            self._setup_device()
            self._init_memory_management()
            self._init_streams()
            
        # Specialized trading kernels
        self._init_trading_kernels()
        
        # Memory tracking
        self.active_allocations = {}
        self.allocation_stats = {
            'total_allocated': 0,
            'max_allocated': 0,
            'allocation_count': 0,
            'peak_usage_percent': 0.0
        }
        
        self._initialized = True
        
        self.logger.info(
            f"GPU Manager initialized: {self.device_count} devices available, "
            f"Using device {self.selected_device}, VRAM: {self.vram_size/1024**3:.2f}GB, "
            f"Compute capability: {self.compute_capability[0]}.{self.compute_capability[1]}"
        )
        
    def _detect_gpus(self) -> None:
        """Detect available GPUs and their capabilities across frameworks."""
        try:
            if HAS_TORCH and torch.cuda.is_available():
                self.has_gpu = True
                self.device_count = torch.cuda.device_count()
                self.devices = list(range(self.device_count))
                
                if self.device_count > 0:
                    self.selected_device = 0  # Default to first device
                    
                    # Get device properties
                    props = torch.cuda.get_device_properties(self.selected_device)
                    self.vram_size = props.total_memory
                    self.compute_capability = (props.major, props.minor)
                    self.device_properties['torch'] = props
                    
                    # Check for tensor cores (Volta+ architecture, cc >= 7.0)
                    self.tensor_cores_available = self.compute_capability[0] >= 7
                    
                    # Check for FP16 support (Pascal+, cc >= 6.0)
                    self.half_precision_supported = self.compute_capability[0] >= 6
                    
            elif HAS_GPU:
                self.has_gpu = True
                self.device_count = cp.cuda.runtime.getDeviceCount()
                self.devices = list(range(self.device_count))
                
                if self.device_count > 0:
                    self.selected_device = 0
                    # Get device properties with CuPy
                    cp.cuda.Device(self.selected_device).use()
                    self.vram_size = cp.cuda.Device(self.selected_device).mem_info[1]
                    attrs = cp.cuda.Device(self.selected_device).attributes
                    self.compute_capability = (
                        attrs.get('ComputeCapabilityMajor', 0),
                        attrs.get('ComputeCapabilityMinor', 0)
                    )
                    self.device_properties['cupy'] = attrs
                    self.tensor_cores_available = self.compute_capability[0] >= 7
                    self.half_precision_supported = self.compute_capability[0] >= 6
                    
            elif HAS_PYCUDA:
                self.has_gpu = True
                self.device_count = drv.Device.count()
                self.devices = list(range(self.device_count))
                
                if self.device_count > 0:
                    self.selected_device = 0
                    # Get device properties with PyCUDA
                    device = drv.Device(self.selected_device)
                    self.vram_size = device.total_memory()
                    cc = device.compute_capability()
                    self.compute_capability = cc
                    self.device_properties['pycuda'] = {
                        'name': device.name(),
                        'compute_capability': cc,
                        'total_memory': device.total_memory()
                    }
                    self.tensor_cores_available = cc[0] >= 7
                    self.half_precision_supported = cc[0] >= 6
            
            # If no GPU is detected with any framework
            if not self.has_gpu:
                self.logger.warning("No CUDA-compatible GPU detected")
                
        except Exception as e:
            self.logger.error(f"Error detecting GPUs: {str(e)}")
            self.has_gpu = False
    
    def _setup_device(self) -> None:
        """Configure the selected GPU device for optimal trading performance."""
        try:
            if not self.has_gpu:
                return
                
            # Set device in all frameworks
            if HAS_TORCH:
                torch.cuda.set_device(self.selected_device)
                # Optimize performance
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # Check available memory
                self.vram_available = torch.cuda.get_device_properties(self.selected_device).total_memory
                memory_allocated = torch.cuda.memory_allocated(self.selected_device)
                memory_reserved = torch.cuda.memory_reserved(self.selected_device)
                
                self.vram_available = self.vram_available - memory_allocated - memory_reserved
                
            if HAS_GPU:
                cp.cuda.Device(self.selected_device).use()
                # Get memory info
                free, total = cp.cuda.Device(self.selected_device).mem_info
                self.vram_available = min(self.vram_available, total) if self.vram_available > 0 else total
                
            if HAS_PYCUDA:
                # PyCUDA uses the device set by the context, which is handled in autoinit
                # Ensure we're using the correct device
                drv.Context.pop()  # Remove current context
                device = drv.Device(self.selected_device)
                ctx = device.make_context()
                
                # Get memory info
                free, total = driver.mem_get_info()
                self.vram_available = min(self.vram_available, total) if self.vram_available > 0 else total
                
                # Need to re-push the context
                ctx.push()
                
            # RTX 3050 optimization profile
            # The RTX 3050 has 8GB VRAM and is optimized for CUDA compute capability 7.5
            if "3050" in str(self.device_properties).lower() and self.vram_size <= 8.5 * 1024**3:
                self.logger.info("Detected RTX 3050 GPU, applying specialized optimizations")
                # Conservative batch sizing for 8GB VRAM
                self.max_batch_size = 256
                # Reserve more memory to prevent OOM errors
                self.reserved_memory = 0.15  # 15% reservation
                
            self.logger.info(
                f"Using GPU device {self.selected_device} with "
                f"{self.vram_available/1024**3:.2f}GB available VRAM"
            )
            
            # Log GPU capabilities
            caps = []
            if self.tensor_cores_available and self.use_tensor_cores:
                caps.append("Tensor Cores")
            if self.half_precision_supported and self.use_half_precision:
                caps.append("FP16 precision")
                
            if caps:
                self.logger.info(f"GPU acceleration features enabled: {', '.join(caps)}")
                
        except Exception as e:
            self.logger.error(f"Error setting up GPU device: {str(e)}")
            raise HardwareError(f"GPU setup failed: {str(e)}")
    
    def _init_memory_management(self) -> None:
        """Initialize memory management for optimal VRAM utilization."""
        if not self.has_gpu:
            return
            
        try:
            # Create memory pools for faster allocation/deallocation
            if HAS_TORCH:
                # Enable PyTorch's memory pools
                torch.cuda.empty_cache()
                
                # Set memory allocation fraction (considering reserved memory)
                max_memory = int(self.vram_size * (1 - self.reserved_memory))
                torch.cuda.set_per_process_memory_fraction(max_memory / self.vram_size)
                
            if HAS_GPU:
                # Create memory pool
                self.memory_pool = cp.cuda.MemoryPool()
                cp.cuda.set_allocator(self.memory_pool.malloc)
                
                # Limit memory usage
                pool_limit = int(self.vram_size * (1 - self.reserved_memory))
                self.memory_pool.set_limit(size=pool_limit)
                
            # Log memory configuration
            self.logger.info(
                f"GPU memory management initialized with {self.reserved_memory * 100:.1f}% reserved, "
                f"leaving {(1 - self.reserved_memory) * self.vram_size/1024**3:.2f}GB for computations"
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing GPU memory management: {str(e)}")
            
    def _init_streams(self) -> None:
        """Initialize CUDA streams for parallel execution."""
        if not self.has_gpu:
            return
            
        try:
            # Create multiple streams for concurrent operations
            if HAS_TORCH:
                # Create primary stream and worker streams
                self.stream_pool = [torch.cuda.Stream(device=self.selected_device) for _ in range(4)]
                
            if HAS_GPU:
                # Create CuPy streams
                self.stream_pool.extend([cp.cuda.Stream() for _ in range(4)])
                
            if HAS_PYCUDA:
                # Create PyCUDA streams
                self.stream_pool.extend([drv.Stream() for _ in range(4)])
                
            self.logger.debug(f"Created {len(self.stream_pool)} CUDA streams for parallel execution")
            
        except Exception as e:
            self.logger.error(f"Error initializing CUDA streams: {str(e)}")
    
    def _init_trading_kernels(self) -> None:
        """Initialize specialized CUDA kernels for common trading operations."""
        if not self.has_gpu or not HAS_PYCUDA:
            return
            
        try:
            # Define custom CUDA kernels for common trading operations
            # These specialized kernels significantly outperform CPU implementations
            
            # 1. Moving average calculation kernel
            ma_kernel_code = """
            extern "C" {
                __global__ void moving_average(float *input, float *output, int window_size, int n) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < n) {
                        float sum = 0.0f;
                        int count = 0;
                        int start = max(0, idx - window_size + 1);
                        
                        for (int i = start; i <= idx; i++) {
                            sum += input[i];
                            count++;
                        }
                        
                        output[idx] = sum / count;
                    }
                }
            }
            """
            
            # 2. Bollinger Bands calculation kernel
            bb_kernel_code = """
            extern "C" {
                __global__ void bollinger_bands(float *prices, float *sma, float *upper, float *lower, 
                                                int window_size, float num_std, int n) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < n) {
                        float sum = 0.0f;
                        float sum_sq = 0.0f;
                        int count = 0;
                        int start = max(0, idx - window_size + 1);
                        
                        for (int i = start; i <= idx; i++) {
                            sum += prices[i];
                            sum_sq += prices[i] * prices[i];
                            count++;
                        }
                        
                        float mean = sum / count;
                        float variance = (sum_sq / count) - (mean * mean);
                        float std_dev = sqrt(variance);
                        
                        sma[idx] = mean;
                        upper[idx] = mean + (num_std * std_dev);
                        lower[idx] = mean - (num_std * std_dev);
                    }
                }
            }
            """
            
            # 3. RSI calculation helper kernels
            rsi_kernel_code = """
            extern "C" {
                __global__ void price_changes(float *prices, float *changes, int n) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx > 0 && idx < n) {
                        changes[idx] = prices[idx] - prices[idx-1];
                    } else if (idx == 0) {
                        changes[idx] = 0.0f;
                    }
                }
                
                __global__ void gains_losses(float *changes, float *gains, float *losses, int n) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < n) {
                        gains[idx] = max(0.0f, changes[idx]);
                        losses[idx] = max(0.0f, -changes[idx]);
                    }
                }
                
                __global__ void calc_rsi(float *avg_gains, float *avg_losses, float *rsi, int n) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < n) {
                        float rs = avg_losses[idx] > 0.0f ? avg_gains[idx] / avg_losses[idx] : 100.0f;
                        rsi[idx] = 100.0f - (100.0f / (1.0f + rs));
                    }
                }
            }
            """
            
            # 4. Pattern detection kernel
            pattern_detection_code = """
            extern "C" {
                __global__ void detect_engulfing(float *open, float *close, float *high, float *low, 
                                                int *patterns, int n) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx > 0 && idx < n-1) {
                        // Bullish engulfing
                        if (close[idx-1] < open[idx-1] && 
                            open[idx] < close[idx-1] && 
                            close[idx] > open[idx-1]) {
                            patterns[idx] = 1;
                        }
                        // Bearish engulfing
                        else if (close[idx-1] > open[idx-1] && 
                                open[idx] > close[idx-1] && 
                                close[idx] < open[idx-1]) {
                            patterns[idx] = -1;
                        }
                        else {
                            patterns[idx] = 0;
                        }
                    } else {
                        patterns[idx] = 0;
                    }
                }
            }
            """
            
            # 5. VWAP calculation kernel
            vwap_kernel_code = """
            extern "C" {
                __global__ void vwap_calc(float *high, float *low, float *close, float *volume, float *vwap, int n) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < n) {
                        float typical_price = (high[idx] + low[idx] + close[idx]) / 3.0f;
                        float price_volume = typical_price * volume[idx];
                        
                        float cum_price_volume = 0.0f;
                        float cum_volume = 0.0f;
                        
                        for (int i = 0; i <= idx; i++) {
                            cum_price_volume += ((high[i] + low[i] + close[i]) / 3.0f) * volume[i];
                            cum_volume += volume[i];
                        }
                        
                        vwap[idx] = cum_volume > 0.0f ? cum_price_volume / cum_volume : 0.0f;
                    }
                }
            }
            """
            
            # 6. Trend strength estimation kernel (based on consecutive price movements)
            trend_kernel_code = """
            extern "C" {
                __global__ void trend_strength(float *prices, float *strength, int window, int n) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < n) {
                        int start = max(0, idx - window + 1);
                        int up_count = 0;
                        int down_count = 0;
                        
                        for (int i = start; i < idx; i++) {
                            if (prices[i+1] > prices[i]) {
                                up_count++;
                            } else if (prices[i+1] < prices[i]) {
                                down_count++;
                            }
                        }
                        
                        int total = idx - start;
                        if (total > 0) {
                            float up_ratio = (float)up_count / total;
                            float down_ratio = (float)down_count / total;
                            
                            // Range from -1 (strong downtrend) to 1 (strong uptrend)
                            strength[idx] = up_ratio - down_ratio;
                        } else {
                            strength[idx] = 0.0f;
                        }
                    }
                }
            }
            """
            
            # Compile all kernels
            if HAS_PYCUDA:
                # Create dictionary of compiled modules
                self.kernel_cache['moving_average'] = SourceModule(ma_kernel_code)
                self.kernel_cache['bollinger_bands'] = SourceModule(bb_kernel_code)
                self.kernel_cache['rsi'] = SourceModule(rsi_kernel_code)
                self.kernel_cache['pattern_detection'] = SourceModule(pattern_detection_code)
                self.kernel_cache['vwap'] = SourceModule(vwap_kernel_code)
                self.kernel_cache['trend_strength'] = SourceModule(trend_kernel_code)
                
                self.logger.info("Compiled specialized CUDA kernels for trading operations")
                
        except Exception as e:
            self.logger.error(f"Error initializing trading kernels: {str(e)}")
            self.logger.info("Will fall back to standard calculations")
    
    def get_device(self):
        """Return current device information for the framework in use."""
        if not self.has_gpu:
            return None
            
        if HAS_TORCH:
            return torch.device(f"cuda:{self.selected_device}")
        elif HAS_GPU:
            return cp.cuda.Device(self.selected_device)
        elif HAS_PYCUDA:
            return self.selected_device
        else:
            return None
            
    def memory_status(self) -> Dict[str, Any]:
        """Get current GPU memory usage statistics."""
        if not self.has_gpu:
            return {"error": "No GPU available"}
            
        try:
            status = {
                "total_vram": self.vram_size / (1024**3),  # GB
                "reserved_percent": self.reserved_memory * 100,
                "allocations": len(self.active_allocations),
                "stats": self.allocation_stats.copy()
            }
            
            # Get current usage
            if HAS_TORCH:
                allocated = torch.cuda.memory_allocated(self.selected_device)
                reserved = torch.cuda.memory_reserved(self.selected_device)
                status["allocated_mb"] = allocated / (1024**2)
                status["reserved_mb"] = reserved / (1024**2)
                status["free_mb"] = (self.vram_size - reserved) / (1024**2)
                status["percent_used"] = (allocated / self.vram_size) * 100
                
            elif HAS_GPU:
                free, total = cp.cuda.Device(self.selected_device).mem_info
                used = total - free
                status["allocated_mb"] = used / (1024**2)
                status["free_mb"] = free / (1024**2)
                status["percent_used"] = (used / total) * 100
                
            status["active_tensors"] = len(self.active_allocations)
            status["peak_usage_percent"] = self.allocation_stats["peak_usage_percent"]
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting memory status: {str(e)}")
            return {"error": str(e)}
            
    def clear_cache(self) -> None:
        """Clear all cached memory to free up VRAM."""
        if not self.has_gpu:
            return
            
        try:
            if HAS_TORCH:
                torch.cuda.empty_cache()
                
            if HAS_GPU:
                cp.get_default_memory_pool().free_all_blocks()
                
            # Reset tracking
            self.active_allocations = {}
            self.allocation_stats["total_allocated"] = 0
            
            self.logger.info("Cleared GPU memory cache")
            
        except Exception as e:
            self.logger.error(f"Error clearing GPU cache: {str(e)}")
            
    def synchronize(self) -> None:
        """Synchronize all CUDA streams and wait for completion."""
        if not self.has_gpu:
            return
            
        try:
            if HAS_TORCH:
                torch.cuda.synchronize()
                
            if HAS_GPU:
                cp.cuda.Stream.null.synchronize()
                
            for stream in self.stream_pool:
                if hasattr(stream, 'synchronize'):
                    stream.synchronize()
                    
            self.logger.debug("GPU operations synchronized")
            
        except Exception as e:
            self.logger.error(f"Error synchronizing GPU: {str(e)}")
    
    def get_optimal_batch_size(self, tensor_size: Tuple[int, ...], dtype: str = 'float32') -> int:
        """Calculate optimal batch size based on available VRAM and tensor dimensions."""
        if not self.has_gpu:
            return self.max_batch_size
            
        try:
            # Calculate memory per sample
            element_size = 4  # Default float32
            if dtype == 'float16' or dtype == 'half':
                element_size = 2
            elif dtype == 'int32':
                element_size = 4
            elif dtype == 'int64' or dtype == 'double' or dtype == 'float64':
                element_size = 8
                
            # Calculate tensor size excluding batch dimension
            tensor_elements = 1
            for dim in tensor_size[1:]:  # Skip batch dimension
                tensor_elements *= dim
                
            bytes_per_sample = tensor_elements * element_size
            
            # Get available memory with safety margin
            if HAS_TORCH:
                available = self.vram_size - torch.cuda.memory_allocated() - torch.cuda.memory_reserved()
                # Apply safety margin
                available = available * 0.8  # 80% of available memory
            elif HAS_GPU:
                free, total = cp.cuda.Device(self.selected_device).mem_info
                available = free * 0.8
            else:
                # Conservative estimate
                available = self.vram_size * 0.5
                
            # Calculate batch size
            batch_size = int(available / bytes_per_sample)
            
            # Apply constraints
            batch_size = min(batch_size, self.max_batch_size)
            batch_size = max(batch_size, 1)  # Ensure at least 1
            
            self.logger.debug(
                f"Calculated optimal batch size: {batch_size} for tensor {tensor_size} "
                f"({bytes_per_sample/1024:.2f}KB per sample)"
            )
            
            return batch_size
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal batch size: {str(e)}")
            return min(32, self.max_batch_size)  # Conservative fallback
            
    def run_kernel(self, kernel_name: str, *args, **kwargs) -> Any:
        """Run a specialized CUDA kernel for trading calculations."""
        if not self.has_gpu or not HAS_PYCUDA or kernel_name not in self.kernel_cache:
            return None
            
        try:
            # Get compiled module
            module = self.kernel_cache[kernel_name]
            
            # Map kernel names to their functions and required arguments
            kernel_map = {
                'moving_average': {
                    'func': module.get_function('moving_average'),
                    'required_args': ['input', 'output', 'window_size', 'n']
                },
                'bollinger_bands': {
                    'func': module.get_function('bollinger_bands'),
                    'required_args': ['prices', 'sma', 'upper', 'lower', 'window_size', 'num_std', 'n']
                },
                'rsi': {
                    'func_changes': module.get_function('price_changes'),
                    'func_gains_losses': module.get_function('gains_losses'),
                    'func_rsi': module.get_function('calc_rsi'),
                    'required_args': ['prices', 'rsi', 'window_size', 'n']
                },
                'pattern_detection': {
                    'func': module.get_function('detect_engulfing'),
                    'required_args': ['open', 'close', 'high', 'low', 'patterns', 'n']
                },
                'vwap': {
                    'func': module.get_function('vwap_calc'),
                    'required_args': ['high', 'low', 'close', 'volume', 'vwap', 'n']
                },
                'trend_strength': {
                    'func': module.get_function('trend_strength'),
                    'required_args': ['prices', 'strength', 'window', 'n']
                }
            }
            
            # Get kernel info
            kernel_info = kernel_map[kernel_name]
            
            # Special case for RSI which uses multiple kernels
            if kernel_name == 'rsi':
                # Unpack arguments
                prices = kwargs.get('prices')
                window_size = kwargs.get('window_size', 14)
                n = len(prices)
                
                # Allocate memory for intermediate arrays
                changes = gpuarray.zeros(n, dtype=np.float32)
                gains = gpuarray.zeros(n, dtype=np.float32)
                losses = gpuarray.zeros(n, dtype=np.float32)
                avg_gains = gpuarray.zeros(n, dtype=np.float32)
                avg_losses = gpuarray.zeros(n, dtype=np.float32)
                rsi = gpuarray.zeros(n, dtype=np.float32)
                
                # Calculate price changes
                block_size = 256
                grid_size = (n + block_size - 1) // block_size
                
                kernel_info['func_changes'](
                    prices, changes, np.int32(n),
                    block=(block_size, 1, 1), grid=(grid_size, 1)
                )
                
                # Calculate gains and losses
                kernel_info['func_gains_losses'](
                    changes, gains, losses, np.int32(n),
                    block=(block_size, 1, 1), grid=(grid_size, 1)
                )
                
                # Calculate average gains and losses (this part is implemented in Python for flexibility)
                avg_gains_np = gains.get()
                avg_losses_np = losses.get()
                
                # First average
                avg_gains_np[window_size-1] = sum(avg_gains_np[:window_size]) / window_size
                avg_losses_np[window_size-1] = sum(avg_losses_np[:window_size]) / window_size
                
                # Subsequent smoothed averages
                for i in range(window_size, n):
                    avg_gains_np[i] = (avg_gains_np[i-1] * (window_size-1) + avg_gains_np[i]) / window_size
                    avg_losses_np[i] = (avg_losses_np[i-1] * (window_size-1) + avg_losses_np[i]) / window_size
                
                # Copy back to GPU
                avg_gains = gpuarray.to_gpu(avg_gains_np)
                avg_losses = gpuarray.to_gpu(avg_losses_np)
                
                # Calculate RSI
                kernel_info['func_rsi'](
                    avg_gains, avg_losses, rsi, np.int32(n),
                    block=(block_size, 1, 1), grid=(grid_size, 1)
                )
                
                return rsi.get()
                
            else:
                # Standard kernel execution
                func = kernel_info['func']
                grid_size = (kwargs.get('n') + 255) // 256
                
                func(
                    *[kwargs.get(arg) for arg in kernel_info['required_args']],
                    block=(256, 1, 1),
                    grid=(grid_size, 1)
                )
                
                # The result will be in the output array provided in kwargs
                return True
                
        except Exception as e:
            self.logger.error(f"Error running kernel {kernel_name}: {str(e)}")
            return None


class GPUAccelerator(HardwareAccelerator):
    """GPU acceleration implementation for machine learning operations."""
    
    def __init__(self):
        super().__init__(accelerator_type="gpu")
        self.logger = get_logger(f"{__name__}.GPUAccelerator")
        self.gpu = GPUManager()
        self.scaler = None
        
        # Initialize mixed precision if supported
        if HAS_TORCH and self.gpu.half_precision_supported and self.gpu.use_half_precision:
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled with gradient scaling")
            
        # Register metrics
        self.metrics.register_gauge("gpu_memory_used_mb")
        self.metrics.register_gauge("gpu_utilization_percent")
        self.metrics.register_counter("gpu_operations_count")
        self.metrics.register_counter("gpu_errors_count")
        
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.gpu.has_gpu
        
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed information about the GPU device."""
        if not self.gpu.has_gpu:
            return {"status": "No GPU available"}
            
        return {
            "device_count": self.gpu.device_count,
            "selected_device": self.gpu.selected_device,
            "device_name": str(self.gpu.device_properties).split('\n')[0] if self.gpu.device_properties else "Unknown",
            "compute_capability": f"{self.gpu.compute_capability[0]}.{self.gpu.compute_capability[1]}",
            "vram_total_gb": self.gpu.vram_size / (1024**3),
            "tensor_cores_available": self.gpu.tensor_cores_available,
            "half_precision_supported": self.gpu.half_precision_supported,
            "memory_status": self.gpu.memory_status(),
            "features_enabled": {
                "tensor_cores": self.gpu.use_tensor_cores and self.gpu.tensor_cores_available,
                "half_precision": self.gpu.use_half_precision and self.gpu.half_precision_supported,
                "adaptive_batch_sizing": self.gpu.adaptive_batch_sizing,
                "async_execution": self.gpu.async_execution
            }
        }
        
    def to_device(self, data: Any) -> Any:
        """Move data to GPU device."""
        if not self.gpu.has_gpu:
            return data
            
        try:
            # Track GPU memory usage
            self.metrics.set_gauge("gpu_memory_used_mb", 
                                 self.gpu.memory_status().get("allocated_mb", 0))
            
            # Handle different data types
            if HAS_TORCH and isinstance(data, torch.Tensor):
                return data.cuda(self.gpu.selected_device)
                
            elif HAS_TORCH and isinstance(data, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in data):
                return [x.cuda(self.gpu.selected_device) for x in data]
                
            elif HAS_GPU and isinstance(data, np.ndarray):
                return cp.asarray(data)
                
            elif HAS_TORCH and isinstance(data, np.ndarray):
                return torch.from_numpy(data).cuda(self.gpu.selected_device)
                
            # Return as is if can't be moved
            return data
            
        except Exception as e:
            self.logger.error(f"Error moving data to device: {str(e)}")
            self.metrics.increment_counter("gpu_errors_count")
            return data
            
    def to_host(self, data: Any) -> Any:
        """Move data from GPU to CPU."""
        if not self.gpu.has_gpu:
            return data
            
        try:
            # Handle different data types
            if HAS_TORCH and isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
                
            elif HAS_TORCH and isinstance(data, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in data):
                return [x.detach().cpu().numpy() for x in data]
                
            elif HAS_GPU and isinstance(data, cp.ndarray):
                return cp.asnumpy(data)
                
            # Return as is if already on CPU
            return data
            
        except Exception as e:
            self.logger.error(f"Error moving data to host: {str(e)}")
            self.metrics.increment_counter("gpu_errors_count")
            return data
    
    @contextlib.contextmanager
    def mixed_precision_context(self):
        """Context manager for mixed precision operations."""
        if not self.gpu.has_gpu or not HAS_TORCH:
            yield
            return
            
        try:
            if self.gpu.half_precision_supported and self.gpu.use_half_precision:
                with autocast():
                    yield
            else:
                yield
                
        except Exception as e:
            self.logger.error(f"Error in mixed precision context: {str(e)}")
            self.metrics.increment_counter("gpu_errors_count")
            yield
            
    def optimize_batch_size(self, shape: Tuple[int, ...], dtype: str = "float32") -> int:
        """Calculate optimal batch size for GPU memory."""
        return self.gpu.get_optimal_batch_size(shape, dtype)
        
    def run_specialized_operation(self, operation: str, **kwargs) -> Any:
        """Run specialized GPU-accelerated operations for trading algorithms."""
        if not self.gpu.has_gpu:
            return None
            
        self.metrics.increment_counter("gpu_operations_count")
        
        try:
            # Map operation names to kernel functions
            if operation in ["moving_average", "bollinger_bands", "rsi", 
                           "pattern_detection", "vwap", "trend_strength"]:
                return self.gpu.run_kernel(operation, **kwargs)
                
            # Custom pytorch operations
            elif operation == "normalize_tensor" and HAS_TORCH:
                tensor = kwargs.get("tensor")
                if tensor is not None:
                    mean = tensor.mean()
                    std = tensor.std()
                    return (tensor - mean) / (std + 1e-8)
                    
            elif operation == "fast_correlation" and HAS_TORCH:
                x = kwargs.get("x")
                y = kwargs.get("y")
                if x is not None and y is not None:
                    # Fast GPU correlation
                    x_norm = (x - x.mean()) / (x.std() + 1e-8)
                    y_norm = (y - y.mean()) / (y.std() + 1e-8)
                    return (x_norm * y_norm).mean()
                    
            self.logger.warning(f"Specialized operation '{operation}' not implemented")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in specialized operation '{operation}': {str(e)}")
            self.metrics.increment_counter("gpu_errors_count")
            return None
            
    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        self.gpu.clear_cache()
        
    def synchronize(self) -> None:
        """Synchronize GPU operations."""
        self.gpu.synchronize()
        
    def create_tensor(self, data: np.ndarray, dtype: str = "float32") -> Any:
        """Create a tensor on the GPU with the specified data and dtype."""
        if not self.gpu.has_gpu:
            return data
            
        try:
            if HAS_TORCH:
                # Map dtype string to torch dtype
                dtype_map = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "int32": torch.int32,
                    "int64": torch.int64,
                    "float64": torch.float64,
                    "bool": torch.bool
                }
                
                torch_dtype = dtype_map.get(dtype, torch.float32)
                return torch.tensor(data, dtype=torch_dtype, device=f"cuda:{self.gpu.selected_device}")
                
            elif HAS_GPU:
                # Map dtype string to numpy dtype
                dtype_map = {
                    "float32": np.float32,
                    "float16": np.float16,
                    "int32": np.int32,
                    "int64": np.int64,
                    "float64": np.float64,
                    "bool": np.bool_
                }
                
                np_dtype = dtype_map.get(dtype, np.float32)
                return cp.array(data, dtype=np_dtype)
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating tensor: {str(e)}")
            self.metrics.increment_counter("gpu_errors_count")
            return data
            
    def optimize_model_for_inference(self, model: Any) -> Any:
        """Optimize a PyTorch model for inference on GPU."""
        if not self.gpu.has_gpu or not HAS_TORCH:
            return model
            
        try:
            # Move model to GPU
            model = model.cuda(self.gpu.selected_device)
            
            # Set to evaluation mode
            model.eval()
            
            # Use half precision if supported
            if self.gpu.half_precision_supported and self.gpu.use_half_precision:
                model = model.half()
                
            # Torchscript compilation for faster inference
            try:
                # Try to script the model
                scripted_model = torch.jit.script(model)
                return scripted_model
            except Exception as script_error:
                self.logger.warning(f"Could not script model: {str(script_error)}")
                # Fall back to regular model
                return model
                
        except Exception as e:
            self.logger.error(f"Error optimizing model for inference: {str(e)}")
            self.metrics.increment_counter("gpu_errors_count")
            return model


def setup_gpu() -> bool:
    """Initialize the GPU manager and return availability status."""
    manager = GPUManager()
    return manager.has_gpu


def get_gpu_memory_usage() -> float:
    """Return current allocated GPU memory in MB if available."""
    manager = GPUManager()
    status = manager.memory_status()
    return status.get("allocated_mb", 0)
