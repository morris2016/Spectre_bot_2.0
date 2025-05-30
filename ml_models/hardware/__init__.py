#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
ML Hardware Acceleration Module

This module provides hardware optimization and acceleration for machine learning models,
focusing on GPU acceleration through CUDA, CPU optimization, and memory management to
ensure the system runs efficiently on the specified hardware (RTX 3050 with 8GB GPU RAM).
"""

import os
import sys
import logging
import platform
import gc
import numpy as np
import warnings
from typing import Dict, Optional, Union, List, Any, Tuple
import json
import psutil

# Conditionally import hardware-specific libraries
try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from feature_service.processor_utils import cudf, cp, HAS_GPU
CUPY_AVAILABLE = HAS_GPU

try:
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Internal imports
from common.logger import get_logger
from common.exceptions import HardwareError
from common.constants import GPU_MEMORY_LIMIT, GPU_MEMORY_GROWTH, DEFAULT_GPU_ID

logger = get_logger(__name__)

# Global state for hardware configuration
_hardware_config = {
    "gpu_enabled": False,
    "cuda_available": False,
    "gpu_id": DEFAULT_GPU_ID,
    "gpu_memory_limit": GPU_MEMORY_LIMIT,
    "gpu_memory_growth": GPU_MEMORY_GROWTH,
    "cpu_threads": os.cpu_count(),
    "memory_efficient": False,
    "hardware_info": {},
    "optimization_level": "auto"
}

def initialize_hardware(
    gpu_enabled: bool = True,
    gpu_id: int = DEFAULT_GPU_ID,
    gpu_memory_limit: Optional[int] = None,
    gpu_memory_growth: bool = GPU_MEMORY_GROWTH,
    cpu_threads: Optional[int] = None,
    memory_efficient: bool = False,
    optimization_level: str = "auto"
) -> Dict[str, Any]:
    """
    Initialize hardware settings for optimal ML performance.
    
    Args:
        gpu_enabled: Whether to enable GPU acceleration
        gpu_id: GPU device ID to use
        gpu_memory_limit: Memory limit for GPU in MB (None for no limit)
        gpu_memory_growth: Allow memory growth for TensorFlow
        cpu_threads: Number of CPU threads to use (None for all available)
        memory_efficient: Use memory-efficient algorithms
        optimization_level: Optimization level ("low", "medium", "high", "auto")
    
    Returns:
        Dictionary with hardware configuration information
    """
    global _hardware_config
    
    try:
        # Detect hardware capabilities
        hardware_info = detect_hardware()
        _hardware_config["hardware_info"] = hardware_info
        
        # Determine if CUDA is available
        cuda_available = detect_cuda_availability()
        _hardware_config["cuda_available"] = cuda_available
        
        # Configure GPU settings
        if gpu_enabled and cuda_available:
            _hardware_config["gpu_enabled"] = True
            _hardware_config["gpu_id"] = gpu_id
            
            # Set GPU memory limits
            if gpu_memory_limit is not None:
                _hardware_config["gpu_memory_limit"] = gpu_memory_limit
            else:
                # If no limit specified, use available GPU memory with a safety margin
                available_gpu_memory = get_available_gpu_memory(gpu_id)
                if available_gpu_memory:
                    # Use 80% of available memory as a safety margin
                    _hardware_config["gpu_memory_limit"] = int(available_gpu_memory * 0.8)
            
            _hardware_config["gpu_memory_growth"] = gpu_memory_growth
            
            # Apply GPU settings
            configure_gpu(
                gpu_id=gpu_id,
                memory_limit=_hardware_config["gpu_memory_limit"],
                memory_growth=gpu_memory_growth
            )
            
            logger.info(f"GPU acceleration enabled (Device {gpu_id})")
        else:
            _hardware_config["gpu_enabled"] = False
            if gpu_enabled and not cuda_available:
                logger.warning("GPU acceleration requested but CUDA is not available. Using CPU instead.")
            else:
                logger.info("Using CPU for computation")
        
        # Configure CPU settings
        if cpu_threads is not None:
            _hardware_config["cpu_threads"] = min(cpu_threads, os.cpu_count())
        else:
            _hardware_config["cpu_threads"] = os.cpu_count()
        
        # Set thread count for various libraries
        configure_cpu_threads(_hardware_config["cpu_threads"])
        
        # Configure memory efficiency settings
        _hardware_config["memory_efficient"] = memory_efficient
        if memory_efficient:
            logger.info("Memory-efficient mode enabled")
        
        # Set optimization level
        if optimization_level == "auto":
            # Auto-detect based on hardware capabilities
            if _hardware_config["gpu_enabled"] and _hardware_config["gpu_memory_limit"] >= 4000:
                _hardware_config["optimization_level"] = "high"
            elif _hardware_config["cpu_threads"] >= 8:
                _hardware_config["optimization_level"] = "medium"
            else:
                _hardware_config["optimization_level"] = "low"
        else:
            _hardware_config["optimization_level"] = optimization_level
        
        logger.info(f"Hardware initialization complete. Optimization level: {_hardware_config['optimization_level']}")
        return _hardware_config
        
    except Exception as e:
        logger.error(f"Error initializing hardware: {str(e)}")
        raise HardwareError(f"Failed to initialize hardware: {str(e)}")

def detect_hardware() -> Dict[str, Any]:
    """
    Detect and report hardware capabilities of the system.
    
    Returns:
        Dictionary with hardware information
    """
    hardware_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "gpu_available": False,
        "gpu_info": []
    }
    
    # Get GPU information if available
    if TORCH_AVAILABLE and torch.cuda.is_available():
        hardware_info["gpu_available"] = True
        hardware_info["cuda_version"] = torch.version.cuda
        
        gpu_count = torch.cuda.device_count()
        hardware_info["gpu_count"] = gpu_count
        
        for i in range(gpu_count):
            gpu_info = {
                "device_id": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory": torch.cuda.get_device_properties(i).total_memory,
            }
            hardware_info["gpu_info"].append(gpu_info)
    
    # TensorFlow GPU info as fallback
    elif TF_AVAILABLE:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            hardware_info["gpu_available"] = True
            hardware_info["gpu_count"] = len(gpus)
            
            for i, gpu in enumerate(gpus):
                gpu_info = {
                    "device_id": i,
                    "name": gpu.name,
                }
                hardware_info["gpu_info"].append(gpu_info)
    
    logger.debug(f"Hardware detection complete: {json.dumps(hardware_info, indent=2)}")
    return hardware_info

def detect_cuda_availability() -> bool:
    """
    Detect if CUDA is available for GPU acceleration.
    
    Returns:
        Boolean indicating if CUDA is available
    """
    cuda_available = False
    
    # Check PyTorch CUDA availability
    if TORCH_AVAILABLE:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(f"CUDA available through PyTorch (version {torch.version.cuda})")
            return True
    
    # Check TensorFlow GPU availability
    if TF_AVAILABLE:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            cuda_available = True
            logger.info(f"GPU devices available through TensorFlow: {len(gpus)}")
            return True
    
    # Check NUMBA CUDA availability
    if NUMBA_AVAILABLE:
        try:
            cuda_available = cuda.is_available()
            if cuda_available:
                logger.info("CUDA available through Numba")
                return True
        except:
            pass
    
    if not cuda_available:
        logger.warning("No CUDA-capable GPU detected")
    
    return cuda_available

def get_available_gpu_memory(gpu_id: int = 0) -> Optional[int]:
    """
    Get available GPU memory in MB.
    
    Args:
        gpu_id: GPU device ID
    
    Returns:
        Available memory in MB or None if unavailable
    """
    try:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # PyTorch method
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            reserved_memory = torch.cuda.memory_reserved(gpu_id)
            allocated_memory = torch.cuda.memory_allocated(gpu_id)
            available_memory = (total_memory - reserved_memory - allocated_memory) / (1024 * 1024)  # Convert to MB
            return int(available_memory)
        
        # Fallback to nvidia-smi through subprocess
        elif shutil.which('nvidia-smi'):
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader', '-i', str(gpu_id)],
                encoding='utf-8',
                capture_output=True,
                check=True
            )
            return int(result.stdout.strip())
    
    except Exception as e:
        logger.warning(f"Failed to get available GPU memory: {str(e)}")
    
    return None

def configure_gpu(
    gpu_id: int = 0,
    memory_limit: Optional[int] = None,
    memory_growth: bool = True
) -> None:
    """
    Configure GPU settings for optimal performance.
    
    Args:
        gpu_id: GPU device ID to use
        memory_limit: Memory limit in MB (None for no limit)
        memory_growth: Allow memory growth for TensorFlow
    """
    # Set CUDA device for PyTorch
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            torch.cuda.set_device(gpu_id)
            logger.debug(f"Set PyTorch CUDA device to {gpu_id}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Failed to set PyTorch CUDA device: {str(e)}")
    
    # Configure TensorFlow GPU settings
    if TF_AVAILABLE:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                # Set visible devices
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                
                # Configure memory growth
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                    logger.debug(f"Enabled memory growth for TensorFlow GPU {gpu_id}")
                
                # Set memory limit
                if memory_limit:
                    tf.config.set_logical_device_configuration(
                        gpus[gpu_id],
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
                    logger.debug(f"Set TensorFlow GPU memory limit to {memory_limit}MB")
        except Exception as e:
            logger.warning(f"Failed to configure TensorFlow GPU: {str(e)}")
    
    # Set CUDA_VISIBLE_DEVICES environment variable for other libraries
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    logger.info(f"GPU {gpu_id} configured for machine learning acceleration")

def configure_cpu_threads(num_threads: int) -> None:
    """
    Configure thread count for various libraries.
    
    Args:
        num_threads: Number of threads to use
    """
    # Set OpenMP thread count (used by scikit-learn, XGBoost, etc.)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    
    # Configure NumPy threading
    try:
        import numpy as np
        np.set_num_threads(num_threads)
    except:
        pass
    
    # Configure PyTorch CPU threading
    if TORCH_AVAILABLE:
        try:
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
        except:
            pass
    
    logger.info(f"CPU thread count configured to {num_threads} threads")

def get_hardware_config() -> Dict[str, Any]:
    """
    Get current hardware configuration.
    
    Returns:
        Dictionary with current hardware configuration
    """
    return _hardware_config

def cleanup_gpu_memory() -> None:
    """
    Clean up GPU memory to prevent memory leaks.
    """
    if _hardware_config["gpu_enabled"]:
        # PyTorch cleanup
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.debug("PyTorch GPU memory cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear PyTorch GPU memory: {str(e)}")
        
        # TensorFlow cleanup
        if TF_AVAILABLE:
            try:
                if hasattr(tf.keras.backend, 'clear_session'):
                    tf.keras.backend.clear_session()
                    logger.debug("TensorFlow session cleared")
            except Exception as e:
                logger.warning(f"Failed to clear TensorFlow session: {str(e)}")
        
        # CuPy cleanup
        if HAS_GPU:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                logger.debug("CuPy memory pool cleared")
            except Exception as e:
                logger.warning(f"Failed to clear CuPy memory: {str(e)}")
        
        # General garbage collection
        gc.collect()
        logger.debug("General garbage collection performed")

def optimize_tensor_operations(
    array: Union[np.ndarray, 'torch.Tensor', 'tf.Tensor'],
    operation: str = "default"
) -> Union[np.ndarray, 'torch.Tensor', 'tf.Tensor']:
    """
    Optimize tensor operations based on available hardware.
    
    Args:
        array: Input array or tensor
        operation: Operation type for specialized handling
    
    Returns:
        The input tensor, potentially moved to the appropriate device
    """
    # Return early if not GPU-enabled
    if not _hardware_config["gpu_enabled"]:
        return array
    
    try:
        # Handle PyTorch tensors
        if TORCH_AVAILABLE and isinstance(array, torch.Tensor):
            device = torch.device(f"cuda:{_hardware_config['gpu_id']}" if torch.cuda.is_available() else "cpu")
            return array.to(device)
        
        # Handle TensorFlow tensors
        elif TF_AVAILABLE and isinstance(array, tf.Tensor):
            # TensorFlow manages device placement automatically
            return array
        
        # Handle NumPy arrays
        elif isinstance(array, np.ndarray):
            if HAS_GPU and _hardware_config["gpu_enabled"]:
                # Convert to CuPy array for GPU acceleration
                return cp.asarray(array)
            return array
        
    except Exception as e:
        logger.warning(f"Failed to optimize tensor operation: {str(e)}")
        # Return original array on error
        return array
    
    # Default case - return the original array
    return array

def get_recommended_batch_size(
    input_shape: Tuple[int, ...],
    model_complexity: str = "medium",
    dtype: str = "float32"
) -> int:
    """
    Calculate recommended batch size based on GPU memory and model complexity.
    
    Args:
        input_shape: Shape of input data (excluding batch dimension)
        model_complexity: Complexity of the model ("low", "medium", "high")
        dtype: Data type of the input data
    
    Returns:
        Recommended batch size
    """
    if not _hardware_config["gpu_enabled"]:
        # CPU-based recommendation
        return 64  # Default conservative batch size for CPU
    
    try:
        # Get element size in bytes based on dtype
        dtype_sizes = {
            "float16": 2,
            "float32": 4,
            "float64": 8,
            "int8": 1,
            "int16": 2,
            "int32": 4,
            "int64": 8
        }
        element_size = dtype_sizes.get(dtype, 4)  # Default to float32 (4 bytes)
        
        # Calculate input size
        input_elements = np.prod(input_shape)
        input_size = input_elements * element_size
        
        # Model memory multipliers based on complexity
        # This accounts for model parameters, activations, gradients, optimizer states
        complexity_multipliers = {
            "low": 10,      # Simple models (linear, shallow trees)
            "medium": 50,   # Medium models (CNNs, RNNs)
            "high": 100     # Complex models (Transformers, large CNNs)
        }
        multiplier = complexity_multipliers.get(model_complexity, 50)
        
        # Get available GPU memory (in bytes)
        available_memory = _hardware_config["gpu_memory_limit"] * 1024 * 1024  # Convert MB to bytes
        
        # Apply a safety factor to account for memory fragmentation and overhead
        safety_factor = 0.8
        available_memory *= safety_factor
        
        # Calculate batch size
        max_batch_size = int(available_memory / (input_size * multiplier))
        
        # Ensure batch size is at least 1
        batch_size = max(1, max_batch_size)
        
        # Round to power of 2 for better performance
        batch_size = 2 ** int(np.log2(batch_size))
        
        return batch_size
        
    except Exception as e:
        logger.warning(f"Failed to calculate recommended batch size: {str(e)}")
        # Return a conservative default
        return 32

def is_gpu_suitable_for_model(
    model_type: str,
    input_shape: Optional[Tuple[int, ...]] = None,
    parameter_count: Optional[int] = None
) -> bool:
    """
    Determine if the GPU is suitable for a given model.
    
    Args:
        model_type: Type of model
        input_shape: Shape of input data
        parameter_count: Number of model parameters
    
    Returns:
        Boolean indicating if GPU is suitable
    """
    if not _hardware_config["gpu_enabled"]:
        return False
    
    # Get GPU memory
    gpu_memory_mb = _hardware_config["gpu_memory_limit"]
    
    # Memory requirements for different model types
    memory_requirements = {
        "linear": 100,  # Small linear models
        "tree": 200,    # Decision trees, random forests
        "xgboost": 500, # XGBoost models
        "lightgbm": 500, # LightGBM models
        "catboost": 800, # CatBoost models
        "neural_network_small": 1000,  # Small neural networks
        "neural_network_medium": 2000, # Medium neural networks
        "neural_network_large": 4000,  # Large neural networks
        "rnn": 2000,    # Recurrent neural networks
        "lstm": 3000,   # LSTM networks
        "gru": 2500,    # GRU networks
        "cnn_small": 1500, # Small convolutional networks
        "cnn_large": 4000, # Large convolutional networks
        "transformer_small": 3000, # Small transformer models
        "transformer_large": 6000, # Large transformer models
    }
    
    # Get memory requirement for the model type
    base_requirement = memory_requirements.get(model_type, 1000)  # Default 1GB if unknown
    
    # Adjust based on parameter count if provided
    if parameter_count is not None:
        # Rough estimation: each parameter needs storage for value, gradient, and optimizer state
        # 4 bytes per float32 parameter × 3 (param, gradient, optimizer) = 12 bytes per parameter
        param_memory_mb = (parameter_count * 12) / (1024 * 1024)
        base_requirement = max(base_requirement, param_memory_mb)
    
    # Adjust based on input shape if provided
    if input_shape is not None:
        input_size = np.prod(input_shape) * 4  # Assuming float32
        # Multiple by a factor to account for intermediate activations
        activation_factor = 10
        input_memory_mb = (input_size * activation_factor) / (1024 * 1024)
        base_requirement += input_memory_mb
    
    # Add a safety margin
    required_memory = base_requirement * 1.5
    
    # Check if GPU has enough memory
    has_enough_memory = gpu_memory_mb >= required_memory
    
    if not has_enough_memory:
        logger.warning(
            f"GPU memory ({gpu_memory_mb}MB) may be insufficient for {model_type} "
            f"model requiring approximately {required_memory:.1f}MB"
        )
    
    return has_enough_memory

def monitor_hardware_usage() -> Dict[str, Any]:
    """
    Monitor current hardware resource usage.
    
    Returns:
        Dictionary with resource usage information
    """
    usage = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used": psutil.virtual_memory().used,
        "memory_available": psutil.virtual_memory().available,
        "gpu_utilization": None,
        "gpu_memory_used": None,
        "gpu_memory_total": None
    }
    
    # Get GPU information if available
    if _hardware_config["gpu_enabled"]:
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_id = _hardware_config["gpu_id"]
                # Get GPU memory usage
                usage["gpu_memory_total"] = torch.cuda.get_device_properties(gpu_id).total_memory
                usage["gpu_memory_reserved"] = torch.cuda.memory_reserved(gpu_id)
                usage["gpu_memory_allocated"] = torch.cuda.memory_allocated(gpu_id)
                usage["gpu_memory_used"] = usage["gpu_memory_reserved"]
                
                # Calculate utilization percentage
                if usage["gpu_memory_total"] > 0:
                    usage["gpu_utilization"] = (usage["gpu_memory_used"] / usage["gpu_memory_total"]) * 100
        except Exception as e:
            logger.warning(f"Failed to get GPU usage information: {str(e)}")
    
    return usage

# Module initialization
logger.info("ML hardware module imported")

# Import submodules for easier access
if TORCH_AVAILABLE:
    from . import gpu

# Make hardware config available at module level
hardware_config = _hardware_config

