#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
FPGA Acceleration Module

This module provides FPGA acceleration for time-critical trading operations,
using PyOpenCL for FPGA interaction. It enables ultra-low latency processing
for specific trading algorithms through hardware acceleration.
"""

import os
import logging
import numpy as np
import time
import threading
import queue
from typing import Dict, List, Union, Tuple, Optional, Any, Callable

# Conditional imports for FPGA support
try:
    import pyopencl as cl
    import pyopencl.array
    HAS_PYOPENCL = True
except ImportError:
    HAS_PYOPENCL = False

try:
    from pynq import Overlay
    HAS_PYNQ = True
except ImportError:
    HAS_PYNQ = False

# Local imports
from common.logger import get_logger
from common.exceptions import HardwareError, ResourceError
from common.metrics import MetricsCollector
from ml_models.hardware.base import HardwareAccelerator

logger = get_logger(__name__)
metrics = MetricsCollector.get_instance()


class FPGAManager:
    """Manages FPGA resources and handles acceleration of specific trading algorithms."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FPGAManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = get_logger(f"{__name__}.FPGAManager")
        self.metrics = MetricsCollector.get_instance()
        
        # Default values
        self.has_fpga = False
        self.device_type = None  # OpenCL or PYNQ
        self.platform = None
        self.device = None
        self.context = None
        self.queue = None
        self.program = None
        self.kernels = {}
        self.bitstream_path = None
        self.overlay = None
        
        # Task queue for asynchronous processing
        self.task_queue = queue.Queue()
        self.result_queues = {}
        self.worker_thread = None
        self.shutdown_flag = False
        
        # Trading algorithm implementations
        self.trading_algorithms = {}
        
        # Initialize FPGA environment
        self._detect_fpga()
        if self.has_fpga:
            self._setup_device()
            self._init_kernels()
            self._start_worker_thread()
            
        self._initialized = True
        
        self.logger.info(
            f"FPGA Manager initialized: {self.device_type} device"
            if self.has_fpga else "No FPGA device detected"
        )
        
    def _detect_fpga(self) -> None:
        """Detect available FPGA devices using PyOpenCL or PYNQ."""
        try:
            # First try OpenCL-based FPGAs
            if HAS_PYOPENCL:
                platforms = cl.get_platforms()
                for platform in platforms:
                    for device in platform.get_devices():
                        # Check if device is FPGA
                        if ("fpga" in platform.name.lower() or 
                            "fpga" in device.name.lower() or
                            "xilinx" in platform.name.lower() or
                            "intel" in platform.name.lower() and "fpga" in device.name.lower()):
                            
                            self.has_fpga = True
                            self.device_type = "OpenCL"
                            self.platform = platform
                            self.device = device
                            
                            self.logger.info(f"Detected OpenCL FPGA: {platform.name} - {device.name}")
                            break
                            
                    if self.has_fpga:
                        break
                        
            # If no OpenCL FPGA, try PYNQ
            if not self.has_fpga and HAS_PYNQ:
                # Check for PYNQ bitstream
                default_bitstream_paths = [
                    "./bitstreams/trading_algorithms.bit",
                    "/home/xilinx/trading_algorithms.bit",
                    os.path.join(os.path.dirname(__file__), "../bitstreams/trading_algorithms.bit")
                ]
                
                for path in default_bitstream_paths:
                    if os.path.exists(path):
                        self.bitstream_path = path
                        self.has_fpga = True
                        self.device_type = "PYNQ"
                        
                        self.logger.info(f"Detected PYNQ FPGA with bitstream: {path}")
                        break
                        
            if not self.has_fpga:
                self.logger.info("No FPGA devices detected")
                
        except Exception as e:
            self.logger.error(f"Error detecting FPGA devices: {str(e)}")
            self.has_fpga = False
    
    def _setup_device(self) -> None:
        """Set up the FPGA device for trading operations."""
        try:
            if not self.has_fpga:
                return
                
            if self.device_type == "OpenCL":
                # Create OpenCL context and queue
                self.context = cl.Context([self.device])
                self.queue = cl.CommandQueue(self.context)
                
                self.logger.info(f"Set up OpenCL FPGA: {self.device.name}")
                
            elif self.device_type == "PYNQ":
                # Load the overlay (bitstream)
                self.overlay = Overlay(self.bitstream_path)
                
                self.logger.info(f"Loaded PYNQ overlay from {self.bitstream_path}")
                
                # Log available IP cores
                if hasattr(self.overlay, 'ip_dict'):
                    ip_cores = list(self.overlay.ip_dict.keys())
                    self.logger.info(f"Available IP cores: {', '.join(ip_cores)}")
                    
        except Exception as e:
            self.logger.error(f"Error setting up FPGA device: {str(e)}")
            self.has_fpga = False
            raise HardwareError(f"FPGA setup failed: {str(e)}")
    
    def _init_kernels(self) -> None:
        """Initialize OpenCL kernels or PYNQ IP cores for trading algorithms."""
        if not self.has_fpga:
            return
            
        try:
            if self.device_type == "OpenCL":
                # Load kernel sources
                kernel_source_path = os.path.join(os.path.dirname(__file__), "kernels/trading_kernels.cl")
                
                if not os.path.exists(kernel_source_path):
                    # Create a basic kernel for testing if none exists
                    kernel_source = """
                    __kernel void moving_average(
                        __global const float* input,
                        __global float* output,
                        const unsigned int window_size,
                        const unsigned int data_size
                    ) {
                        int gid = get_global_id(0);
                        if (gid < data_size) {
                            float sum = 0.0f;
                            int count = 0;
                            int start_idx = (gid >= window_size) ? (gid - window_size + 1) : 0;
                            
                            for (int i = start_idx; i <= gid; i++) {
                                sum += input[i];
                                count++;
                            }
                            
                            output[gid] = sum / count;
                        }
                    }
                    
                    __kernel void exponential_moving_average(
                        __global const float* input,
                        __global float* output,
                        const float alpha,
                        const unsigned int data_size
                    ) {
                        int gid = get_global_id(0);
                        if (gid < data_size) {
                            if (gid == 0) {
                                output[gid] = input[gid];
                            } else {
                                output[gid] = alpha * input[gid] + (1.0f - alpha) * output[gid-1];
                            }
                        }
                    }
                    
                    __kernel void macd(
                        __global const float* input,
                        __global float* fast_ema,
                        __global float* slow_ema,
                        __global float* macd_line,
                        __global float* signal_line,
                        __global float* histogram,
                        const float fast_alpha,
                        const float slow_alpha,
                        const float signal_alpha,
                        const unsigned int data_size
                    ) {
                        int gid = get_global_id(0);
                        if (gid < data_size) {
                            // Calculate EMAs
                            if (gid == 0) {
                                fast_ema[gid] = input[gid];
                                slow_ema[gid] = input[gid];
                                macd_line[gid] = 0.0f;
                                signal_line[gid] = 0.0f;
                                histogram[gid] = 0.0f;
                            } else {
                                fast_ema[gid] = fast_alpha * input[gid] + (1.0f - fast_alpha) * fast_ema[gid-1];
                                slow_ema[gid] = slow_alpha * input[gid] + (1.0f - slow_alpha) * slow_ema[gid-1];
                                
                                // MACD line = fast EMA - slow EMA
                                macd_line[gid] = fast_ema[gid] - slow_ema[gid];
                                
                                // Signal line = EMA of MACD line
                                if (gid == 1) {
                                    signal_line[gid] = macd_line[gid];
                                } else {
                                    signal_line[gid] = signal_alpha * macd_line[gid] + (1.0f - signal_alpha) * signal_line[gid-1];
                                }
                                
                                // Histogram = MACD line - signal line
                                histogram[gid] = macd_line[gid] - signal_line[gid];
                            }
                        }
                    }
                    
                    __kernel void market_order_evaluation(
                        __global const float* prices,
                        __global const float* indicators,
                        __global const float* thresholds,
                        __global int* signals,
                        const unsigned int data_size,
                        const unsigned int indicator_count
                    ) {
                        int gid = get_global_id(0);
                        if (gid < data_size) {
                            int buy_signals = 0;
                            int sell_signals = 0;
                            
                            for (int i = 0; i < indicator_count; i++) {
                                float indicator_value = indicators[gid * indicator_count + i];
                                float buy_threshold = thresholds[i * 2];
                                float sell_threshold = thresholds[i * 2 + 1];
                                
                                if (indicator_value > buy_threshold) {
                                    buy_signals++;
                                } else if (indicator_value < sell_threshold) {
                                    sell_signals++;
                                }
                            }
                            
                            // Decision logic - simple majority voting
                            if (buy_signals > sell_signals && buy_signals > indicator_count / 3) {
                                signals[gid] = 1;  // Buy signal
                            } else if (sell_signals > buy_signals && sell_signals > indicator_count / 3) {
                                signals[gid] = -1;  // Sell signal
                            } else {
                                signals[gid] = 0;  // No signal
                            }
                        }
                    }
                    """
                else:
                    with open(kernel_source_path, 'r') as f:
                        kernel_source = f.read()
                        
                # Build program
                self.program = cl.Program(self.context, kernel_source).build()
                
                # Register kernels
                self.kernels["moving_average"] = self.program.moving_average
                self.kernels["exponential_moving_average"] = self.program.exponential_moving_average
                self.kernels["macd"] = self.program.macd
                self.kernels["market_order_evaluation"] = self.program.market_order_evaluation
                
                self.logger.info(f"Initialized OpenCL kernels: {', '.join(self.kernels.keys())}")
                
            elif self.device_type == "PYNQ":
                # Register IP cores as trading algorithms
                if hasattr(self.overlay, 'moving_average'):
                    self.trading_algorithms["moving_average"] = self._pynq_moving_average
                    
                if hasattr(self.overlay, 'ema'):
                    self.trading_algorithms["exponential_moving_average"] = self._pynq_ema
                    
                if hasattr(self.overlay, 'macd'):
                    self.trading_algorithms["macd"] = self._pynq_macd
                    
                if hasattr(self.overlay, 'market_eval'):
                    self.trading_algorithms["market_order_evaluation"] = self._pynq_market_eval
                    
                self.logger.info(f"Initialized PYNQ IP cores: {', '.join(self.trading_algorithms.keys())}")
                
        except Exception as e:
            self.logger.error(f"Error initializing FPGA kernels: {str(e)}")
            self.logger.info("Will fall back to CPU implementations")
            
    def _start_worker_thread(self) -> None:
        """Start a worker thread for asynchronous FPGA processing."""
        if not self.has_fpga:
            return
            
        self.shutdown_flag = False
        self.worker_thread = threading.Thread(target=self._process_task_queue, daemon=True)
        self.worker_thread.start()
        
        self.logger.debug("Started FPGA worker thread for asynchronous processing")
        
    def _process_task_queue(self) -> None:
        """Worker thread function that processes the task queue."""
        while not self.shutdown_flag:
            try:
                # Get task with 0.1 second timeout
                task_id, algorithm, args, kwargs = self.task_queue.get(timeout=0.1)
                
                try:
                    # Execute the algorithm
                    if algorithm in self.trading_algorithms:
                        result = self.trading_algorithms[algorithm](*args, **kwargs)
                    else:
                        result = self._execute_algorithm(algorithm, *args, **kwargs)
                        
                    # Put result in result queue
                    if task_id in self.result_queues:
                        self.result_queues[task_id].put((True, result))
                        
                except Exception as e:
                    # Put error in result queue
                    if task_id in self.result_queues:
                        self.result_queues[task_id].put((False, str(e)))
                        
                finally:
                    # Mark task as done
                    self.task_queue.task_done()
                    
            except queue.Empty:
                # Queue is empty, just continue
                continue
                
            except Exception as e:
                self.logger.error(f"Error in FPGA worker thread: {str(e)}")
                
        self.logger.debug("FPGA worker thread terminated")
        
    def shutdown(self) -> None:
        """Shutdown the FPGA manager and worker thread."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.shutdown_flag = True
            self.worker_thread.join(timeout=2.0)
            
        self.logger.info("FPGA Manager shutdown")
        
    def execute_algorithm(self, algorithm: str, *args, **kwargs) -> Any:
        """Execute a trading algorithm on the FPGA synchronously."""
        if not self.has_fpga:
            return None
            
        try:
            if self.device_type == "OpenCL":
                return self._execute_algorithm(algorithm, *args, **kwargs)
            elif self.device_type == "PYNQ" and algorithm in self.trading_algorithms:
                return self.trading_algorithms[algorithm](*args, **kwargs)
            else:
                self.logger.warning(f"Algorithm '{algorithm}' not implemented on FPGA")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing algorithm '{algorithm}': {str(e)}")
            return None
            
    def execute_algorithm_async(self, algorithm: str, *args, **kwargs) -> str:
        """Execute a trading algorithm on the FPGA asynchronously."""
        if not self.has_fpga:
            return None
            
        try:
            # Generate a task ID
            task_id = f"task_{time.time()}_{id(algorithm)}"
            
            # Create a result queue for this task
            self.result_queues[task_id] = queue.Queue()
            
            # Add task to queue
            self.task_queue.put((task_id, algorithm, args, kwargs))
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error queueing algorithm '{algorithm}': {str(e)}")
            return None
            
    def get_result(self, task_id: str, timeout: float = None) -> Tuple[bool, Any]:
        """Get the result of an asynchronous task."""
        if task_id not in self.result_queues:
            return (False, "Unknown task ID")
            
        try:
            # Wait for result
            success, result = self.result_queues[task_id].get(timeout=timeout)
            
            # Clean up
            self.result_queues[task_id].task_done()
            del self.result_queues[task_id]
            
            return (success, result)
            
        except queue.Empty:
            return (False, "Timeout waiting for result")
            
        except Exception as e:
            self.logger.error(f"Error getting result for task {task_id}: {str(e)}")
            return (False, str(e))
            
    def _execute_algorithm(self, algorithm: str, *args, **kwargs) -> Any:
        """Execute an OpenCL kernel for the specified algorithm."""
        if algorithm not in self.kernels:
            raise ValueError(f"Algorithm '{algorithm}' not implemented")
            
        kernel = self.kernels[algorithm]
        
        if algorithm == "moving_average":
            # Unpack arguments
            input_data = kwargs.get('input_data')
            window_size = kwargs.get('window_size', 20)
            
            if input_data is None:
                raise ValueError("input_data is required")
                
            # Convert to numpy array if not already
            if not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data, dtype=np.float32)
                
            # Create output buffer
            output_data = np.zeros_like(input_data, dtype=np.float32)
            
            # Create OpenCL buffers
            input_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_data)
            output_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, output_data.nbytes)
            
            # Execute kernel
            kernel(self.queue, (len(input_data),), None, input_buf, output_buf, np.uint32(window_size), np.uint32(len(input_data)))
            
            # Read result
            cl.enqueue_copy(self.queue, output_data, output_buf)
            
            return output_data
            
        elif algorithm == "exponential_moving_average":
            # Unpack arguments
            input_data = kwargs.get('input_data')
            alpha = kwargs.get('alpha', 0.2)
            
            if input_data is None:
                raise ValueError("input_data is required")
                
            # Convert to numpy array if not already
            if not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data, dtype=np.float32)
                
            # Create output buffer
            output_data = np.zeros_like(input_data, dtype=np.float32)
            
            # Create OpenCL buffers
            input_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_data)
            output_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, output_data.nbytes)
            
            # Execute kernel
            kernel(self.queue, (len(input_data),), None, input_buf, output_buf, np.float32(alpha), np.uint32(len(input_data)))
            
            # Read result
            cl.enqueue_copy(self.queue, output_data, output_buf)
            
            return output_data
            
        elif algorithm == "macd":
            # Unpack arguments
            input_data = kwargs.get('input_data')
            fast_period = kwargs.get('fast_period', 12)
            slow_period = kwargs.get('slow_period', 26)
            signal_period = kwargs.get('signal_period', 9)
            
            if input_data is None:
                raise ValueError("input_data is required")
                
            # Convert to numpy array if not already
            if not isinstance(input_data, np.ndarray):
                input_data = np.array(input_data, dtype=np.float32)
                
            # Calculate alpha values
            fast_alpha = 2.0 / (fast_period + 1)
            slow_alpha = 2.0 / (slow_period + 1)
            signal_alpha = 2.0 / (signal_period + 1)
            
            # Create output buffers
            fast_ema = np.zeros_like(input_data, dtype=np.float32)
            slow_ema = np.zeros_like(input_data, dtype=np.float32)
            macd_line = np.zeros_like(input_data, dtype=np.float32)
            signal_line = np.zeros_like(input_data, dtype=np.float32)
            histogram = np.zeros_like(input_data, dtype=np.float32)
            
            # Create OpenCL buffers
            input_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_data)
            fast_ema_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, fast_ema.nbytes)
            slow_ema_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, slow_ema.nbytes)
            macd_line_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, macd_line.nbytes)
            signal_line_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, signal_line.nbytes)
            histogram_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, histogram.nbytes)
            
            # Execute kernel
            kernel(
                self.queue, (len(input_data),), None,
                input_buf, fast_ema_buf, slow_ema_buf, macd_line_buf, signal_line_buf, histogram_buf,
                np.float32(fast_alpha), np.float32(slow_alpha), np.float32(signal_alpha),
                np.uint32(len(input_data))
            )
            
            # Read results
            cl.enqueue_copy(self.queue, fast_ema, fast_ema_buf)
            cl.enqueue_copy(self.queue, slow_ema, slow_ema_buf)
            cl.enqueue_copy(self.queue, macd_line, macd_line_buf)
            cl.enqueue_copy(self.queue, signal_line, signal_line_buf)
            cl.enqueue_copy(self.queue, histogram, histogram_buf)
            
            return {
                'fast_ema': fast_ema,
                'slow_ema': slow_ema,
                'macd_line': macd_line,
                'signal_line': signal_line,
                'histogram': histogram
            }
            
        elif algorithm == "market_order_evaluation":
            # Unpack arguments
            prices = kwargs.get('prices')
            indicators = kwargs.get('indicators')
            thresholds = kwargs.get('thresholds')
            
            if prices is None or indicators is None or thresholds is None:
                raise ValueError("prices, indicators, and thresholds are required")
                
            # Convert to numpy arrays if not already
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices, dtype=np.float32)
                
            if not isinstance(indicators, np.ndarray):
                indicators = np.array(indicators, dtype=np.float32)
                
            if not isinstance(thresholds, np.ndarray):
                thresholds = np.array(thresholds, dtype=np.float32)
                
            # Check shapes
            data_size = len(prices)
            indicator_count = indicators.shape[1] if len(indicators.shape) > 1 else 1
            
            # Ensure indicators is 2D
            if len(indicators.shape) == 1:
                indicators = indicators.reshape(data_size, 1)
                
            # Create output buffer
            signals = np.zeros(data_size, dtype=np.int32)
            
            # Create OpenCL buffers
            prices_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=prices)
            indicators_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=indicators)
            thresholds_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=thresholds)
            signals_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, signals.nbytes)
            
            # Execute kernel
            kernel(
                self.queue, (data_size,), None,
                prices_buf, indicators_buf, thresholds_buf, signals_buf,
                np.uint32(data_size), np.uint32(indicator_count)
            )
            
            # Read result
            cl.enqueue_copy(self.queue, signals, signals_buf)
            
            return signals
            
        else:
            raise ValueError(f"Algorithm '{algorithm}' implementation not defined")
            
    # PYNQ IP core implementations
    def _pynq_moving_average(self, *args, **kwargs) -> np.ndarray:
        """Execute moving average using PYNQ IP core."""
        # Unpack arguments
        input_data = kwargs.get('input_data')
        window_size = kwargs.get('window_size', 20)
        
        if input_data is None:
            raise ValueError("input_data is required")
            
        # Convert to numpy array if not already
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)
            
        # Get IP core
        ma_ip = self.overlay.moving_average
        
        # Create output buffer
        output_data = np.zeros_like(input_data, dtype=np.float32)
        
        # Configure IP core
        ma_ip.write(0x10, window_size)  # Assuming register address 0x10 for window size
        ma_ip.write(0x18, len(input_data))  # Assuming register address 0x18 for data size
        
        # Write input data to IP core memory
        for i, value in enumerate(input_data):
            ma_ip.write(0x20 + i*4, int(value.view(np.int32)))  # Assuming register address 0x20 for input data
            
        # Start processing
        ma_ip.write(0x00, 0x01)  # Assuming register address 0x00 for control register, 0x01 to start
        
        # Wait for completion
        while (ma_ip.read(0x00) & 0x02) == 0:  # Assuming bit 1 indicates completion
            time.sleep(0.001)
            
        # Read output data
        for i in range(len(output_data)):
            int_val = ma_ip.read(0x40 + i*4)  # Assuming register address 0x40 for output data
            output_data[i] = np.float32(int_val).view(np.float32)
            
        return output_data
        
    def _pynq_ema(self, *args, **kwargs) -> np.ndarray:
        """Execute exponential moving average using PYNQ IP core."""
        # Unpack arguments
        input_data = kwargs.get('input_data')
        alpha = kwargs.get('alpha', 0.2)
        
        if input_data is None:
            raise ValueError("input_data is required")
            
        # Convert to numpy array if not already
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)
            
        # Get IP core
        ema_ip = self.overlay.ema
        
        # Create output buffer
        output_data = np.zeros_like(input_data, dtype=np.float32)
        
        # Convert alpha to fixed point representation (assuming Q8.24 format)
        alpha_fixed = int(alpha * (1 << 24))
        
        # Configure IP core
        ema_ip.write(0x10, alpha_fixed)  # Assuming register address 0x10 for alpha
        ema_ip.write(0x18, len(input_data))  # Assuming register address 0x18 for data size
        
        # Write input data to IP core memory
        for i, value in enumerate(input_data):
            ema_ip.write(0x20 + i*4, int(value.view(np.int32)))
            
        # Start processing
        ema_ip.write(0x00, 0x01)
        
        # Wait for completion
        while (ema_ip.read(0x00) & 0x02) == 0:
            time.sleep(0.001)
            
        # Read output data
        for i in range(len(output_data)):
            int_val = ema_ip.read(0x40 + i*4)
            output_data[i] = np.float32(int_val).view(np.float32)
            
        return output_data
        
    def _pynq_macd(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """Execute MACD calculation using PYNQ IP core."""
        # Unpack arguments
        input_data = kwargs.get('input_data')
        fast_period = kwargs.get('fast_period', 12)
        slow_period = kwargs.get('slow_period', 26)
        signal_period = kwargs.get('signal_period', 9)
        
        if input_data is None:
            raise ValueError("input_data is required")
            
        # Convert to numpy array if not already
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data, dtype=np.float32)
            
        # Get IP core
        macd_ip = self.overlay.macd
        
        # Create output buffers
        fast_ema = np.zeros_like(input_data, dtype=np.float32)
        slow_ema = np.zeros_like(input_data, dtype=np.float32)
        macd_line = np.zeros_like(input_data, dtype=np.float32)
        signal_line = np.zeros_like(input_data, dtype=np.float32)
        histogram = np.zeros_like(input_data, dtype=np.float32)
        
        # Calculate alpha values (convert to fixed point, assuming Q8.24 format)
        fast_alpha = int((2.0 / (fast_period + 1)) * (1 << 24))
        slow_alpha = int((2.0 / (slow_period + 1)) * (1 << 24))
        signal_alpha = int((2.0 / (signal_period + 1)) * (1 << 24))
        
        # Configure IP core
        macd_ip.write(0x10, fast_alpha)
        macd_ip.write(0x14, slow_alpha)
        macd_ip.write(0x18, signal_alpha)
        macd_ip.write(0x1C, len(input_data))
        
        # Write input data to IP core memory
        for i, value in enumerate(input_data):
            macd_ip.write(0x20 + i*4, int(value.view(np.int32)))
            
        # Start processing
        macd_ip.write(0x00, 0x01)
        
        # Wait for completion
        while (macd_ip.read(0x00) & 0x02) == 0:
            time.sleep(0.001)
            
        # Read output data
        for i in range(len(input_data)):
            fast_ema[i] = np.float32(macd_ip.read(0x40 + i*4)).view(np.float32)
            slow_ema[i] = np.float32(macd_ip.read(0x60 + i*4)).view(np.float32)
            macd_line[i] = np.float32(macd_ip.read(0x80 + i*4)).view(np.float32)
            signal_line[i] = np.float32(macd_ip.read(0xA0 + i*4)).view(np.float32)
            histogram[i] = np.float32(macd_ip.read(0xC0 + i*4)).view(np.float32)
            
        return {
            'fast_ema': fast_ema,
            'slow_ema': slow_ema,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
        
    def _pynq_market_eval(self, *args, **kwargs) -> np.ndarray:
        """Execute market order evaluation using PYNQ IP core."""
        # Unpack arguments
        prices = kwargs.get('prices')
        indicators = kwargs.get('indicators')
        thresholds = kwargs.get('thresholds')
        
        if prices is None or indicators is None or thresholds is None:
            raise ValueError("prices, indicators, and thresholds are required")
            
        # Convert to numpy arrays if not already
        if not isinstance(prices, np.ndarray):
            prices = np.array(prices, dtype=np.float32)
            
        if not isinstance(indicators, np.ndarray):
            indicators = np.array(indicators, dtype=np.float32)
            
        if not isinstance(thresholds, np.ndarray):
            thresholds = np.array(thresholds, dtype=np.float32)
            
        # Check shapes
        data_size = len(prices)
        indicator_count = indicators.shape[1] if len(indicators.shape) > 1 else 1
        
        # Ensure indicators is 2D
        if len(indicators.shape) == 1:
            indicators = indicators.reshape(data_size, 1)
            
        # Get IP core
        eval_ip = self.overlay.market_eval
        
        # Create output buffer
        signals = np.zeros(data_size, dtype=np.int32)
        
        # Configure IP core
        eval_ip.write(0x10, data_size)
        eval_ip.write(0x14, indicator_count)
        
        # Write prices to IP core memory
        for i, value in enumerate(prices):
            eval_ip.write(0x20 + i*4, int(value.view(np.int32)))
            
        # Write indicators to IP core memory
        for i in range(data_size):
            for j in range(indicator_count):
                eval_ip.write(0x40 + (i*indicator_count + j)*4, int(indicators[i, j].view(np.int32)))
                
        # Write thresholds to IP core memory
        for i in range(indicator_count * 2):  # Buy and sell thresholds for each indicator
            eval_ip.write(0x60 + i*4, int(thresholds[i].view(np.int32)))
            
        # Start processing
        eval_ip.write(0x00, 0x01)
        
        # Wait for completion
        while (eval_ip.read(0x00) & 0x02) == 0:
            time.sleep(0.001)
            
        # Read output data
        for i in range(data_size):
            signals[i] = eval_ip.read(0x80 + i*4)
            
        return signals


class FPGAAccelerator(HardwareAccelerator):
    """FPGA acceleration implementation for time-critical trading operations."""
    
    def __init__(self):
        super().__init__(accelerator_type="fpga")
        self.logger = get_logger(f"{__name__}.FPGAAccelerator")
        self.fpga = FPGAManager()
        
        # Register metrics
        self.metrics.register_counter("fpga_operations_count")
        self.metrics.register_counter("fpga_errors_count")
        self.metrics.register_gauge("fpga_latency_us")
        
        # Define available algorithms
        self.available_algorithms = set([
            "moving_average",
            "exponential_moving_average",
            "macd",
            "market_order_evaluation"
        ])
        
        if self.fpga.has_fpga:
            # Add actual available algorithms from FPGA
            if self.fpga.device_type == "OpenCL":
                self.available_algorithms.update(self.fpga.kernels.keys())
            elif self.fpga.device_type == "PYNQ":
                self.available_algorithms.update(self.fpga.trading_algorithms.keys())
                
    def is_available(self) -> bool:
        """Check if FPGA acceleration is available."""
        return self.fpga.has_fpga
        
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed information about the FPGA device."""
        if not self.fpga.has_fpga:
            return {"status": "No FPGA available"}
            
        info = {
            "device_type": self.fpga.device_type,
            "available_algorithms": list(self.available_algorithms)
        }
        
        if self.fpga.device_type == "OpenCL":
            info.update({
                "platform_name": self.fpga.platform.name if self.fpga.platform else "Unknown",
                "device_name": self.fpga.device.name if self.fpga.device else "Unknown"
            })
        elif self.fpga.device_type == "PYNQ":
            info.update({
                "bitstream_path": self.fpga.bitstream_path,
                "ip_cores": list(self.fpga.overlay.ip_dict.keys()) if hasattr(self.fpga.overlay, 'ip_dict') else []
            })
            
        return info
        
    def run_algorithm(self, algorithm: str, **kwargs) -> Any:
        """Run a trading algorithm on the FPGA."""
        if not self.fpga.has_fpga or algorithm not in self.available_algorithms:
            self.logger.warning(f"Algorithm '{algorithm}' not available on FPGA")
            return None
            
        try:
            start_time = time.time()
            
            # Execute algorithm
            result = self.fpga.execute_algorithm(algorithm, **kwargs)
            
            # Calculate latency
            latency_us = (time.time() - start_time) * 1_000_000
            self.metrics.set_gauge("fpga_latency_us", latency_us)
            self.metrics.increment_counter("fpga_operations_count")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running algorithm '{algorithm}' on FPGA: {str(e)}")
            self.metrics.increment_counter("fpga_errors_count")
            return None
            
    def run_algorithm_async(self, algorithm: str, **kwargs) -> Tuple[str, bool]:
        """Run a trading algorithm on the FPGA asynchronously."""
        if not self.fpga.has_fpga or algorithm not in self.available_algorithms:
            self.logger.warning(f"Algorithm '{algorithm}' not available on FPGA")
            return None, False
            
        try:
            # Queue algorithm execution
            task_id = self.fpga.execute_algorithm_async(algorithm, **kwargs)
            self.metrics.increment_counter("fpga_operations_count")
            
            return task_id, True
            
        except Exception as e:
            self.logger.error(f"Error queueing algorithm '{algorithm}' on FPGA: {str(e)}")
            self.metrics.increment_counter("fpga_errors_count")
            return None, False
            
    def get_result(self, task_id: str, timeout: float = None) -> Tuple[bool, Any]:
        """Get the result of an asynchronous FPGA operation."""
        if not self.fpga.has_fpga:
            return False, "FPGA not available"
            
        return self.fpga.get_result(task_id, timeout)
        
    def is_algorithm_available(self, algorithm: str) -> bool:
        """Check if a specific algorithm is available on the FPGA."""
        return self.fpga.has_fpga and algorithm in self.available_algorithms
        
    def shutdown(self) -> None:
        """Shutdown the FPGA accelerator."""
        if self.fpga.has_fpga:
            self.fpga.shutdown()
            
    def get_algorithm_info(self, algorithm: str) -> Dict[str, Any]:
        """Get detailed information about a specific algorithm."""
        if not self.fpga.has_fpga or algorithm not in self.available_algorithms:
            return {"status": "Algorithm not available"}
            
        # Algorithm details and parameters
        algorithm_info = {
            "algorithm": algorithm,
            "accelerator_type": "FPGA",
            "device_type": self.fpga.device_type,
            "parameters": {}
        }
        
        # Add algorithm-specific parameter information
        if algorithm == "moving_average":
            algorithm_info["parameters"] = {
                "input_data": "Required. Array of price data",
                "window_size": "Optional, default=20. Size of the moving window"
            }
        elif algorithm == "exponential_moving_average":
            algorithm_info["parameters"] = {
                "input_data": "Required. Array of price data",
                "alpha": "Optional, default=0.2. Smoothing factor (0 < alpha < 1)"
            }
        elif algorithm == "macd":
            algorithm_info["parameters"] = {
                "input_data": "Required. Array of price data",
                "fast_period": "Optional, default=12. Fast EMA period",
                "slow_period": "Optional, default=26. Slow EMA period",
                "signal_period": "Optional, default=9. Signal line period"
            }
        elif algorithm == "market_order_evaluation":
            algorithm_info["parameters"] = {
                "prices": "Required. Array of price data",
                "indicators": "Required. Array or 2D array of indicator values",
                "thresholds": "Required. Array of buy/sell thresholds for each indicator"
            }
            
        return algorithm_info
