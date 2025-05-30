#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Feature Service - Processor

This module implements the core feature processing capabilities, handling
multi-threaded and GPU-accelerated feature computation for high-performance
pattern recognition and analysis.
"""

import logging
import os
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time
import traceback
try:
    import dask  # type: ignore
    import dask.dataframe as dd  # type: ignore
except Exception:  # pragma: no cover - optional
    dask = None
    dd = None
from feature_service.processor_utils import cudf, HAS_GPU

# Try to import GPU-related libraries
try:
    from numba import cuda, jit
    HAS_CUDA = True
except ImportError:
    from numba import jit
    HAS_CUDA = False
    # Create a dummy cuda module for compatibility
    class DummyCuda:
        def get_device_count(self):
            return 0
        
        def get_current_device(self):
            raise ValueError("No CUDA-capable devices found")
    
    cuda = DummyCuda()

import traceback
from common.logger import get_logger
from common.metrics import MetricsCollector
from common.utils import time_execution, chunks
from common.async_utils import run_in_threadpool, gather_with_concurrency
from common.constants import FEATURE_PRIORITY_LEVELS
# Removed FEATURE_BATCH_SIZE, FEATURE_MAX_WORKERS, FEATURE_CHUNK_SIZE,
# GPU_ENABLED, GPU_MEMORY_LIMIT as they are now sourced from config
# or were unused (FEATURE_CHUNK_SIZE).
from common.exceptions import (
    FeatureCalculationError, ResourceExhaustionError,
    InvalidFeatureDefinitionError, FeatureTimeoutError
)

from data_storage.time_series import TimeSeriesManager
from feature_service.feature_extraction import FeatureExtractor

logger = get_logger(__name__)
metrics = MetricsCollector.get_instance("feature_service.processor")


class FeatureProcessor:
    """
    Advanced feature processor with parallel computation capabilities.
    
    This class manages the computation of features across multiple assets, timeframes,
    and feature types with optimized resource utilization. It supports both CPU and
    GPU-accelerated processing paths with intelligent workload distribution.
    """
    
    def __init__(
        self,
        config=None,
        redis_client=None,
        db_client=None,
        time_series_store: Optional[TimeSeriesManager] = None
        # Removed max_workers, batch_size, use_gpu from signature
    ):
        """
        Initialize the feature processor.
        
        Args:
            config: Configuration object or dictionary
            redis_client: Redis client instance
            db_client: Database client instance
            time_series_store: Data store for accessing market data
            # max_workers, batch_size, use_gpu are now derived from config
        """
        self.config = config
        self.redis_client = redis_client
        self.db_client = db_client
        self.time_series_store = time_series_store
        
        # Get settings from config object
        if self.config:
            self.max_workers = self.config.get("feature_service.max_workers", os.cpu_count() or 4)
            self.batch_size = self.config.get("feature_service.batch_size", 1024) # Default 1024 if not in config
            self.use_gpu = self.config.get("system.gpu_enabled", True) and self.config.get("feature_service.use_gpu", True) and HAS_GPU and HAS_CUDA
        else:
            # Fallback defaults if no config is provided (less ideal)
            self.max_workers = os.cpu_count() or 4
            self.batch_size = 1024
            self.use_gpu = HAS_GPU and HAS_CUDA # Default to True only if GPU and CUDA are available
        
        # Initialize executors
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=max(2, self.max_workers // 2))
        
        # Initialize feature extractors
        self.extractors: Dict[str, FeatureExtractor] = {}
        
        # Set up GPU context if enabled
        if self.use_gpu:
            try:
                self.gpu_context = self._initialize_gpu()
                logger.info(f"GPU acceleration enabled with {self.gpu_context['device_count']} devices")
                metrics.gauge("feature.gpu.device_count", self.gpu_context['device_count'])
                metrics.gauge("feature.gpu.memory_available", self.gpu_context['memory_available'])
            except Exception as e:
                logger.warning(f"Failed to initialize GPU acceleration: {str(e)}")
                logger.warning("Falling back to CPU processing")
                self.use_gpu = False
        
        # Processing queues by priority
        self.processing_queues = {
            level: asyncio.Queue() 
            for level in FEATURE_PRIORITY_LEVELS
        }
        
        # Task tracking
        self.active_tasks = {}
        self.task_lock = asyncio.Lock()
        
        logger.info(f"Feature processor initialized with {self.max_workers} workers, GPU: {self.use_gpu}")
    
    def _initialize_gpu(self) -> Dict[str, Any]:
        """Initialize and configure GPU resources."""
        gpu_context = {}
        
        # Check if CUDA is available
        if not HAS_CUDA:
            raise ValueError("CUDA is not available")
        
        # Get device information
        device_count = cuda.get_device_count()
        gpu_context['device_count'] = device_count
        
        if device_count == 0:
            raise ValueError("No CUDA-capable devices found")
        
        try:
            # Get primary device info
            device = cuda.get_current_device()
            gpu_context['device_id'] = device.id
            gpu_context['device_name'] = device.name
            gpu_context['memory_available'] = device.memory_info()[0]  # Free memory
        except Exception as e:
            logger.warning(f"Error getting CUDA device info: {str(e)}")
            # Provide default values
            gpu_context['device_id'] = 0
            gpu_context['device_name'] = "Unknown"
            gpu_context['memory_available'] = 1000000000  # 1GB default
        
        # Configure memory limits
        # Get GPU memory limit fraction from config, default to 0.8 (80%)
        gpu_memory_fraction = self.config.get("machine_learning.gpu_memory_limit", 0.8) if self.config else 0.8
        
        # Calculate memory limit based on available memory and the fraction
        # Ensure the fraction is between 0.1 and 1.0 for safety
        gpu_memory_fraction = max(0.1, min(1.0, gpu_memory_fraction))
        memory_limit_from_config = gpu_context['memory_available'] * gpu_memory_fraction
        
        # If GPU_MEMORY_LIMIT constant was intended as an absolute cap, it would need to be handled.
        # For now, we assume the fraction from config is the primary control.
        # If an absolute cap (like the old GPU_MEMORY_LIMIT) is still desired from constants,
        # it would need to be imported and used here, e.g.,
        # memory_limit = min(ABSOLUTE_GPU_CAP_CONSTANT, memory_limit_from_config)
        # However, since GPU_MEMORY_LIMIT is not defined in constants.py, we'll use the config fraction.
        memory_limit = memory_limit_from_config
        
        gpu_context['memory_limit'] = memory_limit
        
        # Configure cuDF for memory efficiency
        cudf.set_option('gpu.memory_limit', memory_limit)
        
        return gpu_context
    
    async def start(self):
        """Start the feature processor service."""
        logger.info("Starting feature processor service")
        
        try:
            # Check if time_series_store is initialized
            if self.time_series_store is None:
                logger.error("time_series_store is not initialized")
                raise ValueError("time_series_store is not initialized")
                
            # Start processing workers for each priority level
            self.processing_tasks = []
            for level in FEATURE_PRIORITY_LEVELS:
                logger.debug(f"Creating processing task for priority level: {level}")
                task = asyncio.create_task(self._process_queue(level))
                self.processing_tasks.append(task)
            
            # Create a main task that the service manager can monitor
            self.task = asyncio.create_task(self._keep_alive())
            
            logger.info("Feature processor service started successfully")
        except Exception as e:
            logger.error(f"Failed to start feature processor service: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    async def _keep_alive(self):
        """
        Keep the processor service alive with a long-running task.
        This prevents the service from completing unexpectedly.
        """
        logger.info("Feature processor keep-alive task started")
        try:
            while True:
                # Perform periodic health check
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Feature processor keep-alive task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in feature processor keep-alive task: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def stop(self):
        """Stop the feature processor service."""
        logger.info("Stopping feature processor service")
        
        # Cancel the keep-alive task if it exists
        if hasattr(self, 'task') and self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        try:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("Feature processor service stopped")
    
    async def _process_queue(self, priority_level: str):
        """
        Process feature calculation requests from the queue for a specific priority level.
        
        Args:
            priority_level: Priority level to process
        """
        queue = self.processing_queues[priority_level]
        logger.info(f"Starting feature processing queue for priority: {priority_level}")
        
        while True:
            try:
                # Get next task from queue
                task_id, feature_request = await queue.get()
                
                try:
                    # Process the feature request
                    logger.debug(f"Processing feature request {task_id} with priority {priority_level}")
                    
                    result = await self._calculate_features(
                        feature_request['asset'],
                        feature_request['timeframe'],
                        feature_request['features'],
                        feature_request['start_time'],
                        feature_request['end_time'],
                        feature_request['params']
                    )
                    
                    # Store result
                    if task_id in self.active_tasks:
                        future = self.active_tasks[task_id]
                        if not future.done():
                            future.set_result(result)
                    
                except Exception as e:
                    logger.error(f"Error processing feature request {task_id}: {str(e)}")
                    # Set exception on future
                    if task_id in self.active_tasks:
                        future = self.active_tasks[task_id]
                        if not future.done():
                            future.set_exception(e)
                
                finally:
                    # Mark task as complete
                    queue.task_done()
                    
                    # Remove from active tasks
                    async with self.task_lock:
                        if task_id in self.active_tasks:
                            del self.active_tasks[task_id]
            
            except asyncio.CancelledError:
                logger.info(f"Feature processing queue for priority {priority_level} cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in feature processing queue ({priority_level}): {str(e)}")
                continue
    
    async def request_features(
        self,
        asset: str,
        timeframe: str,
        features: List[str],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        priority: str = "normal"
    ) -> pd.DataFrame:
        """
        Request feature calculation for a specific asset and timeframe.
        
        Args:
            asset: Asset symbol to calculate features for
            timeframe: Timeframe to calculate features on
            features: List of features to calculate
            start_time: Start time for data (epoch ms)
            end_time: End time for data (epoch ms)
            params: Additional parameters for feature calculation
            priority: Priority level for the calculation
            
        Returns:
            DataFrame containing calculated features
        """
        if priority not in FEATURE_PRIORITY_LEVELS:
            raise ValueError(f"Invalid priority level: {priority}")
        
        # Create feature request
        feature_request = {
            'asset': asset,
            'timeframe': timeframe,
            'features': features,
            'start_time': start_time,
            'end_time': end_time,
            'params': params or {}
        }
        
        # Create future for result
        future = asyncio.Future()
        
        # Generate unique task ID
        task_id = f"{asset}_{timeframe}_{int(time.time() * 1000)}_{id(future)}"
        
        # Store in active tasks
        async with self.task_lock:
            self.active_tasks[task_id] = future
        
        # Add to processing queue
        await self.processing_queues[priority].put((task_id, feature_request))
        metrics.increment(f"feature.requests.{priority}")
        
        # Wait for result
        try:
            result = await future
            return result
        except Exception as e:
            metrics.increment("feature.calculations.failed")
            raise FeatureCalculationError(f"Feature calculation failed: {str(e)}") from e
    
    async def _calculate_features(
        self,
        asset: str,
        timeframe: str,
        features: List[str],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Calculate requested features for an asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for calculations
            features: List of features to calculate
            start_time: Start time for data range
            end_time: End time for data range
            params: Additional parameters for calculations
            
        Returns:
            DataFrame with calculated features
        """
        start_ts = time.time()
        metrics.increment("feature.calculations.started")
        logger.debug(f"Calculating features for {asset} ({timeframe}): {features}")
        
        try:
            # Get market data
            logger.debug(f"Fetching candles for {asset} {timeframe} from time_series_store")
            if self.time_series_store is None:
                logger.error("time_series_store is None, cannot fetch candles")
                raise ValueError("time_series_store is not initialized")
                
            try:
                data = await run_in_threadpool(
                    self.time_series_store.get_candles,
                    asset=asset,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )
                logger.debug(f"Retrieved {len(data) if not data.empty else 0} candles")
            except Exception as e:
                logger.error(f"Error fetching candles: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
            if data.empty:
                logger.warning(f"No data available for {asset} ({timeframe})")
                return pd.DataFrame()
            
            # Prepare parameters
            params = params or {}
            
            # Determine optimal processing path (CPU vs GPU)
            if self.use_gpu and len(data) > 1000:
                # GPU path for larger datasets
                result = await self._calculate_features_gpu(data, features, params)
            else:
                # CPU path
                result = await self._calculate_features_cpu(data, features, params)
            
            metrics.timing("feature.calculation.time", (time.time() - start_ts) * 1000)
            metrics.increment("feature.calculations.completed")
            
            return result
        
        except Exception as e:
            logger.error(f"Feature calculation error for {asset} ({timeframe}): {str(e)}")
            metrics.increment("feature.calculations.failed")
            raise FeatureCalculationError(f"Error calculating features: {str(e)}") from e
    
    async def _calculate_features_cpu(
        self,
        data: pd.DataFrame,
        features: List[str],
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Calculate features using CPU-based processing."""
        logger.debug(f"Using CPU-based feature calculation for {len(data)} rows")
        
        # Create feature extractor if not already created
        extractor_key = "_".join(sorted(features))
        if extractor_key not in self.extractors:
            self.extractors[extractor_key] = FeatureExtractor(features)
        
        extractor = self.extractors[extractor_key]
        
        # For smaller datasets, process directly
        if len(data) <= self.batch_size:
            return await run_in_threadpool(extractor.extract_features, data, params)
        
        # For larger datasets, use parallel processing with chunks
        chunks = chunks(data, self.batch_size)
        logger.debug(f"Processing {len(chunks)} chunks in parallel")
        
        # Define processing function
        def process_chunk(chunk):
            return extractor.extract_features(chunk, params)
        
        # Process chunks in thread pool
        results = await gather_with_concurrency(
            self.max_workers,
            *(run_in_threadpool(process_chunk, chunk) for chunk in chunks)
        )
        
        # Combine results
        if not results:
            return pd.DataFrame()
        
        combined = pd.concat(results, ignore_index=False)
        return combined
    
    async def _calculate_features_gpu(
        self,
        data: pd.DataFrame,
        features: List[str],
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Calculate features using GPU-accelerated processing."""
        logger.debug(f"Using GPU-accelerated feature calculation for {len(data)} rows")
        
        try:
            # Create feature extractor if not already created
            extractor_key = f"gpu_{'_'.join(sorted(features))}"
            if extractor_key not in self.extractors:
                self.extractors[extractor_key] = FeatureExtractor(features, use_gpu=True)
            
            extractor = self.extractors[extractor_key]
            
            # Convert to cuDF DataFrame for GPU processing
            async def convert_to_cudf():
                return cudf.DataFrame.from_pandas(data)
            
            gpu_data = await run_in_threadpool(lambda: cudf.DataFrame.from_pandas(data))
            
            # Process on GPU
            result_gpu = await run_in_threadpool(extractor.extract_features_gpu, gpu_data, params)
            
            # Convert back to pandas DataFrame
            result = await run_in_threadpool(lambda: result_gpu.to_pandas())
            
            return result
            
        except Exception as e:
            logger.warning(f"GPU processing failed, falling back to CPU: {str(e)}")
            return await self._calculate_features_cpu(data, features, params)
    
    async def get_feature_metadata(self) -> Dict[str, Any]:
        """Get metadata about available features."""
        all_extractors = set()
        
        # Create a basic extractor to get feature info
        temp_extractor = FeatureExtractor([])
        feature_info = await run_in_threadpool(temp_extractor.get_all_feature_info)
        
        return {
            "available_features": feature_info,
            "gpu_enabled": self.use_gpu,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size
        }
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get the current status of processing queues."""
        return {
            level: {
                "queue_size": self.processing_queues[level].qsize(),
                "active_tasks": len([
                    task_id for task_id, future in self.active_tasks.items() 
                    if not future.done()
                ])
            }
            for level in FEATURE_PRIORITY_LEVELS
        }


class FeatureProcessorPool:
    """
    Pool of feature processors optimized for different workloads.
    
    This class manages multiple feature processors to optimize resource usage
    for different types of feature calculations (e.g., short-term vs. long-term).
    """
    
    def __init__(self, time_series_store: TimeSeriesManager):
        """
        Initialize the feature processor pool.
        
        Args:
            time_series_store: Data store for accessing market data
        """
        self.time_series_store = time_series_store
        self.processors = {}
        self.lock = asyncio.Lock()
        
        # Default processor configurations
        self.default_configs = {
            "realtime": {
                "max_workers": max(2, FEATURE_MAX_WORKERS // 4),
                "batch_size": FEATURE_BATCH_SIZE // 2,
                "use_gpu": GPU_ENABLED
            },
            "standard": {
                "max_workers": FEATURE_MAX_WORKERS,
                "batch_size": FEATURE_BATCH_SIZE,
                "use_gpu": GPU_ENABLED
            },
            "batch": {
                "max_workers": max(2, FEATURE_MAX_WORKERS // 2),
                "batch_size": FEATURE_BATCH_SIZE * 2,
                "use_gpu": GPU_ENABLED
            }
        }
        
        logger.info(f"Initializing feature processor pool with {len(self.default_configs)} configurations")
    
    async def start(self):
        """Start the feature processor pool."""
        logger.info("Starting feature processor pool")
        
        # Initialize default processors
        for processor_type, config in self.default_configs.items():
            await self.get_processor(processor_type)
        
        logger.info("Feature processor pool started")
    
    async def stop(self):
        """Stop all processors in the pool."""
        logger.info("Stopping feature processor pool")
        
        stop_tasks = []
        for processor in self.processors.values():
            stop_tasks.append(processor.stop())
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        self.processors.clear()
        logger.info("Feature processor pool stopped")
    
    async def get_processor(self, processor_type: str = "standard") -> FeatureProcessor:
        """
        Get a feature processor of the specified type.
        
        Args:
            processor_type: Type of processor to get
            
        Returns:
            Feature processor instance
        """
        async with self.lock:
            if processor_type not in self.processors:
                # Create new processor with configuration
                config = self.default_configs.get(
                    processor_type, self.default_configs["standard"]
                )
                
                processor = FeatureProcessor(
                    self.time_series_store,
                    max_workers=config["max_workers"],
                    batch_size=config["batch_size"],
                    use_gpu=config["use_gpu"]
                )
                
                await processor.start()
                self.processors[processor_type] = processor
                logger.info(f"Created new processor of type '{processor_type}'")
            
            return self.processors[processor_type]
    
    async def request_features(
        self,
        asset: str,
        timeframe: str,
        features: List[str],
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        priority: str = "normal",
        processor_type: str = "standard"
    ) -> pd.DataFrame:
        """
        Request feature calculation using appropriate processor.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for calculations
            features: List of features to calculate
            start_time: Start time for data range
            end_time: End time for data range
            params: Additional parameters for calculations
            priority: Priority level for calculation
            processor_type: Type of processor to use
            
        Returns:
            DataFrame with calculated features
        """
        # Select appropriate processor based on request characteristics
        if processor_type == "auto":
            # Auto-select processor based on request characteristics
            if end_time is None or (end_time and int(time.time() * 1000) - end_time < 60000):
                # Recent data - use realtime processor
                processor_type = "realtime"
            elif start_time and end_time and end_time - start_time > 86400000 * 30:
                # Long historical range - use batch processor
                processor_type = "batch"
            else:
                # Default to standard processor
                processor_type = "standard"
        
        # Get appropriate processor
        processor = await self.get_processor(processor_type)
        
        # Request features
        metrics.increment(f"feature.requests.{processor_type}")
        result = await processor.request_features(
            asset, timeframe, features, start_time, end_time, params, priority
        )
        
        return result
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get status information about all processors in the pool."""
        status = {}
        
        for processor_type, processor in self.processors.items():
            queue_status = await processor.get_queue_status()
            status[processor_type] = {
                "queue_status": queue_status,
                "max_workers": processor.max_workers,
                "batch_size": processor.batch_size,
                "gpu_enabled": processor.use_gpu
            }
        
        return status
