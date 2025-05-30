#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Data Ingest Service

This module implements the main Data Ingest Service, responsible for ingesting
data from various sources and preprocessing it for consumption by other services.
"""

import os
import sys
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Set
import importlib

from common.logger import get_logger
from common.metrics import MetricsCollector
from common.exceptions import (
    DataIngestionError, ProcessorNotFoundError, SourceNotFoundError,
    ConfigurationError, ServiceStartupError, ServiceShutdownError
)
from common.async_utils import run_with_timeout, run_in_threadpool
from common.utils import import_submodules

from data_ingest.processor import DataProcessor
from data_ingest.sources.data_source import DataSource


class DataIngestService:
    """Main service for data ingestion and preprocessing."""
    
    def __init__(self, config, loop=None, redis_client=None, db_client=None):
        """
        Initialize the data ingest service.
        
        Args:
            config: Configuration object
            loop: Asyncio event loop
            redis_client: Redis client instance
            db_client: Database client instance
        """
        self.config = config
        self.loop = loop or asyncio.get_event_loop()
        self.redis_client = redis_client
        self.db_client = db_client
        self.logger = get_logger("DataIngestService")
        self.running = False
        self.shutting_down = False
        self.task = None
        
        # Metrics collector
        self.metrics = MetricsCollector("data_ingest")
        
        # Processors and sources
        self.processors = {}
        self.sources = {}
        self.source_tasks = {}
        
        # Data queues
        self.ingest_queue = asyncio.Queue()
        self.process_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        
        # Processing stats
        self.stats = {
            "ingested": 0,
            "processed": 0,
            "errors": 0,
            "processing_time": 0,
        }
    
    async def start(self):
        """Start the data ingest service."""
        self.logger.info("Starting Data Ingest Service")
        
        # Load processors and sources
        await self._load_processors()
        await self._load_sources()
        
        # Start data pipeline
        self.processing_task = asyncio.create_task(self._process_data_pipeline())
        self.output_task = asyncio.create_task(self._output_data_pipeline())
        
        # Start sources
        for source_name, source in self.sources.items():
            self.logger.info(f"Starting source: {source_name}")
            source_task = asyncio.create_task(self._run_source(source_name, source))
            self.source_tasks[source_name] = source_task
        
        self.running = True
        self.task = asyncio.create_task(self._monitor_tasks())
        self.logger.info("Data Ingest Service started successfully")
    
    async def stop(self):
        """Stop the data ingest service."""
        self.logger.info("Stopping Data Ingest Service")
        self.shutting_down = True
        
        # Cancel monitor task
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        # Stop sources
        for source_name, task in self.source_tasks.items():
            if not task.done():
                self.logger.info(f"Stopping source: {source_name}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cancel processing and output tasks
        for task in [getattr(self, "processing_task", None), getattr(self, "output_task", None)]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.running = False
        self.logger.info("Data Ingest Service stopped successfully")
    
    async def health_check(self):
        """Perform a health check on the data ingest service."""
        if not self.running:
            return False
        
        # Check if tasks are running
        task = getattr(self, "task", None)
        processing = getattr(self, "processing_task", None)
        output = getattr(self, "output_task", None)

        all_tasks_running = (
            task is not None and not task.done() and
            processing is not None and not processing.done() and
            output is not None and not output.done()
        )
        
        # Check that at least some sources are running
        sources_running = any(not task.done() for task in self.source_tasks.values())
        
        return all_tasks_running and sources_running
    
    async def _load_processors(self):
        """Load all available data processors."""
        self.logger.info("Loading data processors")
        
        # First, check configuration for explicitly enabled processors
        processor_configs = self.config.data_ingest.get("processors", {})
        
        # Import processor modules
        try:
            # Import all processor modules for auto-discovery
            import data_ingest.processors
            processor_modules = import_submodules(data_ingest.processors)
            
            # Find all processor classes
            processor_classes = {}
            for module in processor_modules.values():
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                            issubclass(attr, DataProcessor) and 
                            attr is not DataProcessor):
                        processor_name = attr.__name__.lower()
                        if processor_name.endswith('processor'):
                            processor_name = processor_name[:-9]  # Remove 'processor' suffix
                        processor_classes[processor_name] = attr
            
            # Initialize enabled processors
            for processor_name, processor_class in processor_classes.items():
                processor_config = processor_configs.get(processor_name, {})
                if processor_config.get("enabled", True):
                    self.logger.info(f"Initializing processor: {processor_name}")
                    try:
                        processor = processor_class(
                            config=processor_config,
                            logger=get_logger(f"DataProcessor.{processor_name}")
                        )
                        self.processors[processor_name] = processor
                    except Exception as e:
                        self.logger.error(f"Failed to initialize processor {processor_name}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error loading processors: {str(e)}")
            raise ServiceStartupError(f"Failed to load processors: {str(e)}")
        
        self.logger.info(f"Loaded {len(self.processors)} processors: {', '.join(self.processors.keys())}")
    
    async def _load_sources(self):
        """Load all available data sources."""
        self.logger.info("Loading data sources")
        
        # Check configuration for explicitly enabled sources
        source_configs = self.config.data_ingest.get("sources", {})
        
        try:
            # Import all source modules for auto-discovery
            import data_ingest.sources
            source_modules = import_submodules(data_ingest.sources)
            
            # Find all source classes
            source_classes = {}
            for module in source_modules.values():
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                            issubclass(attr, DataSource) and 
                            attr is not DataSource):
                        source_name = attr.__name__.lower()
                        if source_name.endswith('source'):
                            source_name = source_name[:-6]  # Remove 'source' suffix
                        source_classes[source_name] = attr
            
            # Initialize enabled sources
            for source_name, source_class in source_classes.items():
                source_config = source_configs.get(source_name, {})
                if source_config.get("enabled", True):
                    self.logger.info(f"Initializing source: {source_name}")
                    try:
                        source = source_class(
                            config=source_config,
                            queue=self.ingest_queue,
                            redis_client=self.redis_client,
                            db_client=self.db_client,
                            logger=get_logger(f"DataSource.{source_name}")
                        )
                        self.sources[source_name] = source
                    except Exception as e:
                        self.logger.error(f"Failed to initialize source {source_name}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error loading sources: {str(e)}")
            raise ServiceStartupError(f"Failed to load sources: {str(e)}")
        
        self.logger.info(f"Loaded {len(self.sources)} sources: {', '.join(self.sources.keys())}")
    
    async def _run_source(self, source_name, source):
        """Run a data source and handle its output."""
        self.logger.info(f"Running source: {source_name}")
        try:
            await source.start()
            self.logger.info(f"Source {source_name} started successfully")
            
            # Sources push data to the ingest_queue directly
            # Just keep the task alive until shutdown
            while not self.shutting_down:
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            self.logger.info(f"Source {source_name} task cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in source {source_name}: {str(e)}")
            self.metrics.increment("source.error")
        finally:
            try:
                await source.stop()
            except Exception as e:
                self.logger.error(f"Error stopping source {source_name}: {str(e)}")
    
    async def _process_data_pipeline(self):
        """Process data from the ingest queue."""
        self.logger.info("Starting data processing pipeline")
        while not self.shutting_down:
            try:
                # Get data from the ingest queue
                data = await self.ingest_queue.get()
                self.metrics.increment("data.ingested")
                self.stats["ingested"] += 1
                
                # Determine which processor to use
                data_type = data.get("type", "unknown")
                processor = self._get_processor_for_type(data_type)
                
                if processor:
                    # Process the data
                    start_time = time.time()
                    try:
                        processed_data = await processor.process(data)
                        processing_time = time.time() - start_time
                        self.metrics.histogram("processing_time", processing_time)
                        self.stats["processing_time"] += processing_time
                        
                        # Queue for output
                        await self.output_queue.put(processed_data)
                        self.metrics.increment("data.processed")
                        self.stats["processed"] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error processing data: {str(e)}")
                        self.metrics.increment("processing.error")
                        self.stats["errors"] += 1
                else:
                    self.logger.warning(f"No processor found for data type: {data_type}")
                    self.metrics.increment("processing.no_processor")
                
                # Mark task as done
                self.ingest_queue.task_done()
                
            except asyncio.CancelledError:
                self.logger.info("Data processing pipeline cancelled")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in data processing pipeline: {str(e)}")
                self.metrics.increment("processing.unexpected_error")
                await asyncio.sleep(1)  # Avoid tight error loop
    
    async def _output_data_pipeline(self):
        """Handle processed data output."""
        self.logger.info("Starting data output pipeline")
        while not self.shutting_down:
            try:
                # Get processed data
                processed_data = await self.output_queue.get()
                
                # Publish to Redis for other services to consume
                if self.redis_client:
                    data_type = processed_data.get("type", "unknown")
                    channel = f"data.{data_type}"
                    await self.redis_client.publish(channel, processed_data)
                    self.metrics.increment("data.published")
                
                # Store in database if configured
                if self.db_client and self.config.data_ingest.get("store_processed_data", False):
                    await self._store_data(processed_data)
                
                # Mark task as done
                self.output_queue.task_done()
                
            except asyncio.CancelledError:
                self.logger.info("Data output pipeline cancelled")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in data output pipeline: {str(e)}")
                self.metrics.increment("output.error")
                await asyncio.sleep(1)  # Avoid tight error loop
    
    async def _store_data(self, data):
        """Store processed data in the database."""
        try:
            data_type = data.get("type", "unknown")
            collection = f"processed_{data_type}"
            await self.db_client.insert(collection, data)
            self.metrics.increment("data.stored")
        except Exception as e:
            self.logger.error(f"Error storing data in database: {str(e)}")
            self.metrics.increment("storage.error")
    
    def _get_processor_for_type(self, data_type):
        """Get the appropriate processor for a data type."""
        # Map of data types to processor names
        processor_map = {
            "market": "market",
            "ohlcv": "market",
            "order_book": "market",
            "trade": "market",
            "news": "news",
            "article": "news",
            "social": "social",
            "tweet": "social",
            "reddit": "social",
            "onchain": "blockchain"
        }
        
        processor_name = processor_map.get(data_type, data_type)
        return self.processors.get(processor_name)
    
    async def _monitor_tasks(self):
        """Monitor all tasks and restart if necessary."""
        self.logger.info("Starting task monitoring")
        while not self.shutting_down:
            try:
                # Log statistics periodically
                self.logger.info(
                    f"Data ingest stats: Ingested: {self.stats['ingested']}, "
                    f"Processed: {self.stats['processed']}, "
                    f"Errors: {self.stats['errors']}, "
                    f"Queue sizes: Ingest: {self.ingest_queue.qsize()}, "
                    f"Output: {self.output_queue.qsize()}"
                )
                
                # Set metrics
                self.metrics.set("queue.ingest", self.ingest_queue.qsize())
                self.metrics.set("queue.output", self.output_queue.qsize())
                
                # Check source tasks and restart if needed
                for source_name, task in list(self.source_tasks.items()):
                    if task.done() and not self.shutting_down:
                        try:
                            # Get result to propagate any exceptions
                            await task
                        except Exception as e:
                            self.logger.error(f"Source {source_name} task failed: {str(e)}")
                        
                        # Restart the source if auto_restart is enabled
                        source_config = self.config.data_ingest.get("sources", {}).get(source_name, {})
                        if source_config.get("auto_restart", True):
                            self.logger.info(f"Restarting source: {source_name}")
                            source = self.sources[source_name]
                            self.source_tasks[source_name] = asyncio.create_task(
                                self._run_source(source_name, source)
                            )
                
                # Check processing and output tasks
                for task_name, task in [
                    ("processing", self.processing_task),
                    ("output", self.output_task)
                ]:
                    if task.done() and not self.shutting_down:
                        try:
                            # Get result to propagate any exceptions
                            await task
                        except Exception as e:
                            self.logger.error(f"{task_name} task failed: {str(e)}")
                        
                        # Restart the task
                        self.logger.info(f"Restarting {task_name} task")
                        if task_name == "processing":
                            self.processing_task = asyncio.create_task(self._process_data_pipeline())
                        else:
                            self.output_task = asyncio.create_task(self._output_data_pipeline())
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                self.logger.info("Task monitoring cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in task monitoring: {str(e)}")
                await asyncio.sleep(5)  # Avoid tight error loop
