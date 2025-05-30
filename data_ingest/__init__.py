

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Data Ingestion Module Initialization

This module initializes the data ingestion package, registering components
and establishing the necessary infrastructure for high-performance data acquisition.
"""

import os
import sys
from typing import Dict, List, Set, Any, Optional
import importlib
import pkgutil
import inspect

# Version information
__version__ = '1.0.0'
__author__ = 'QuantumSpectre Team'

# Import common utilities
from common.logger import get_logger
from common.metrics import MetricsCollector
from common.constants import DATA_INGEST_METRICS_PREFIX

# Initialize package-level logger
logger = get_logger('data_ingest')

# Initialize metrics collector for this module
metrics = MetricsCollector(DATA_INGEST_METRICS_PREFIX)

# Dictionary to store registered data processors
registered_processors = {}

# Dictionary to store registered data sources
registered_sources = {}

# Component registration
def register_processor(name: str, processor_class):
    """
    Register a data processor component.
    
    Args:
        name: Unique identifier for the processor
        processor_class: The processor class to register
    """
    if name in registered_processors:
        logger.warning(f"Processor '{name}' already registered. Overwriting.")
    
    registered_processors[name] = processor_class
    logger.info(f"Registered data processor: {name}")
    metrics.increment('processor_registration')

def register_source(name: str, source_class):
    """
    Register a data source component.
    
    Args:
        name: Unique identifier for the source
        source_class: The source class to register
    """
    if name in registered_sources:
        logger.warning(f"Source '{name}' already registered. Overwriting.")
    
    registered_sources[name] = source_class
    logger.info(f"Registered data source: {name}")
    metrics.increment('source_registration')

def get_processor(name: str):
    """
    Get a registered processor by name.
    
    Args:
        name: Identifier of the processor to retrieve
        
    Returns:
        The registered processor class
        
    Raises:
        KeyError: If no processor is registered with the given name
    """
    if name not in registered_processors:
        raise KeyError(f"No processor registered with name '{name}'")
    
    return registered_processors[name]

def get_source(name: str):
    """
    Get a registered data source by name.
    
    Args:
        name: Identifier of the source to retrieve
        
    Returns:
        The registered source class
        
    Raises:
        KeyError: If no source is registered with the given name
    """
    if name not in registered_sources:
        raise KeyError(f"No source registered with name '{name}'")
    
    return registered_sources[name]

def list_processors():
    """
    List all registered data processors.
    
    Returns:
        A list of registered processor names
    """
    return list(registered_processors.keys())

def list_sources():
    """
    List all registered data sources.
    
    Returns:
        A list of registered source names
    """
    return list(registered_sources.keys())

# Auto-discovery and registration of components
def _discover_and_register_components():
    """
    Auto-discover and register data processing components from submodules.
    """
    from data_ingest import processors, sources
    
    # Find and register processor components
    for _, name, _ in pkgutil.iter_modules(processors.__path__):
        try:
            module = importlib.import_module(f'data_ingest.processors.{name}')
            for item_name, item in inspect.getmembers(module):
                # Register classes that end with 'Processor' and aren't imported
                if (inspect.isclass(item) and 
                    item_name.endswith('Processor') and 
                    item.__module__ == module.__name__):
                    register_processor(item_name, item)
        except Exception as e:
            logger.error(f"Error loading processor module {name}: {str(e)}")
    
    # Find and register source components
    for _, name, _ in pkgutil.iter_modules(sources.__path__):
        try:
            module = importlib.import_module(f'data_ingest.sources.{name}')
            for item_name, item in inspect.getmembers(module):
                # Register classes that end with 'Source' and aren't imported
                if (inspect.isclass(item) and 
                    item_name.endswith('Source') and 
                    item.__module__ == module.__name__):
                    register_source(item_name, item)
        except Exception as e:
            logger.error(f"Error loading source module {name}: {str(e)}")

# Initialize component discovery when module is imported
_discover_and_register_components()

# Export public API
__all__ = [
    'register_processor',
    'register_source',
    'get_processor',
    'get_source',
    'list_processors',
    'list_sources',
    'registered_processors',
    'registered_sources',
]

logger.info(f"Data Ingest module initialized with {len(registered_processors)} processors and {len(registered_sources)} sources")

