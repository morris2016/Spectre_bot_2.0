"""
QuantumSpectre Elite Trading System
Data Storage Module

This module provides the data storage capabilities for the QuantumSpectre Elite Trading System,
including time series databases, feature stores, and pattern libraries.
"""

import os
import logging
from typing import Dict, Any, List, Optional

from common.logger import get_logger
from common.utils import singleton
from common.exceptions import DatabaseConnectionError

# Initialize logger
logger = get_logger(__name__)

# Component registry
registered_components = {}

def register_component(name: str, component: Any) -> None:
    """
    Register a data storage component for system-wide access.
    
    Args:
        name: Unique name for the component
        component: The component instance to register
    """
    if name in registered_components:
        logger.warning(f"Component {name} already registered, overwriting")
    registered_components[name] = component
    logger.debug(f"Registered data storage component: {name}")

def get_component(name: str) -> Any:
    """
    Retrieve a registered data storage component.
    
    Args:
        name: The name of the component to retrieve
        
    Returns:
        The registered component instance
        
    Raises:
        KeyError: If the component is not registered
    """
    if name not in registered_components:
        raise KeyError(f"Component {name} not registered")
    return registered_components[name]

def initialize_storage(config: Dict[str, Any]) -> None:
    """
    Initialize all data storage components based on configuration.
    
    Args:
        config: Configuration dictionary for storage components
        
    Raises:
        DatabaseConnectionError: If database connection fails
    """
    logger.info("Initializing data storage components")
    
    # Import here to avoid circular imports
    from data_storage.database import DatabaseManager
    from data_storage.time_series import TimeSeriesManager
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(config.get('database', {}))
        register_component('db_manager', db_manager)
        
        # Initialize time series manager
        ts_manager = TimeSeriesManager(config.get('time_series', {}))
        register_component('ts_manager', ts_manager)
        
        logger.info("Data storage components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize data storage: {str(e)}")
        raise DatabaseConnectionError(f"Storage initialization failed: {str(e)}")

def get_all_components() -> Dict[str, Any]:
    """
    Get all registered data storage components.
    
    Returns:
        Dictionary of all registered components
    """
    return registered_components.copy()

def shutdown() -> None:
    """
    Properly shut down all data storage components.
    """
    logger.info("Shutting down data storage components")
    for name, component in registered_components.items():
        try:
            if hasattr(component, 'shutdown'):
                component.shutdown()
                logger.debug(f"Shut down component: {name}")
        except Exception as e:
            logger.error(f"Error shutting down {name}: {str(e)}")
    
    registered_components.clear()
    logger.info("All data storage components shut down")

