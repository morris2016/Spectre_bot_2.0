#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Feature Service - Module Initialization

This module initializes the feature service components that calculate technical indicators,
volatility measures, pattern recognition features, and other trading signals.
"""

import os
import sys
import logging
from typing import Dict, List, Set, Any, Optional, Type

from common.logger import get_logger
from common.utils import register_component, get_registered_components

# Initialize module logger
logger = get_logger("feature_service")

# Version and metadata
__version__ = "1.0.0"
__author__ = "QuantumSpectre Team"
__description__ = "Feature calculation and pattern recognition service"

# Available feature groups - expanded for comprehensive market analysis
FEATURE_GROUPS = {
    "technical": "Basic technical indicators",
    "volatility": "Volatility and risk metrics",
    "volume": "Volume analysis features",
    "pattern": "Pattern recognition features",
    "sentiment": "Market sentiment analysis",
    "market_structure": "Market structure identification",
    "order_flow": "Order flow analysis",
    "harmonic": "Harmonic pattern recognition",
    "support_resistance": "Support and resistance levels",
    "correlation": "Asset correlation features",
    "cyclical": "Market cycles and seasonality",
    "regime": "Market regime indicators",
    "divergence": "Divergence detection features",
    "auction": "Market auction theory metrics",
    "liquidity": "Liquidity analysis features",
    "vwap": "Volume-weighted metrics",
    "fundamental": "Fundamental data indicators",
    "options": "Options-derived indicators",
    "anomaly": "Market anomaly detectors",
    "momentum": "Advanced momentum indicators"
}

# Component registries
_feature_calculators: Dict[str, Any] = {}
_feature_transformers: Dict[str, Any] = {}
_feature_detectors: Dict[str, Any] = {}
_processors: Dict[str, Any] = {}

def register_feature_calculator(name: str, calculator: Any) -> None:
    """
    Register a feature calculator component.
    
    Args:
        name: Unique identifier for the calculator
        calculator: The calculator component to register
    """
    if name in _feature_calculators:
        logger.warning(f"Overwriting existing feature calculator: {name}")
    _feature_calculators[name] = calculator
    logger.debug(f"Registered feature calculator: {name}")
    register_component(f"feature_calculator.{name}", calculator)

def register_feature_transformer(name: str, transformer: Any) -> None:
    """
    Register a feature transformer component.
    
    Args:
        name: Unique identifier for the transformer
        transformer: The transformer component to register
    """
    if name in _feature_transformers:
        logger.warning(f"Overwriting existing feature transformer: {name}")
    _feature_transformers[name] = transformer
    logger.debug(f"Registered feature transformer: {name}")
    register_component(f"feature_transformer.{name}", transformer)

def register_feature_detector(name: str, detector: Any) -> None:
    """
    Register a feature detector component.
    
    Args:
        name: Unique identifier for the detector
        detector: The detector component to register
    """
    if name in _feature_detectors:
        logger.warning(f"Overwriting existing feature detector: {name}")
    _feature_detectors[name] = detector
    logger.debug(f"Registered feature detector: {name}")
    register_component(f"feature_detector.{name}", detector)

def register_processor(name: str, processor: Any) -> None:
    """
    Register a feature processor component.
    
    Args:
        name: Unique identifier for the processor
        processor: The processor component to register
    """
    if name in _processors:
        logger.warning(f"Overwriting existing processor: {name}")
    _processors[name] = processor
    logger.debug(f"Registered processor: {name}")
    register_component(f"feature_processor.{name}", processor)

def get_feature_calculator(name: str) -> Any:
    """
    Retrieve a registered feature calculator by name.
    
    Args:
        name: The name of the calculator to retrieve
        
    Returns:
        The requested calculator component
        
    Raises:
        KeyError: If the calculator is not registered
    """
    if name not in _feature_calculators:
        raise KeyError(f"Feature calculator not found: {name}")
    return _feature_calculators[name]

def get_feature_transformer(name: str) -> Any:
    """
    Retrieve a registered feature transformer by name.
    
    Args:
        name: The name of the transformer to retrieve
        
    Returns:
        The requested transformer component
        
    Raises:
        KeyError: If the transformer is not registered
    """
    if name not in _feature_transformers:
        raise KeyError(f"Feature transformer not found: {name}")
    return _feature_transformers[name]

def get_feature_detector(name: str) -> Any:
    """
    Retrieve a registered feature detector by name.
    
    Args:
        name: The name of the detector to retrieve
        
    Returns:
        The requested detector component
        
    Raises:
        KeyError: If the detector is not registered
    """
    if name not in _feature_detectors:
        raise KeyError(f"Feature detector not found: {name}")
    return _feature_detectors[name]

def get_processor(name: str) -> Any:
    """
    Retrieve a registered processor by name.
    
    Args:
        name: The name of the processor to retrieve
        
    Returns:
        The requested processor component
        
    Raises:
        KeyError: If the processor is not registered
    """
    if name not in _processors:
        raise KeyError(f"Processor not found: {name}")
    return _processors[name]

def get_all_feature_calculators() -> Dict[str, Any]:
    """
    Get all registered feature calculators.
    
    Returns:
        Dictionary of all registered feature calculators
    """
    return _feature_calculators.copy()

def get_all_feature_transformers() -> Dict[str, Any]:
    """
    Get all registered feature transformers.
    
    Returns:
        Dictionary of all registered feature transformers
    """
    return _feature_transformers.copy()

def get_all_feature_detectors() -> Dict[str, Any]:
    """
    Get all registered feature detectors.
    
    Returns:
        Dictionary of all registered feature detectors
    """
    return _feature_detectors.copy()

def get_all_processors() -> Dict[str, Any]:
    """
    Get all registered processors.
    
    Returns:
        Dictionary of all registered processors
    """
    return _processors.copy()

# Auto-import all feature modules to register components
def _auto_import_features():
    """
    Auto-import all feature modules to ensure components are registered.
    """
    try:
        # Import all feature implementation modules
        from feature_service.features import (
            technical, volatility, volume, sentiment, 
            market_structure, order_flow, pattern
        )
        
        # Import all transformer modules
        from feature_service.transformers import (
            normalizers, filters, aggregators
        )
        
        # Import core modules
        from feature_service import (
            processor, feature_extraction, multi_timeframe
        )
        
        logger.info("Feature service modules loaded successfully")
    except ImportError as e:
        logger.error(f"Error importing feature modules: {e}")
        raise


# In feature_service/__init__.py

def _auto_import_features():
    """
    Automatically import all feature modules
    """
    feature_modules = []
    
    try:
        # Import base classes first
        from feature_service.features.base_feature import BaseFeature
        
        # Try to import feature modules one by one
        try:
            from feature_service.features import order_flow
            feature_modules.append(order_flow)
        except ImportError as e:
            logger.warning(f"Could not import order_flow module: {e}")
            
        try:
            from feature_service.features import pattern
            feature_modules.append(pattern)
        except ImportError as e:
            logger.warning(f"Could not import pattern module: {e}")
            
        try:
            from feature_service.features import sentiment
            feature_modules.append(sentiment)
        except ImportError as e:
            logger.warning(f"Could not import sentiment module: {e}")
            
        try:
            from feature_service.features import volume
            feature_modules.append(volume)
        except ImportError as e:
            logger.warning(f"Could not import volume module: {e}")
        
    except ImportError as e:
        logger.error(f"Error importing feature modules: {e}")
    
    return feature_modules

# Add the shutdown function if it's missing
def shutdown_feature_service():
    """
    Gracefully shut down the feature service and release resources
    """
    logger.info("Shutting down feature service...")
    # Implementation of shutdown logic
# Call auto-import function when module is loaded
_auto_import_features()

# Service initialization flag
initialized = False



def init_feature_service():
    """
    Initialize the feature service.
    Sets up connections to data sources and prepares feature calculators.
    """
    global initialized
    if initialized:
        logger.warning("Feature service already initialized")
        return
    
    logger.info("Initializing feature service")
    
    # Perform any necessary initialization steps
    
    initialized = True
    logger.info("Feature service initialized successfully")

def shutdown_feature_service():
    """
    Shutdown the feature service.
    Clean up resources and close connections.
    """
    global initialized
    if not initialized:
        logger.warning("Feature service not initialized")
        return
    
    logger.info("Shutting down feature service")
    
    # Perform any necessary cleanup
    
    initialized = False
    logger.info("Feature service shutdown complete")
