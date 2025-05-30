#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Features Module Initialization

This module initializes the feature service components and provides a registry
for all available feature calculators. It serves as the entry point for feature
calculation and management, enabling dynamic discovery and usage of different
feature types throughout the system.
"""

import importlib
import inspect
import os
import sys
from typing import Dict, List, Callable, Type, Any
import logging
from common.logger import get_logger
from common.exceptions import FeatureNotFoundError

# Set up logger
logger = get_logger('features')

# Registry to store all feature calculators
FEATURE_REGISTRY: Dict[str, Dict[str, Callable]] = {}

# Forward declaration to avoid circular imports
def register_feature(feature_cls):
    """
    Decorator to register a feature calculator with the system.
    
    Args:
        feature_cls: The feature calculator class to register
        
    Returns:
        The original class
    """
    # Avoid circular import
    from .base_feature import BaseFeatureCalculator
    
    if not issubclass(feature_cls, BaseFeatureCalculator):
        raise TypeError(f"{feature_cls.__name__} must inherit from BaseFeatureCalculator")
    
    # Extract feature name and group
    name = feature_cls.name
    group = feature_cls.group
    
    # Create group if it doesn't exist
    if group not in FEATURE_REGISTRY:
        FEATURE_REGISTRY[group] = {}
    
    # Register the feature
    FEATURE_REGISTRY[group][name] = feature_cls
    logger.debug(f"Registered feature {name} in group {group}")
    
    return feature_cls

# Import base feature after defining register_feature
from .base_feature import BaseFeature

# Make it available at the module level
__all__ = ['BaseFeature', 'register_feature']

# Import modules to make them available - do this after defining register_feature
from . import order_flow
from . import pattern
from . import cross_asset
# We already have pattern.py and order_flow.py, no need to import patterns or order_flows
# Groups for organizing features
FEATURE_GROUPS = {
    'technical': 'Technical indicators like RSI, MACD, Bollinger Bands, etc.',
    'volatility': 'Volatility measurements and related indicators',
    'volume': 'Volume-based analysis indicators',
    'sentiment': 'Market sentiment indicators from news, social, etc.',
    'market_structure': 'Market structure identification features',
    'order_flow': 'Order flow and market microstructure features',
    'pattern': 'Pattern recognition and classification features',
    'fundamental': 'Fundamental analysis features for stocks and crypto',
    'regime': 'Market regime identification features',
    'correlation': 'Correlation-based features across instruments',
    'custom': 'Custom user-defined features'
}

# Base class for feature calculators
class BaseFeatureCalculator:
    """Base class for all feature calculators"""
    
    # Feature metadata
    name: str = "base_feature"
    description: str = "Base feature calculator class"
    group: str = "technical"
    requires_columns: List[str] = ["close"]
    output_columns: List[str] = []
    parameters: Dict[str, Any] = {}
    min_data_points: int = 1
    
    def __init__(self, **kwargs):
        """
        Initialize with optional parameter overrides.
        
        Args:
            **kwargs: Parameter overrides
        """
        # Set parameters from kwargs, falling back to defaults
        self.params = self.parameters.copy()
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                logger.warning(f"Unknown parameter '{key}' for feature {self.name}")
        
        # Set up logger
        self.logger = get_logger(f"feature.{self.name}")
        
    def calculate(self, data):
        """
        Calculate the feature from input data.
        
        Args:
            data: DataFrame with required columns
            
        Returns:
            DataFrame with added feature columns
        """
        raise NotImplementedError("Subclasses must implement calculate()")
        
    def validate_data(self, data):
        """
        Validate input data has required columns and sufficient length.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required columns
        for col in self.requires_columns:
            if col not in data.columns:
                self.logger.error(f"Missing required column '{col}' for {self.name}")
                return False
        
        # Check data length
        if len(data) < self.min_data_points:
            self.logger.error(
                f"Insufficient data for {self.name}. "
                f"Need at least {self.min_data_points} data points, got {len(data)}"
            )
            return False
            
        return True
        
    def __str__(self):
        return f"<{self.name}: {self.description}>"
        
    def __repr__(self):
        return self.__str__()


def register_feature(feature_cls):
    """
    Decorator to register a feature calculator with the system.
    
    Args:
        feature_cls: The feature calculator class to register
        
    Returns:
        The original class
    """
    if not issubclass(feature_cls, BaseFeatureCalculator):
        raise TypeError(f"{feature_cls.__name__} must inherit from BaseFeatureCalculator")
    
    # Extract feature name and group
    name = feature_cls.name
    group = feature_cls.group
    
    # Create group if it doesn't exist
    if group not in FEATURE_REGISTRY:
        FEATURE_REGISTRY[group] = {}
    
    # Register the feature
    FEATURE_REGISTRY[group][name] = feature_cls
    logger.debug(f"Registered feature {name} in group {group}")
    
    return feature_cls


def get_feature_calculator(name: str, group: str = None, **params) -> BaseFeatureCalculator:
    """
    Get a feature calculator instance by name and optional group.
    
    Args:
        name: Name of the feature
        group: Optional group to look in
        **params: Parameters to initialize the feature calculator with
        
    Returns:
        Initialized feature calculator instance
        
    Raises:
        FeatureNotFoundError: If feature not found
    """
    # If group specified, look only in that group
    if group is not None:
        if group not in FEATURE_REGISTRY:
            raise FeatureNotFoundError(f"Feature group '{group}' not found")
        
        if name not in FEATURE_REGISTRY[group]:
            raise FeatureNotFoundError(f"Feature '{name}' not found in group '{group}'")
            
        return FEATURE_REGISTRY[group][name](**params)
    
    # Look in all groups
    for group_name, features in FEATURE_REGISTRY.items():
        if name in features:
            return features[name](**params)
            
    raise FeatureNotFoundError(f"Feature '{name}' not found in any group")


def get_all_features() -> Dict[str, Dict[str, Type[BaseFeatureCalculator]]]:
    """
    Get all registered features.
    
    Returns:
        Dictionary of all registered features grouped by category
    """
    return FEATURE_REGISTRY


def get_group_features(group: str) -> Dict[str, Type[BaseFeatureCalculator]]:
    """
    Get all features in a specific group.
    
    Args:
        group: Feature group name
        
    Returns:
        Dictionary of features in the group
        
    Raises:
        FeatureNotFoundError: If group not found
    """
    if group not in FEATURE_REGISTRY:
        raise FeatureNotFoundError(f"Feature group '{group}' not found")
        
    return FEATURE_REGISTRY[group]


def list_features() -> Dict[str, List[str]]:
    """
    List all available features by group.
    
    Returns:
        Dictionary mapping group names to lists of feature names
    """
    result = {}
    for group, features in FEATURE_REGISTRY.items():
        result[group] = list(features.keys())
    return result


def create_feature_group(group_name: str, description: str = ""):
    """
    Create a new feature group.
    
    Args:
        group_name: Name of the group
        description: Optional description
        
    Returns:
        None
    """
    if group_name in FEATURE_GROUPS:
        logger.warning(f"Feature group '{group_name}' already exists")
        return
        
    FEATURE_GROUPS[group_name] = description or f"Custom feature group: {group_name}"
    FEATURE_REGISTRY[group_name] = {}
    logger.info(f"Created new feature group '{group_name}'")


# Automatically discover and load all feature modules
def autodiscover_features():
    """
    Automatically discover and import all feature modules.
    
    This searches through the features package for Python modules and imports them
    to trigger the feature registration decorators.
    """
    logger.info("Discovering feature modules...")
    
    # Get the directory of this file
    feature_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all Python files in the directory
    for filename in os.listdir(feature_dir):
        # Skip this init file and non-Python files
        if (filename == "__init__.py" or 
            not filename.endswith(".py") or 
            filename.startswith("_")):
            continue
            
        module_name = filename[:-3]  # Remove .py extension
        full_module_name = f"{__name__}.{module_name}"
        
        try:
            # Import the module to trigger registration
            importlib.import_module(full_module_name)
            logger.debug(f"Imported feature module: {module_name}")
        except Exception as e:
            logger.error(f"Error importing feature module {module_name}: {str(e)}")


# Initialize by discovering features
autodiscover_features()

# Log the discovered features
feature_count = sum(len(features) for features in FEATURE_REGISTRY.values())
logger.info(f"Registered {feature_count} features across {len(FEATURE_REGISTRY)} groups")

# Export public API
__all__ = [
    'register_feature',
    'BaseFeatureCalculator',
    'get_feature_calculator',
    'get_all_features',
    'get_group_features',
    'list_features',
    'create_feature_group',
    'FEATURE_GROUPS',
    'FEATURE_REGISTRY',
    'order_flow',
    'pattern',
    'cross_asset'
]

from .volume import (
    analyze_volume_profile,
    detect_volume_climax,
    calculate_volume_profile,
)
from .pattern import detect_patterns

__all__.extend([
    'analyze_volume_profile',
    'detect_volume_climax',
    'calculate_volume_profile',
    'detect_patterns',
])
