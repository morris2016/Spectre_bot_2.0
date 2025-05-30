#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
ML Models - Preprocessing Module Initialization

This module initializes the preprocessing components for the ML subsystem.
It provides utilities for data preprocessing, feature transformation, and scaling.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union

# Initialize logger
logger = logging.getLogger('quantumspectre.ml_models.preprocessing')

# Registry for all preprocessors
PREPROCESSOR_REGISTRY = {}

# Default scaling configurations
DEFAULT_SCALING_CONFIGS = {
    'price': {
        'method': 'min_max',
        'feature_range': (0, 1),
        'window': 1000
    },
    'volume': {
        'method': 'log',
        'base': 10,
        'add_constant': 1
    },
    'technical': {
        'method': 'standard',
        'with_mean': True,
        'with_std': True,
        'window': 500
    },
    'sentiment': {
        'method': 'min_max',
        'feature_range': (-1, 1),
        'window': 200
    }
}

# Default encoding configurations
DEFAULT_ENCODING_CONFIGS = {
    'categorical': {
        'method': 'one_hot',
        'handle_unknown': 'ignore'
    },
    'ordinal': {
        'method': 'ordinal',
        'categories': 'auto'
    },
    'cyclic': {
        'method': 'cyclic',
        'period': {
            'hour': 24,
            'day_of_week': 7,
            'month': 12
        }
    },
    'target': {
        'method': 'label',
        'threshold': 0
    }
}

# Default sampling configurations
DEFAULT_SAMPLING_CONFIGS = {
    'imbalanced': {
        'method': 'smote',
        'k_neighbors': 5,
        'sampling_strategy': 'auto'
    },
    'outliers': {
        'method': 'isolation_forest',
        'contamination': 'auto',
        'n_estimators': 100,
        'max_samples': 'auto'
    },
    'noise': {
        'method': 'gaussian',
        'mean': 0,
        'std': 0.01
    }
}

# Import submodules to register preprocessors

# Register all preprocessors automatically
def register_preprocessor(name: str, preprocessor: Any) -> None:
    """
    Register a preprocessor in the global registry.
    
    Args:
        name: Unique identifier for the preprocessor
        preprocessor: The preprocessor class or function
    """
    if name in PREPROCESSOR_REGISTRY:
        logger.warning(f"Preprocessor '{name}' already registered. Overwriting.")
    
    PREPROCESSOR_REGISTRY[name] = preprocessor
    logger.debug(f"Registered preprocessor: {name}")

def get_preprocessor(name: str) -> Any:
    """
    Retrieve a preprocessor from the registry.
    
    Args:
        name: Name of the preprocessor to retrieve
        
    Returns:
        The requested preprocessor
        
    Raises:
        KeyError: If the preprocessor is not found
    """
    if name not in PREPROCESSOR_REGISTRY:
        raise KeyError(f"Preprocessor '{name}' not found in registry")
    
    return PREPROCESSOR_REGISTRY[name]

def list_preprocessors() -> List[str]:
    """
    List all available preprocessors.
    
    Returns:
        List of preprocessor names
    """
    return list(PREPROCESSOR_REGISTRY.keys())

def get_default_config(config_type: str, feature_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific preprocessor type and feature.
    
    Args:
        config_type: Type of configuration ('scaling', 'encoding', 'sampling')
        feature_type: Type of feature to get configuration for
        
    Returns:
        Dictionary with default configuration
        
    Raises:
        ValueError: If config_type or feature_type is invalid
    """
    if config_type == 'scaling':
        if feature_type not in DEFAULT_SCALING_CONFIGS:
            raise ValueError(f"Unknown feature type '{feature_type}' for scaling configuration")
        return DEFAULT_SCALING_CONFIGS[feature_type].copy()
        
    elif config_type == 'encoding':
        if feature_type not in DEFAULT_ENCODING_CONFIGS:
            raise ValueError(f"Unknown feature type '{feature_type}' for encoding configuration")
        return DEFAULT_ENCODING_CONFIGS[feature_type].copy()
        
    elif config_type == 'sampling':
        if feature_type not in DEFAULT_SAMPLING_CONFIGS:
            raise ValueError(f"Unknown feature type '{feature_type}' for sampling configuration")
        return DEFAULT_SAMPLING_CONFIGS[feature_type].copy()
        
    else:
        raise ValueError(f"Unknown configuration type: {config_type}")

# Version information
__version__ = '1.0.0'

# Import submodules after registry functions are defined
from . import scaling  # noqa
