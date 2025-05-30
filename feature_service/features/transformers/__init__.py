

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Feature Transformers Module

This module provides various transformer classes for processing and transforming
feature data in the QuantumSpectre Elite Trading System. These transformers handle
normalization, scaling, filtering, and other data transformations required for
accurate pattern recognition and signal generation aimed at high win rates.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import logging

from common.logger import get_logger
from common.utils import timing_decorator, parallelize

logger = get_logger(__name__)

# Import transformer implementations
# These will be imported in the actual implementation
# For now, just define the base class and interfaces

class BaseTransformer(ABC):
    """
    Base transformer class that defines the interface for all feature transformers.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the transformer.
        
        Args:
            name: Unique name for this transformer instance
            **kwargs: Additional transformer-specific parameters
        """
        self.name = name
        self.params = kwargs
        self._is_fitted = False
        logger.debug(f"Initialized {self.__class__.__name__} transformer '{name}'")
    
    @property
    def is_fitted(self) -> bool:
        """Check if the transformer has been fitted to data."""
        return self._is_fitted
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BaseTransformer':
        """
        Fit the transformer to the data.
        
        Args:
            data: Input data to fit the transformer
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Fit the transformer to the data and then transform it.
        
        Args:
            data: Input data to fit and transform
            **kwargs: Additional fitting parameters
            
        Returns:
            Transformed data
        """
        return self.fit(data, **kwargs).transform(data)
    
    def reset(self) -> None:
        """Reset the transformer to its initial state."""
        self._is_fitted = False
        logger.debug(f"Reset {self.__class__.__name__} transformer '{self.name}'")
    
    def __repr__(self) -> str:
        """String representation of the transformer."""
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}(name='{self.name}', {params_str})"


class TransformerRegistry:
    """
    Registry for managing and accessing transformer implementations.
    """
    
    _transformers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, transformer_class: type) -> type:
        """
        Register a transformer class.
        
        Args:
            transformer_class: The transformer class to register
            
        Returns:
            The registered transformer class (for decorator use)
        """
        cls._transformers[transformer_class.__name__] = transformer_class
        logger.debug(f"Registered transformer class: {transformer_class.__name__}")
        return transformer_class
    
    @classmethod
    def get(cls, transformer_name: str) -> Optional[type]:
        """
        Get a transformer class by name.
        
        Args:
            transformer_name: Name of the transformer class
            
        Returns:
            The transformer class if found, None otherwise
        """
        return cls._transformers.get(transformer_name)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available transformer classes.
        
        Returns:
            List of transformer class names
        """
        return list(cls._transformers.keys())
    
    @classmethod
    def create(cls, transformer_name: str, instance_name: str, **kwargs) -> Optional[BaseTransformer]:
        """
        Create a new transformer instance.
        
        Args:
            transformer_name: Name of the transformer class
            instance_name: Name for the new instance
            **kwargs: Additional parameters for the transformer
            
        Returns:
            New transformer instance if class found, None otherwise
        """
        transformer_class = cls.get(transformer_name)
        if transformer_class:
            return transformer_class(instance_name, **kwargs)
        else:
            logger.error(f"Transformer class not found: {transformer_name}")
            return None


class TransformerPipeline:
    """
    Pipeline for chaining multiple transformers together.
    """
    
    def __init__(self, name: str):
        """
        Initialize a transformer pipeline.
        
        Args:
            name: Name for this pipeline
        """
        self.name = name
        self.transformers: List[BaseTransformer] = []
        self._is_fitted = False
        logger.debug(f"Initialized transformer pipeline '{name}'")
    
    def add(self, transformer: BaseTransformer) -> 'TransformerPipeline':
        """
        Add a transformer to the pipeline.
        
        Args:
            transformer: Transformer instance to add
            
        Returns:
            Self for method chaining
        """
        self.transformers.append(transformer)
        self._is_fitted = False
        logger.debug(f"Added transformer '{transformer.name}' to pipeline '{self.name}'")
        return self
    
    def remove(self, transformer_name: str) -> 'TransformerPipeline':
        """
        Remove a transformer from the pipeline by name.
        
        Args:
            transformer_name: Name of the transformer to remove
            
        Returns:
            Self for method chaining
        """
        self.transformers = [t for t in self.transformers if t.name != transformer_name]
        self._is_fitted = False
        logger.debug(f"Removed transformer '{transformer_name}' from pipeline '{self.name}'")
        return self
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'TransformerPipeline':
        """
        Fit all transformers in the pipeline to the data.
        
        Args:
            data: Input data to fit the transformers
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        logger.debug(f"Fitting transformer pipeline '{self.name}'")
        
        current_data = data.copy()
        for transformer in self.transformers:
            current_data = transformer.fit_transform(current_data, **kwargs)
        
        self._is_fitted = True
        logger.debug(f"Transformer pipeline '{self.name}' fitted successfully")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformers in the pipeline to the data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        if not self._is_fitted:
            logger.warning(f"Transformer pipeline '{self.name}' used without fitting")
        
        logger.debug(f"Transforming data with pipeline '{self.name}'")
        
        current_data = data.copy()
        for transformer in self.transformers:
            current_data = transformer.transform(current_data)
        
        return current_data
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Fit all transformers to the data and then transform it.
        
        Args:
            data: Input data to fit and transform
            **kwargs: Additional fitting parameters
            
        Returns:
            Transformed data
        """
        return self.fit(data, **kwargs).transform(data)
    
    def reset(self) -> None:
        """Reset all transformers in the pipeline."""
        for transformer in self.transformers:
            transformer.reset()
        
        self._is_fitted = False
        logger.debug(f"Reset transformer pipeline '{self.name}'")
    
    def describe(self) -> Dict[str, Any]:
        """
        Get a description of the pipeline.
        
        Returns:
            Dictionary with pipeline details
        """
        return {
            'name': self.name,
            'num_transformers': len(self.transformers),
            'is_fitted': self._is_fitted,
            'transformers': [
                {
                    'name': t.name,
                    'type': t.__class__.__name__,
                    'is_fitted': t.is_fitted,
                    'params': t.params
                }
                for t in self.transformers
            ]
        }


# Import transformer implementations
try:
    from feature_service.transformers.normalizers import *
    from feature_service.transformers.filters import *
    from feature_service.transformers.aggregators import *
    logger.info("Loaded transformer implementations successfully")
except ImportError as e:
    logger.warning(f"Failed to load some transformer implementations: {e}")

