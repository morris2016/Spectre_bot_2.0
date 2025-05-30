#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Base Feature

This module provides the base class for all feature calculations.
"""

import time
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

from common.metrics import MetricsCollector
from common.exceptions import FeatureCalculationError

class BaseFeature(ABC):
    """Base class for feature calculation components."""
    
    def __init__(self, name=None, config=None):
        """
        Initialize the feature calculator.
        
        Args:
            name: Optional feature name (defaults to class name)
            config: Optional configuration dictionary
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = logging.getLogger(f"feature.{self.name}")
        self.metrics = MetricsCollector(f"feature.{self.name}")
        self.cache = {}
        
    @abstractmethod
    async def calculate(self, data, **kwargs):
        """
        Calculate the feature value from input data.
        
        Args:
            data: Input data (usually OHLCV data)
            **kwargs: Additional parameters
            
        Returns:
            Calculated feature value(s)
        """
        pass
    
    async def calculate_with_timing(self, data, **kwargs):
        """
        Calculate feature with performance timing.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Calculated feature value(s)
        """
        start_time = time.time()
        
        try:
            result = await self.calculate(data, **kwargs)
            
            # Record timing metrics
            elapsed_time = time.time() - start_time
            self.metrics.record_timer(f"calculation_time", elapsed_time)
            self.metrics.increment(f"calculation_count")
            
            return result
            
        except Exception as e:
            # Record error metrics
            self.metrics.increment(f"calculation_error")
            self.logger.error(f"Error calculating feature: {str(e)}")
            raise FeatureCalculationError(f"Failed to calculate {self.name}: {str(e)}")
            
    async def is_applicable(self, data, **kwargs):
        """
        Check if this feature is applicable to the given data.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            bool: True if applicable, False otherwise
        """
        # Default implementation - override in subclasses
        return True
    
    def get_dependencies(self):
        """
        Get the dependencies of this feature.
        
        Returns:
            List of feature names this feature depends on
        """
        # Default implementation - override in subclasses
        return []
    
    def get_metadata(self):
        """
        Get metadata about this feature.
        
        Returns:
            Dict: Feature metadata
        """
        return {
            "name": self.name,
            "description": self.__doc__,
            "dependencies": self.get_dependencies(),
            "config": {k: v for k, v in self.config.items() if not k.startswith('_')}
        }
    
    async def clear_cache(self):
        """Clear the feature cache."""
        self.cache = {}
        
        
    def compute(self, data):
        """
        Compute the feature value
        
        Parameters:
        -----------
        data : dict or pandas.DataFrame
            Input data
            
        Returns:
        --------
        dict or pandas.DataFrame
            Computed feature
        """
        raise NotImplementedError("Subclasses must implement compute method")
    
    def __str__(self):
        return f"{self.name}: {self.description}"
