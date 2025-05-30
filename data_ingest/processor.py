#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Data Processor Base Class

This module defines the base class for all data processors in the system.
Data processors transform raw data into a format usable by other system components.
"""

import time
import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

from common.logger import get_logger
from common.exceptions import DataProcessorError, ProcessorNotFoundError, DataValidationError
from common.metrics import MetricsCollector


class DataProcessor(ABC):
    """Base class for all data processors."""
    
    def __init__(self, config, logger=None):
        """
        Initialize the data processor.
        
        Args:
            config: Processor configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or get_logger(self.__class__.__name__)
        self.metrics = MetricsCollector(f"processor.{self.__class__.__name__.lower()}")
        self.validation_rules = {}
        self.initialize_validation_rules()
    
    def initialize_validation_rules(self):
        """Initialize data validation rules. Override in subclasses."""
        pass
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the data. Main entry point for data processing.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processed data dictionary
            
        Raises:
            DataValidationError: If data validation fails
            DataProcessorError: If processing fails
        """
        try:
            # Validate the data
            if not self.validate(data):
                raise DataValidationError(f"Data validation failed")
            
            # Process the data
            start_time = time.time()
            result = await self._process(data)
            processing_time = time.time() - start_time
            
            # Add metadata
            result['processed_at'] = time.time()
            result['processing_time'] = processing_time
            result['processor'] = self.__class__.__name__
            
            # Update metrics
            self.metrics.increment("data.processed")
            self.metrics.histogram("processing_time", processing_time)
            
            return result
            
        except DataValidationError:
            self.metrics.increment("validation.error")
            raise
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            self.metrics.increment("processing.error")
            raise DataProcessorError(f"Processing failed: {str(e)}")
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate the data against the validation rules.
        
        Args:
            data: Input data dictionary
            
        Returns:
            True if validation succeeds, False otherwise
        """
        if not isinstance(data, dict):
            self.logger.error(f"Data is not a dictionary: {type(data)}")
            return False
        
        for field, rule in self.validation_rules.items():
            # Check if required field exists
            if rule.get('required', False) and field not in data:
                self.logger.error(f"Required field '{field}' missing")
                return False
            
            # Skip validation if field is not present and not required
            if field not in data:
                continue
            
            # Validate field type
            field_type = rule.get('type')
            if field_type and not isinstance(data[field], field_type):
                self.logger.error(f"Field '{field}' has wrong type: {type(data[field])}, expected {field_type}")
                return False
            
            # Validate field value if validator function exists
            validator = rule.get('validator')
            if validator and not validator(data[field]):
                self.logger.error(f"Field '{field}' failed validation")
                return False
        
        return True
    
    @abstractmethod
    async def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the data. Must be implemented by subclasses.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processed data dictionary
        """
        pass

def normalize_instrument_id(instrument_id: str) -> str:
    """
    Normalize an instrument ID to a standard format.
    
    Args:
        instrument_id: The instrument ID to normalize
        
    Returns:
        Normalized instrument ID
    """
    # Remove any whitespace
    normalized = instrument_id.strip()
    
    # Convert to uppercase
    normalized = normalized.upper()
    
    # Handle common formats
    
    # Format: BTC-USD, ETH-USD, etc.
    if re.match(r'^[A-Z0-9]{2,10}-[A-Z]{3,4}$', normalized):
        # Already in standard format
        return normalized
    
    # Format: BTCUSD, ETHUSD, etc.
    if re.match(r'^[A-Z0-9]{2,10}[A-Z]{3,4}$', normalized):
        # Extract base and quote
        for i in range(len(normalized) - 3, 0, -1):
            if normalized[i:] in ['USD', 'EUR', 'GBP', 'JPY', 'BTC', 'ETH', 'USDT', 'USDC']:
                return f"{normalized[:i]}-{normalized[i:]}"
        
        # Default to assuming last 3 chars are quote currency
        return f"{normalized[:-3]}-{normalized[-3:]}"
    
    # Format: BTC/USD, ETH/USD, etc.
    if '/' in normalized:
        base, quote = normalized.split('/', 1)
        return f"{base}-{quote}"
    
    # Format: BTC_USD, ETH_USD, etc.
    if '_' in normalized:
        base, quote = normalized.split('_', 1)
        return f"{base}-{quote}"
    
    # If no specific format detected, return as is
    return normalized

def standardize_ticker_symbol(symbol: str) -> str:
    """
    Standardize ticker symbol to a common format.
    
    Args:
        symbol: The ticker symbol to standardize
        
    Returns:
        Standardized ticker symbol
    """
    # Remove any whitespace
    standardized = symbol.strip()
    
    # Convert to uppercase
    standardized = standardized.upper()
    
    # Handle prefixes like $ or #
    if standardized.startswith('$') or standardized.startswith('#'):
        standardized = standardized[1:]
    
    # Handle suffixes like .X
    if '.' in standardized:
        base, extension = standardized.split('.', 1)
        # Keep known exchange extensions
        if extension in ['L', 'N', 'OQ', 'P', 'B']:
            return standardized
        return base
    
    return standardized

def extract_price(data: Dict, price_field: str = 'price') -> Optional[float]:
    """
    Extract price from data safely.
    
    Args:
        data: Data dictionary
        price_field: Field name for price
        
    Returns:
        Price as float or None if not found
    """
    price = data.get(price_field)
    
    if price is None:
        return None
        
    try:
        return float(price)
    except (ValueError, TypeError):
        return None

def normalize_timestamp(timestamp: Union[int, float, str]) -> int:
    """
    Normalize timestamp to milliseconds since epoch.
    
    Args:
        timestamp: Timestamp in seconds, milliseconds, or ISO format
        
    Returns:
        Timestamp in milliseconds
    """
    import time
    from datetime import datetime
    
    # If string, try to parse as ISO format
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        except ValueError:
            # If not ISO format, try to convert to float
            try:
                timestamp = float(timestamp)
            except ValueError:
                # If all else fails, use current time
                return int(time.time() * 1000)
    
    # If already numeric, convert to milliseconds if needed
    if isinstance(timestamp, (int, float)):
        # If timestamp is in seconds (typical Unix timestamp)
        if timestamp < 10000000000:  # Threshold for seconds vs milliseconds
            return int(timestamp * 1000)
        # If timestamp is already in milliseconds
        return int(timestamp)
    
    # Default to current time
    return int(time.time() * 1000)

def calculate_vwap(trades: List[Dict[str, Any]]) -> Optional[float]:
    """
    Calculate Volume Weighted Average Price (VWAP) from a list of trades.
    
    Args:
        trades: List of trade dictionaries with price and size fields
        
    Returns:
        VWAP or None if no valid trades
    """
    total_volume = 0.0
    volume_price_sum = 0.0
    
    for trade in trades:
        price = trade.get('price')
        size = trade.get('size')
        
        if price is not None and size is not None:
            try:
                price_float = float(price)
                size_float = float(size)
                
                if price_float > 0 and size_float > 0:
                    volume_price_sum += price_float * size_float
                    total_volume += size_float
            except (ValueError, TypeError):
                continue
    
    if total_volume > 0:
        return volume_price_sum / total_volume
    
    return None
