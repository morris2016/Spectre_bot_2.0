#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Core Utility Functions

This module provides a collection of utility functions used throughout the system,
including time handling, data processing, JSON manipulation, validation, and more.
"""

import os
import re
import gzip
import io
import time
import json
import uuid
import hmac
import hashlib
import base64
import pickle
import itertools


class TradingMode:
    """
    Singleton class to manage the current trading mode of the system.
    
    This class ensures that all components have a consistent view of the
    current trading mode (live, paper, backtest, etc.)
    """
    
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    SIMULATION = "simulation"
    OPTIMIZATION = "optimization"
    STRESS_TEST = "stress_test"
    
    def __init__(self):
        self._mode = self.PAPER  # Default to paper trading
        self._observers = []
        
    def get_mode(self) -> str:
        """Get the current trading mode."""
        return self._mode
        
    def set_mode(self, mode: str) -> None:
        """
        Set the current trading mode.
        
        Args:
            mode: The trading mode to set
        """
        if mode not in [self.LIVE, self.PAPER, self.BACKTEST,
                        self.SIMULATION, self.OPTIMIZATION, self.STRESS_TEST]:
            raise ValueError(f"Invalid trading mode: {mode}")
            
        old_mode = self._mode
        self._mode = mode
        
        # Notify observers of mode change
        for observer in self._observers:
            observer(old_mode, mode)
            
    def register_observer(self, observer: callable) -> None:
        """
        Register an observer to be notified of mode changes.
        
        Args:
            observer: Callable that takes (old_mode, new_mode) parameters
        """
        self._observers.append(observer)
        
    def unregister_observer(self, observer: callable) -> None:
        """
        Unregister an observer.
        
        Args:
            observer: The observer to unregister
        """
        if observer in self._observers:
            self._observers.remove(observer)


def normalize_weights(weights: dict) -> dict:
    """
    Normalize a dictionary of weights so they sum to 1.0.
    
    Args:
        weights: Dictionary of items and their weights
        
    Returns:
        Dictionary with normalized weights
    """
    total = sum(weights.values())
    if total == 0:
        # If all weights are zero, assign equal weights
        return {k: 1.0 / len(weights) for k in weights}
    return {k: v / total for k, v in weights.items()}


def format_price(price: float, precision: int = 2) -> str:
    """
    Format a price value with appropriate precision.
    
    Args:
        price: The price to format
        precision: Number of decimal places
        
    Returns:
        Formatted price string
    """
    return f"{price:.{precision}f}"


class CircularBuffer:
    """
    Fixed-size buffer that overwrites oldest data when full.
    
    This is useful for maintaining a sliding window of recent data
    without growing memory usage.
    """
    
    def __init__(self, size: int):
        """
        Initialize a circular buffer.
        
        Args:
            size: Maximum number of items to store
        """
        self.size = size
        self.buffer = [None] * size
        self.position = 0
        self.is_full = False
        
    def append(self, item):
        """Add an item to the buffer."""
        self.buffer[self.position] = item
        self.position = (self.position + 1) % self.size
        if self.position == 0:
            self.is_full = True
            
    def get_all(self):
        """Get all items in the buffer."""
        if self.is_full:
            return self.buffer
        else:
            return self.buffer[:self.position]
            
    def clear(self):
        """Clear the buffer."""
        self.buffer = [None] * self.size
        self.position = 0
        self.is_full = False
        
    def __len__(self):
        """Return the number of items in the buffer."""
        if self.is_full:
            return self.size
        return self.position


def circular_buffer(size: int) -> CircularBuffer:
    """
    Create a new circular buffer.
    
    Args:
        size: Maximum number of items to store
        
    Returns:
        A new CircularBuffer instance
    """
    return CircularBuffer(size)


def chunked_iterable(iterable, chunk_size):
    """
    Break an iterable into chunks of a specified size.
    
    Args:
        iterable: The iterable to chunk
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of the iterable
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: The value to format (0.1 = 10%)
        precision: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


def calculate_atr(high_prices, low_prices, close_prices, period=14):
    """
    Calculate Average True Range (ATR).
    
    Args:
        high_prices: List of high prices
        low_prices: List of low prices
        close_prices: List of close prices
        period: ATR period
        
    Returns:
        List of ATR values
    """
    if len(high_prices) < period + 1:
        return [0] * len(high_prices)
        
    # Calculate True Range
    tr = []
    for i in range(len(high_prices)):
        if i == 0:
            tr.append(high_prices[i] - low_prices[i])
        else:
            tr.append(max(
                high_prices[i] - low_prices[i],
                abs(high_prices[i] - close_prices[i-1]),
                abs(low_prices[i] - close_prices[i-1])
            ))
    
    # Calculate ATR
    atr = [0] * len(tr)
    atr[period-1] = sum(tr[:period]) / period
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
    return atr


def find_peaks(data, window=5):
    """
    Find peaks in a data series.
    
    Args:
        data: List of values
        window: Window size for peak detection
        
    Returns:
        List of peak indices
    """
    peaks = []
    for i in range(window, len(data) - window):
        is_peak = True
        for j in range(1, window + 1):
            if data[i] <= data[i - j] or data[i] <= data[i + j]:
                is_peak = False
                break
        if is_peak:
            peaks.append(i)
    return peaks


def merge_configs(base_config, override_config):
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result


def is_higher_high(current_value, previous_value, threshold=0.0):
    """
    Check if current value is a higher high compared to previous value.
    
    Args:
        current_value: Current value to check
        previous_value: Previous value to compare against
        threshold: Minimum percentage difference required
        
    Returns:
        True if current value is a higher high, False otherwise
    """
    if previous_value <= 0:
        return current_value > 0
        
    diff_pct = (current_value - previous_value) / previous_value
    return diff_pct > threshold


def is_lower_low(current_value, previous_value, threshold=0.0):
    """
    Check if current value is a lower low compared to previous value.
    
    Args:
        current_value: Current value to check
        previous_value: Previous value to compare against
        threshold: Minimum percentage difference required
        
    Returns:
        True if current value is a lower low, False otherwise
    """
    if previous_value <= 0:
        return current_value < 0
        
    diff_pct = (previous_value - current_value) / previous_value
    return diff_pct > threshold


def truncate_float(value, decimals=2):
    """
    Truncate a float to a specified number of decimal places.
    
    Args:
        value: Float value to truncate
        decimals: Number of decimal places
        
    Returns:
        Truncated float value
    """
    factor = 10 ** decimals
    return int(value * factor) / factor


def get_user_preference(key, default=None):
    """
    Get a user preference from the configuration.
    
    Args:
        key: Preference key
        default: Default value if preference not found
        
    Returns:
        Preference value or default
    """
    # In a real implementation, this would load from a user preferences file
    # For now, return the default value
    return default


def zigzag_identification(prices, deviation=0.05):
    """
    Identify zigzag points in a price series.
    
    Args:
        prices: List of price values
        deviation: Minimum percentage deviation for a zigzag point
        
    Returns:
        List of indices of zigzag points
    """
    if len(prices) < 3:
        return []
        
    # Initialize with first point
    zigzag_points = [0]
    trend = None
    
    for i in range(1, len(prices)):
        if trend is None:
            # Determine initial trend
            if prices[i] > prices[0]:
                trend = "up"
            elif prices[i] < prices[0]:
                trend = "down"
            else:
                continue
                
            zigzag_points.append(i)
            continue
        
        last_zigzag_price = prices[zigzag_points[-1]]
        
        if trend == "up":
            # Check for reversal
            if prices[i] < last_zigzag_price * (1 - deviation):
                trend = "down"
                zigzag_points.append(i)
            # Check for new high
            elif prices[i] > last_zigzag_price:
                zigzag_points[-1] = i
        else:  # trend == "down"
            # Check for reversal
            if prices[i] > last_zigzag_price * (1 + deviation):
                trend = "up"
                zigzag_points.append(i)
            # Check for new low
            elif prices[i] < last_zigzag_price:
                zigzag_points[-1] = i
    
    return zigzag_points


class Singleton(type):
    """
    Metaclass for implementing the Singleton pattern.
    
    This ensures only one instance of a class exists throughout the application.
    
    Usage:
        class MyClass(metaclass=Singleton):
            pass
    """
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
import random
import zlib
import socket
import string
import decimal
import datetime
try:
    from dateutil import parser as date_parser  # type: ignore
    DATEUTIL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    DATEUTIL_AVAILABLE = False
    date_parser = None  # type: ignore
import threading
import functools
from concurrent.futures import ThreadPoolExecutor
import itertools
import collections
import urllib.parse
try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    class _DummyNumpy:
        ndarray = list
        integer = int
        floating = float

        def __getattr__(self, name: str):
            raise ImportError("NumPy is required for this functionality")

    np = _DummyNumpy()  # type: ignore

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    class _DummyPandas:
        class Series(list):
            pass

        class DataFrame(dict):
            pass

        class Timestamp:
            pass

        def __getattr__(self, name: str):
            raise ImportError("pandas is required for this functionality")

    pd = _DummyPandas()  # type: ignore
import logging
try:
    import nltk  # type: ignore
    NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    nltk = None  # type: ignore
    NLTK_AVAILABLE = False
import sys
import asyncio
import importlib
import pkgutil
import enum
from pathlib import Path
from functools import wraps
from contextlib import suppress, asynccontextmanager, contextmanager
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Union,
    Callable,
    Tuple,
    Generator,
    Set,
    Type,
    Sequence,
)
import inspect
from common.logger import get_logger, performance_log
from common.constants import OrderSide, OrderType, TimeInForce

# Configure module logger
logger = get_logger(__name__)

# Constants for time utilities
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
DATE_FORMAT = "%Y-%m-%d"
US_TIMESTAMP_FORMAT = "%Y-%m-%d %I:%M:%S %p"
MILLISECONDS_IN_SECOND = 1000


def merge_deep(source, destination):
    """
    Deep merge two dictionaries.

    The source dictionary values will override destination values if there's a conflict.
    Lists will be combined (not overwritten).

    Args:
        source: Source dictionary
        destination: Destination dictionary

    Returns:
        Merged dictionary
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # Get node or create one
            node = destination.setdefault(key, {})
            if isinstance(node, dict):
                merge_deep(value, node)
            else:
                destination[key] = value
        elif isinstance(value, list):
            if key in destination and isinstance(destination[key], list):
                # Combine lists without duplicates
                destination[key] = list(set(destination[key] + value))
            else:
                destination[key] = value
        else:
            destination[key] = value

    return destination


MICROSECONDS_IN_SECOND = 1000000
NANOSECONDS_IN_SECOND = 1000000000




def import_submodules(package_name):
    """
    Import all submodules of a package, recursively.

    Args:
        package_name (str): The name of the package to import submodules from.

    Returns:
        dict: A dictionary mapping module names to the imported modules.
    """
    if isinstance(package_name, str):
        package = importlib.import_module(package_name)
    else:
        package = package_name
        package_name = package.__name__

    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(
        package.__path__, package_name + "."
    ):
        try:
            results[name] = importlib.import_module(name)
        except Exception as e:
            logger.exception(f"Failed to import module {name}: {e}")
    return results


# ======================================
# Time and Date Utilities
# ======================================


def timestamp_ms() -> int:
    """
    Get current timestamp in milliseconds.

    Returns:
        int: Current timestamp in milliseconds
    """
    return int(time.time() * MILLISECONDS_IN_SECOND)


def current_timestamp() -> int:
    """
    Get current timestamp in milliseconds.

    Returns:
        int: Current timestamp in milliseconds
    """
    return int(time.time() * MILLISECONDS_IN_SECOND)


def current_timestamp_micros() -> int:
    """
    Get current timestamp in microseconds.

    Returns:
        int: Current timestamp in microseconds
    """
    return int(time.time() * MICROSECONDS_IN_SECOND)


def current_timestamp_nanos() -> int:
    """
    Get current timestamp in nanoseconds.

    Returns:
        int: Current timestamp in nanoseconds
    """
    return int(time.time() * NANOSECONDS_IN_SECOND)


def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime.datetime:
    """
    Convert timestamp to datetime object, auto-detecting format.

    Args:
        timestamp: Timestamp value (seconds, milliseconds, microseconds, or nanoseconds)

    Returns:
        datetime.datetime: Datetime object
    """
    # Detect timestamp resolution based on value magnitude
    if timestamp > 1e18:  # nanoseconds
        return datetime.datetime.fromtimestamp(timestamp / NANOSECONDS_IN_SECOND)
    elif timestamp > 1e15:  # microseconds
        return datetime.datetime.fromtimestamp(timestamp / MICROSECONDS_IN_SECOND)
    elif timestamp > 1e12:  # milliseconds
        return datetime.datetime.fromtimestamp(timestamp / MILLISECONDS_IN_SECOND)
    else:  # seconds
        return datetime.datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(
    dt: Union[datetime.datetime, str], resolution: str = "ms"
) -> int:
    """
    Convert datetime to timestamp with specified resolution.

    Args:
        dt: Datetime object or string
        resolution: Timestamp resolution ('s', 'ms', 'us', or 'ns')

    Returns:
        int: Timestamp in specified resolution
    """
    if isinstance(dt, str):
        dt = parse_datetime(dt)

    ts = dt.timestamp()

    if resolution == "s":
        return int(ts)
    elif resolution == "ms":
        return int(ts * MILLISECONDS_IN_SECOND)
    elif resolution == "us":
        return int(ts * MICROSECONDS_IN_SECOND)
    elif resolution == "ns":
        return int(ts * NANOSECONDS_IN_SECOND)
    else:
        raise ValueError(
            f"Invalid resolution: {resolution}. Use 's', 'ms', 'us', or 'ns'"
        )


def parse_datetime(date_string: str) -> datetime.datetime:
    """
    Parse datetime from string using flexible formats.

    Args:
        date_string: Datetime string in various formats

    Returns:
        datetime.datetime: Parsed datetime object
    """
    try:
        if DATEUTIL_AVAILABLE:
            return date_parser.parse(date_string)  # type: ignore[attr-defined]
        return datetime.datetime.fromisoformat(date_string)
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(f"Failed to parse datetime: {date_string}", exc_info=e)
        raise


def format_datetime(
    dt: Union[datetime.datetime, str, int, float], fmt: str = TIMESTAMP_FORMAT
) -> str:
    """
    Format datetime object to string.

    Args:
        dt: Datetime object, timestamp, or date string
        fmt: Output format string

    Returns:
        str: Formatted datetime string
    """
    if isinstance(dt, (int, float)):
        dt = timestamp_to_datetime(dt)
    elif isinstance(dt, str):
        dt = parse_datetime(dt)

    return dt.strftime(fmt)


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert timeframe string to seconds.

    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')

    Returns:
        int: Timeframe in seconds
    """
    match = re.match(r"^(\d+)([smhdwM])$", timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    value, unit = match.groups()
    value = int(value)

    if unit == "s":
        return value
    elif unit == "m":
        return value * 60
    elif unit == "h":
        return value * 60 * 60
    elif unit == "d":
        return value * 60 * 60 * 24
    elif unit == "w":
        return value * 60 * 60 * 24 * 7
    elif unit == "M":
        return value * 60 * 60 * 24 * 30  # Approximate
    else:
        raise ValueError(f"Invalid timeframe unit: {unit}")


def timeframe_to_timedelta(timeframe: str) -> datetime.timedelta:
    """
    Convert timeframe string to timedelta.

    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')

    Returns:
        datetime.timedelta: Timeframe as timedelta
    """
    seconds = timeframe_to_seconds(timeframe)
    return datetime.timedelta(seconds=seconds)


def round_timestamp(
    timestamp: Union[int, float], timeframe: str, resolution: str = "ms"
) -> int:
    """
    Round timestamp to nearest timeframe boundary.

    Args:
        timestamp: Timestamp value
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')
        resolution: Timestamp resolution ('s', 'ms', 'us', or 'ns')

    Returns:
        int: Rounded timestamp in original resolution
    """
    # Normalize to seconds
    if resolution == "ms":
        ts_seconds = timestamp / MILLISECONDS_IN_SECOND
    elif resolution == "us":
        ts_seconds = timestamp / MICROSECONDS_IN_SECOND
    elif resolution == "ns":
        ts_seconds = timestamp / NANOSECONDS_IN_SECOND
    else:  # 's'
        ts_seconds = timestamp

    # Get timeframe in seconds
    tf_seconds = timeframe_to_seconds(timeframe)

    # Round
    rounded_seconds = int(ts_seconds // tf_seconds * tf_seconds)

    # Convert back to original resolution
    if resolution == "ms":
        return rounded_seconds * MILLISECONDS_IN_SECOND
    elif resolution == "us":
        return rounded_seconds * MICROSECONDS_IN_SECOND
    elif resolution == "ns":
        return rounded_seconds * NANOSECONDS_IN_SECOND
    else:  # 's'
        return rounded_seconds


def generate_timeframes(
    start_dt: Union[datetime.datetime, str, int],
    end_dt: Union[datetime.datetime, str, int],
    timeframe: str,
) -> List[int]:
    """
    Generate a list of timestamp boundaries between start and end.

    Args:
        start_dt: Start datetime (object, string, or timestamp)
        end_dt: End datetime (object, string, or timestamp)
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')

    Returns:
        List[int]: List of timestamp boundaries in milliseconds
    """
    # Convert inputs to datetime objects
    if isinstance(start_dt, (int, float)):
        start_dt = timestamp_to_datetime(start_dt)
    elif isinstance(start_dt, str):
        start_dt = parse_datetime(start_dt)

    if isinstance(end_dt, (int, float)):
        end_dt = timestamp_to_datetime(end_dt)
    elif isinstance(end_dt, str):
        end_dt = parse_datetime(end_dt)

    # Get timeframe as timedelta
    tf_delta = timeframe_to_timedelta(timeframe)

    # Round start_dt to timeframe boundary
    tf_seconds = timeframe_to_seconds(timeframe)
    start_timestamp = int(start_dt.timestamp())
    rounded_start = int(start_timestamp // tf_seconds * tf_seconds)
    current_dt = datetime.datetime.fromtimestamp(rounded_start)

    # Generate boundaries
    boundaries = []
    while current_dt <= end_dt:
        boundaries.append(int(current_dt.timestamp() * MILLISECONDS_IN_SECOND))
        current_dt += tf_delta

    return boundaries


def create_timeframes(start_dt: Union[datetime.datetime, str, int], end_dt: Union[datetime.datetime, str, int], timeframe: str) -> List[int]:
    """Backward-compatible alias for :func:`generate_timeframes`."""
    return generate_timeframes(start_dt, end_dt, timeframe)


def parse_timeframe(timeframe_str: str) -> Tuple[int, str]:
    """
    Parse a timeframe string into value and unit parts.

    Args:
        timeframe_str: Timeframe string (e.g., '1m', '5m', '1h', '1d')

    Returns:
        Tuple[int, str]: Value and unit parts
    """
    match = re.match(r"^(\d+)([smhdwM])$", timeframe_str)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe_str}")

    value, unit = match.groups()
    return int(value), unit


# ======================================
# Data Handling Utilities
# ======================================


def parse_decimal(value: Union[str, float, int, None]) -> Optional[decimal.Decimal]:
    """
    Parse a value to Decimal with safe handling.

    Args:
        value: Value to parse

    Returns:
        decimal.Decimal or None if input is None
    """
    if value is None:
        return None

    try:
        return decimal.Decimal(str(value))
    except (decimal.InvalidOperation, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse decimal: {value}", exc_info=e)
        return None


def safe_divide(
    numerator: Union[float, int, decimal.Decimal],
    denominator: Union[float, int, decimal.Decimal],
    default: Union[float, int, decimal.Decimal] = 0,
) -> Union[float, decimal.Decimal]:
    """
    Safely divide two numbers, returning a default on division by zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return on error

    Returns:
        Result of division or default on error
    """
    try:
        # Handle division by zero or very small numbers
        if denominator == 0 or (
            isinstance(denominator, float) and abs(denominator) < 1e-10
        ):
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError, ValueError) as e:
        logger.debug(f"Division error: {numerator} / {denominator}", exc_info=e)
        return default


def round_to_tick(
    value: Union[float, decimal.Decimal], tick_size: Union[float, decimal.Decimal]
) -> Union[float, decimal.Decimal]:
    """
    Round a value to the nearest tick size.

    Args:
        value: Value to round
        tick_size: Tick size

    Returns:
        Rounded value
    """
    if tick_size == 0:
        return value

    # Determine if we're working with Decimal or float
    if isinstance(value, decimal.Decimal) or isinstance(tick_size, decimal.Decimal):
        value = parse_decimal(value)
        tick_size = parse_decimal(tick_size)
        return (value / tick_size).quantize(
            decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP
        ) * tick_size
    else:
        # Default float implementation
        return round(value / tick_size) * tick_size


def round_to_tick_size(
    value: Union[float, decimal.Decimal], tick_size: Union[float, decimal.Decimal]
) -> Union[float, decimal.Decimal]:
    """Alias for :func:`round_to_tick` for backward compatibility."""
    return round_to_tick(value, tick_size)


def calculate_change_percent(
    current: Union[float, decimal.Decimal], previous: Union[float, decimal.Decimal]
) -> float:
    """
    Calculate percent change between two values.

    Args:
        current: Current value
        previous: Previous value

    Returns:
        float: Percent change
    """
    return 100.0 * safe_divide(current - previous, previous)


def normalize_value(
    value: float,
    min_value: float,
    max_value: float,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> float:
    """
    Normalize value to a new range.

    Args:
        value: Value to normalize
        min_value: Minimum value in original range
        max_value: Maximum value in original range
        new_min: Minimum value in new range
        new_max: Maximum value in new range

    Returns:
        float: Normalized value
    """
    if min_value == max_value:
        return (new_min + new_max) / 2

    normalized = (value - min_value) / (max_value - min_value)
    return new_min + normalized * (new_max - new_min)


def moving_average(data: List[Union[float, int]], window: int) -> List[float]:
    """
    Calculate simple moving average.

    Args:
        data: List of numeric values
        window: Moving average window size

    Returns:
        List[float]: List of moving averages
    """
    if not data or window <= 0 or window > len(data):
        return []

    result = []
    cumsum = 0

    for i, value in enumerate(data):
        cumsum += value

        if i >= window:
            cumsum -= data[i - window]

        if i >= window - 1:
            result.append(cumsum / window)

    return result


def exponential_moving_average(data: List[Union[float, int]], span: int) -> List[float]:
    """
    Calculate exponential moving average.

    Args:
        data: List of numeric values
        span: EMA span (like pandas implementation)

    Returns:
        List[float]: List of exponential moving averages
    """
    if not data or span <= 0:
        return []

    alpha = 2 / (span + 1)
    result = []

    # Initialize with SMA
    sma = sum(data[:span]) / span if len(data) >= span else sum(data) / len(data)
    ema = sma
    result.append(ema)

    # Calculate EMA for remaining data
    for value in data[span:]:
        ema = value * alpha + ema * (1 - alpha)
        result.append(ema)

    return result


def normalize_data(sequence: Sequence[float]) -> List[float]:
    """Normalize a numeric sequence to the range [0, 1]."""
    arr = np.asarray(sequence, dtype=float)
    if arr.size == 0:
        return []
    min_v = arr.min()
    max_v = arr.max()
    if min_v == max_v:
        return [0.0 for _ in arr]
    return ((arr - min_v) / (max_v - min_v)).tolist()


def calculate_dynamic_threshold(data: Sequence[float], window: int = 100, multiplier: float = 1.5) -> float:
    """Calculate a simple dynamic threshold using rolling mean and std."""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return 0.0
    window_data = arr[-window:]
    return float(window_data.mean() + multiplier * window_data.std())


def rolling_window(sequence, window_size):
    """
    Create rolling windows of data for time series analysis.

    Args:
        sequence: Input sequence (list, numpy array, etc.)
        window_size: Size of the rolling window

    Returns:
        Iterator of sliding windows
    """
    if isinstance(sequence, np.ndarray):
        # Use numpy's stride_tricks for efficient rolling windows
        return np.lib.stride_tricks.sliding_window_view(sequence, window_size)
    else:
        # For other sequence types, use a generator
        if len(sequence) < window_size:
            return []
        return (
            sequence[i : i + window_size]
            for i in range(len(sequence) - window_size + 1)
        )


def efficient_rolling_window(sequence, window_size):
    """
    Alias for rolling_window function.
    Creates efficient rolling windows of data for time series analysis.

    Args:
        sequence: Input sequence (list, numpy array, etc.)
        window_size: Size of the rolling window

    Returns:
        Iterator of sliding windows
    """
    return rolling_window(sequence, window_size)


def is_time_series(df: 'pd.DataFrame', time_column: str | None = None) -> bool:
    """Return True if *df* appears to be a time series ordered by datetime."""
    if time_column and time_column in df.columns:
        series = pd.to_datetime(df[time_column], errors="coerce")
    else:
        if isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            series = df.index
        else:
            series = pd.to_datetime(df.index, errors="coerce")
    if series.isnull().any():
        return False
    return series.is_monotonic_increasing


def create_window_samples(
    data: 'pd.DataFrame | np.ndarray | Sequence',
    window_size: int,
    step_size: int = 1,
    flatten: bool = False,
):
    """Generate sliding windows from *data* with the given window and step size."""
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")

    arr = data.values if hasattr(data, "values") else np.asarray(data)
    if len(arr) < window_size:
        return np.empty((0, window_size))
    num_windows = 1 + (len(arr) - window_size) // step_size
    windows = []
    for idx in range(0, num_windows * step_size, step_size):
        window = arr[idx : idx + window_size]
        windows.append(window.flatten() if flatten else window)
    return np.array(windows)


# ======================================
# String and Data Format Utilities
# ======================================


def camel_to_snake(name: str) -> str:
    """
    Convert camelCase to snake_case.

    Args:
        name: String in camelCase

    Returns:
        str: String in snake_case
    """
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def snake_to_camel(name: str) -> str:
    """
    Convert snake_case to camelCase.

    Args:
        name: String in snake_case

    Returns:
        str: String in camelCase
    """
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def format_number(
    number: Union[float, int, decimal.Decimal], precision: int = None
) -> str:
    """
    Format number with specified precision and thousands separator.

    Args:
        number: Number to format
        precision: Decimal precision (None for automatic)

    Returns:
        str: Formatted number string
    """
    if number is None:
        return "N/A"

    if precision is None:
        # Use string formatting for automatic precision
        if abs(number) >= 1000:
            return f"{number:,.0f}"
        elif abs(number) >= 100:
            return f"{number:,.1f}"
        elif abs(number) >= 10:
            return f"{number:,.2f}"
        elif abs(number) >= 1:
            return f"{number:,.3f}"
        elif abs(number) >= 0.1:
            return f"{number:,.4f}"
        elif abs(number) >= 0.01:
            return f"{number:,.5f}"
        elif abs(number) >= 0.001:
            return f"{number:,.6f}"
        elif abs(number) >= 0.0001:
            return f"{number:,.7f}"
        else:
            return f"{number:,.8f}"
    else:
        # Use specified precision
        format_str = f"{{:,.{precision}f}}"
        return format_str.format(number)


def format_currency(
    amount: Union[float, int, decimal.Decimal],
    currency: str = "USD",
    precision: int = None,
) -> str:
    """
    Format amount as currency with symbol.

    Args:
        amount: Amount to format
        currency: Currency code
        precision: Decimal precision (None for automatic)

    Returns:
        str: Formatted currency string
    """
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "BTC": "₿", "ETH": "Ξ"}

    # Get symbol or use currency code
    symbol = symbols.get(currency, f"{currency} ")

    # Format the amount
    formatted_amount = format_number(amount, precision)

    # Combine symbol and amount
    return f"{symbol}{formatted_amount}"


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to max length with optional suffix.

    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        str: Truncated string
    """
    if len(s) <= max_length:
        return s

    return s[: max_length - len(suffix)] + suffix


def pluralize(count: int, singular: str, plural: str = None) -> str:
    """
    Return singular or plural form based on count.

    Args:
        count: Count value
        singular: Singular form
        plural: Plural form (default is singular + 's')

    Returns:
        str: Appropriate form based on count
    """
    if plural is None:
        plural = singular + "s"

    return singular if count == 1 else plural


# ======================================
# JSON and Data Structure Utilities
# ======================================


class EnhancedJSONEncoder(json.JSONEncoder):
    """Extended JSON encoder that handles additional Python types."""

    def default(self, obj):
        # Handle datetime
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()

        # Handle Decimal
        if isinstance(obj, decimal.Decimal):
            return float(obj)

        # Handle sets
        if isinstance(obj, set):
            return list(obj)

        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle pandas Timestamp
        if hasattr(pd, "Timestamp") and isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        # Handle objects with to_json method
        if hasattr(obj, "to_json"):
            return obj.to_json()

        # Handle objects with __dict__ attribute
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        # Let parent class handle or raise TypeError
        return super().default(obj)


class JsonEncoder(json.JSONEncoder):
    """
    Extended JSON encoder for serializing Python objects to JSON.
    This is an alias for EnhancedJSONEncoder with the same functionality.
    """

    def default(self, obj):
        # Handle datetime
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()

        # Handle Decimal
        if isinstance(obj, decimal.Decimal):
            return float(obj)

        # Handle sets
        if isinstance(obj, set):
            return list(obj)

        # Handle numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle pandas Timestamp
        if hasattr(pd, "Timestamp") and isinstance(obj, pd.Timestamp):
            return obj.isoformat()

        # Handle objects with to_json method
        if hasattr(obj, "to_json"):
            return obj.to_json()

        # Handle objects with __dict__ attribute
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        # Let parent class handle or raise TypeError
        return super().default(obj)


def json_dumps(obj: Any, **kwargs) -> str:
    """
    Enhanced JSON dumps with additional type handling.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps

    Returns:
        str: JSON string
    """
    return json.dumps(obj, cls=EnhancedJSONEncoder, **kwargs)


def json_loads(s: str, **kwargs) -> Any:
    """
    Safe JSON loads with error handling.

    Args:
        s: JSON string
        **kwargs: Additional arguments for json.loads

    Returns:
        Deserialized object
    """
    try:
        return json.loads(s, **kwargs)
    except json.JSONDecodeError as e:
        # Log the error with truncated string
        logger.error(f"JSON decode error: {e} for string: {truncate_string(s)}")
        raise


def load_json_file(path: str) -> Any:
    """Load JSON data from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(data: Any, path: str) -> None:
    """Save JSON data to a file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def deep_update(original: Dict, update: Dict) -> Dict:
    """
    Recursively update a dictionary.

    Args:
        original: Original dictionary
        update: Dictionary with updates

    Returns:
        Dict: Updated dictionary
    """
    for key, value in update.items():
        if (
            key in original
            and isinstance(original[key], dict)
            and isinstance(value, dict)
        ):
            original[key] = deep_update(original[key], value)
        else:
            original[key] = value

    return original


def deep_get(
    dictionary: Dict,
    keys: Union[str, List[str]],
    default: Any = None,
    separator: str = ".",
) -> Any:
    """
    Get a value from nested dictionary using dot notation or key list.

    Args:
        dictionary: Dictionary to search
        keys: Key path (string with separators or list of keys)
        default: Default value if key not found
        separator: Key separator if keys is a string

    Returns:
        Value at key path or default
    """
    if not dictionary:
        return default

    if isinstance(keys, str):
        keys = keys.split(separator)

    for key in keys:
        try:
            if isinstance(dictionary, dict):
                dictionary = dictionary[key]
            else:
                return default
        except (KeyError, TypeError):
            return default

    return dictionary


def flatten_dict(d: Dict, parent_key: str = "", separator: str = ".") -> Dict:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursive calls
        separator: Key separator in flattened keys

    Returns:
        Dict: Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{separator}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, separator).items())
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(d: Dict, separator: str = ".") -> Dict:
    """
    Restore a flattened dictionary to nested form.

    Args:
        d: Flattened dictionary
        separator: Key separator in flattened keys

    Returns:
        Dict: Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(separator)

        # Start at the root
        current = result

        # Work through the key parts
        for part in parts[:-1]:
            # Create middle dictionaries as needed
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value at the leaf
        current[parts[-1]] = value

    return result


def dict_to_object(d: Dict) -> object:
    """
    Convert a dictionary to an object with attributes.

    Args:
        d: Dictionary to convert

    Returns:
        object: Object with attributes from dictionary
    """

    class DictObject:
        def __init__(self, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    value = dict_to_object(value)
                setattr(self, key, value)

        def __repr__(self):
            return str(self.__dict__)

    return DictObject(d)


def dict_to_namedtuple(name: str, data: Dict[str, Any]) -> Any:
    """Convert a dictionary to a namedtuple."""
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")
    NT = collections.namedtuple(name, data.keys())
    return NT(**data)


def group_by(items: List[Any], key_func: Callable) -> Dict:
    """
    Group items by key function.

    Args:
        items: List of items to group
        key_func: Function to extract group key

    Returns:
        Dict: Items grouped by key
    """
    result = {}
    for item in items:
        key = key_func(item)
        if key not in result:
            result[key] = []
        result[key].append(item)

    return result


def chunks(lst: List[Any], n: int) -> Generator[List[Any], None, None]:
    """
    Split list into chunks of size n.

    Args:
        lst: List to split
        n: Chunk size

    Returns:
        Generator of list chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def make_async(func: Callable) -> Callable:
    """Convert a synchronous function into an asynchronous one."""

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


def filter_none_values(d: Dict) -> Dict:
    """
    Remove None values from dictionary.

    Args:
        d: Dictionary to filter

    Returns:
        Dict: Dictionary without None values
    """
    return {k: v for k, v in d.items() if v is not None}


def find_duplicate_items(items: List[Any]) -> List[Any]:
    """
    Find duplicate items in a list.

    Args:
        items: List to check for duplicates

    Returns:
        List: List of duplicate items
    """
    seen = set()
    duplicates = set()

    for item in items:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)


def merge_lists(list1: List, list2: List, key: str = None) -> List:
    """
    Merge two lists of dictionaries by a common key field.

    Args:
        list1: First list of dictionaries
        list2: Second list of dictionaries
        key: Key field for matching (None to merge by position)

    Returns:
        List: Merged list
    """
    if key is None:
        # Merge by position
        result = []
        for i in range(max(len(list1), len(list2))):
            item = {}
            if i < len(list1):
                item.update(list1[i])
            if i < len(list2):
                item.update(list2[i])
            result.append(item)
        return result
    else:
        # Merge by key
        result = list1.copy()

        # Create lookup for list1
        lookup = {item[key]: i for i, item in enumerate(result) if key in item}

        # Process list2
        for item2 in list2:
            if key in item2 and item2[key] in lookup:
                # Merge with existing item
                i = lookup[item2[key]]
                result[i].update(item2)
            else:
                # Add as new item
                result.append(item2)

        return result


# ======================================
# Security and Validation Utilities
# ======================================


def generate_secure_random_string(length: int = 32) -> str:
    """
    Generate a secure random string.

    Args:
        length: Length of the random string

    Returns:
        str: Secure random string
    """
    # Use all printable characters except whitespace
    chars = string.ascii_letters + string.digits + string.punctuation
    return "".join(random.SystemRandom().choice(chars) for _ in range(length))


def generate_uuid() -> str:
    """
    Generate a UUID string.

    Returns:
        str: UUID string
    """
    return str(uuid.uuid4())


def generate_hmac_signature(key: str, message: str, algorithm: str = "sha256") -> str:
    """
    Generate HMAC signature for a message.

    Args:
        key: Secret key
        message: Message to sign
        algorithm: Hash algorithm

    Returns:
        str: Hex-encoded signature
    """
    if isinstance(key, str):
        key = key.encode()
    if isinstance(message, str):
        message = message.encode()

    if algorithm == "sha256":
        digest = hmac.new(key, message, hashlib.sha256).hexdigest()
    elif algorithm == "sha512":
        digest = hmac.new(key, message, hashlib.sha512).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return digest


def hash_content(content: str, algorithm: str = "sha256") -> str:
    """
    Create a hash of content for deduplication and identification purposes.

    Args:
        content: The content to hash
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        str: Hexadecimal hash string
    """
    # Convert to bytes if string
    if isinstance(content, str):
        content = content.encode()

    # Create the hash based on specified algorithm
    if algorithm == "sha256":
        hash_obj = hashlib.sha256(content)
    elif algorithm == "md5":
        hash_obj = hashlib.md5(content)
    elif algorithm == "sha1":
        hash_obj = hashlib.sha1(content)
    else:
        # Default to sha256 if unknown algorithm
        hash_obj = hashlib.sha256(content)

    # Return hexadecimal digest
    return hash_obj.hexdigest()


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.

    Args:
        url: URL to check

    Returns:
        bool: True if URL is valid
    """
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def is_valid_email(email: str) -> bool:
    """
    Check if an email address is valid.

    Args:
        email: Email address to check

    Returns:
        bool: True if email is valid
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for all operating systems.

    Args:
        filename: Filename to sanitize

    Returns:
        str: Sanitized filename
    """
    # Remove invalid characters
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized = "".join(c for c in filename if c in valid_chars)

    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")

    # Ensure it's not empty and doesn't start with a dot
    if not sanitized or sanitized.startswith("."):
        sanitized = "f" + sanitized

    return sanitized


def validate_required_keys(d: Dict, required_keys: List[str]) -> List[str]:
    """
    Validate that dictionary contains all required keys.

    Args:
        d: Dictionary to validate
        required_keys: List of required keys

    Returns:
        List[str]: List of missing keys (empty if all present)
    """
    return [key for key in required_keys if key not in d]


def mask_sensitive_data(data: Dict, sensitive_keys: List[str]) -> Dict:
    """
    Mask sensitive data in a dictionary.

    Args:
        data: Dictionary containing sensitive data
        sensitive_keys: List of sensitive key patterns (supports wildcards)

    Returns:
        Dict: Dictionary with masked sensitive data
    """
    result = data.copy()

    def mask_value(value: str) -> str:
        if not value:
            return value
        if len(value) <= 4:
            return "*" * len(value)
        return value[:2] + "*" * (len(value) - 4) + value[-2:]

    for key, value in data.items():
        # Check if this key should be masked
        should_mask = any(
            key == pattern or (pattern.endswith("*") and key.startswith(pattern[:-1]))
            for pattern in sensitive_keys
        )

        if should_mask and isinstance(value, str):
            result[key] = mask_value(value)
        elif isinstance(value, dict):
            result[key] = mask_sensitive_data(value, sensitive_keys)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            result[key] = [
                (
                    mask_sensitive_data(item, sensitive_keys)
                    if isinstance(item, dict)
                    else item
                )
                for item in value
            ]

    return result


# ======================================
# Network and System Utilities
# ======================================


def get_host_info() -> Dict[str, Any]:
    """
    Get information about the host system.

    Returns:
        Dict: System information
    """
    info = {
        "hostname": socket.gethostname(),
        "ip": socket.gethostbyname(socket.gethostname()),
        "platform": sys.platform,
        "python_version": sys.version,
        "time": datetime.datetime.now().isoformat(),
        "pid": os.getpid(),
    }

    try:
        import psutil

        process = psutil.Process(os.getpid())
        info.update(
            {
                "cpu_count": os.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "process_memory": process.memory_info().rss,
                "process_cpu_percent": process.cpu_percent(),
                "disk_usage": psutil.disk_usage("/").percent,
            }
        )
    except ImportError:
        # psutil not available
        pass

    return info


def gpu_info() -> Dict[str, Any]:
    """Return basic GPU information using torch if available."""
    info: Dict[str, Any] = {}
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            info["device_count"] = device_count
            info["devices"] = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append(
                    {
                        "id": i,
                        "name": props.name,
                        "total_memory": props.total_memory,
                    }
                )
    except Exception:
        pass

    return info


def gpu_stats() -> Dict[str, Any]:
    """Return current GPU usage statistics if available."""
    stats: Dict[str, Any] = {}
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            stats["device_count"] = torch.cuda.device_count()
            stats["allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
    except Exception:
        pass

    return stats

def is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """
    Check if a network port is open.

    Args:
        host: Host to check
        port: Port to check
        timeout: Connection timeout in seconds

    Returns:
        bool: True if port is open
    """
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((host, port))
        s.close()
        return result == 0
    except:
        return False


def rate_limit(max_calls: int, period: float, call_limit_reached: Callable = None):
    """
    Decorator for rate limiting function calls.

    Args:
        max_calls: Maximum number of calls in period
        period: Time period in seconds
        call_limit_reached: Function to call when rate limit is reached

    Returns:
        Decorator function
    """

    def decorator(func):
        # Initialize timestamps list
        timestamps = collections.deque(maxlen=max_calls)
        lock = threading.RLock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()

                # Remove expired timestamps
                while timestamps and timestamps[0] < now - period:
                    timestamps.popleft()

                # Check if we're at the limit
                if len(timestamps) >= max_calls:
                    if call_limit_reached:
                        return call_limit_reached(*args, **kwargs)
                    else:
                        # Sleep until we can make another call
                        sleep_time = timestamps[0] - (now - period)
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                # Add current timestamp and call the function
                timestamps.append(time.time())
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Alias for backward compatibility
rate_limited = rate_limit


def retry(
    max_attempts: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple = (Exception,),
):
    """
    Decorator for retrying a function on exception.

    Args:
        max_attempts: Maximum number of attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Delay multiplier for subsequent retries
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorator function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = retry_delay

            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_attempts:
                        # Max attempts reached, re-raise
                        logger.warning(
                            f"Max retry attempts ({max_attempts}) reached for {func.__name__}"
                        )
                        raise

                    # Log the exception and retry
                    logger.info(
                        f"Retry {attempt}/{max_attempts} for {func.__name__} in {delay:.2f}s: {str(e)}"
                    )
                    time.sleep(delay)

                    # Increase delay for next retry
                    delay *= backoff_factor

        return wrapper

    return decorator


def timer(func):
    """
    Decorator for timing function execution.

    Args:
        func: Function to time

    Returns:
        Decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(
            f"Function {func.__name__} took {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.start = 0.0
        self.end = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end = time.time()
        logger.debug(f"{self.name} took {self.end - self.start:.4f} seconds")


def create_uuid() -> str:
    """Generate a new UUID string."""

    return UuidUtils.generate()


def create_unique_id() -> str:
    """Generate a short unique identifier."""

    return UuidUtils.generate_short()


# Retry with exponential backoff for async functions
async def retry_with_backoff(
    coro,
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
    exceptions=(Exception,),
    logger=None,
):
    """
    Retry an async coroutine with exponential backoff.

    Args:
        coro: Async function to retry (as a callable that returns a coroutine)
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exceptions to catch and retry on
        logger: Optional logger for logging retries

    Returns:
        Result of the coroutine if successful

    Raises:
        RuntimeError: If all retry attempts fail
    """
    for attempt in range(max_retries):
        try:
            return await coro()
        except exceptions as e:
            delay = min(max_delay, base_delay * (2**attempt))
            if logger:
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s"
                )
            await asyncio.sleep(delay)
    raise RuntimeError(f"All {max_retries} retries failed")


def retry_with_backoff_decorator(
    max_retries=5, base_delay=1.0, max_delay=60.0, exceptions=(Exception,), logger=None
):
    """
    Decorator for retrying an async function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exceptions: Tuple of exceptions to catch and retry on
        logger: Optional logger for logging retries

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async def _func():
                return await func(*args, **kwargs)

            return await retry_with_backoff(
                _func, max_retries, base_delay, max_delay, exceptions, logger
            )

        return wrapper

    return decorator


def exponential_backoff(base_delay=1.0, max_delay=60.0, factor=2.0):
    """
    Decorator for async functions to retry with exponential backoff.

    Args:
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        factor: Multiplier for delay after each retry

    Returns:
        Decorated async function
    """

    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            delay = base_delay
            while True:
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Exception: {e}, retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    delay = min(delay * factor, max_delay)

        return wrapper

    return decorator


def time_execution(label):
    """
    Decorator for timing execution of functions.

    Args:
        label: Label to use in log message

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(f"{label} took {end - start:.2f}s")
            return result

        return wrapper

    return decorator


def time_function(label):
    """Deprecated alias for :func:`time_execution`."""
    return time_execution(label)


def calculate_checksum(data: bytes, method="sha256") -> str:
    """
    Calculate checksum from binary data.

    Args:
        data: Binary data to hash
        method: Hash algorithm to use

    Returns:
        str: Hexadecimal hash string
    """
    if method == "sha256":
        return hashlib.sha256(data).hexdigest()
    elif method == "md5":
        return hashlib.md5(data).hexdigest()
    else:
        raise ValueError(f"Unsupported checksum method: {method}")


def ensure_future(coro_or_future, loop=None, name=None):
    """
    Enhanced version of asyncio.ensure_future with additional functionality.
    Creates a Task from a coroutine or returns the Future directly.

    Args:
        coro_or_future: Coroutine or Future object
        loop: Optional event loop (defaults to current loop)
        name: Optional name for the task

    Returns:
        asyncio.Task or Future object
    """
    if loop is None:
        loop = asyncio.get_event_loop()

    task = asyncio.ensure_future(coro_or_future, loop=loop)

    # Set name if provided and it's a Task (not a Future)
    if name is not None and isinstance(task, asyncio.Task):
        task.set_name(name)

    return task


def create_task_name(prefix="task"):
    """
    Generate a unique name for asyncio tasks.

    Args:
        prefix: Prefix for the task name

    Returns:
        str: Unique task name
    """
    # Use a UUID for uniqueness
    unique_id = str(uuid.uuid4()).split("-")[0]
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}_{unique_id}"


# Thread-safe atomic counter
class AtomicCounter:
    """
    Thread-safe counter for atomic operations.
    Useful for generating unique IDs or keeping track of counts across threads.
    """

    def __init__(self, initial_value=0):
        """
        Initialize the atomic counter.

        Args:
            initial_value: Starting value for the counter (default: 0)
        """
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self, delta=1):
        """
        Atomically increment the counter and return the new value.

        Args:
            delta: Value to add to the counter (default: 1)

        Returns:
            int: New counter value after incrementing
        """
        with self._lock:
            self._value += delta
            return self._value

    def decrement(self, delta=1):
        """
        Atomically decrement the counter and return the new value.

        Args:
            delta: Value to subtract from the counter (default: 1)

        Returns:
            int: New counter value after decrementing
        """
        with self._lock:
            self._value -= delta
            return self._value

    def get(self):
        """
        Get the current counter value.

        Returns:
            int: Current counter value
        """
        with self._lock:
            return self._value

    def set(self, new_value):
        """
        Set the counter to a new value.

        Args:
            new_value: New value for the counter

        Returns:
            int: New counter value
        """
        with self._lock:
            self._value = new_value
            return self._value

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return f"AtomicCounter(value={self.get()})"


# SafeDict for missing key handling
class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


# ======================================
# Trading-Specific Utilities
# ======================================


def calculate_order_size(
    balance: float,
    price: float,
    risk_percent: float,
    stop_loss_percent: float = None,
    stop_loss_price: float = None,
) -> float:
    """
    Calculate position size based on account balance and risk parameters.

    Args:
        balance: Account balance
        price: Current price
        risk_percent: Percentage of balance to risk (0-100)
        stop_loss_percent: Stop loss percentage from entry (optional)
        stop_loss_price: Specific stop loss price (optional)

    Returns:
        float: Position size in base units
    """
    if not balance or not price or not risk_percent:
        return 0

    risk_amount = balance * (risk_percent / 100)

    if stop_loss_price:
        # Calculate using specific stop price
        price_distance = abs(price - stop_loss_price)
        if price_distance <= 0:
            return 0
        return risk_amount / price_distance
    elif stop_loss_percent:
        # Calculate using stop loss percentage
        price_distance = price * (stop_loss_percent / 100)
        if price_distance <= 0:
            return 0
        return risk_amount / price_distance
    else:
        # Simple percentage of balance at market
        return risk_amount / price


def calculate_position_value(size: float, price: float) -> float:
    """
    Calculate position value.

    Args:
        size: Position size
        price: Price

    Returns:
        float: Position value
    """
    return size * price


def calculate_profit_after_fees(
    entry_price: float,
    exit_price: float,
    position_size: float,
    fee_rate: float,
    is_long: bool = True,
) -> float:
    """
    Calculate the profit or loss after fees for a trade.

    Args:
        entry_price: Entry price of the position
        exit_price: Exit price of the position
        position_size: Size of the position in base units
        fee_rate: Fee rate as a decimal (e.g., 0.001 for 0.1%)
        is_long: Whether this is a long position (True) or short position (False)

    Returns:
        float: Profit or loss after fees
    """
    # Calculate position values
    entry_value = position_size * entry_price
    exit_value = position_size * exit_price

    # Calculate fees
    entry_fee = entry_value * fee_rate
    exit_fee = exit_value * fee_rate
    total_fees = entry_fee + exit_fee

    # Calculate profit/loss
    if is_long:
        profit = exit_value - entry_value
    else:
        profit = entry_value - exit_value

    # Subtract fees
    net_profit = profit - total_fees

    return net_profit


def calculate_pip_value(
    size: float, pip_size: float, price: float, quote_currency_rate: float = 1.0
) -> float:
    """
    Calculate pip value for a position.

    Args:
        size: Position size
        pip_size: Size of 1 pip in decimal
        price: Current price
        quote_currency_rate: Exchange rate to account currency

    Returns:
        float: Pip value in account currency
    """
    pip_value = size * pip_size * quote_currency_rate

    # For currency pairs where base is the account currency
    if "USD/" in price or "/USD" in price:
        if "/USD" in price:  # XXX/USD pairs
            pip_value = size * pip_size
        else:  # USD/XXX pairs
            pip_value = size * pip_size / price

    return pip_value


def calculate_arbitrage_profit(
    buy_price: float,
    sell_price: float,
    quantity: float,
    buy_fee_rate: float = 0.0,
    sell_fee_rate: float = 0.0,
) -> float:
    """Calculate net arbitrage profit after fees."""
    if quantity <= 0:
        return 0.0
    gross = (sell_price - buy_price) * quantity
    fees = (buy_price * buy_fee_rate + sell_price * sell_fee_rate) * quantity
    return gross - fees


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_percent: float,
) -> float:
    """Basic position sizing using risk percentage and stop loss distance."""
    if account_balance <= 0 or risk_percent <= 0 or stop_loss_percent <= 0:
        return 0.0
    risk_amount = account_balance * risk_percent
    return risk_amount / stop_loss_percent


def detect_market_condition(prices: Sequence[float]) -> str:
    """Simple market condition detection based on price direction."""
    if len(prices) < 2:
        return "unknown"
    if prices[-1] > prices[0]:
        return "bullish"
    if prices[-1] < prices[0]:
        return "bearish"
    return "sideways"


def calculate_risk_reward(*args: Union[str, float]) -> float:
    """Compute risk-reward ratio for a trade.

    This helper accepts either ``(entry_price, stop_loss, take_profit)`` or
    ``(action, entry_price, stop_loss, take_profit)``. The ``action`` argument
    can be ``"buy"`` or ``"sell"`` and is optional.  The ratio is calculated as
    ``abs(take_profit - entry_price) / abs(entry_price - stop_loss)``.

    Args:
        *args: Arguments as described above.

    Returns:
        float: Risk-reward ratio. ``0`` if inputs are invalid or risk is ``0``.
    """

    if len(args) == 3:
        entry_price, stop_loss, take_profit = args
        action = "buy"
    elif len(args) == 4:
        action, entry_price, stop_loss, take_profit = args
    else:
        raise ValueError("calculate_risk_reward expects 3 or 4 arguments")

    try:
        if str(action).lower() == "sell":

            risk = abs(stop_loss - entry_price)
            reward = abs(stop_loss - take_profit)
        else:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)

    except Exception:
        return 0.0

    if risk == 0:
        return 0.0

    return reward / risk


def calculate_confidence_score(
    votes: Dict[str, float], reasoning_data: Dict[str, Dict[str, Any]]
) -> float:
    """Calculate overall confidence score from council votes.

    Each entry in ``votes`` represents the normalized weight for an action
    (e.g., ``{"buy": 0.6, "sell": 0.4}``). ``reasoning_data`` contains per
    council information with the council's chosen action and confidence.  The
    final score is a weighted average of council confidences using their
    corresponding vote weights.

    Args:
        votes: Normalized vote weights per action.
        reasoning_data: Mapping of council name to action/confidence data.

    Returns:
        float: Confidence score in the range ``0`` to ``1``.
    """

    if not votes or not reasoning_data:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0

    for info in reasoning_data.values():
        action = info.get("action")
        confidence = info.get("confidence", 0.0)
        weight = votes.get(action, 0.0)
        weighted_sum += weight * confidence
        weight_total += weight

    if weight_total == 0:
        return 0.0

    return weighted_sum / weight_total


def normalize_probability(value: float) -> float:
    """Normalize a probability value to the ``0-1`` range."""

    if value is None:
        return 0.0

    if value < 0:
        return 0.0
    if value > 1:
        if value <= 100:
            return value / 100.0
        return 1.0
    return float(value)


def weighted_average(values: List[float], weights: List[float]) -> float:
    """Return the weighted average of *values* using *weights*."""

    if not values or not weights or len(values) != len(weights):
        raise ValueError("values and weights must be non-empty and the same length")

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    return sum(v * w for v, w in zip(values, weights)) / total_weight


def time_weighted_average(values: List[float], timestamps: List[float]) -> float:
    """Compute a simple time weighted average for a series."""

    if not values or not timestamps or len(values) != len(timestamps):
        raise ValueError("values and timestamps must be the same length")

    if len(values) == 1:
        return float(values[0])

    durations = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    durations.insert(0, durations[0])

    total_duration = sum(durations)
    if total_duration == 0:
        return float(np.mean(values))

    return sum(v * d for v, d in zip(values, durations)) / total_duration


def validate_signal(signal: Dict[str, Any]) -> bool:
    """Basic validation for a trading signal dictionary."""

    required = {
        "symbol",
        "action",
        "entry_price",
        "stop_loss",
        "take_profit",
        "confidence",
    }

    if not isinstance(signal, dict):
        return False

    for field in required:
        if field not in signal:
            return False

    if (
        not isinstance(signal["confidence"], (int, float))
        or not 0 <= signal["confidence"] <= 1
    ):
        return False

    numeric_fields = ["entry_price", "stop_loss", "take_profit"]
    for field in numeric_fields:
        if not isinstance(signal[field], (int, float)):
            return False

    return True


def calculate_win_rate(wins: int, losses: int) -> float:
    """
    Calculate win rate.

    Args:
        wins: Number of winning trades
        losses: Number of losing trades

    Returns:
        float: Win rate percentage
    """
    total = wins + losses
    if total == 0:
        return 0
    return (wins / total) * 100


def calculate_success_rate(successes: int, attempts: int) -> float:
    """
    Calculate the success rate as a percentage.

    Args:
        successes: Number of successful outcomes
        attempts: Total number of attempts

    Returns:
        float: Success rate as a percentage (0-100)
    """
    if attempts == 0:
        return 0.0
    return (successes / attempts) * 100.0


def calculate_risk_reward_ratio(risk: float, reward: float) -> float:
    """
    Calculate risk-reward ratio.

    Args:
        risk: Risk amount
        reward: Reward amount

    Returns:
        float: Risk-reward ratio
    """
    if risk <= 0:
        return 0
    return reward / risk


def calculate_expectancy(win_rate: float,
                        avg_win: float,
                        avg_loss: float) -> float:

    """
    Calculate system expectancy.

    Args:
        win_rate: Win rate as percentage (0-100)
        avg_win: Average win amount
        avg_loss: Average loss amount (positive value)

    Returns:
        float: System expectancy
    """
    win_decimal = win_rate / 100
    return (win_decimal * avg_win) - ((1 - win_decimal) * avg_loss)


def calculate_expected_value(trades: List[Union[float, Dict[str, float]]]) -> float:
    """Calculate the expected value from a sequence of trades.

    Each trade can be provided as a numeric profit/loss value or as a dictionary
    containing a ``pnl`` or ``profit`` key. Positive values indicate winning
    trades while negative values indicate losses.

    Args:
        trades: Collection of trade results.

    Returns:
        Expected value per trade.
    """
    if not trades:
        return 0.0

    pnl_values = []
    for trade in trades:
        if isinstance(trade, dict):
            value = trade.get("pnl", trade.get("profit"))
        else:
            value = trade
        if value is None:
            continue
        pnl_values.append(float(value))

    if not pnl_values:
        return 0.0

    wins = [v for v in pnl_values if v > 0]
    losses = [abs(v) for v in pnl_values if v <= 0]
    win_rate = calculate_success_rate(len(wins), len(pnl_values))
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return calculate_expectancy(win_rate, avg_win, avg_loss)


def calculate_kelly_criterion(win_rate: float,
                             avg_win_loss_ratio: float) -> float:

    """
    Calculate Kelly criterion for optimal position sizing.

    Args:
        win_rate: Win rate as decimal (0-1)
        avg_win_loss_ratio: Ratio of average win to average loss

    Returns:
        float: Kelly percentage as decimal
    """
    # Convert win rate to decimal if it's a percentage
    if win_rate > 1:
        win_rate = win_rate / 100

    lose_rate = 1 - win_rate

    # Full Kelly formula: K = W/A - (1-W)/B where:
    # W = win rate, 1-W = lose rate
    # A = amount lost per trade (as positive number)
    # B = amount won per trade
    # Simplified when expressing as a ratio B/A:
    # K = W - (1-W)/(B/A)

    # Check valid inputs
    if lose_rate == 0 or avg_win_loss_ratio == 0:
        return 0

    kelly = win_rate - (lose_rate / avg_win_loss_ratio)

    # Limit to sensible range
    return max(0, min(kelly, 1))


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for a series of returns.

    Args:
        returns: List of period returns (not cumulative)
        risk_free_rate: Risk-free rate for the period

    Returns:
        float: Sharpe ratio
    """
    if not returns:
        return 0

    # Convert to numpy array for calculations
    returns_array = np.array(returns)

    # Calculate excess return
    excess_returns = returns_array - risk_free_rate

    # Calculate mean and standard deviation
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns, ddof=1)  # Use sample std

    if std_excess_return == 0:
        return 0

    # Calculate and return Sharpe ratio
    return mean_excess_return / std_excess_return


def calculate_sharpe(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """Backward-compatible alias for :func:`calculate_sharpe_ratio`."""
    return calculate_sharpe_ratio(returns, risk_free_rate)


def calculate_sortino_ratio(
    returns: List[float], risk_free_rate: float = 0.0, target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio for a series of returns.

    Args:
        returns: List of period returns (not cumulative)
        risk_free_rate: Risk-free rate for the period
        target_return: Target return (defaults to 0)

    Returns:
        float: Sortino ratio
    """
    if not returns:
        return 0

    # Convert to numpy array for calculations
    returns_array = np.array(returns)

    # Calculate excess return over risk-free rate
    excess_returns = returns_array - risk_free_rate

    # Calculate mean excess return
    mean_excess_return = np.mean(excess_returns)

    # Calculate downside deviation (only negative returns against target)
    downside_returns = excess_returns[excess_returns < target_return] - target_return

    if len(downside_returns) == 0:
        return float("inf")  # No downside, perfect Sortino

    downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))

    if downside_deviation == 0:
        return 0

    # Calculate and return Sortino ratio
    return mean_excess_return / downside_deviation


def calculate_sortino(returns: List[float], target_return: float = 0.0, risk_free_rate: float = 0.0) -> float:
    """Backward-compatible alias for :func:`calculate_sortino_ratio`."""
    return calculate_sortino_ratio(returns, risk_free_rate, target_return)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown from an equity curve.

    Args:
        equity_curve: List of equity values over time

    Returns:
        float: Maximum drawdown as a percentage
    """
    if not equity_curve:
        return 0

    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_curve)

    # Calculate drawdown in percent terms
    drawdowns = (equity_curve - running_max) / running_max * 100

    # Get the maximum drawdown
    max_drawdown = np.min(drawdowns)

    return abs(max_drawdown)


def calculate_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """
    Calculate Calmar ratio.

    Args:
        annual_return: Annual return as a percentage
        max_drawdown: Maximum drawdown as a percentage

    Returns:
        float: Calmar ratio
    """
    if max_drawdown == 0:
        return float("inf")  # No drawdown, perfect Calmar

    return annual_return / max_drawdown


def calculate_volatility(
    prices: Union[pd.Series, List[float]], window: int = 20
) -> float:
    """Calculate historical volatility based on log returns."""
    if isinstance(prices, list):
        prices = pd.Series(prices)
    returns = np.log(prices).diff().dropna()
    if returns.empty:
        return 0.0
    if len(returns) > window:
        returns = returns[-window:]
    return float(returns.std() * np.sqrt(len(returns)))


def is_categorical(series: pd.Series) -> bool:
    """Determine if a pandas Series is categorical."""
    return series.dtype.name == 'category' or series.dtype == object


def is_cyclical(series: pd.Series, name: str = "") -> bool:
    """Heuristic check for cyclical data like hours or months."""
    if series.empty:
        return False
    unique = series.dropna().unique()
    if series.dtype.kind in {'i', 'u'}:
        if name.lower() in {"month", "day", "hour", "minute", "second"}:
            return True
        if unique.min() == 0 and unique.max() in {11, 23, 59}:
            return True
    return False


def is_ordinal(series: pd.Series) -> bool:
    """Check if a Series represents ordinal values."""
    return pd.api.types.is_integer_dtype(series) and series.nunique() / len(series) < 0.5


def calculate_correlation(
    series1: Union[pd.Series, List[float]],
    series2: Union[pd.Series, List[float]],
) -> float:
    """Compute the correlation coefficient between two data series."""
    if isinstance(series1, list):
        series1 = pd.Series(series1)
    if isinstance(series2, list):
        series2 = pd.Series(series2)
    if series1.empty or series2.empty:
        return 0.0
    min_len = min(len(series1), len(series2))
    if min_len == 0:
        return 0.0
    return float(series1[-min_len:].corr(series2[-min_len:]))


def calculate_correlation_matrix(
    symbol_dict: Dict[str, Union[np.ndarray, pd.Series]],
) -> pd.DataFrame:
    """Compute a correlation matrix from closing price data.

    Parameters
    ----------
    symbol_dict : Dict[str, Union[np.ndarray, pd.Series]]
        Mapping of symbols to arrays or Series of closing prices.

    Returns
    -------
    pandas.DataFrame
        Correlation matrix indexed and labeled by symbol.
    """

    if not symbol_dict:
        return pd.DataFrame()

    df = pd.DataFrame({k: pd.Series(v) for k, v in symbol_dict.items()})
    return df.corr()


def calculate_drawdown(
    equity_curve: Union[pd.Series, List[float]],
) -> Tuple[float, float]:
    """Calculate maximum and current drawdown percentages."""
    if isinstance(equity_curve, list):
        equity_curve = pd.Series(equity_curve)
    if equity_curve.empty:
        return 0.0, 0.0
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max * 100
    max_drawdown = drawdowns.min()
    current_drawdown = drawdowns.iloc[-1]
    return abs(float(max_drawdown)), abs(float(current_drawdown))


def calculate_liquidation_price(
    side: Union[str, enum.Enum],
    entry_price: float,
    leverage: float,
    maintenance_margin: float = 0.005,
) -> float:
    """Estimate liquidation price for a leveraged position."""
    if leverage <= 0 or entry_price <= 0:
        return 0.0
    if isinstance(side, enum.Enum):
        side = side.value
    side = str(side).lower()
    if side == "long":
        return entry_price * (1 - 1 / leverage + maintenance_margin)
    return entry_price * (1 + 1 / leverage - maintenance_margin)


def z_score(value: float, mean: float, std_dev: float) -> float:
    """
    Calculate z-score for a value.

    Args:
        value: Value to calculate z-score for
        mean: Mean of the distribution
        std_dev: Standard deviation of the distribution

    Returns:
        float: Z-score
    """
    if std_dev == 0:
        return 0
    return (value - mean) / std_dev


def calculate_z_score(value: float, data_series: List[float]) -> float:
    """
    Calculate the z-score of a value relative to a data series.

    Args:
        value: The value to calculate the z-score for
        data_series: List of historical values to compare against

    Returns:
        float: The z-score (number of standard deviations from the mean)
    """
    if not data_series or len(data_series) < 2:
        return 0.0

    mean = sum(data_series) / len(data_series)
    variance = sum((x - mean) ** 2 for x in data_series) / len(data_series)
    std_dev = variance**0.5

    if std_dev == 0:
        return 0.0

    return (value - mean) / std_dev


def is_price_consolidating(prices: List[float], threshold_percent: float = 2.0) -> bool:
    """
    Check if price is consolidating within a range.

    Args:
        prices: List of prices
        threshold_percent: Maximum percentage difference for consolidation

    Returns:
        bool: True if price is consolidating
    """
    if not prices:
        return False

    price_min = min(prices)
    price_max = max(prices)
    price_range = price_max - price_min
    threshold = price_min * (threshold_percent / 100)

    return price_range <= threshold


def is_breaking_out(
    prices: List[float], lookback: int = 20, threshold_percent: float = 2.0
) -> bool:
    """
    Check if price is breaking out of a consolidation.

    Args:
        prices: List of prices (most recent last)
        lookback: Number of prices to look back for consolidation
        threshold_percent: Minimum percentage change for breakout

    Returns:
        bool: True if price is breaking out
    """
    if len(prices) < lookback + 1:
        return False

    # Get consolidation period prices
    consolidation_prices = prices[-lookback - 1 : -1]
    current_price = prices[-1]

    # Calculate consolidation range
    price_min = min(consolidation_prices)
    price_max = max(consolidation_prices)

    # Check if current price is breaking out
    threshold = price_max * (threshold_percent / 100)

    return (
        current_price > price_max + threshold or current_price < price_min - threshold
    )


def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Calculate pivot points (floor method).

    Args:
        high: High price
        low: Low price
        close: Close price

    Returns:
        Dict: Calculated pivot points
    """
    pivot = (high + low + close) / 3

    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)

    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)

    return {"pivot": pivot, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


# Backwards compatibility alias
pivot_points = calculate_pivot_points


def obfuscate_sensitive_data(data: Union[str, Dict, List], level: int = 1) -> Union[str, Dict, List]:

    """
    Obfuscate sensitive data to prevent leakage of confidential information.
    Different from mask_sensitive_data, this focuses on scrubbing content rather than masking keys.

    Args:
        data: String, dictionary, or list to obfuscate
        level: Obfuscation level (1-3, higher means more aggressive)

    Returns:
        Obfuscated data in the same format as the input
    """
    if isinstance(data, str):
        # Obfuscate email addresses
        data = re.sub(r"[\w\.-]+@[\w\.-]+", "[EMAIL]", data)

        # Obfuscate phone numbers
        data = re.sub(
            r"\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "[PHONE]", data
        )

        # Obfuscate IP addresses
        data = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]", data)

        # Obfuscate URLs if level > 1
        if level > 1:
            data = re.sub(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", "[URL]", data)

        # Obfuscate financial values if level > 2
        if level > 2:
            # Match common currency patterns
            data = re.sub(r"[$€£¥]?\s?\d+(?:,\d{3})*(?:\.\d+)?", "[AMOUNT]", data)

    elif isinstance(data, dict):
        # Recursively process dictionary values
        return {k: obfuscate_sensitive_data(v, level) for k, v in data.items()}

    elif isinstance(data, list):
        # Recursively process list items
        return [obfuscate_sensitive_data(item, level) for item in data]

    return data


def exponential_smoothing(data: List[float], alpha: float = 0.3) -> List[float]:
    """
    Apply simple exponential smoothing to a time series.

    Args:
        data: List of data points (time series)
        alpha: Smoothing factor (0 < alpha < 1)

    Returns:
        List of smoothed values
    """
    if not data:
        return []

    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1 (exclusive)")

    smoothed = [data[0]]  # Initialize with first value

    for i in range(1, len(data)):
        # Formula: s_t = alpha * x_t + (1 - alpha) * s_{t-1}
        smoothed_val = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
        smoothed.append(smoothed_val)

    return smoothed


def exponential_smooth(data: List[float], alpha: float = 0.3) -> List[float]:
    """
    Alias for exponential_smoothing with a shorter name.

    Args:
        data: List of values to smooth
        alpha: Smoothing factor (0-1)

    Returns:
        List[float]: Smoothed values
    """
    return exponential_smoothing(data, alpha)



def async_retry_with_backoff_decorator(
    max_retries=3,
    backoff_factor=2,
    initial_wait=1.0,
    max_wait=60.0,
    retry_exceptions=(Exception,),
):
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_retries: Maximum number of retries
        backoff_factor: Multiplier for each successive backoff
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        retry_exceptions: Exceptions to catch and retry on

    Returns:
        Decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await async_retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                initial_wait=initial_wait,
                max_wait=max_wait,
                retry_exceptions=retry_exceptions,
                **kwargs,
            )

        return wrapper

    return decorator


# Alias for backward compatibility
async_retry_with_backoff = async_retry_with_backoff_decorator


# Define the interface for external usage
def cache_with_ttl(ttl_seconds=300):
    """
    Decorator for caching function results with a time-to-live (TTL).

    Args:
        ttl_seconds: Time-to-live in seconds (default: 300)

    Returns:
        Decorated function
    """
    import time
    import functools

    def decorator(func):
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key from the function arguments
            key = str(args) + str(sorted(kwargs.items()))

            # Check if result is in cache and not expired
            current_time = time.time()
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return result

            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)

            # Clean up expired entries
            for k in list(cache.keys()):
                if current_time - cache[k][1] > ttl_seconds:
                    del cache[k]

            return result

        return wrapper

    return decorator


def safe_execute(func, *args, default=None, log_error=True, **kwargs):
    """
    Execute a function safely, catching any exceptions.

    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        default: Default value to return on error
        log_error: Whether to log the error
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Function result or default value on error
    """
    import logging

    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            logger = logging.getLogger("utils")
            logger.error(f"Error executing {func.__name__}: {str(e)}")
        return default


def periodic_reset(seconds=None, minutes=None, hours=None, days=None):
    """
    Decorator for functions that should reset their state periodically.

    Args:
        seconds: Reset interval in seconds
        minutes: Reset interval in minutes
        hours: Reset interval in hours
        days: Reset interval in days

    Returns:
        Decorated function
    """
    import time
    import functools

    # Calculate total seconds
    total_seconds = 0
    if seconds:
        total_seconds += seconds
    if minutes:
        total_seconds += minutes * 60
    if hours:
        total_seconds += hours * 3600
    if days:
        total_seconds += days * 86400

    if total_seconds <= 0:
        total_seconds = 3600  # Default: 1 hour

    def decorator(func):
        # Store state
        state = {"last_reset": time.time(), "cache": {}}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if reset is needed
            current_time = time.time()
            if current_time - state["last_reset"] > total_seconds:
                state["cache"] = {}
                state["last_reset"] = current_time

            # Generate cache key
            key = str(args) + str(sorted(kwargs.items()))

            # Check if result is cached
            if key in state["cache"]:
                return state["cache"][key]

            # Call function and cache result
            result = func(*args, **kwargs)
            state["cache"][key] = result
            return result

        return wrapper

    return decorator


# Dictionary to store registered components
_REGISTERED_COMPONENTS = {}


def register_component(name, component):
    """
    Register a component by name for later retrieval.

    Args:
        name: Component name
        component: Component object or class

    Returns:
        The registered component
    """
    _REGISTERED_COMPONENTS[name] = component
    return component


def get_registered_components():
    """
    Get all registered components.

    Returns:
        Dictionary of registered components
    """
    return _REGISTERED_COMPONENTS.copy()


def get_registered_component(name, default=None):
    """
    Get a registered component by name.

    Args:
        name: Component name
        default: Default value if component not found

    Returns:
        The component or default value
    """
    return _REGISTERED_COMPONENTS.get(name, default)


def timeit(func):
    """Decorator for timing function execution."""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(f"Function {func.__name__} took {elapsed_time:.4f} seconds")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(f"Function {func.__name__} took {elapsed_time:.4f} seconds")
        return result

    # Use appropriate wrapper based on whether the function is async or not
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Alias for backward compatibility
timing_decorator = timeit


def execution_time_ms(func):
    """
    Decorator that measures execution time and returns it along with the result.

    Args:
        func: Function to measure execution time for

    Returns:
        Tuple of (result, execution_time_ms)
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # ms
        return result, execution_time

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # ms
        return result, execution_time

    # Use appropriate wrapper based on whether the function is async or not
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def rolling_apply(data, window, func):
    """
    Apply a function to rolling windows of a data series.

    Args:
        data: Data series (list or numpy array)
        window: Window size
        func: Function to apply to each window

    Returns:
        List of results for each window
    """
    if len(data) < window:
        return []

    result = []
    for i in range(len(data) - window + 1):
        window_data = data[i : i + window]
        result.append(func(window_data))

    return result


def numpy_rolling_window(arr, window):
    """
    Create rolling windows of a numpy array.

    Args:
        arr: Numpy array
        window: Window size

    Returns:
        Array of rolling windows
    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def parallelize_calculation(func, data_list, num_processes=None):
    """
    Parallelize a calculation across multiple processes.

    Args:
        func: Function to parallelize
        data_list: List of data items to process
        num_processes: Number of processes to use (None = CPU count)

    Returns:
        List of results
    """
    import multiprocessing

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    if num_processes <= 1 or len(data_list) <= 1:
        # For small data or requested serial processing, don't use multiprocessing
        return [func(item) for item in data_list]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(func, data_list)

    return results


# Alias for backward compatibility
parallelize = parallelize_calculation


def exponential_decay(values, decay_factor=0.9):
    """
    Apply exponential decay to a series of values.

    Args:
        values: List of values
        decay_factor: Decay factor (0-1)

    Returns:
        List of values with exponential decay applied
    """
    if not values:
        return []

    result = [values[0]]
    for i in range(1, len(values)):
        result.append(values[i] * decay_factor + result[i - 1] * (1 - decay_factor))

    return result


def window_calculation(data, window, func, min_periods=None):
    """
    Apply a function to sliding windows of data.

    Args:
        data: List or numpy array of data
        window: Window size
        func: Function to apply to each window
        min_periods: Minimum number of observations required

    Returns:
        List of results
    """
    if isinstance(data, np.ndarray):
        # Use numpy functions for efficiency
        if min_periods is None:
            min_periods = window

        result = []
        for i in range(len(data)):
            if i < window - 1:
                if i >= min_periods - 1:
                    window_data = data[: i + 1]
                    result.append(func(window_data))
                else:
                    result.append(None)
            else:
                window_data = data[i - window + 1 : i + 1]
                result.append(func(window_data))

        return result
    else:
        # For regular lists
        if min_periods is None:
            min_periods = window

        result = []
        for i in range(len(data)):
            if i < window - 1:
                if i >= min_periods - 1:
                    window_data = data[: i + 1]
                    result.append(func(window_data))
                else:
                    result.append(None)
            else:
                window_data = data[i - window + 1 : i + 1]
                result.append(func(window_data))

        return result


def exponential_decay_weights(window, decay_factor=0.94):
    """
    Generate exponential decay weights for a window.

    Args:
        window: Window size
        decay_factor: Decay factor (0-1)

    Returns:
        Numpy array of weights that sum to 1
    """
    weights = np.array([decay_factor**i for i in range(window)])
    weights = weights[::-1]  # Reverse to give higher weight to recent values
    return weights / weights.sum()  # Normalize to sum to 1


def sigmoid(x):
    """
    Calculate the sigmoid function for the input.

    Args:
        x: Input value

    Returns:
        Sigmoid value
    """
    import numpy as np

    return 1 / (1 + np.exp(-x))


def singleton(cls):
    """
    Decorator to implement the singleton pattern.

    Args:
        cls: Class to make singleton

    Returns:
        Singleton class
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def setup_event_loop(debug=False, thread_name_prefix="", max_tasks=None):
    """
    Set up an asyncio event loop with proper configuration.

    Args:
        debug: Whether to enable debug mode
        thread_name_prefix: Prefix for thread names in the executor
        max_tasks: Maximum number of tasks in the event loop

    Returns:
        Configured event loop
    """
    import asyncio

    # Create a new event loop
    loop = asyncio.new_event_loop()

    # Configure the loop
    loop.set_debug(debug)

    # Configure thread pool executor
    if thread_name_prefix or max_tasks:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(
            thread_name_prefix=thread_name_prefix, max_workers=max_tasks
        )
        loop.set_default_executor(executor)

    # Set as the current event loop
    asyncio.set_event_loop(loop)

    return loop


async def cancel_all_tasks(tasks, timeout=10.0):
    """
    Cancel all provided tasks and wait for them to complete.

    Args:
        tasks: Set or list of tasks to cancel
        timeout: Maximum time to wait for tasks to complete cancellation

    Returns:
        Set of tasks that did not complete within the timeout
    """
    import asyncio

    if not tasks:
        return set()

    # Convert to set if it's not already
    tasks = set(tasks)

    # Request cancellation for all tasks
    for task in tasks:
        if not task.done():
            task.cancel()

    # Wait for all tasks to complete cancellation
    pending = tasks
    try:
        # Use wait_for to limit the time we wait
        done, pending = await asyncio.wait(
            tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED
        )
    except asyncio.CancelledError:
        # If this function itself is cancelled, re-cancel all tasks
        for task in pending:
            if not task.done():
                task.cancel()
        # Re-raise to propagate cancellation
        raise

    # Log warnings for tasks that didn't complete
    if pending:
        logger.warning(
            f"{len(pending)} tasks did not complete cancellation within {timeout}s"
        )

    return pending


async def create_async_task(coro, name=None, logger=None):
    """
    Create and schedule an asyncio task with proper error handling.

    Args:
        coro: Coroutine to schedule as a task
        name: Optional name for the task
        logger: Optional logger for error reporting

    Returns:
        asyncio.Task: The created task
    """
    if name is None:
        name = create_task_name()

    if logger is None:
        logger = get_logger(__name__)

    async def _wrapped_coro():
        try:
            return await coro
        except asyncio.CancelledError:
            logger.debug(f"Task {name} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in async task {name}: {str(e)}", exc_info=True)
            raise

    task = asyncio.create_task(_wrapped_coro(), name=name)
    return task


def validate_timeframe(timeframe: str) -> str:
    """
    Validate that a timeframe string is in the correct format.

    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')

    Returns:
        str: The validated timeframe string

    Raises:
        ValueError: If the timeframe format is invalid
    """
    match = re.match(r"^(\d+)([smhdwM])$", timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    value, unit = match.groups()
    value = int(value)

    # Validate that the value is positive
    if value <= 0:
        raise ValueError(f"Timeframe value must be positive: {timeframe}")

    # Validate the unit
    if unit not in "smhdwM":
        raise ValueError(f"Invalid timeframe unit: {unit}")

    return timeframe


def get_higher_timeframes(timeframe: str, count: int = 3) -> List[str]:
    """
    Generate a list of higher timeframes based on a given timeframe.

    Args:
        timeframe: Base timeframe string (e.g., '1m', '5m', '1h', '1d')
        count: Number of higher timeframes to generate

    Returns:
        List[str]: List of higher timeframe strings
    """
    # Validate the input timeframe
    validate_timeframe(timeframe)

    # Parse the timeframe
    value, unit = parse_timeframe(timeframe)

    # Define the unit progression
    unit_progression = ["m", "h", "d", "w", "M"]

    # Define standard timeframe values for each unit
    standard_values = {
        "s": [1, 5, 15, 30],
        "m": [1, 5, 15, 30],
        "h": [1, 2, 4, 6, 8, 12],
        "d": [1, 3, 7],
        "w": [1, 2],
        "M": [1, 3, 6],
    }

    result = []
    current_unit_idx = unit_progression.index(unit) if unit in unit_progression else -1

    # If the unit is seconds or not in the progression, start with minutes
    if unit == "s" or current_unit_idx == -1:
        current_unit_idx = 0  # Start with minutes

    # Generate higher timeframes
    remaining = count
    current_unit = unit

    while remaining > 0 and current_unit_idx < len(unit_progression):
        current_unit = unit_progression[current_unit_idx]

        # If we're still on the same unit as the input timeframe,
        # only consider values higher than the input value
        values_to_consider = [
            v
            for v in standard_values[current_unit]
            if current_unit != unit or v > value
        ]

        # If no higher values in this unit, move to the next unit
        if not values_to_consider:
            current_unit_idx += 1
            continue

        # Add timeframes from this unit
        for val in values_to_consider:
            if remaining <= 0:
                break

            result.append(f"{val}{current_unit}")
            remaining -= 1

        # Move to the next unit
        current_unit_idx += 1

    return result


def calculate_rolling_correlation(series1, series2, window=20):
    """
    Calculate rolling correlation between two series.

    Args:
        series1: First data series
        series2: Second data series
        window: Rolling window size

    Returns:
        List of correlation values
    """
    if len(series1) != len(series2):
        raise ValueError("Series must be of equal length")

    if len(series1) < window:
        return []

    correlations = []
    for i in range(window, len(series1) + 1):
        window_s1 = series1[i - window : i]
        window_s2 = series2[i - window : i]

        # Calculate correlation
        mean_s1 = sum(window_s1) / window
        mean_s2 = sum(window_s2) / window

        num = sum(
            (window_s1[j] - mean_s1) * (window_s2[j] - mean_s2) for j in range(window)
        )
        den1 = sum((window_s1[j] - mean_s1) ** 2 for j in range(window))
        den2 = sum((window_s2[j] - mean_s2) ** 2 for j in range(window))

        if den1 == 0 or den2 == 0:
            correlations.append(0)
        else:
            correlations.append(num / ((den1 * den2) ** 0.5))

    return correlations


def get_submodules(package_name):
    """
    Get all submodules of a package.

    Args:
        package_name: Name of the package

    Returns:
        List of submodule names
    """
    if isinstance(package_name, str):
        package = importlib.import_module(package_name)
    else:
        package = package_name
        package_name = package.__name__

    submodules = []
    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        submodules.append(name)
        if is_pkg:
            submodules.extend(get_submodules(name))

    return submodules



def compress_data(data: bytes) -> bytes:
    """Compress binary data using gzip."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
        f.write(data)
    return buffer.getvalue()


def decompress_data(data: bytes) -> bytes:
    """Decompress gzip-compressed binary data."""
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode='rb') as f:
        return f.read()


def create_directory(path, exist_ok=True):
    """
    Create a directory and any necessary parent directories.

    Args:
        path: Directory path to create
        exist_ok: If True, don't raise an error if directory already exists

    Returns:
        Path to the created directory
    """
    try:
        os.makedirs(path, exist_ok=exist_ok)
        return path
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {str(e)}")
        raise


def create_directory_if_not_exists(path: str) -> str:
    """Create directory if it does not already exist."""
    return create_directory(path, exist_ok=True)



ASSET_PRECISION_MAP = {
    'BTC': 8,
    'ETH': 8,
    'USDT': 2,
}


def calculate_zscore(data: Sequence[float]) -> float:
    """Calculate the z-score of the latest data point."""
    series = np.asarray(data, dtype=float)
    return (series[-1] - series.mean()) / (series.std() + 1e-10)


def detect_outliers(data: Sequence[float], threshold: float = 3.0) -> List[int]:
    """Return indices of values beyond ``threshold`` standard deviations."""
    series = np.asarray(data, dtype=float)
    mean = series.mean()
    std = series.std() + 1e-10
    return [i for i, x in enumerate(series) if abs(x - mean) > threshold * std]



def safe_nltk_download(resource: str, quiet: bool = True) -> bool:
    """Check for an NLTK resource without downloading.

    If the resource is not found locally, log a warning and return ``False``
    instead of attempting a network download. This prevents network timeouts
    when running in restricted environments.

    Args:
        resource: Name of the NLTK resource (e.g. ``'vader_lexicon'``).
        quiet: Unused, maintained for API compatibility.

    Returns:
        ``True`` if the resource is available locally, otherwise ``False``.
    """
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        logger = logging.getLogger(__name__)
        logger.warning("NLTK resource '%s' not available; skipping download", resource)
        return False


def compress_object(data: Any) -> bytes:
    """Serialize and gzip-compress arbitrary Python objects."""
    serialized = pickle.dumps(data)
    return gzip.compress(serialized)


def calculate_metrics(y_true: Sequence[Any], y_pred: Sequence[Any]) -> Dict[str, float]:
    """Compute basic classification metrics using scikit-learn."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def decompress_object(data: bytes) -> Any:
    """Decompress and deserialize data produced by :func:`compress_object`."""
    decompressed = gzip.decompress(data)
    return pickle.loads(decompressed)


def save_to_file(obj: Any, path: str) -> None:
    """Serialize an object to a binary file using ``pickle``."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_from_file(path: str) -> Any:
    """Load a serialized object from a binary file."""
    with open(path, "rb") as f:
        return pickle.load(f)




class ThreadSafeDict:
    """
    Thread-safe dictionary implementation using a lock.
    """

    def __init__(self, initial_data=None):
        """
        Initialize the thread-safe dictionary.

        Args:
            initial_data: Optional initial dictionary data
        """
        self._dict = initial_data.copy() if initial_data else {}
        self._lock = threading.RLock()

    def __getitem__(self, key):
        """Get item with thread safety."""
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key, value):
        """Set item with thread safety."""
        with self._lock:
            self._dict[key] = value

    def __delitem__(self, key):
        """Delete item with thread safety."""
        with self._lock:
            del self._dict[key]

    def __contains__(self, key):
        """Check if key exists with thread safety."""
        with self._lock:
            return key in self._dict

    def __len__(self):
        """Get dictionary length with thread safety."""
        with self._lock:
            return len(self._dict)

    def __iter__(self):
        """Iterate over keys with thread safety."""
        with self._lock:
            return iter(self._dict.copy())

    def get(self, key, default=None):
        """Get item with default value and thread safety."""
        with self._lock:
            return self._dict.get(key, default)

    def pop(self, key, default=None):
        """Pop item with thread safety."""
        with self._lock:
            return self._dict.pop(key, default)

    def clear(self):
        """Clear dictionary with thread safety."""
        with self._lock:
            self._dict.clear()

    def update(self, other=None, **kwargs):
        """Update dictionary with thread safety."""
        with self._lock:
            if other:
                self._dict.update(other)
            if kwargs:
                self._dict.update(kwargs)

    def items(self):
        """Get items with thread safety."""
        with self._lock:
            return list(self._dict.items())

    def keys(self):
        """Get keys with thread safety."""
        with self._lock:
            return list(self._dict.keys())

    def values(self):
        """Get values with thread safety."""
        with self._lock:
            return list(self._dict.values())

    def copy(self):
        """Get a copy of the dictionary with thread safety."""
        with self._lock:
            return self._dict.copy()


class ClassRegistry:
    """Simple registry for dynamically loaded classes."""

    def __init__(self) -> None:
        # Mapping of class name to the actual class reference
        self._classes: Dict[str, type] = {}

    def register(self, cls: Type) -> None:
        """Register a class reference using its ``__name__``."""
        self._classes[cls.__name__] = cls

    def get(self, name: str) -> type:
        """Retrieve a previously registered class by name.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        try:
            return self._classes[name]
        except KeyError as exc:
            raise KeyError(f"{name!r} not registered") from exc

    def get_all(self) -> List[type]:
        """Return all registered class references."""
        return list(self._classes.values())


class AsyncService:
    """Minimal async service base class used by services throughout the system."""

    def __init__(
        self,
        name: str = "",
        config: Optional[Any] = None,
        signal_bus: Optional["SignalBus"] = None,
    ) -> None:
        self.name = name
        self.config = config or {}
        self.signal_bus = signal_bus or SignalBus()

    async def start(self) -> None:
        """Start the service.  Subclasses should override."""
        return None

    async def stop(self) -> None:
        """Stop the service.  Subclasses should override."""
        return None


class Signal(str, enum.Enum):
    """Enumeration of basic system-wide events for the signal bus."""

    ACCOUNT_BALANCE_UPDATED = "account_balance_updated"
    MARKET_DATA_UPDATED = "market_data_updated"
    TRADE_EXECUTED = "trade_executed"
    TRADE_CLOSED = "trade_closed"
    POSITION_SIZE_REQUESTED = "position_size_requested"
    POSITION_SIZE_RESPONSE = "position_size_response"
    STOP_LOSS_REQUESTED = "stop_loss_requested"
    STOP_LOSS_RESPONSE = "stop_loss_response"
    TAKE_PROFIT_REQUESTED = "take_profit_requested"
    TAKE_PROFIT_RESPONSE = "take_profit_response"
    RISK_ASSESSMENT_REQUESTED = "risk_assessment_requested"
    RISK_ASSESSMENT_RESPONSE = "risk_assessment_response"
    MARKET_REGIME_CHANGED = "market_regime_changed"
    VOLATILITY_SPIKE_DETECTED = "volatility_spike_detected"
    CIRCUIT_BREAKER_ACTIVATED = "circuit_breaker_activated"
    CIRCUIT_BREAKER_DEACTIVATED = "circuit_breaker_deactivated"
    RISK_LEVEL_CHANGED = "risk_level_changed"
    DRAWDOWN_PROTECTION_ACTIVATED = "drawdown_protection_activated"
    ADJUST_STOP_LOSS = "adjust_stop_loss"
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"


class SignalBus:
    """Simple synchronous signal bus for decoupled communication."""

    def __init__(self) -> None:
        self._subscribers: Dict[Signal, List[Callable]] = {}

    def register(self, signal: Signal, callback: Callable) -> None:
        """Register a callback for the given signal."""
        self._subscribers.setdefault(signal, []).append(callback)

    def emit(self, signal: Signal, *args: Any, **kwargs: Any) -> None:
        """Invoke callbacks registered for the given signal."""
        for cb in list(self._subscribers.get(signal, [])):
            if asyncio.iscoroutinefunction(cb):
                asyncio.create_task(cb(*args, **kwargs))
            else:
                cb(*args, **kwargs)

    # ``get_signal`` for backward compatibility with previous API
    def get_signal(self, name: str) -> "Signal":
        return Signal(name)

    # ``get`` is kept for backwards compatibility
    get = get_signal


# Additional utility functions needed by intelligence modules


def create_event_loop(debug=False, thread_name_prefix="", max_tasks=None):
    """
    Create and configure an asyncio event loop.

    This is a wrapper around setup_event_loop for backward compatibility.

    Args:
        debug: Enable asyncio debug mode
        thread_name_prefix: Prefix for thread names
        max_tasks: Maximum number of tasks

    Returns:
        Configured asyncio event loop
    """
    return setup_event_loop(debug, thread_name_prefix, max_tasks)


def run_in_executor(loop, executor, func, *args, **kwargs):
    """
    Run a function in an executor.

    Args:
        loop: Asyncio event loop
        executor: Executor to run the function in
        func: Function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Future representing the execution of the function
    """
    if loop is None:
        loop = asyncio.get_event_loop()
    return loop.run_in_executor(executor, lambda: func(*args, **kwargs))


def benchmark(func):
    """
    Decorator to benchmark function execution time.

    Args:
        func: Function to benchmark

    Returns:
        Wrapped function that logs execution time
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.debug(
            f"Function {func.__name__} took {(end_time - start_time) * 1000:.2f}ms to execute"
        )
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(
            f"Function {func.__name__} took {(end_time - start_time) * 1000:.2f}ms to execute"
        )
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def serialize_numpy(obj):
    """
    Serialize numpy arrays and other objects for JSON serialization.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation of the object
    """
    if isinstance(obj, np.ndarray):
        return {
            "__type__": "numpy.ndarray",
            "data": obj.tolist(),
            "dtype": str(obj.dtype),
        }
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return {
            "__type__": "pandas.DataFrame",
            "data": obj.to_dict(orient="records"),
            "index": (
                obj.index.tolist() if not isinstance(obj.index, pd.RangeIndex) else None
            ),
        }
    elif isinstance(obj, pd.Series):
        return {
            "__type__": "pandas.Series",
            "data": obj.tolist(),
            "index": (
                obj.index.tolist() if not isinstance(obj.index, pd.RangeIndex) else None
            ),
            "name": obj.name,
        }
    return obj


def serialize_dict(d: Dict[str, Any], exclude_keys: List[str] = None) -> str:
    """
    Serialize a dictionary to a JSON string with special handling for numpy types.

    Args:
        d: Dictionary to serialize
        exclude_keys: Optional list of keys to exclude from serialization

    Returns:
        str: JSON string representation of the dictionary
    """
    if exclude_keys:
        d = {k: v for k, v in d.items() if k not in exclude_keys}

    return json.dumps(d, cls=EnhancedJSONEncoder)


def deserialize_dict(s: str) -> Dict[str, Any]:
    """
    Deserialize a JSON string back to a dictionary.

    Args:
        s: JSON string to deserialize

    Returns:
        Dict[str, Any]: Deserialized dictionary
    """
    if not s:
        return {}

    try:
        return json.loads(s, object_hook=deserialize_numpy)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to deserialize dictionary: {str(e)}")
        return {}


def deserialize_numpy(obj):
    """
    Deserialize objects serialized by serialize_numpy.

    Args:
        obj: Serialized object

    Returns:
        Deserialized object
    """
    if isinstance(obj, dict) and "__type__" in obj:
        if obj["__type__"] == "numpy.ndarray":
            return np.array(obj["data"], dtype=obj["dtype"] if "dtype" in obj else None)
        elif obj["__type__"] == "pandas.DataFrame":
            df = pd.DataFrame(obj["data"])
            if obj.get("index") is not None:
                df.index = obj["index"]
            return df
        elif obj["__type__"] == "pandas.Series":
            s = pd.Series(obj["data"], name=obj.get("name"))
            if obj.get("index") is not None:
                s.index = obj["index"]
            return s
    return obj


class TimeFrame(enum.Enum):
    """
    Enum for standard timeframes.
    This is an alias for the Timeframe enum for backward compatibility.
    """

    M1 = "1m"  # 1 minute
    M5 = "5m"  # 5 minutes
    M15 = "15m"  # 15 minutes
    M30 = "30m"  # 30 minutes
    H1 = "1h"  # 1 hour
    H4 = "4h"  # 4 hours
    D1 = "1d"  # 1 day
    W1 = "1w"  # 1 week
    MN1 = "1M"  # 1 month


def calculate_body_size(open_price, close_price):
    """
    Calculate the body size of a candlestick.

    Args:
        open_price: Opening price
        close_price: Closing price

    Returns:
        Absolute size of the candlestick body
    """
    return abs(close_price - open_price)


def get_timestamp():
    """
    Get current timestamp in milliseconds.
    Alias for current_timestamp for backward compatibility.

    Returns:
        int: Current timestamp in milliseconds
    """
    return current_timestamp()


def get_current_time():
    """
    Get current time as a datetime object.

    Returns:
        datetime.datetime: Current time
    """
    return datetime.datetime.now()


def calculate_shadow_size(open_price, close_price, high_price, low_price):
    """
    Calculate the upper and lower shadow sizes of a candlestick.

    Args:
        open_price: Opening price
        close_price: Closing price
        high_price: Highest price
        low_price: Lowest price

    Returns:
        tuple: (upper_shadow_size, lower_shadow_size)
    """
    body_high = max(open_price, close_price)
    body_low = min(open_price, close_price)

    upper_shadow = high_price - body_high
    lower_shadow = body_low - low_price

    return upper_shadow, lower_shadow


def calculate_vwap(
    prices: List[float], volumes: List[float], window: int = None
) -> List[float]:
    """
    Calculate Volume Weighted Average Price (VWAP).

    Args:
        prices: List of prices (typically (high+low+close)/3)
        volumes: List of volumes
        window: Optional rolling window size (None for cumulative VWAP)

    Returns:
        List[float]: VWAP values
    """
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have the same length")

    if not prices:
        return []

    # Calculate price * volume
    pv = [p * v for p, v in zip(prices, volumes)]

    if window is None:
        # Cumulative VWAP
        cum_pv = np.cumsum(pv)
        cum_volume = np.cumsum(volumes)

        # Avoid division by zero
        cum_volume = np.where(cum_volume == 0, 1, cum_volume)

        vwap = cum_pv / cum_volume
    else:
        # Rolling VWAP
        window_pv = np.convolve(pv, np.ones(window), "valid") / window
        window_volume = np.convolve(volumes, np.ones(window), "valid") / window

        # Avoid division by zero
        window_volume = np.where(window_volume == 0, 1, window_volume)

        vwap = window_pv / window_volume

        # Pad the beginning to match input length
        padding = len(prices) - len(vwap)
        vwap = np.pad(vwap, (padding, 0), "constant", constant_values=np.nan)

    return vwap.tolist()


def calculate_distance_percentage(price1: float, price2: float) -> float:
    """
    Calculate the percentage distance between two price points.

    Args:
        price1: First price point
        price2: Second price point

    Returns:
        float: Percentage distance between prices (absolute value)
    """
    if price1 == 0 or price2 == 0:
        return 0.0

    # Use average of two prices as the base to calculate percentage
    avg_price = (price1 + price2) / 2

    # Calculate absolute percentage difference
    distance_pct = abs(price1 - price2) / avg_price * 100.0

    return distance_pct


def calculate_distance(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple

    Returns:
        float: Euclidean distance between points
    """
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def float_round(value: float, precision: int = 2) -> float:
    """Round a floating point number to a given precision."""
    return round(float(value), precision)


def calculate_trade_imbalance(buy_volumes: List[float], sell_volumes: List[float]) -> float:
    """Compute normalized trade imbalance between buys and sells."""
    total_buy = sum(buy_volumes)
    total_sell = sum(sell_volumes)
    if total_buy + total_sell == 0:
        return 0.0
    return (total_buy - total_sell) / (total_buy + total_sell)


def is_higher_timeframe(higher_tf: str, lower_tf: str) -> bool:
    """
    Check if one timeframe is higher than another.

    Args:
        higher_tf: Potentially higher timeframe (e.g., '1h')
        lower_tf: Potentially lower timeframe (e.g., '5m')

    Returns:
        bool: True if higher_tf is actually higher than lower_tf
    """
    tf_seconds = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
        "M": 2592000,  # Approximate month
    }

    # Parse timeframes
    higher_match = re.match(r"^(\d+)([smhdwM])$", higher_tf)
    lower_match = re.match(r"^(\d+)([smhdwM])$", lower_tf)

    if not higher_match or not lower_match:
        raise ValueError(f"Invalid timeframe format: {higher_tf} or {lower_tf}")

    higher_value, higher_unit = int(higher_match.group(1)), higher_match.group(2)
    lower_value, lower_unit = int(lower_match.group(1)), lower_match.group(2)

    higher_seconds = higher_value * tf_seconds[higher_unit]
    lower_seconds = lower_value * tf_seconds[lower_unit]

    return higher_seconds > lower_seconds


def calculate_imbalance(
    bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], depth: int = 10
) -> float:
    """
    Calculate order book imbalance ratio.

    Args:
        bids: List of (price, volume) tuples for bids
        asks: List of (price, volume) tuples for asks
        depth: Depth of order book to consider

    Returns:
        Imbalance ratio (-1 to 1, negative means more asks, positive means more bids)
    """
    if not bids or not asks:
        return 0.0

    # Limit to specified depth
    bids = bids[:depth] if len(bids) > depth else bids
    asks = asks[:depth] if len(asks) > depth else asks

    # Calculate total volume
    bid_volume = sum(vol for _, vol in bids)
    ask_volume = sum(vol for _, vol in asks)

    total_volume = bid_volume + ask_volume

    # Avoid division by zero
    if total_volume == 0:
        return 0.0

    # Calculate imbalance ratio
    imbalance = (bid_volume - ask_volume) / total_volume

    return imbalance


def get_market_hours(exchange: str, asset_class: str = None) -> Dict[str, Any]:
    """
    Get market hours for a specific exchange and asset class.

    Args:
        exchange: Exchange name (e.g., 'binance', 'deriv')
        asset_class: Optional asset class (e.g., 'crypto', 'forex')

    Returns:
        Dict with market hours information:
            - is_open: Whether market is currently open
            - open_time: Daily opening time (UTC)
            - close_time: Daily closing time (UTC)
            - timezone: Timezone of the exchange
            - is_24h: Whether market is open 24/7
    """
    # Default to 24/7 for crypto exchanges
    if exchange.lower() in ["binance", "coinbase", "kraken", "kucoin"]:
        return {
            "is_open": True,
            "open_time": "00:00:00",
            "close_time": "00:00:00",
            "timezone": "UTC",
            "is_24h": True,
        }

    # Forex markets
    if asset_class and asset_class.lower() == "forex":
        # Check if current time is in forex trading hours (approx. Sunday 5PM ET to Friday 5PM ET)
        now_utc = datetime.datetime.utcnow()
        weekday = now_utc.weekday()  # 0=Monday, 6=Sunday

        # Forex is closed from Friday 5PM to Sunday 5PM ET (approx. 9PM-9PM UTC)
        is_weekend_closure = (
            (weekday == 4 and now_utc.hour >= 21)
            or weekday == 5
            or (weekday == 6 and now_utc.hour < 21)
        )

        return {
            "is_open": not is_weekend_closure,
            "open_time": "21:00:00" if weekday == 6 else "00:00:00",
            "close_time": "21:00:00" if weekday == 4 else "00:00:00",
            "timezone": "UTC",
            "is_24h": False,
        }

    # Default for other markets (conservative estimate)
    return {
        "is_open": True,  # Assume open by default
        "open_time": "00:00:00",
        "close_time": "00:00:00",
        "timezone": "UTC",
        "is_24h": True,
    }


def threaded_calculation(func, items, max_workers=None, *args, **kwargs):
    """
    Execute a function on multiple items using thread pool.

    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Maximum number of worker threads
        *args, **kwargs: Additional arguments to pass to func

    Returns:
        List of results in the same order as items
    """
    if not items:
        return []

    if max_workers is None:
        max_workers = min(32, os.cpu_count() * 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item, *args, **kwargs) for item in items]
        return [future.result() for future in futures]


def create_batches(items, batch_size):
    """
    Split a list of items into batches of specified size.

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Returns:
        List of batches, where each batch is a list of items
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


class UuidUtils:
    """Utility class for UUID operations."""

    @staticmethod
    def generate() -> str:
        """Generate a new UUID string."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_short() -> str:
        """Generate a shorter UUID (first 8 chars)."""
        return str(uuid.uuid4())[:8]

    @staticmethod
    def is_valid(uuid_str: str) -> bool:
        """Check if a string is a valid UUID."""
        try:
            uuid.UUID(uuid_str)
            return True
        except ValueError:
            return False

    @staticmethod
    def generate_deterministic(namespace: str, name: str) -> str:
        """Generate a deterministic UUID based on namespace and name."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace}:{name}"))


class HashUtils:
    """Utility class for hashing operations."""

    @staticmethod
    def md5(data: Union[str, bytes]) -> str:
        """Generate MD5 hash of data."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.md5(data).hexdigest()

    @staticmethod
    def sha1(data: Union[str, bytes]) -> str:
        """Generate SHA1 hash of data."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha1(data).hexdigest()

    @staticmethod
    def sha256(data: Union[str, bytes]) -> str:
        """Generate SHA256 hash of data."""
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def file_hash(filepath: str, algorithm: str = "sha256") -> str:
        """Generate hash of file contents."""
        hash_func = getattr(hashlib, algorithm)()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()


def generate_uid(prefix: str = "uid") -> str:
    """
    Generate a unique identifier with optional prefix.

    Args:
        prefix: Optional prefix for the UID

    Returns:
        str: Unique identifier string
    """
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    return f"{prefix}_{timestamp}_{unique_id[:8]}"


def generate_id(prefix: str = "id") -> str:
    """
    Generate a unique identifier with optional prefix.
    Alias for generate_uid for backward compatibility.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        str: Unique identifier string
    """
    return generate_uid(prefix)


def memoize(func):
    """
    Decorator to memoize function results for faster repeated calls.

    Args:
        func: Function to memoize

    Returns:
        Wrapped function with memoization
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from the function arguments
        key = str(args) + str(sorted(kwargs.items()))

        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    # Add a clear_cache method to the wrapper function
    wrapper.clear_cache = lambda: cache.clear()

    return wrapper


class TimestampUtils:
    """Utility class for timestamp operations and conversions."""

    @staticmethod
    def to_milliseconds(dt: Union[datetime, str, int, float]) -> int:
        """Convert various time formats to millisecond timestamp."""
        if isinstance(dt, datetime):
            return int(dt.timestamp() * 1000)
        elif isinstance(dt, str):
            return int(parse_datetime(dt).timestamp() * 1000)
        elif isinstance(dt, (int, float)):
            # If already a timestamp, ensure it's in milliseconds
            if dt > 1e16:  # nanoseconds
                return int(dt / 1000000)
            elif dt > 1e13:  # microseconds
                return int(dt / 1000)
            elif dt > 1e10:  # milliseconds
                return int(dt)
            else:  # seconds
                return int(dt * 1000)
        else:
            raise ValueError(f"Unsupported timestamp format: {type(dt)}")

    @staticmethod
    def from_milliseconds(ms: int) -> datetime:
        """Convert millisecond timestamp to datetime object."""
        return datetime.fromtimestamp(ms / 1000)

    @staticmethod
    def now_ms() -> int:
        """Get current time in milliseconds."""
        return int(time.time() * 1000)

    @staticmethod
    def format_ms(ms: int, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Convert milliseconds timestamp to formatted string."""
        dt = datetime.datetime.fromtimestamp(ms / 1000)
        return dt.strftime(fmt)


# Trading-specific utility functions
def calculate_price_precision(symbol: str, exchange: str = None) -> int:
    """
    Calculate the price precision for a given symbol and exchange.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        exchange: Exchange name (optional)

    Returns:
        int: Number of decimal places for price
    """
    # Default precisions for common symbols
    default_precisions = {
        "BTC": 2,
        "ETH": 2,
        "LTC": 2,
        "XRP": 5,
        "ADA": 6,
        "DOT": 4,
        "BNB": 3,
        "SOL": 3,
        "DOGE": 7,
        "USDT": 2,
        "USDC": 2,
        "DEFAULT": 8,
    }

    # Extract base currency from symbol
    base = symbol.split("/")[0] if "/" in symbol else symbol

    # Return default precision for the base currency or DEFAULT if not found
    return default_precisions.get(base, default_precisions["DEFAULT"])


def calculate_quantity_precision(symbol: str, exchange: str = None) -> int:
    """
    Calculate the quantity precision for a given symbol and exchange.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        exchange: Exchange name (optional)

    Returns:
        int: Number of decimal places for quantity
    """
    # Default precisions for common symbols
    default_precisions = {
        "BTC": 6,
        "ETH": 5,
        "LTC": 4,
        "XRP": 1,
        "ADA": 1,
        "DOT": 2,
        "BNB": 3,
        "SOL": 2,
        "DOGE": 0,
        "USDT": 2,
        "USDC": 2,
        "DEFAULT": 8,
    }

    # Extract base currency from symbol
    base = symbol.split("/")[0] if "/" in symbol else symbol

    # Return default precision for the base currency or DEFAULT if not found
    return default_precisions.get(base, default_precisions["DEFAULT"])


def get_asset_precision(symbol: str, exchange: str | None = None) -> int:
    """Return precision for an asset symbol."""
    return calculate_quantity_precision(symbol, exchange)


def round_to_precision(value: float, precision: int) -> float:
    """
    Round a value to a specific number of decimal places.

    Args:
        value: Value to round
        precision: Number of decimal places

    Returns:
        float: Rounded value
    """
    factor = 10**precision
    return round(value * factor) / factor


def convert_timeframe(timeframe: str, to_format: str = "seconds") -> Union[int, str]:
    """
    Convert a timeframe string to different formats.

    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')
        to_format: Target format ('seconds', 'minutes', 'hours', 'days', or 'pandas')

    Returns:
        Union[int, str]: Converted timeframe
    """
    # Parse timeframe
    match = re.match(r"^(\d+)([smhdwM])$", timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")

    value, unit = match.groups()
    value = int(value)

    # Convert to seconds first
    if unit == "s":
        seconds = value
    elif unit == "m":
        seconds = value * 60
    elif unit == "h":
        seconds = value * 60 * 60
    elif unit == "d":
        seconds = value * 60 * 60 * 24
    elif unit == "w":
        seconds = value * 60 * 60 * 24 * 7
    elif unit == "M":
        seconds = value * 60 * 60 * 24 * 30  # Approximate
    else:
        raise ValueError(f"Invalid timeframe unit: {unit}")

    # Convert to target format
    if to_format == "seconds":
        return seconds
    elif to_format == "minutes":
        return seconds / 60
    elif to_format == "hours":
        return seconds / (60 * 60)
    elif to_format == "days":
        return seconds / (60 * 60 * 24)
    elif to_format == "pandas":
        # Convert to pandas frequency string
        if unit == "s":
            return f"{value}S"
        elif unit == "m":
            return f"{value}T"
        elif unit == "h":
            return f"{value}H"
        elif unit == "d":
            return f"{value}D"
        elif unit == "w":
            return f"{value}W"
        elif unit == "M":
            return f"{value}M"
    else:
        raise ValueError(f"Invalid target format: {to_format}")


def calculate_order_cost(
    price: float, quantity: float, fee_rate: float = 0.001
) -> float:
    """
    Calculate the total cost of an order including fees.

    Args:
        price: Order price
        quantity: Order quantity
        fee_rate: Fee rate as a decimal (default: 0.1%)

    Returns:
        float: Total order cost
    """
    base_cost = price * quantity
    fee = base_cost * fee_rate
    return base_cost + fee


def calculate_order_risk(
    price: float, stop_loss: float, quantity: float, account_balance: float
) -> float:
    """
    Calculate the risk percentage of an order relative to account balance.

    Args:
        price: Entry price
        stop_loss: Stop loss price
        quantity: Order quantity
        account_balance: Total account balance

    Returns:
        float: Risk as a percentage of account balance
    """
    if price <= 0 or account_balance <= 0:
        return 0.0

    # Calculate potential loss
    loss_per_unit = abs(price - stop_loss)
    total_loss = loss_per_unit * quantity

    # Calculate risk percentage
    risk_percentage = (total_loss / account_balance) * 100

    return risk_percentage


def normalize_price(price: float, tick_size: float) -> float:
    """
    Normalize a price to comply with exchange tick size requirements.

    Args:
        price: Raw price
        tick_size: Minimum price increment

    Returns:
        float: Normalized price
    """
    if tick_size <= 0:
        return price

    return round(price / tick_size) * tick_size


def normalize_price_series(series: 'pd.Series', tick_size: float) -> 'pd.Series':
    """Normalize a price series to comply with exchange tick size requirements."""
    if tick_size <= 0:
        return series
    return (series / tick_size).round() * tick_size


def normalize_quantity(
    quantity: float, step_size: float, min_quantity: float = 0
) -> float:
    """
    Normalize a quantity to comply with exchange step size requirements.

    Args:
        quantity: Raw quantity
        step_size: Minimum quantity increment
        min_quantity: Minimum allowed quantity

    Returns:
        float: Normalized quantity
    """
    if step_size <= 0:
        return max(quantity, min_quantity)

    normalized = round(quantity / step_size) * step_size

    # Ensure the quantity is at least the minimum
    return max(normalized, min_quantity)






def format_timestamp(timestamp: Union[int, float, datetime.datetime], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp to a human-readable string.
    
    Args:
        timestamp: Unix timestamp, datetime object, or timestamp in milliseconds
        format_str: Format string for datetime formatting
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, datetime.datetime):
        dt = timestamp
    elif isinstance(timestamp, (int, float)):
        # Handle both seconds and milliseconds timestamps
        if timestamp > 1e10:  # Likely milliseconds
            timestamp = timestamp / 1000
        dt = datetime.datetime.fromtimestamp(timestamp)
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
    
    return dt.strftime(format_str)


def escape_html(text: str) -> str:
    """
    Escape HTML special characters in text.
    
    Args:
        text: Text to escape
        
    Returns:
        HTML-escaped text
    """
    if not isinstance(text, str):
        text = str(text)
    
    html_escape_table = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
        "/": "&#x2F;",
    }
    
    return "".join(html_escape_table.get(c, c) for c in text)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN values and ensure DataFrame input."""
    if pd is None:  # pragma: no cover - optional dependency
        raise ImportError('pandas is required for validate_data')
    if not isinstance(df, pd.DataFrame):
        raise TypeError('validate_data expects a pandas DataFrame')
    return df.dropna()


__all__ = [
    # Time utilities
    'timestamp_ms', 'current_timestamp', 'current_timestamp_micros', 'current_timestamp_nanos',
    'timestamp_to_datetime', 'datetime_to_timestamp', 'parse_datetime',
    'format_datetime', 'timeframe_to_seconds', 'timeframe_to_timedelta',
    'round_timestamp', 'generate_timeframes', 'create_timeframes', 'parse_timeframe', 'validate_timeframe',
    'get_higher_timeframes', 'TimestampUtils',
    
    # Data handling and trading utilities
    'calculate_price_precision', 'calculate_quantity_precision',
    'round_to_precision', 'convert_timeframe', 'calculate_order_cost',
    'calculate_order_risk', 'normalize_price', 'normalize_quantity',
    'parse_decimal', 'safe_divide', 'round_to_tick', 'round_to_tick_size', 'calculate_change_percent',
    'normalize_value', 'moving_average', 'exponential_moving_average', 'rolling_window',
    'efficient_rolling_window', 'is_time_series', 'create_window_samples',
    
    # String and format
    'camel_to_snake', 'snake_to_camel', 'format_number', 'format_currency', 'truncate_string',
    'pluralize',
    
    # JSON and data structures
    'EnhancedJSONEncoder', 'JsonEncoder', 'json_dumps', 'json_loads', 'deep_update', 'deep_get',
    'flatten_dict', 'unflatten_dict', 'dict_to_object', 'dict_to_namedtuple', 'group_by', 'chunks',
    'filter_none_values', 'find_duplicate_items', 'merge_lists',
    
    # Security and validation
    'generate_secure_random_string', 'generate_uuid', 'generate_hmac_signature',
    'is_valid_url', 'is_valid_email', 'sanitize_filename', 'validate_required_keys',
    'mask_sensitive_data', 'hash_content', 'generate_uid',
    
    # Network and system
    'get_host_info', 'is_port_open', 'rate_limit', 'rate_limited', 'retry', 'timer',
    'retry_with_backoff', 'exponential_backoff', 'time_execution', 'time_function', 'calculate_checksum',
    
    # Async utilities
    'ensure_future', 'create_task_name',
    
    # Thread-safe utilities
    'AtomicCounter', 'SafeDict',
    
    # Trading-specific
    'calculate_order_size', 'calculate_position_value', 'calculate_pip_value', 'calculate_arbitrage_profit',
    'calculate_position_size', 'calculate_volatility', 'calculate_correlation', 'calculate_drawdown',
    'calculate_liquidation_price', 'calculate_risk_reward', 'calculate_win_rate',
    'calculate_risk_reward_ratio', 'calculate_confidence_score', 'normalize_probability',
    'normalize_price_series', 'detect_market_condition', 'normalize_data', 'calculate_dynamic_threshold',
    'weighted_average', 'time_weighted_average', 'validate_signal', 'validate_data', 'calculate_expectancy',
    'calculate_kelly_criterion', 'calculate_sharpe_ratio', 'calculate_sortino_ratio', 'calculate_metrics',
    'calculate_max_drawdown', 'calculate_calmar_ratio', 'z_score',

    'is_price_consolidating', 'is_breaking_out', 'calculate_pivot_points',
    'pivot_points',
    'periodic_reset', 'obfuscate_sensitive_data', 'exponential_smoothing',
    'calculate_distance', 'calculate_distance_percentage', 'memoize',
    'is_higher_timeframe', 'threaded_calculation', 'create_batches',
    'create_directory', 'create_directory_if_not_exists',
    'UuidUtils', 'HashUtils', 'SecurityUtils',
    'OrderSide', 'OrderType', 'TimeInForce',
    'ClassRegistry', 'AsyncService', 'Signal', 'SignalBus'
]


class SecurityUtils:
    """Utility class for security operations."""

    @staticmethod
    def encrypt(data: str, key: str) -> str:
        """
        Encrypt data using a key.

        Args:
            data: Data to encrypt
            key: Encryption key

        Returns:
            str: Encrypted data in base64 format
        """
        if not data:
            return ""

        # Create a simple encryption using HMAC and base64
        if isinstance(data, str):
            data = data.encode("utf-8")
        if isinstance(key, str):
            key = key.encode("utf-8")

        h = hmac.new(key, data, hashlib.sha256)
        signature = h.digest()

        # Combine data and signature and encode
        result = base64.b64encode(data + signature).decode("utf-8")
        return result

    @staticmethod
    def decrypt(encrypted_data: str, key: str) -> str:
        """
        Decrypt data using a key.

        Args:
            encrypted_data: Encrypted data in base64 format
            key: Decryption key

        Returns:
            str: Decrypted data
        """
        if not encrypted_data:
            return ""

        try:
            # Decode from base64
            if isinstance(key, str):
                key = key.encode("utf-8")

            decoded = base64.b64decode(encrypted_data)

            # Extract data and signature
            data = decoded[:-32]  # SHA256 digest is 32 bytes
            signature = decoded[-32:]

            # Verify signature
            h = hmac.new(key, data, hashlib.sha256)
            calculated_signature = h.digest()

            if not hmac.compare_digest(signature, calculated_signature):
                raise ValueError("Invalid signature, data may be tampered")

            # Return decrypted data
            return data.decode("utf-8")
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            return ""

    @staticmethod
    def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
        """
        Hash a password with optional salt.

        Args:
            password: Password to hash
            salt: Optional salt (generated if not provided)

        Returns:
            Tuple[str, str]: (hashed_password, salt)
        """
        if salt is None:
            salt = os.urandom(16).hex()

        if isinstance(password, str):
            password = password.encode("utf-8")
        if isinstance(salt, str):
            salt = salt.encode("utf-8")

        # Use PBKDF2 for password hashing
        key = hashlib.pbkdf2_hmac("sha256", password, salt, 100000)
        hashed = base64.b64encode(key).decode("utf-8")

        return hashed, salt.decode("utf-8") if isinstance(salt, bytes) else salt

    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        """
        Verify a password against a hash.

        Args:
            password: Password to verify
            hashed_password: Stored hash
            salt: Salt used for hashing

        Returns:
            bool: True if password matches
        """
        if isinstance(password, str):
            password = password.encode("utf-8")
        if isinstance(salt, str):
            salt = salt.encode("utf-8")

        # Hash the input password with the same salt
        key = hashlib.pbkdf2_hmac("sha256", password, salt, 100000)
        calculated_hash = base64.b64encode(key).decode("utf-8")

        # Compare hashes
        return hashed_password == calculated_hash


