#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Metrics Collection Module

This module provides tools for collecting and reporting metrics about system performance,
resource usage, and trading activity.
"""

import time
import json
import asyncio
import statistics
from typing import Dict, List, Any, Optional, Tuple
import psutil
from contextlib import contextmanager
from functools import wraps
from collections import defaultdict

from common.logger import get_logger

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    class _DummyNumpy:
        ndarray = list

        def __getattr__(self, name: str):
            raise ImportError("NumPy is required for metrics calculations")

    np = _DummyNumpy()  # type: ignore

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    class _DummyPandas:
        class Series(list):
            pass

        class DataFrame(dict):
            pass

        def __getattr__(self, name: str):
            raise ImportError("pandas is required for metrics calculations")

    pd = _DummyPandas()  # type: ignore
import math


class Timer:
    """Utility for timing operations."""

    def __init__(self, metrics_collector, metric_name):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.metrics_collector.record_timer(self.metric_name, elapsed_time)

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.metrics_collector.record_timer(self.metric_name, elapsed_time)


class MetricsCollector:
    """Collects and manages system and trading metrics."""

    def __init__(self, namespace, subsystem=None):

        """
        Initialize metrics collector.

        Args:
            namespace: Namespace for metrics
            subsystem: Optional subsystem name appended to the namespace
        """
        self.namespace = namespace if subsystem is None else f"{namespace}.{subsystem}"

        self.counters = {}
        self.gauges = {}
        self.timers = defaultdict(list)
        self.histograms = defaultdict(list)
        self.start_time = time.time()
        self.logger = get_logger(f"Metrics.{namespace}")

    def increment(self, metric_name, value=1):
        """
        Increment a counter metric.

        Args:
            metric_name: Metric name
            value: Increment value (default: 1)

        Returns:
            New counter value
        """
        full_name = f"{self.namespace}.{metric_name}"
        if full_name not in self.counters:
            self.counters[full_name] = 0
        self.counters[full_name] += value
        return self.counters[full_name]

    def decrement(self, metric_name, value=1):
        """
        Decrement a counter metric.

        Args:
            metric_name: Metric name
            value: Decrement value (default: 1)

        Returns:
            New counter value
        """
        return self.increment(metric_name, -value)

    def set(self, metric_name, value):
        """
        Set a gauge metric to a specific value.

        Args:
            metric_name: Metric name
            value: Gauge value

        Returns:
            Set value
        """
        full_name = f"{self.namespace}.{metric_name}"
        self.gauges[full_name] = value
        return value

    def register_gauge(self, metric_name, description=None, initial_value=0):
        """
        Register a new gauge with an initial value.

        Args:
            metric_name: Metric name
            description: Optional description of the gauge
            initial_value: Initial gauge value (default: 0)

        Returns:
            Initial gauge value
        """
        full_name = f"{self.namespace}.{metric_name}"
        if full_name not in self.gauges:
            self.gauges[full_name] = initial_value

        # Store description if provided
        if description and not hasattr(self, '_metric_descriptions'):
            self._metric_descriptions = {}
        if description:
            self._metric_descriptions[full_name] = description

        return self.gauges[full_name]

    def register_counter(self, metric_name, initial_value=0):
        """
        Register a new counter with an initial value.

        Args:
            metric_name: Metric name
            initial_value: Initial counter value (default: 0)

        Returns:
            Initial counter value
        """
        full_name = f"{self.namespace}.{metric_name}"
        if full_name not in self.counters:
            self.counters[full_name] = initial_value
        return self.counters[full_name]

    def record_timer(self, metric_name, value):
        """
        Record a timer value.

        Args:
            metric_name: Metric name
            value: Timer value (in seconds)
        """
        full_name = f"{self.namespace}.{metric_name}"
        self.timers[full_name].append(value)
        # Keep only the last 1000 values to limit memory usage
        if len(self.timers[full_name]) > 1000:
            self.timers[full_name] = self.timers[full_name][-1000:]

    def register_histogram(self, metric_name, initial_values=None):
        """
        Register a new histogram with initial values.

        Args:
            metric_name: Metric name
            initial_values: Initial histogram values (default: None)

        Returns:
            Histogram data structure
        """
        full_name = f"{self.namespace}.{metric_name}"
        if full_name not in self.histograms:
            self.histograms[full_name] = []

        if initial_values:
            self.histograms[full_name].extend(initial_values)

        return self.histograms[full_name]

    def get(self, metric_name, default=None):
        """
        Get the current value of a metric.

        Args:
            metric_name: Metric name
            default: Default value if metric doesn't exist

        Returns:
            Metric value or default
        """
        full_name = f"{self.namespace}.{metric_name}"
        if full_name in self.counters:
            return self.counters[full_name]
        if full_name in self.gauges:
            return self.gauges[full_name]
        return default

    def get_timer_stats(self, metric_name):
        """
        Get statistics for a timer metric.

        Args:
            metric_name: Metric name

        Returns:
            Dictionary of timer statistics (count, min, max, mean, median, p95)
        """
        full_name = f"{self.namespace}.{metric_name}"
        if full_name not in self.timers or not self.timers[full_name]:
            return {}

        values = self.timers[full_name]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
        }

    def timer(self, metric_name):
        """
        Create a timer context manager.

        Args:
            metric_name: Metric name

        Returns:
            Timer context manager
        """
        return Timer(self, metric_name)

    def get_all_metrics(self):
        """
        Get all metrics.

        Returns:
            Dictionary of all metrics
        """
        metrics = {
            "counters": self.counters.copy(),
            "gauges": self.gauges.copy()
        }

        # Add timer statistics
        timer_stats = {}
        for name in self.timers:
            short_name = name.replace(f"{self.namespace}.", "")
            timer_stats[short_name] = self.get_timer_stats(short_name)
        metrics["timers"] = timer_stats

        # Add histogram statistics
        histogram_stats = {}
        for name in self.histograms:
            short_name = name.replace(f"{self.namespace}.", "")
            histogram_stats[short_name] = self.get_histogram_stats(short_name)
        metrics["histograms"] = histogram_stats

        # Add system metrics
        metrics["system"] = self.get_system_metrics()

        return metrics

    def get_system_metrics(self):
        """
        Get system metrics (CPU, memory, disk).

        Returns:
            Dictionary of system metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "memory_total_mb": memory.total / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024 * 1024 * 1024),
                "disk_total_gb": disk.total / (1024 * 1024 * 1024),
                "uptime_seconds": time.time() - self.start_time
            }
        except Exception as e:
            self.logger.warning(f"Error getting system metrics: {str(e)}")
            return {}

    def reset(self, metric_type=None):
        """
        Reset metrics.

        Args:
            metric_type: Optional metric type to reset ('counters', 'gauges', 'timers', 'histograms')
        """
        if metric_type is None or metric_type == 'counters':
            self.counters = {}
        if metric_type is None or metric_type == 'gauges':
            self.gauges = {}
        if metric_type is None or metric_type == 'timers':
            self.timers = defaultdict(list)
        if metric_type is None or metric_type == 'histograms':
            self.histograms = {}

    def export_json(self, file_path=None):
        """
        Export metrics to JSON.

        Args:
            file_path: Optional file path to write to

        Returns:
            JSON string
        """
        metrics = self.get_all_metrics()
        json_data = json.dumps(metrics, indent=2)

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(json_data)
            except Exception as e:
                self.logger.error(f"Error exporting metrics to {file_path}: {str(e)}")

        return json_data

    def record_timing(self, metric_name, duration):
        """Record a timing measurement."""
        full_name = f"{self.namespace}.{metric_name}"
        if full_name not in self.timers:
            self.timers[full_name] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': 0,
                'avg': 0,
                'samples': []
            }

        timer = self.timers[full_name]
        timer['count'] += 1
        timer['sum'] += duration
        timer['min'] = min(timer['min'], duration)
        timer['max'] = max(timer['max'], duration)
        timer['avg'] = timer['sum'] / timer['count']

        # Keep limited samples for percentile calculations
        timer['samples'].append(duration)
        if len(timer['samples']) > 100:
            timer['samples'].pop(0)

        return timer

    def get_percentile(self, metric_name, percentile):
        """Get a percentile value for a timing metric."""
        full_name = f"{self.namespace}.{metric_name}"
        if full_name in self.timers and self.timers[full_name]:
            samples = sorted(self.timers[full_name])
            idx = int(round((len(samples) - 1) * (percentile / 100)))
            return samples[idx]
        return None

    def record_histogram(self, metric_name, value):
        """Record a value in a histogram."""
        full_name = f"{self.namespace}.{metric_name}"
        if full_name not in self.histograms:
            self.histograms[full_name] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0,
                'values': []
            }

        hist = self.histograms[full_name]
        hist['count'] += 1
        hist['sum'] += value
        hist['min'] = min(hist['min'], value)
        hist['max'] = max(hist['max'], value)
        hist['avg'] = hist['sum'] / hist['count']

        # Keep values for distribution analysis
        hist['values'].append(value)
        if len(hist['values']) > 1000:
            hist['values'].pop(0)

        return hist

    def get_histogram_stats(self, metric_name):
        """Get statistical information about a histogram."""
        full_name = f"{self.namespace}.{metric_name}"
        if full_name in self.histograms:
            hist = self.histograms[full_name]
            values = hist['values']

            if not values:
                return None

            # Calculate standard deviation
            mean = hist['avg']
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5

            # Calculate percentiles
            sorted_values = sorted(values)
            p50 = sorted_values[int(len(sorted_values) * 0.5)]
            p90 = sorted_values[int(len(sorted_values) * 0.9)]
            p95 = sorted_values[int(len(sorted_values) * 0.95)]
            p99 = sorted_values[int(len(sorted_values) * 0.99)]

            return {
                'count': hist['count'],
                'min': hist['min'],
                'max': hist['max'],
                'avg': hist['avg'],
                'std_dev': std_dev,
                'p50': p50,
                'p90': p90,
                'p95': p95,
                'p99': p99
            }
        return None

    def export_metrics(self):
        """Export all metrics as a dictionary for serialization."""
        result = {
            'timestamp': time.time(),
            'uptime': time.time() - self.start_time,
            'counters': self.counters.copy(),
            'gauges': self.gauges.copy(),
            'timers': {},
            'histograms': {},
        }

        for name in self.timers:
            short_name = name.replace(f"{self.namespace}.", "")
            result['timers'][name] = self.get_timer_stats(short_name)

        for name in self.histograms:
            short_name = name.replace(f"{self.namespace}.", "")
            result['histograms'][name] = self.get_histogram_stats(short_name)

        # Simplify timer data for export
        for name, timer in self.timers.items():
            result['timers'][name] = {
                'count': timer['count'],
                'avg_ms': timer['avg'] * 1000,  # Convert to ms
                'min_ms': timer['min'] * 1000,
                'max_ms': timer['max'] * 1000,
                'p95_ms': self.get_percentile(name.replace(f"{self.namespace}.", ""), 95) * 1000 if timer['samples'] else None,
                'p99_ms': self.get_percentile(name.replace(f"{self.namespace}.", ""), 99) * 1000 if timer['samples'] else None
            }

        # Simplify histogram data for export
        for name, hist in self.histograms.items():
            stats = self.get_histogram_stats(name.replace(f"{self.namespace}.", ""))
            if stats:
                result['histograms'][name] = stats

        return result

    # Context manager for timing operations
        # Add this static method for singleton access
    _instances = {}

    @classmethod
    def get_instance(cls, namespace="default"):
        """
        Get or create a MetricsCollector instance for the given namespace.
        Implements the singleton pattern.

        Args:
            namespace: Namespace for metrics

        Returns:
            MetricsCollector instance
        """
        if namespace not in cls._instances:
            cls._instances[namespace] = cls(namespace)
        return cls._instances[namespace]

    @contextmanager
    def timing(self, metric_name, duration: float | None = None):
        """Record timing either as context manager or direct call."""
        if duration is not None:
            self.record_timing(metric_name, duration)
            yield
            return

        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(metric_name, duration)

    async def collect_metrics_task(self, interval=10):
        """
        Background task to periodically collect system metrics.

        Args:
            interval: Collection interval in seconds
        """
        while True:
            try:
                # Update system metrics
                system_metrics = self.get_system_metrics()
                for key, value in system_metrics.items():
                    self.set(f"system.{key}", value)

                # Wait for next interval
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                self.logger.info("Metrics collection task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection task: {str(e)}")
                await asyncio.sleep(interval)

    def record_latency(self, name, value_ms, tags=None):
        """
        Record a latency metric.

        Args:
            name: Metric name
            value_ms: Latency value in milliseconds
            tags: Optional dictionary of tags
        """
        # Convert to seconds for internal storage
        value_sec = value_ms / 1000.0
        self.record_timer(name, value_sec)

        # Log the metric if tags are provided
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in tags.items())
            self.logger.debug(f"Latency: {name} {value_ms}ms {tag_str}")
            
    def gauge(self, name, value, tags=None):
        """
        Set a gauge metric to a specific value.
        This is an alias for the set() method for compatibility.

        Args:
            name: Metric name
            value: Gauge value
            tags: Optional tags (ignored in this implementation)

        Returns:
            Set value
        """
        return self.set(name, value)
    
    def counter(self, name, value=1, tags=None):
        """
        Increment a counter metric.
        This is an alias for the increment() method for compatibility.

        Args:
            name: Metric name
            value: Increment value (default: 1)
            tags: Optional tags (ignored in this implementation)

        Returns:
            New counter value
        """
        return self.increment(name, value)


# Module-level instance for global use
def get_default_collector():
    """Get the default metrics collector instance."""
    return MetricsCollector.get_instance("default")


# Global collector instances

performance_tracker = MetricsCollector.get_instance("performance")
performance_metrics = MetricsCollector.get_instance("performance_metrics")
_default_collector = get_default_collector()


def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: List of period returns (e.g., daily returns)
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Factor to annualize returns (252 for daily, 52 for weekly, 12 for monthly)
    
    Returns:
        float: Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    # Convert to numpy array for calculations
    returns_array = np.array(returns)
    
    # Calculate mean return and standard deviation
    mean_return = np.mean(returns_array)
    std_dev = np.std(returns_array, ddof=1)  # Use sample standard deviation
    
    if std_dev == 0:
        return 0.0  # Avoid division by zero
    
    # Calculate daily excess return
    daily_risk_free = risk_free_rate / annualization_factor
    excess_return = mean_return - daily_risk_free
    
    # Calculate daily Sharpe ratio
    daily_sharpe = excess_return / std_dev
    
    # Annualize Sharpe ratio
    sharpe_ratio = daily_sharpe * math.sqrt(annualization_factor)
    
    return sharpe_ratio


def sortino_ratio(returns: List[float], risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate the Sortino ratio, which measures the risk-adjusted return using downside deviation.
    
    Args:
        returns: List of period returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Risk-free rate of return (default: 0.0)
        annualization_factor: Number of periods in a year (default: 252 for daily returns)
    
    Returns:
        Sortino ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    # Convert to numpy array
    returns_array = np.array(returns)
    
    # Calculate excess returns
    daily_risk_free = risk_free_rate / annualization_factor
    excess_returns = returns_array - daily_risk_free
    
    # Calculate average excess return
    avg_excess_return = np.mean(excess_returns)
    
    # Calculate downside deviation (standard deviation of negative returns only)
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        # No negative returns, avoid division by zero
        return float('inf') if avg_excess_return > 0 else 0.0
    
    downside_deviation = np.sqrt(np.mean(negative_returns**2))
    
    if downside_deviation == 0:
        return float('inf') if avg_excess_return > 0 else 0.0
    
    # Calculate annualized Sortino ratio
    sortino_ratio = (avg_excess_return / downside_deviation) * np.sqrt(annualization_factor)
    
    return sortino_ratio


def expectancy(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate the expectancy (average profit/loss per trade).
    
    Args:
        trades: List of trade dictionaries, each containing at least a 'profit' key
               with the profit/loss value for the trade
    
    Returns:
        float: Expectancy value (average profit/loss per trade)
    """
    if not trades:
        return 0.0
    
    total_profit = sum(trade.get('profit', 0) for trade in trades)
    return total_profit / len(trades)


def drawdown(returns: List[float]) -> Tuple[float, int, int]:
    """
    Calculate the maximum drawdown, drawdown duration, and drawdown start index.
    
    Args:
        returns: List of period returns (as decimals, e.g., 0.01 for 1%)
    
    Returns:
        Tuple containing:
        - Maximum drawdown as a positive decimal (e.g., 0.25 for 25%)
        - Drawdown duration in periods
        - Start index of the drawdown
    """
    if not returns or len(returns) < 2:
        return 0.0, 0, 0
    
    # Calculate cumulative returns
    cum_returns = np.cumprod(1 + np.array(returns))
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdowns
    drawdowns = (cum_returns - running_max) / running_max
    
    # Find maximum drawdown and its index
    max_dd_idx = np.argmin(drawdowns)
    max_dd = abs(drawdowns[max_dd_idx])
    
    # Find the start of the drawdown (index of the last peak before the drawdown)
    peak_idx = np.where(cum_returns[:max_dd_idx+1] == running_max[max_dd_idx])[0][-1]
    
    # Calculate drawdown duration
    dd_duration = max_dd_idx - peak_idx
    
    return max_dd, dd_duration, peak_idx


def profit_factor(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate the profit factor (gross profit / gross loss).
    
    Args:
        trades: List of trade dictionaries, each containing at least a 'profit' key
               with the profit/loss value for the trade
    
    Returns:
        float: Profit factor (> 1.0 is profitable)
    """
    if not trades:
        return 0.0
    
    gross_profit = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
    gross_loss = sum(abs(trade.get('profit', 0)) for trade in trades if trade.get('profit', 0) < 0)
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def win_rate(trades: List[Dict[str, Any]]) -> float:
    """
    Calculate the win rate (percentage of profitable trades).
    
    Args:
        trades: List of trade dictionaries, each containing at least a 'profit' key
               with the profit/loss value for the trade
    
    Returns:
        float: Win rate as a decimal (e.g., 0.65 for 65%)
    """
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
    return winning_trades / len(trades)


def calmar_ratio(returns: List[float], drawdowns: List[float] = None, annualization_factor: int = 252) -> float:
    """
    Calculate the Calmar ratio, which is the annualized return divided by the maximum drawdown.
    
    Args:
        returns: List of period returns (as decimals, e.g., 0.01 for 1%)
        drawdowns: Optional list of drawdown values. If not provided, will be calculated from returns.
        annualization_factor: Number of periods in a year (default: 252 for daily returns)
    
    Returns:
        Calmar ratio
    """
    if not returns or len(returns) < 2:
        return 0.0
    
    # Calculate annualized return
    returns_array = np.array(returns)
    total_return = np.prod(1 + returns_array) - 1
    periods = len(returns)
    annualized_return = (1 + total_return) ** (annualization_factor / periods) - 1
    
    # Calculate maximum drawdown if not provided
    if drawdowns is None:
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns_array)
        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)
        # Calculate drawdowns
        drawdowns = (cum_returns - running_max) / running_max
    
    # Get maximum drawdown (as a positive number)
    max_drawdown = abs(min(drawdowns)) if drawdowns else 0.0
    
    # Avoid division by zero
    if max_drawdown == 0:
        return float('inf') if annualized_return > 0 else 0.0
    
    # Calculate Calmar ratio
    calmar = annualized_return / max_drawdown
    
    return calmar


def calculate_timing(func=None, *, metric_name: Optional[str] = None,
                     collector: Optional[MetricsCollector] = None):
    """Decorator to measure execution time and record via ``MetricsCollector``.

    This decorator supports both synchronous and asynchronous callables. The
    execution duration (in seconds) is recorded using ``collector.record_timing``.

    Args:
        func: The function to decorate when used without arguments.
        metric_name: Optional metric name. Defaults to the function's ``__name__``.
        collector: Optional ``MetricsCollector`` instance. Defaults to the
            ``performance_tracker`` collector.

    Returns:
        Wrapped function that records timing information.
    """

    if func is None:
        return lambda f: calculate_timing(
            f, metric_name=metric_name, collector=collector
        )

    metric = metric_name or func.__name__
    coll = collector or performance_tracker

    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                coll.record_timer(metric, time.perf_counter() - start)

        return async_wrapper

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            coll.record_timer(metric, time.perf_counter() - start)

    return sync_wrapper


def record_latency(name, value_ms, tags=None):
    """
    Record a latency metric using the default collector.

    Args:
        name: Metric name
        value_ms: Latency value in milliseconds
        tags: Optional dictionary of tags
    """
    _default_collector.record_latency(name, value_ms, tags)


def increment_counter(name, value=1, tags=None):
    """
    Increment a counter metric using the default collector.

    Args:
        name: Metric name
        value: Increment value (default: 1)
        tags: Optional dictionary of tags

    Returns:
        New counter value
    """
    return _default_collector.increment(name, value)


def record_value(name, value, tags=None):
    """
    Record a gauge value using the default collector.

    Args:
        name: Metric name
        value: Value to record
        tags: Optional dictionary of tags

    Returns:
        Recorded value
    """
    return _default_collector.set(name, value)


def record_success(name, tags=None):
    """
    Record a success event using the default collector.

    Args:
        name: Metric name
        tags: Optional dictionary of tags
    """
    _default_collector.increment(f"{name}.success", 1)

    if tags:
        tag_str = ",".join(f"{k}={v}" for k, v in tags.items())
        _default_collector.logger.debug(f"Success: {name} {tag_str}")


def record_failure(name, error=None, tags=None):
    """
    Record a failure event using the default collector.

    Args:
        name: Metric name
        error: Optional error message or exception
        tags: Optional dictionary of tags
    """
    _default_collector.increment(f"{name}.failure", 1)

    # Create log message
    log_msg = f"Failure: {name}"
    if error:
        log_msg += f" - {str(error)}"

    if tags:
        tag_str = ",".join(f"{k}={v}" for k, v in tags.items())
        log_msg += f" {tag_str}"

    _default_collector.logger.warning(log_msg)


def compute_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.

    Args:
        returns: List of period returns (e.g., daily returns)
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Factor to annualize returns (252 for daily, 52 for weekly, 12 for monthly)

    Returns:
        float: Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0

    # Convert to numpy array for calculations
    returns_array = np.array(returns)

    # Calculate mean return and standard deviation
    mean_return = np.mean(returns_array)
    std_dev = np.std(returns_array, ddof=1)  # Use sample standard deviation

    if std_dev == 0:
        return 0.0  # Avoid division by zero

    # Calculate daily excess return
    daily_risk_free = risk_free_rate / annualization_factor
    excess_return = mean_return - daily_risk_free

    # Calculate daily Sharpe ratio
    daily_sharpe = excess_return / std_dev

    # Annualize Sharpe ratio
    sharpe_ratio = daily_sharpe * math.sqrt(annualization_factor)

    return sharpe_ratio


def compute_sortino_ratio(returns: List[float], risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    """
    Calculate the Sortino ratio, which measures the risk-adjusted return using downside deviation.

    Args:
        returns: List of period returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Risk-free rate of return (default: 0.0)
        annualization_factor: Number of periods in a year (default: 252 for daily returns)

    Returns:
        Sortino ratio
    """
    if not returns or len(returns) < 2:
        return 0.0

    # Convert to numpy array
    returns_array = np.array(returns)

    # Calculate excess returns
    excess_returns = returns_array - risk_free_rate

    # Calculate average excess return
    avg_excess_return = np.mean(excess_returns)

    # Calculate downside deviation (standard deviation of negative returns only)
    negative_returns = excess_returns[excess_returns < 0]

    if len(negative_returns) == 0:
        # No negative returns, avoid division by zero
        return float('inf') if avg_excess_return > 0 else 0.0

    downside_deviation = np.sqrt(np.mean(negative_returns**2))

    if downside_deviation == 0:
        return float('inf') if avg_excess_return > 0 else 0.0

    # Calculate annualized Sortino ratio
    sortino_ratio = (avg_excess_return / downside_deviation) * np.sqrt(annualization_factor)

    return sortino_ratio


def calculate_trading_metrics(returns: List[float], trades: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive trading performance metrics.

    Args:
        returns: List of period returns (as decimals, e.g., 0.01 for 1%)
        trades: Optional list of trade dictionaries with details

    Returns:
        Dictionary of trading metrics
    """
    if not returns or len(returns) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_return": 0.0,
            "volatility": 0.0,
            "num_trades": 0
        }

    # Convert returns to numpy array for calculations
    returns_array = np.array(returns)

    # Basic return metrics
    total_return = np.prod(1 + returns_array) - 1
    avg_return = np.mean(returns_array)
    volatility = np.std(returns_array)

    # Annualized metrics (assuming daily returns by default)
    trading_days = 252  # Standard assumption for trading days in a year
    periods = len(returns)
    annualized_return = (1 + total_return) ** (trading_days / periods) - 1
    annualized_volatility = volatility * np.sqrt(trading_days)

    # Risk metrics
    try:
        sharpe_ratio = compute_sharpe_ratio(returns)
    except Exception:
        sharpe_ratio = 0.0

    try:
        sortino_ratio = compute_sortino_ratio(returns)
    except Exception:
        sortino_ratio = 0.0

    # Calculate drawdown
    cumulative_returns = np.cumprod(1 + returns_array)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0.0

    # Trade-specific metrics
    win_rate = 0.0
    profit_factor = 0.0
    num_trades = 0

    if trades and len(trades) > 0:
        num_trades = len(trades)
        winning_trades = [t for t in trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit', 0) <= 0]

        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0

        total_profit = sum(t.get('profit', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('profit', 0) for t in losing_trades))

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return": avg_return,
        "volatility": volatility,
        "annualized_volatility": annualized_volatility,
        "num_trades": num_trades,
    }


class ExecutionMetrics:
    """Simple execution metrics wrapper."""

    def __init__(self) -> None:
        self.collector = MetricsCollector('execution')

    def record_order(self, metric: str) -> None:
        self.collector.increment(metric)
