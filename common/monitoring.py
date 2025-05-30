"""
QuantumSpectre Core Infrastructure Module.

This module provides the foundation for the entire QuantumSpectre trading system,
including configuration management, service registry, and core utilities.
"""

__version__ = "1.0.0"

from .application import Application
from .config import Config, load_config
from .constants import EXCHANGE_TYPES, ORDER_TYPES, TIMEFRAMES
from .exceptions import (
    QuantumSpectreError, ConfigurationError, ExchangeError,
    ServiceError, ValidationError
)
from .logging_config import configure_logging, get_logger
from .service_registry import ServiceRegistry, service
from .utils import (
    async_retry, timeit, safe_execute, 
    create_id, current_timestamp_ms, 
    format_timestamp, serialize_to_json
)

# Initialize core components
configure_logging()
logger = get_logger(__name__)
config = load_config()
service_registry = ServiceRegistry()

__all__ = [
    "Application",
    "Config", "load_config", "config",
    "EXCHANGE_TYPES", "ORDER_TYPES", "TIMEFRAMES",
    "QuantumSpectreError", "ConfigurationError", "ExchangeError",
    "ServiceError", "ValidationError",
    "configure_logging", "get_logger", "logger",
    "ServiceRegistry", "service", "service_registry",
    "async_retry", "timeit", "safe_execute",
    "create_id", "current_timestamp_ms",
    "format_timestamp", "serialize_to_json",
]

logger.info(f"QuantumSpectre Core v{__version__} initialized")
"""
Monitoring and observability for QuantumSpectre Elite Trading System.

This module provides monitoring, metrics collection, and health checks
for system components and performance tracking.
"""

import os
import time
import asyncio
import logging
import socket
import psutil
import platform
from typing import Dict, List, Any, Optional, Callable, Set
import json
from datetime import datetime, timedelta
import uuid
from functools import wraps
import traceback
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary, REGISTRY
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily

# Local imports
from common.config import settings
from common.logger import get_logger
from common.database import redis_publish, redis_get, redis_set

# Get logger
logger = get_logger('monitoring')

# Global variables
MONITORING_ENABLED = False
start_time = time.time()
last_metrics = {}
health_checks = {}
monitored_events = {}

# Prometheus metrics
system_info = Gauge('quantum_spectre_system_info', 'System information', ['version', 'python_version', 'platform'])
process_cpu_usage = Gauge('quantum_spectre_process_cpu_percent', 'Process CPU usage percentage')
process_memory_usage = Gauge('quantum_spectre_process_memory_mb', 'Process memory usage in MB')
system_cpu_usage = Gauge('quantum_spectre_system_cpu_percent', 'System CPU usage percentage')
system_memory_usage = Gauge('quantum_spectre_system_memory_percent', 'System memory usage percentage')

api_requests = Counter('quantum_spectre_api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])
api_request_duration = Histogram('quantum_spectre_api_request_duration_seconds', 'API request duration in seconds', ['endpoint', 'method'])

exchange_requests = Counter('quantum_spectre_exchange_requests_total', 'Total exchange API requests', ['exchange', 'endpoint', 'status'])
exchange_request_duration = Histogram('quantum_spectre_exchange_request_duration_seconds', 'Exchange API request duration in seconds', ['exchange', 'endpoint'])

websocket_messages = Counter('quantum_spectre_websocket_messages_total', 'Total WebSocket messages', ['exchange', 'channel'])
websocket_reconnects = Counter('quantum_spectre_websocket_reconnects_total', 'Total WebSocket reconnection attempts', ['exchange'])

signals_generated = Counter('quantum_spectre_signals_generated_total', 'Total signals generated', ['strategy', 'signal_type', 'asset'])
signals_executed = Counter('quantum_spectre_signals_executed_total', 'Total signals executed', ['strategy', 'signal_type', 'asset'])

orders_placed = Counter('quantum_spectre_orders_placed_total', 'Total orders placed', ['exchange', 'order_type', 'asset'])
orders_filled = Counter('quantum_spectre_orders_filled_total', 'Total orders filled', ['exchange', 'order_type', 'asset'])
orders_canceled = Counter('quantum_spectre_orders_canceled_total', 'Total orders canceled', ['exchange', 'order_type', 'asset'])

trade_profit_pct = Histogram('quantum_spectre_trade_profit_percent', 'Trade profit percentage', ['strategy', 'asset'])
strategy_win_rate = Gauge('quantum_spectre_strategy_win_rate', 'Strategy win rate (0-1)', ['strategy', 'timeframe'])

brain_council_confidence = Gauge('quantum_spectre_brain_council_confidence', 'Brain council confidence level (0-1)', ['council', 'asset'])
brain_council_agreement = Gauge('quantum_spectre_brain_council_agreement', 'Brain council agreement level (0-1)', ['council', 'asset'])

ml_model_accuracy = Gauge('quantum_spectre_ml_model_accuracy', 'ML model accuracy (0-1)', ['model', 'version', 'dataset'])
ml_inference_duration = Histogram('quantum_spectre_ml_inference_duration_seconds', 'ML model inference duration in seconds', ['model', 'version'])

crawler_requests = Counter('quantum_spectre_crawler_requests_total', 'Total crawler requests', ['crawler', 'status'])
crawler_data_items = Counter('quantum_spectre_crawler_data_items_total', 'Total data items collected by crawlers', ['crawler', 'data_type'])

# Start monitoring
async def start_monitoring() -> bool:
    """
    Start the monitoring system.
    
    Returns:
        True if successful, False otherwise
    """
    global MONITORING_ENABLED
    
    try:
        # Start Prometheus HTTP server
        start_http_server(settings.MONITORING_PORT)
        
        # Set system info
        system_info.labels(
            version=settings.VERSION,
            python_version=platform.python_version(),
            platform=platform.system()
        ).set(1)
        
        # Start metrics collection task
        asyncio.create_task(collect_metrics_task())
        
        # Start health check task
        asyncio.create_task(health_check_task())
        
        MONITORING_ENABLED = True
        logger.info(f"Monitoring started on port {settings.MONITORING_PORT}")
        
        # Publish initial system info to Redis
        system_data = {
            'version': settings.VERSION,
            'python_version': platform.python_version(),
            'platform': platform.system(),
            'hostname': socket.gethostname(),
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'pid': os.getpid()
        }
        await redis_set('system:info', system_data)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {str(e)}")
        return False

# Stop monitoring
async def stop_monitoring() -> bool:
    """
    Stop the monitoring system.
    
    Returns:
        True if successful, False otherwise
    """
    global MONITORING_ENABLED
    
    try:
        # Clear metrics
        REGISTRY.unregister(system_info)
        REGISTRY.unregister(process_cpu_usage)
        REGISTRY.unregister(process_memory_usage)
        REGISTRY.unregister(system_cpu_usage)
        REGISTRY.unregister(system_memory_usage)
        
        MONITORING_ENABLED = False
        logger.info("Monitoring stopped")
        return True
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {str(e)}")
        return False

# Register health check
def register_health_check(name: str, check_func: Callable[[], bool]) -> None:
    """
    Register a health check function.
    
    Args:
        name: Name of the health check
        check_func: Function that returns True if healthy, False otherwise
    """
    global health_checks
    
    health_checks[name] = {
        'func': check_func,
        'status': True,
        'last_check': time.time(),
        'failures': 0
    }
    logger.debug(f"Registered health check: {name}")

# Unregister health check
def unregister_health_check(name: str) -> None:
    """
    Unregister a health check.
    
    Args:
        name: Name of the health check
    """
    global health_checks
    
    if name in health_checks:
        del health_checks[name]
        logger.debug(f"Unregistered health check: {name}")

# Health check task
async def health_check_task() -> None:
    """Periodic health check task."""
    global health_checks
    
    while MONITORING_ENABLED:
        try:
            # Store the current state of health checks
            current_health = {name: check['status'] for name, check in health_checks.items()}
            
            # Run all health checks
            for name, check in health_checks.items():
                try:
                    # Run the check function
                    status = check['func']()
                    
                    # Update check status
                    check['status'] = status
                    check['last_check'] = time.time()
                    
                    # Update failure count
                    if status:
                        check['failures'] = 0
                    else:
                        check['failures'] += 1
                        logger.warning(f"Health check failed: {name} (failures: {check['failures']})")
                        
                        # Publish health check failure event
                        await publish_event('health_check_failure', {
                            'name': name,
                            'failures': check['failures'],
                            'timestamp': datetime.now().isoformat()
                        })
                
                except Exception as e:
                    logger.error(f"Error in health check {name}: {str(e)}")
                    check['status'] = False
                    check['failures'] += 1
            
            # Check for changes in health status
            for name, status in current_health.items():
                if name in health_checks and status != health_checks[name]['status']:
                    if health_checks[name]['status']:
                        logger.info(f"Health check recovered: {name}")
                        
                        # Publish health check recovery event
                        await publish_event('health_check_recovery', {
                            'name': name,
                            'timestamp': datetime.now().isoformat()
                        })
                    
            # Publish overall health status to Redis
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'overall': all(check['status'] for check in health_checks.values()),
                'checks': {name: {
                    'status': check['status'],
                    'last_check': datetime.fromtimestamp(check['last_check']).isoformat(),
                    'failures': check['failures']
                } for name, check in health_checks.items()}
            }
            await redis_set('system:health', health_status)
            
            # Wait for next check
            await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            logger.info("Health check task cancelled")
            break
            
        except Exception as e:
            logger.error(f"Error in health check task: {str(e)}")
            await asyncio.sleep(5)

# Metrics collection task
async def collect_metrics_task() -> None:
    """Periodic metrics collection task."""
    while MONITORING_ENABLED:
        try:
            # Process metrics
            process = psutil.Process(os.getpid())
            process_cpu_usage.set(process.cpu_percent(interval=None))
            process_memory_usage.set(process.memory_info().rss / (1024 * 1024))  # Convert to MB
            
            # System metrics
            system_cpu_usage.set(psutil.cpu_percent(interval=None))
            system_memory_usage.set(psutil.virtual_memory().percent)
            
            # Publish metrics to Redis
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'process_cpu': process.cpu_percent(interval=None),
                'process_memory_mb': process.memory_info().rss / (1024 * 1024),
                'system_cpu': psutil.cpu_percent(interval=None),
                'system_memory': psutil.virtual_memory().percent
            }
            await redis_set('system:metrics', metrics)
            
            # Wait for next collection
            await asyncio.sleep(10)
            
        except asyncio.CancelledError:
            logger.info("Metrics collection task cancelled")
            break
            
        except Exception as e:
            logger.error(f"Error in metrics collection task: {str(e)}")
            await asyncio.sleep(10)

# Track API request
def track_api_request(endpoint: str, method: str, status_code: int, duration: float) -> None:
    """
    Track an API request for metrics.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
        status_code: HTTP status code
        duration: Request duration in seconds
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Increment request counter
        api_requests.labels(endpoint=endpoint, method=method, status=str(status_code)).inc()
        
        # Record request duration
        api_request_duration.labels(endpoint=endpoint, method=method).observe(duration)
        
    except Exception as e:
        logger.error(f"Error tracking API request: {str(e)}")

# Track exchange request
def track_exchange_request(exchange: str, endpoint: str, status: str, duration: float) -> None:
    """
    Track an exchange API request for metrics.
    
    Args:
        exchange: Exchange name
        endpoint: API endpoint
        status: Request status
        duration: Request duration in seconds
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Increment request counter
        exchange_requests.labels(exchange=exchange, endpoint=endpoint, status=status).inc()
        
        # Record request duration
        exchange_request_duration.labels(exchange=exchange, endpoint=endpoint).observe(duration)
        
    except Exception as e:
        logger.error(f"Error tracking exchange request: {str(e)}")

# Track WebSocket message
def track_websocket_message(exchange: str, channel: str) -> None:
    """
    Track a WebSocket message for metrics.
    
    Args:
        exchange: Exchange name
        channel: WebSocket channel
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Increment message counter
        websocket_messages.labels(exchange=exchange, channel=channel).inc()
        
    except Exception as e:
        logger.error(f"Error tracking WebSocket message: {str(e)}")

# Track WebSocket reconnect
def track_websocket_reconnect(exchange: str) -> None:
    """
    Track a WebSocket reconnection for metrics.
    
    Args:
        exchange: Exchange name
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Increment reconnect counter
        websocket_reconnects.labels(exchange=exchange).inc()
        
    except Exception as e:
        logger.error(f"Error tracking WebSocket reconnect: {str(e)}")

# Track signal generated
def track_signal_generated(strategy: str, signal_type: str, asset: str) -> None:
    """
    Track a generated trading signal for metrics.
    
    Args:
        strategy: Strategy name
        signal_type: Signal type
        asset: Asset symbol
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Increment signal counter
        signals_generated.labels(strategy=strategy, signal_type=signal_type, asset=asset).inc()
        
    except Exception as e:
        logger.error(f"Error tracking signal generated: {str(e)}")

# Track signal executed
def track_signal_executed(strategy: str, signal_type: str, asset: str) -> None:
    """
    Track an executed trading signal for metrics.
    
    Args:
        strategy: Strategy name
        signal_type: Signal type
        asset: Asset symbol
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Increment signal counter
        signals_executed.labels(strategy=strategy, signal_type=signal_type, asset=asset).inc()
        
    except Exception as e:
        logger.error(f"Error tracking signal executed: {str(e)}")

# Track order placed
def track_order_placed(exchange: str, order_type: str, asset: str) -> None:
    """
    Track a placed order for metrics.
    
    Args:
        exchange: Exchange name
        order_type: Order type
        asset: Asset symbol
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Increment order counter
        orders_placed.labels(exchange=exchange, order_type=order_type, asset=asset).inc()
        
    except Exception as e:
        logger.error(f"Error tracking order placed: {str(e)}")

# Track order filled
def track_order_filled(exchange: str, order_type: str, asset: str) -> None:
    """
    Track a filled order for metrics.
    
    Args:
        exchange: Exchange name
        order_type: Order type
        asset: Asset symbol
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Increment order counter
        orders_filled.labels(exchange=exchange, order_type=order_type, asset=asset).inc()
        
    except Exception as e:
        logger.error(f"Error tracking order filled: {str(e)}")

# Track order canceled
def track_order_canceled(exchange: str, order_type: str, asset: str) -> None:
    """
    Track a canceled order for metrics.
    
    Args:
        exchange: Exchange name
        order_type: Order type
        asset: Asset symbol
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Increment order counter
        orders_canceled.labels(exchange=exchange, order_type=order_type, asset=asset).inc()
        
    except Exception as e:
        logger.error(f"Error tracking order canceled: {str(e)}")

# Track trade profit
def track_trade_profit(strategy: str, asset: str, profit_pct: float) -> None:
    """
    Track trade profit for metrics.
    
    Args:
        strategy: Strategy name
        asset: Asset symbol
        profit_pct: Profit percentage (e.g., 0.05 for 5%)
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Record profit
        trade_profit_pct.labels(strategy=strategy, asset=asset).observe(profit_pct)
        
    except Exception as e:
        logger.error(f"Error tracking trade profit: {str(e)}")

# Update strategy win rate
def update_strategy_win_rate(strategy: str, timeframe: str, win_rate: float) -> None:
    """
    Update strategy win rate metric.
    
    Args:
        strategy: Strategy name
        timeframe: Timeframe (e.g., '1h', '1d')
        win_rate: Win rate (0-1)
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Set win rate
        strategy_win_rate.labels(strategy=strategy, timeframe=timeframe).set(win_rate)
        
    except Exception as e:
        logger.error(f"Error updating strategy win rate: {str(e)}")

# Update brain council metrics
def update_brain_council_metrics(council: str, asset: str, confidence: float, agreement: float) -> None:
    """
    Update brain council metrics.
    
    Args:
        council: Council name
        asset: Asset symbol
        confidence: Confidence level (0-1)
        agreement: Agreement level (0-1)
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Set confidence and agreement
        brain_council_confidence.labels(council=council, asset=asset).set(confidence)
        brain_council_agreement.labels(council=council, asset=asset).set(agreement)
        
    except Exception as e:
        logger.error(f"Error updating brain council metrics: {str(e)}")

# Update ML model metrics
def update_ml_model_metrics(model: str, version: str, dataset: str, accuracy: float) -> None:
    """
    Update ML model metrics.
    
    Args:
        model: Model name
        version: Model version
        dataset: Dataset name
        accuracy: Accuracy (0-1)
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Set accuracy
        ml_model_accuracy.labels(model=model, version=version, dataset=dataset).set(accuracy)
        
    except Exception as e:
        logger.error(f"Error updating ML model metrics: {str(e)}")

# Track ML model inference
def track_ml_inference(model: str, version: str, duration: float) -> None:
    """
    Track ML model inference duration.
    
    Args:
        model: Model name
        version: Model version
        duration: Inference duration in seconds
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Record inference duration
        ml_inference_duration.labels(model=model, version=version).observe(duration)
        
    except Exception as e:
        logger.error(f"Error tracking ML inference: {str(e)}")

# Track crawler request
def track_crawler_request(crawler: str, status: str) -> None:
    """
    Track a crawler request for metrics.
    
    Args:
        crawler: Crawler name
        status: Request status
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Increment request counter
        crawler_requests.labels(crawler=crawler, status=status).inc()
        
    except Exception as e:
        logger.error(f"Error tracking crawler request: {str(e)}")

# Track crawler data item
def track_crawler_data_item(crawler: str, data_type: str) -> None:
    """
    Track a crawler data item for metrics.
    
    Args:
        crawler: Crawler name
        data_type: Data item type
    """
    if not MONITORING_ENABLED:
        return
    
    try:
        # Increment data item counter
        crawler_data_items.labels(crawler=crawler, data_type=data_type).inc()
        
    except Exception as e:
        logger.error(f"Error tracking crawler data item: {str(e)}")

# Publish event
async def publish_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Publish an event to Redis.
    
    Args:
        event_type: Event type
        data: Event data
    """
    try:
        # Add event metadata
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Publish to Redis
        await redis_publish('events', event)
        
        # Track event
        if event_type in monitored_events:
            monitored_events[event_type] += 1
        else:
            monitored_events[event_type] = 1

    except Exception as e:
        logger.error(f"Error publishing event: {str(e)}")
