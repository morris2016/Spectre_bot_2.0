"""
Common Module for QuantumSpectre Elite Trading System.

This module provides shared utilities, services, and configurations used throughout the system.
"""

__version__ = '1.0.0'

from .logger import get_logger, setup_logging
from .utils import (
    generate_uuid,
    timestamp_ms,
    format_currency,
    retry,
    rate_limit,
    timer,
    chunks,
    parse_timeframe,
    ClassRegistry,
    AsyncService,
    Signal,
    SignalBus
)
from .event_bus import EventBus
from .constants import (
    TIME_FRAMES, 
    EXCHANGE_TYPES, 
    ORDER_TYPES, 
    ORDER_SIDES,
    ORDER_STATUSES
)
from .exceptions import (
    QuantumSpectreError,
    ConfigurationError,
    ExchangeError,
    DataError,
    ExecutionError,
    AuthenticationError,
    RateLimitError
)
from .metrics import MetricsCollector
from .security import encrypt_data, decrypt_data, hash_password, verify_password
from .redis_client import RedisClient, get_redis_pool
from .db_client import DatabaseClient, get_db_client
from .async_utils import gather_with_concurrency, cancel_tasks, create_task_group

__all__ = [
    # Logger
    'get_logger',
    'setup_logging',
    
    # Utils
    'generate_uuid',
    'timestamp_ms',
    'format_currency',
    'retry',
    'rate_limit',
    'timer',
    'chunks',
    'parse_timeframe',
    
    # Constants
    'TIME_FRAMES',
    'EXCHANGE_TYPES',
    'ORDER_TYPES',
    'ORDER_SIDES',
    'ORDER_STATUSES',
    
    # Exceptions
    'QuantumSpectreError',
    'ConfigurationError',
    'ExchangeError',
    'DataError',
    'ExecutionError',
    'AuthenticationError',
    'RateLimitError',
    
    # Services
    'MetricsCollector',
    'encrypt_data',
    'decrypt_data',
    'hash_password',
    'verify_password',
    'RedisClient',
    'get_redis_pool',
    'DatabaseClient',
    'get_db_client',
    
    # Async utilities
    'gather_with_concurrency',
    'cancel_tasks',
    'create_task_group',
    'ClassRegistry',
    'AsyncService',
    'Signal',
    'SignalBus',
    'EventBus'
]
