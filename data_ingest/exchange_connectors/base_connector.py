"""
Base Connector for QuantumSpectre Elite Trading System.

This module defines the base connector interface for all exchange interactions.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
import logging
import uuid
import hmac
import hashlib
import traceback

from common.config import settings
from common.logger import get_logger
from common.models.market_data import MarketData, OrderBook, Trade, Candle
from common.metrics.connector_metrics import ConnectorMetrics
from common.exceptions import (
    ConnectionError, 
    AuthenticationError, 
    RateLimitError,
    OrderNotFoundError,
    InsufficientBalanceError,
    ExchangeError
)
from common.utils.retry import async_retry_with_backoff
from common.utils.rate_limiter import RateLimiter
from common.event_bus import EventBus

logger = get_logger(__name__)

class BaseConnector(ABC):
    """
    Abstract base class for all exchange connectors.
    
    Handles common functionality for exchange communication including:
    - Authentication
    - Rate limiting
    - Connection management
    - Error handling
    - Event dispatching
    """
    
    def __init__(self, 
                 exchange_id: str,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = False,
                 rate_limit_config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize base connector.
        
        Args:
            exchange_id: Unique identifier for the exchange
            api_key: API key for authenticated requests
            api_secret: API secret for signed requests
            testnet: Whether to use testnet/sandbox
            rate_limit_config: Configuration for rate limiting
            event_bus: Event bus for publishing events
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Setup logger
        self.logger = get_logger(f"connector.{exchange_id}")
        
        # Connection state
        self.is_connected = False
        self.last_heartbeat = 0
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            **(rate_limit_config or {
                "max_requests": settings.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
                "time_frame": settings.DEFAULT_RATE_LIMIT_TIME_FRAME
            })
        )
        
        # Sessions (to be initialized by subclasses)
        self.ws_sessions = {}
        self.rest_session = None
        
        # Tasks
        self.tasks = set()
        
        # Metrics
        self.metrics = ConnectorMetrics(exchange_id)
        
        # Event bus
        self.event_bus = event_bus or EventBus.get_instance()
        
        # Order ID tracking
        self.order_map = {}  # Local to exchange order ID mapping
        
        # Setup hooks
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup hooks for lifecycle events."""
        self.pre_connect_hooks = []
        self.post_connect_hooks = []
        self.pre_disconnect_hooks = []
        self.post_disconnect_hooks = []
    
    def add_hook(self, event: str, hook: Callable):
        """
        Add a hook for a specific lifecycle event.
        
        Args:
            event: Event name ('pre_connect', 'post_connect', 'pre_disconnect', 'post_disconnect')
            hook: Callable to execute
        """
        if event == 'pre_connect':
            self.pre_connect_hooks.append(hook)
        elif event == 'post_connect':
            self.post_connect_hooks.append(hook)
        elif event == 'pre_disconnect':
            self.pre_disconnect_hooks.append(hook)
        elif event == 'post_disconnect':
            self.post_disconnect_hooks.append(hook)
        else:
            raise ValueError(f"Unknown hook event: {event}")
    
    async def _execute_hooks(self, hooks: List[Callable]):
        """Execute all hooks in sequence."""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self)
                else:
                    hook(self)
            except Exception as e:
                self.logger.error(f"Error executing hook: {str(e)}")
                self.logger.debug(traceback.format_exc())
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the exchange.
        
        Returns:
            Connection success status
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the exchange.
        
        Returns:
            Disconnection success status
        """
        pass
    
    @abstractmethod
    async def fetch_market_data(self, symbol: str, data_type: str, **kwargs) -> Any:
        """
        Fetch specific market data.
        
        Args:
            symbol: Trading symbol
            data_type: Type of data ('ticker', 'orderbook', 'trades', 'candles')
            **kwargs: Additional parameters
            
        Returns:
            Market data
        """
        pass
    
    @abstractmethod
    async def subscribe_market_data(self, symbol: str, data_types: List[str], callbacks: Dict[str, Callable]) -> bool:
        """
        Subscribe to real-time market data.
        
        Args:
            symbol: Trading symbol
            data_types: Types of data to subscribe to
            callbacks: Callback functions for each data type
            
        Returns:
            Subscription success status
        """
        pass
    
    @abstractmethod
    async def unsubscribe_market_data(self, symbol: str, data_types: List[str]) -> bool:
        """
        Unsubscribe from real-time market data.
        
        Args:
            symbol: Trading symbol
            data_types: Types of data to unsubscribe from
            
        Returns:
            Unsubscription success status
        """
        pass
    
    @abstractmethod
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, 
                         price: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Create a new order.
        
        Args:
            symbol: Trading symbol
            order_type: Order type (market, limit, etc.)
            side: Order side (buy or sell)
            amount: Order quantity
            price: Order price (for limit orders)
            **kwargs: Additional parameters
            
        Returns:
            Order details
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID
            
        Returns:
            Cancellation success status
        """
        pass
    
    @abstractmethod
    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Fetch account balance.
        
        Returns:
            Account balance information
        """
        pass
    
    @abstractmethod
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch open orders.
        
        Args:
            symbol: Optional trading symbol
            
        Returns:
            List of open orders
        """
        pass
    
    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch order details.
        
        Args:
            order_id: Order ID
            symbol: Optional trading symbol
            
        Returns:
            Order details
        """
        pass
    
    async def generate_local_order_id(self) -> str:
        """
        Generate a unique local order ID.
        
        Returns:
            Unique order ID
        """
        return f"{self.exchange_id}-{uuid.uuid4()}"
    
    async def map_order_id(self, local_id: str, exchange_id: str):
        """
        Map local order ID to exchange order ID.
        
        Args:
            local_id: Local order ID
            exchange_id: Exchange order ID
        """
        self.order_map[local_id] = exchange_id
        await self.event_bus.publish(
            'order_id_mapped', 
            {
                'exchange': self.exchange_id,
                'local_id': local_id,
                'exchange_id': exchange_id,
                'timestamp': time.time()
            }
        )
    
    async def get_exchange_order_id(self, local_id: str) -> Optional[str]:
        """
        Get exchange order ID from local order ID.
        
        Args:
            local_id: Local order ID
            
        Returns:
            Exchange order ID if found, None otherwise
        """
        return self.order_map.get(local_id)
    
    async def sign_request(self, endpoint: str, params: Dict[str, Any], method: str = 'GET') -> Dict[str, Any]:
        """
        Sign a request with API credentials.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            method: HTTP method
            
        Returns:
            Signed parameters
        """
        if not self.api_secret:
            return params
        
        # Add timestamp
        params['timestamp'] = int(time.time() * 1000)
        
        # Create signature
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        return params
    
    async def handle_error(self, error: Exception, critical: bool = False, context: Dict[str, Any] = None):
        """
        Handle connector error.
        
        Args:
            error: The error that occurred
            critical: Whether this is a critical error
            context: Additional context information
        """
        error_info = {
            'exchange': self.exchange_id,
            'error': str(error),
            'error_type': type(error).__name__,
            'critical': critical,
            'timestamp': time.time(),
            'context': context or {}
        }
        
        if critical:
            self.logger.critical(f"Critical error in {self.exchange_id} connector: {str(error)}")
        else:
            self.logger.error(f"Error in {self.exchange_id} connector: {str(error)}")
        
        if context:
            self.logger.debug(f"Error context: {json.dumps(context)}")
        
        # Update metrics
        self.metrics.increment_error_count(error_info['error_type'])
        
        # Publish error event
        await self.event_bus.publish('connector_error', error_info)
    
    async def heartbeat(self):
        """Send heartbeat and check connection health."""
        self.last_heartbeat = time.time()
        await self.event_bus.publish(
            'connector_heartbeat', 
            {
                'exchange': self.exchange_id,
                'timestamp': self.last_heartbeat,
                'connected': self.is_connected
            }
        )
    
    def __repr__(self) -> str:
        """String representation of the connector."""
        return f"{self.__class__.__name__}(exchange_id={self.exchange_id}, connected={self.is_connected})"
