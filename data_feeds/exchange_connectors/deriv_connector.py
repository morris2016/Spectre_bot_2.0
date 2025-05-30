"""
Deriv Connector for QuantumSpectre Elite Trading System.

This module provides a specialized connector for Deriv exchange.
"""

import asyncio
import json
import time
import websockets
import logging
from typing import Dict, List, Any, Optional, Union, Callable
import random
import traceback
from datetime import datetime, timezone
import uuid

from common.config import settings
from common.logger import get_logger
from common.models.market_data import MarketData, OrderBook, Trade, Candle, Ticker
from common.exceptions import (
    ConnectionError, 
    AuthenticationError, 
    RateLimitError,
    OrderNotFoundError,
    InsufficientBalanceError,
    ExchangeError
)
from common.utils.retry import async_retry_with_backoff
from common.event_bus import EventBus

from .base_connector import BaseConnector

class DerivConnector(BaseConnector):
    """
    Deriv exchange connector implementing the BaseConnector interface.
    
    Supports WebSocket API for market data and trading.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = False,
                 rate_limit_config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize Deriv connector.
        
        Args:
            api_key: Deriv API token
            api_secret: Not used for Deriv (kept for interface consistency)
            testnet: Whether to use demo environment
            rate_limit_config: Configuration for rate limiting
            event_bus: Event bus for publishing events
        """
        super().__init__(
            exchange_id="deriv",
            api_key=api_key,
            api_secret=None,  # Deriv doesn't use API secret
            testnet=testnet,
            rate_limit_config=rate_limit_config,
            event_bus=event_bus
        )
        
        # Set up URLs based on testnet flag
        self.ws_url = settings.DERIV_DEMO_WS_URL if testnet else settings.DERIV_WS_URL
        
        # WebSocket connection
        self.ws = None
        self.ws_connected = False
        self.request_id = 1
        self.pending_requests = {}
        self.subscriptions = {}
        
        # Callbacks for subscriptions
        self.callbacks = {}
        
        # Connection management
        self.ws_connect_lock = asyncio.Lock()
        self.reconnect_delay = settings.BASE_RECONNECT_DELAY
        self.max_reconnect_delay = settings.MAX_RECONNECT_DELAY
        
        # Common symbols mapping (Deriv has different format)
        self.symbol_map = {
            "BTCUSD": "cryBTCUSD",
            "ETHUSD": "cryETHUSD",
            "EURUSD": "frxEURUSD",
            "GBPUSD": "frxGBPUSD",
            "USDJPY": "frxUSDJPY",
            # Add more symbols as needed
        }
        self.reverse_symbol_map = {v: k for k, v in self.symbol_map.items()}
    
    def _get_deriv_symbol(self, symbol: str) -> str:
        """Convert common symbol to Deriv format."""
        return self.symbol_map.get(symbol, symbol)
    
    def _get_common_symbol(self, deriv_symbol: str) -> str:
        """Convert Deriv symbol to common format."""
        return self.reverse_symbol_map.get(deriv_symbol, deriv_symbol)
    
    async def connect(self) -> bool:
        """
        Establish connection to Deriv.
        
        Returns:
            Connection success status
        """
        try:
            await self._execute_hooks(self.pre_connect_hooks)
            
            self.logger.info("Connecting to Deriv...")
            
            async with self.ws_connect_lock:
                if self.ws_connected:
                    self.logger.debug("Already connected to Deriv")
                    return True
                
                self.ws = await websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5
                )
                
                # Start message handler
                self.ws_task = asyncio.create_task(self._message_handler())
                self.tasks.add(self.ws_task)
                self.ws_task.add_done_callback(self.tasks.discard)
                
                # Wait for connection to be established
                self.ws_connected = True
                
                # Authenticate if API key is provided
                if self.api_key:
                    await self._authenticate()
                
                # Start background tasks
                self._start_background_tasks()
                
                self.is_connected = True
                self.logger.info("Successfully connected to Deriv")
                
                # Update metrics
                self.metrics.record_connection()
                
                await self._execute_hooks(self.post_connect_hooks)
                return True
                
        except Exception as e:
            await self.handle_error(e, critical=True, context={"action": "connect"})
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from Deriv.
        
        Returns:
            Disconnection success status
        """
        try:
            await self._execute_hooks(self.pre_disconnect_hooks)
            
            self.logger.info("Disconnecting from Deriv...")
            
            # Cancel background tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            self.tasks = set()
            
            # Close WebSocket connection
            if self.ws:
                await self.ws.close()
                self.ws = None
            
            self.ws_connected = False
            self.is_connected = False
            
            self.logger.info("Successfully disconnected from Deriv")
            
            await self._execute_hooks(self.post_disconnect_hooks)
            return True
            
        except Exception as e:
            await self.handle_error(e, context={"action": "disconnect"})
            return False
    
    def _start_background_tasks(self):
        """Start background tasks for monitoring and maintenance."""
        # Heartbeat task
        task = asyncio.create_task(self._heartbeat_loop())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        
        # Subscription renewal task
        task = asyncio.create_task(self._subscription_renewal_loop())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats and monitor connection health."""
        while self.is_connected:
            try:
                await self.heartbeat()
                
                # Send ping to keep connection alive
                if self.ws_connected:
                    await self._send_ping()
                
                # Check if connection is stale
                if time.time() - self.last_heartbeat > 60:  # No message for 60 seconds
                    self.logger.warning("No messages received for 60s, reconnecting...")
                    await self._reconnect()
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(5)  # Shorter delay on error
    
    async def _subscription_renewal_loop(self):
        """Renew subscriptions periodically to prevent expiry."""
        while self.is_connected:
            try:
                # Deriv subscriptions expire after some time, so we need to renew them
                for subscription_id, subscription in list(self.subscriptions.items()):
                    # Check if subscription is more than 10 hours old
                    if time.time() - subscription.get('timestamp', 0) > 36000:  # 10 hours
                        self.logger.debug(f"Renewing subscription {subscription_id}")
                        
                        # Get subscription details
                        request = subscription.get('request')
                        if request:
                            # Send subscription request again
                            await self._send_request(request, subscription.get('callback'))
                            
                            # Update timestamp
                            self.subscriptions[subscription_id]['timestamp'] = time.time()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in subscription renewal loop: {str(e)}")
                await asyncio.sleep(300)  # Retry after 5 minutes
    
    async def _message_handler(self):
        """Handle WebSocket messages."""
        try:
            while self.ws_connected:
                try:
                    message = await self.ws.recv()
                    self.last_heartbeat = time.time()
                    
                    # Parse message
                    data = json.loads(message)
                    
                    # Handle subscription notifications
                    if 'subscription' in data:
                        await self._handle_subscription(data)
                    
                    # Handle error messages
                    elif 'error' in data:
                        await self._handle_error_message(data)
                    
                    # Handle ping/pong
                    elif 'ping' in data:
                        await self._handle_ping(data)
                    
                    # Handle regular responses
                    elif 'req_id' in data or 'msg_type' in data:
                        await self._handle_response(data)
                    
                    # Other messages
                    else:
                        self.logger.debug(f"Unhandled message type: {message[:100]}...")
                        
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON in WebSocket message")
                except Exception as e:
                    self.logger.error(f"Error processing WebSocket message: {str(e)}")
                    self.logger.debug(traceback.format_exc())
                    
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.warning(f"WebSocket connection closed: {str(e)}")
            self.ws_connected = False
            
            # Try to reconnect
            if self.is_connected:  # Only reconnect if we haven't explicitly disconnected
                asyncio.create_task(self._reconnect())
                
        except asyncio.CancelledError:
            self.logger.info("WebSocket message handler cancelled")
            raise
            
        except Exception as e:
            self.logger.error(f"Error in WebSocket message handler: {str(e)}")
            self.logger.debug(traceback.format_exc())
            self.ws_connected = False
            
            # Try to reconnect
            if self.is_connected:
                asyncio.create_task(self._reconnect())
    
    async def _reconnect(self):
        """Reconnect to WebSocket."""
        retry_count = 0
        while self.is_connected and retry_count < 10:
            try:
                # Exponential backoff
                backoff = min(
                    self.max_reconnect_delay,
                    self.reconnect_delay * (2 ** retry_count)
                )
                self.logger.info(f"Reconnecting to Deriv in {backoff:.2f}s...")
                await asyncio.sleep(backoff)
                
                # Close existing connection if any
                if self.ws:
                    await self.ws.close()
                
                # Reconnect
                await self.connect()
                
                # Resubscribe to all active subscriptions
                for subscription_id, subscription in list(self.subscriptions.items()):
                    request = subscription.get('request')
                    if request:
                        await self._send_request(request, subscription.get('callback'))
                
                return
                
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Failed to reconnect: {str(e)}")
    
    async def _authenticate(self):
        """Authenticate with API token."""
        if not self.api_key:
            return
        
        try:
            request = {
                "authorize": self.api_key,
                "req_id": self._get_next_req_id()
            }
            
            response = await self._send_request(request)
            
            if 'error' in response:
                raise AuthenticationError(f"Authentication failed: {response['error']['message']}")
            
            if 'authorize' in response and response['authorize'].get('loginid'):
                self.logger.info(f"Authenticated as {response['authorize']['loginid']}")
                return True
            
            return False
            
        except Exception as e:
            await self.handle_error(e, critical=True, context={"action": "authenticate"})
            return False
    
    async def _send_ping(self):
        """Send ping message to keep connection alive."""
        try:
            request = {
                "ping": 1,
                "req_id": self._get_next_req_id()
            }
            
            await self._send_request(request)
            
        except Exception as e:
            self.logger.error(f"Error sending ping: {str(e)}")
    
    async def _handle_ping(self, data: Dict[str, Any]):
        """Handle ping message from server."""
        try:
            pong_msg = {"pong": data.get('ping')}
            await self._send_raw(pong_msg)
        except Exception as e:
            self.logger.error(f"Error sending pong: {str(e)}")
    
    async def _handle_subscription(self, data: Dict[str, Any]):
        """Handle subscription update message."""
        try:
            subscription_id = data.get('subscription', {}).get('id')
            if not subscription_id:
                return
            
            subscription = self.subscriptions.get(subscription_id)
            if not subscription:
                self.logger.warning(f"Received update for unknown subscription: {subscription_id}")
                return
            
            # Call the subscription callback
            callback = subscription.get('callback')
            if callback:
                asyncio.create_task(callback(data))
            
        except Exception as e:
            self.logger.error(f"Error handling subscription update: {str(e)}")
    
    async def _handle_error_message(self, data: Dict[str, Any]):
        """Handle error message from server."""
        error = data.get('error', {})
        error_code = error.get('code')
        error_message = error.get('message')
        
        self.logger.error(f"Deriv API error: {error_code} - {error_message}")
        
        # Handle request-specific errors
        req_id = data.get('req_id')
        if req_id in self.pending_requests:
            # Resolve the pending request with the error
            future = self.pending_requests.pop(req_id)
            future.set_result(data)
        
        # Publish error event
        await self.event_bus.publish('connector_error', {
            'exchange': self.exchange_id,
            'error': error_message,
            'error_code': error_code,
            'timestamp': time.time()
        })
    
    async def _handle_response(self, data: Dict[str, Any]):
        """Handle response message from server."""
        req_id = data.get('req_id')
        
        # If this is a response to a pending request, resolve it
        if req_id in self.pending_requests:
            future = self.pending_requests.pop(req_id)
            future.set_result(data)
    
    def _get_next_req_id(self) -> int:
        """Get the next request ID."""
        req_id = self.request_id
        self.request_id += 1
        return req_id
    
    async def _send_raw(self, data: Dict[str, Any]) -> None:
        """Send raw message to WebSocket."""
        if not self.ws_connected:
            raise ConnectionError("WebSocket not connected")
        
        try:
            message = json.dumps(data)
            await self.ws.send(message)
        except Exception as e:
            self.logger.error(f"Error sending WebSocket message: {str(e)}")
            self.ws_connected = False
            raise
    
    async def _send_request(self, request: Dict[str, Any], callback: Callable = None) -> Dict[str, Any]:
        """
        Send a request and wait for the response.
        
        Args:
            request: Request data
            callback: Optional callback for subscription updates
            
        Returns:
            Response data
        """
        if not self.ws_connected:
            await self.connect()
        
        req_id = request.get('req_id')
        if not req_id:
            req_id = self._get_next_req_id()
            request['req_id'] = req_id
        
        future = asyncio.Future()
        self.pending_requests[req_id] = future
        
        try:
            await self._send_raw(request)
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(future, timeout=30.0)
                
                # Store subscription information if this is a subscription
                if 'subscription' in response and callback:
                    subscription_id = response['subscription']['id']
                    self.subscriptions[subscription_id] = {
                        'request': request,
                        'callback': callback,
                        'timestamp': time.time()
                    }
                
                return response
                
            except asyncio.TimeoutError:
                self.logger.error(f"Request timed out: {request}")
                self.pending_requests.pop(req_id, None)
                raise TimeoutError("Request timed out")
                
        except Exception as e:
            self.pending_requests.pop(req_id, None)
            raise
    
    async def fetch_market_data(self, symbol: str, data_type: str, **kwargs) -> Any:
        """
        Fetch specific market data from Deriv.
        
        Args:
            symbol: Trading symbol
            data_type: Type of data ('ticker', 'candles')
            **kwargs: Additional parameters
            
        Returns:
            Market data
        """
        try:
            # Convert to Deriv symbol format
            deriv_symbol = self._get_deriv_symbol(symbol)
            
            if data_type == 'ticker':
                return await self._fetch_ticker(deriv_symbol)
            elif data_type == 'candles':
                interval = kwargs.get('interval', '60')  # Default 1 minute
                count = kwargs.get('limit', 100)
                return await self._fetch_candles(deriv_symbol, interval, count)
            else:
                raise ValueError(f"Unsupported data type for Deriv: {data_type}")
                
        except Exception as e:
            await self.handle_error(e, context={
                "action": "fetch_market_data",
                "symbol": symbol,
                "data_type": data_type,
                "params": kwargs
            })
            raise
    
    async def _fetch_ticker(self, symbol: str) -> Ticker:
        """Fetch ticker data for a symbol."""
        request = {
            "ticks": symbol,
            "subscribe": 0,
            "req_id": self._get_next_req_id()
        }
        
        response = await self._send_request(request)
        
        if 'error' in response:
            raise ExchangeError(f"Error fetching ticker: {response['error']['message']}")
        
        tick = response.get('tick', {})
        
        # Convert to standard model
        ticker = Ticker(
            symbol=self._get_common_symbol(symbol),
            exchange="deriv",
            last_price=float(tick.get('quote', 0)),
            bid_price=float(tick.get('bid', 0)),
            ask_price=float(tick.get('ask', 0)),
            volume=0,  # Deriv doesn't provide volume
            quote_volume=0,
            timestamp=tick.get('epoch', int(time.time())) * 1000,
            high_24h=0,
            low_24h=0,
            price_change_24h=0,
            price_change_percent_24h=0,
            raw=response
        )
        
        return ticker
    
    async def _fetch_candles(self, symbol: str, interval: str = '60', count: int = 100) -> List[Candle]:
        """Fetch candlestick data for a symbol."""
        request = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": interval,
            "count": count,
            "req_id": self._get_next_req_id()
        }
        
        response = await self._send_request(request)
        
        if 'error' in response:
            raise ExchangeError(f"Error fetching candles: {response['error']['message']}")
        
        candles_data = response.get('candles', [])
        
        # Convert to standard model
        candles = [
            Candle(
                symbol=self._get_common_symbol(symbol),
                exchange="deriv",
                timestamp=candle.get('epoch', 0) * 1000,
                open=float(candle.get('open', 0)),
                high=float(candle.get('high', 0)),
                low=float(candle.get('low', 0)),
                close=float(candle.get('close', 0)),
                volume=0,  # Deriv doesn't provide volume
                close_time=candle.get('epoch', 0) * 1000 + int(interval) * 1000,
                quote_volume=0,
                trades=0,
                taker_buy_base_volume=0,
                taker_buy_quote_volume=0,
                interval=interval,
                raw=candle
            )
            for candle in candles_data
        ]
        
        return candles
    
    async def subscribe_market_data(self, symbol: str, data_types: List[str], callbacks: Dict[str, Callable]) -> bool:
        """
        Subscribe to real-time market data via WebSocket.
        
        Args:
            symbol: Trading symbol
            data_types: Types of data to subscribe to ('ticker', 'candles')
            callbacks: Callback functions for each data type
            
        Returns:
            Subscription success status
        """
        try:
            # Convert to Deriv symbol format
            deriv_symbol = self._get_deriv_symbol(symbol)
            
            for data_type in data_types:
                if data_type == 'ticker':
                    await self._subscribe_ticker(deriv_symbol, callbacks.get(data_type))
                elif data_type == 'candles':
                    interval = callbacks.get('interval', '60')  # Default 1 minute
                    await self._subscribe_candles(deriv_symbol, interval, callbacks.get(data_type))
                else:
                    self.logger.warning(f"Unsupported data type for Deriv: {data_type}")
            
            return True
            
        except Exception as e:
            await self.handle_error(e, context={
                "action": "subscribe_market_data",
                "symbol": symbol,
                "data_types": data_types
            })
            return False
    
    async def _subscribe_ticker(self, symbol: str, callback: Callable) -> Dict[str, Any]:
        """Subscribe to ticker updates."""
        request = {
            "ticks": symbol,
            "subscribe": 1,
            "req_id": self._get_next_req_id()
        }
        
        response = await self._send_request(request, self._handle_ticker_update)
        
        if 'error' in response:
            raise ExchangeError(f"Error subscribing to ticker: {response['error']['message']}")
        
        # Store callback
        stream_id = f"{symbol}:ticker"
        self.callbacks[stream_id] = callback
        
        return response
    
    async def _handle_ticker_update(self, data: Dict[str, Any]) -> None:
        """Handle ticker update from subscription."""
        tick = data.get('tick', {})
        if not tick:
            return
        
        symbol = tick.get('symbol')
        common_symbol = self._get_common_symbol(symbol)
        
        # Convert to standard model
        ticker = Ticker(
            symbol=common_symbol,
            exchange="deriv",
            last_price=float(tick.get('quote', 0)),
            bid_price=float(tick.get('bid', 0)),
            ask_price=float(tick.get('ask', 0)),
            volume=0,  # Deriv doesn't provide volume
            quote_volume=0,
            timestamp=tick.get('epoch', int(time.time())) * 1000,
            high_24h=0,
            low_24h=0,
            price_change_24h=0,
            price_change_percent_24h=0,
            raw=data
        )
        
        # Call user callback
        stream_id = f"{symbol}:ticker"
        callback = self.callbacks.get(stream_id)
        if callback:
            try:
                await callback(ticker)
            except Exception as e:
                self.logger.error(f"Error in ticker callback: {str(e)}")
        
        # Publish event
        await self.event_bus.publish('market_data_ticker', {
            'exchange': self.exchange_id,
            'symbol': common_symbol,
            'ticker': ticker.to_dict(),
            'timestamp': time.time()
        })
    
    async def _subscribe_candles(self, symbol: str, interval: str, callback: Callable) -> Dict[str, Any]:
        """Subscribe to candlestick updates."""
        request = {
            "ticks_history": symbol,
            "style": "candles",
            "granularity": interval,
            "subscribe": 1,
            "req_id": self._get_next_req_id()
        }
        
        response = await self._send_request(request, self._handle_candle_update)
        
        if 'error' in response:
            raise ExchangeError(f"Error subscribing to candles: {response['error']['message']}")
        
        # Store callback and interval
        stream_id = f"{symbol}:candles:{interval}"
        self.callbacks[stream_id] = callback
        
        return response
    
    async def _handle_candle_update(self, data: Dict[str, Any]) -> None:
        """Handle candle update from subscription."""
        ohlc = data.get('ohlc', {})
        if not ohlc:
            return
        
        symbol = ohlc.get('symbol')
        interval = ohlc.get('granularity')
        common_symbol = self._get_common_symbol(symbol)
        
        # Convert to standard model
        candle = Candle(
            symbol=common_symbol,
            exchange="deriv",
            timestamp=ohlc.get('open_time', int(time.time())) * 1000,
            open=float(ohlc.get('open', 0)),
            high=float(ohlc.get('high', 0)),
            low=float(ohlc.get('low', 0)),
            close=float(ohlc.get('close', 0)),
            volume=0,  # Deriv doesn't provide volume
            close_time=ohlc.get('epoch', int(time.time())) * 1000,
            quote_volume=0,
            trades=0,
            taker_buy_base_volume=0,
            taker_buy_quote_volume=0,
            interval=interval,
            raw=data
        )
        
        # Call user callback
        stream_id = f"{symbol}:candles:{interval}"
        callback = self.callbacks.get(stream_id)
        if callback:
            try:
                await callback(candle)
            except Exception as e:
                self.logger.error(f"Error in candle callback: {str(e)}")
        
        # Publish event
        await self.event_bus.publish('market_data_candle', {
            'exchange': self.exchange_id,
            'symbol': common_symbol,
            'candle': candle.to_dict(),
            'interval': interval,
            'timestamp': time.time()
        })
    
    async def unsubscribe_market_data(self, symbol: str, data_types: List[str]) -> bool:
        """
        Unsubscribe from real-time market data.
        
        Args:
            symbol: Trading symbol
            data_types: Types of data to unsubscribe from
            
        Returns:
            Unsubscription success status
        """
        try:
            # Convert to Deriv symbol format
            deriv_symbol = self._get_deriv_symbol(symbol)
            
            for data_type in data_types:
                if data_type == 'ticker':
                    # Find the subscription ID
                    stream_id = f"{deriv_symbol}:ticker"
                    await self._unsubscribe(stream_id)
                elif data_type == 'candles':
                    # Find all candle subscriptions for this symbol
                    for stream_id in list(self.callbacks.keys()):
                        if stream_id.startswith(f"{deriv_symbol}:candles:"):
                            await self._unsubscribe(stream_id)
                else:
                    self.logger.warning(f"Unsupported data type for unsubscription: {data_type}")
            
            return True
            
        except Exception as e:
            await self.handle_error(e, context={
                "action": "unsubscribe_market_data",
                "symbol": symbol,
                "data_types": data_types
            })
            return False
    
    async def _unsubscribe(self, stream_id: str) -> bool:
        """Unsubscribe from a specific stream."""
        try:
            # Find subscription IDs for this stream
            to_remove = []
            for subscription_id, subscription in self.subscriptions.items():
                request = subscription.get('request', {})
                if (stream_id.endswith(':ticker') and 'ticks' in request and request.get('subscribe') == 1) or \
                   (stream_id.startswith(deriv_symbol) and 'ticks_history' in request and request.get('subscribe') == 1):
                    # Send forget request
                    forget_request = {
                        "forget": subscription_id,
                        "req_id": self._get_next_req_id()
                    }
                    await self._send_request(forget_request)
                    to_remove.append(subscription_id)
            
            # Remove subscriptions
            for subscription_id in to_remove:
                self.subscriptions.pop(subscription_id, None)
            
            # Remove callbacks
            if stream_id in self.callbacks:
                del self.callbacks[stream_id]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing from {stream_id}: {str(e)}")
            return False
    
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, 
                         price: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Create a new order on Deriv.
        
        Args:
            symbol: Trading symbol
            order_type: Order type (only MARKET supported for Deriv)
            side: Order side (BUY or SELL)
            amount: Order quantity
            price: Order price (not used for Deriv)
            **kwargs: Additional parameters
            
        Returns:
            Order details
        """
        if not self.api_key:
            raise AuthenticationError("API token required for trading")
        
        try:
            # Generate local order ID
            local_id = await self.generate_local_order_id()
            
            # Convert to Deriv symbol format
            deriv_symbol = self._get_deriv_symbol(symbol)
            
            # Deriv uses different terms for buy/sell
            contract_type = "CALL" if side.upper() == "BUY" else "PUT"
            
            # Prepare buy contract request
            request = {
                "buy": 1,
                "price": amount,  # Amount to buy for
                "parameters": {
                    "contract_type": contract_type,
                    "symbol": deriv_symbol,
                    "duration": kwargs.get("duration", 60),  # Default 1 minute
                    "duration_unit": kwargs.get("duration_unit", "s"),  # Default seconds
                    "currency": kwargs.get("currency", "USD"),
                    "barrier": kwargs.get("barrier")  # Target price for barrier contracts
                },
                "req_id": self._get_next_req_id()
            }
            
            # Send request
            response = await self._send_request(request)
            
            if 'error' in response:
                raise ExchangeError(f"Error creating order: {response['error']['message']}")
            
            # Extract contract/order details
            contract_id = response.get('buy', {}).get('contract_id')
            if contract_id:
                # Map local order ID to contract ID
                await self.map_order_id(local_id, contract_id)
                
                # Add local ID to response
                response['local_id'] = local_id
            
            # Publish order event
            await self.event_bus.publish('order_created', {
                'exchange': self.exchange_id,
                'order': response,
                'local_id': local_id,
                'timestamp': time.time()
            })
            
            return response
            
        except Exception as e:
            await self.handle_error(e, context={
                "action": "create_order",
                "symbol": symbol,
                "order_type": order_type,
                "side": side,
                "amount": amount
            })
            raise
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an existing order on Deriv.
        
        Args:
            symbol: Trading symbol (not used for Deriv, kept for interface consistency)
            order_id: Contract ID
            
        Returns:
            Cancellation success status
        """
        if not self.api_key:
            raise AuthenticationError("API token required for trading")
        
        try:
            # Deriv uses "sell" to close positions
            request = {
                "sell": order_id,
                "price": 0,  # Market price
                "req_id": self._get_next_req_id()
            }
            
            response = await self._send_request(request)
            
            if 'error' in response:
                if 'ContractNotFound' in response['error']['code']:
                    raise OrderNotFoundError(f"Contract not found: {order_id}")
                raise ExchangeError(f"Error cancelling order: {response['error']['message']}")
            
            # Publish order event
            await self.event_bus.publish('order_cancelled', {
                'exchange': self.exchange_id,
                'order_id': order_id,
                'response': response,
                'timestamp': time.time()
            })
            
            return True
            
        except OrderNotFoundError:
            self.logger.warning(f"Contract {order_id} not found for cancellation")
            return False
        except Exception as e:
            await self.handle_error(e, context={
                "action": "cancel_order",
                "order_id": order_id
            })
            return False
    
    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Fetch account balance from Deriv.
        
        Returns:
            Account balance information
        """
        if not self.api_key:
            raise AuthenticationError("API token required for fetching balance")
        
        try:
            request = {
                "balance": 1,
                "req_id": self._get_next_req_id()
            }
            
            response = await self._send_request(request)
            
            if 'error' in response:
                raise ExchangeError(f"Error fetching balance: {response['error']['message']}")
            
            balance_info = response.get('balance', {})
            currency = balance_info.get('currency', 'USD')
            
            # Construct balance object
            balances = {
                currency: {
                    'free': float(balance_info.get('balance', 0)),
                    'used': 0,  # Deriv doesn't provide used balance
                    'total': float(balance_info.get('balance', 0))
                }
            }
            
            return balances
            
        except Exception as e:
            await self.handle_error(e, context={"action": "fetch_balance"})
            raise
    
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch open positions from Deriv.
        
        Args:
            symbol: Optional trading symbol (not used for Deriv)
            
        Returns:
            List of open positions
        """
        if not self.api_key:
            raise AuthenticationError("API token required for fetching orders")
        
        try:
            request = {
                "proposal_open_contract": 1,
                "contract_id": 0,  # 0 means all contracts
                "subscribe": 0,
                "req_id": self._get_next_req_id()
            }
            
            response = await self._send_request(request)
            
            if 'error' in response:
                raise ExchangeError(f"Error fetching open orders: {response['error']['message']}")
            
            # Return raw response as Deriv's format is quite different
            return response.get('proposal_open_contract', [])
            
        except Exception as e:
            await self.handle_error(e, context={
                "action": "fetch_open_orders",
                "symbol": symbol
            })
            raise
    
    async def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch order details from Deriv.
        
        Args:
            order_id: Contract ID
            symbol: Optional trading symbol (not used for Deriv)
            
        Returns:
            Order details
        """
        if not self.api_key:
            raise AuthenticationError("API token required for fetching order")
        
        try:
            request = {
                "proposal_open_contract": 1,
                "contract_id": order_id,
                "subscribe": 0,
                "req_id": self._get_next_req_id()
            }
            
            response = await self._send_request(request)
            
            if 'error' in response:
                if 'ContractNotFound' in response['error']['code']:
                    raise OrderNotFoundError(f"Contract not found: {order_id}")
                raise ExchangeError(f"Error fetching order: {response['error']['message']}")
            
            return response.get('proposal_open_contract', {})
            
        except Exception as e:
            await self.handle_error(e, context={
                "action": "fetch_order",
                "order_id": order_id,
                "symbol": symbol
            })
            raise
