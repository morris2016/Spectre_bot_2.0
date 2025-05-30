"""
Binance Connector for QuantumSpectre Elite Trading System.

This module provides a specialized connector for Binance exchange.
"""

import asyncio
import json
import time
import hmac
import hashlib
import aiohttp
import websockets
import logging
from typing import Dict, List, Any, Optional, Union, Callable
import random
import traceback
from datetime import datetime, timezone

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

class BinanceConnector(BaseConnector):
    """
    Binance exchange connector implementing the BaseConnector interface.
    
    Supports both REST API and WebSocket streams for market data and trading.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = False,
                 rate_limit_config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize Binance connector.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet
            rate_limit_config: Configuration for rate limiting
            event_bus: Event bus for publishing events
        """
        super().__init__(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            rate_limit_config=rate_limit_config,
            event_bus=event_bus
        )
        
        # Set up URLs based on testnet flag
        self.base_url = settings.BINANCE_TESTNET_API_URL if testnet else settings.BINANCE_API_URL
        self.ws_base_url = settings.BINANCE_TESTNET_WS_URL if testnet else settings.BINANCE_WS_URL
        
        # For caching market data
        self.orderbook_cache = {}
        self.ticker_cache = {}
        self.last_update_id = {}
        
        # WebSocket subscriptions
        self.active_subscriptions = {}
        self.callbacks = {}
        
        # Connection management
        self.ws_connect_lock = asyncio.Lock()
        self.reconnect_delay = settings.BASE_RECONNECT_DELAY
        self.max_reconnect_delay = settings.MAX_RECONNECT_DELAY
        
        # Specialized rate limits for Binance
        self.order_rate_limiter = self.rate_limiter.create_child("orders", 10, 1)  # 10 orders per second
        self.market_data_rate_limiter = self.rate_limiter.create_child("market_data", 1200, 60)  # 1200 requests per minute
        
        # Order ID prefix
        self.client_order_id_prefix = "qsex_"
    
    async def connect(self) -> bool:
        """
        Establish connection to Binance.
        
        Returns:
            Connection success status
        """
        try:
            await self._execute_hooks(self.pre_connect_hooks)
            
            self.logger.info("Connecting to Binance...")
            
            # Initialize REST session
            if self.rest_session is None or self.rest_session.closed:
                self.rest_session = aiohttp.ClientSession(
                    headers={"X-MBX-APIKEY": self.api_key} if self.api_key else {}
                )
            
            # Test connection with a simple request
            await self.fetch_exchange_info()
            
            # Initialize WebSocket connections (these will be created on demand)
            self.ws_sessions = {}
            
            # Start background tasks
            self._start_background_tasks()
            
            self.is_connected = True
            self.logger.info("Successfully connected to Binance")
            
            # Update metrics
            self.metrics.record_connection()
            
            await self._execute_hooks(self.post_connect_hooks)
            return True
            
        except Exception as e:
            await self.handle_error(e, critical=True, context={"action": "connect"})
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from Binance.
        
        Returns:
            Disconnection success status
        """
        try:
            await self._execute_hooks(self.pre_disconnect_hooks)
            
            self.logger.info("Disconnecting from Binance...")
            
            # Close all WebSocket connections
            for stream_id, session_info in list(self.ws_sessions.items()):
                try:
                    ws = session_info.get('ws')
                    if ws:
                        await ws.close()
                    task = session_info.get('task')
                    if task and not task.done():
                        task.cancel()
                except Exception as e:
                    self.logger.warning(f"Error closing WebSocket for {stream_id}: {str(e)}")
            
            # Clear WebSocket sessions
            self.ws_sessions = {}
            
            # Close REST session
            if self.rest_session and not self.rest_session.closed:
                await self.rest_session.close()
                self.rest_session = None
            
            # Cancel background tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            self.tasks = set()
            
            self.is_connected = False
            self.logger.info("Successfully disconnected from Binance")
            
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
        
        # Order status monitor
        task = asyncio.create_task(self._order_status_monitor())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats and monitor connection health."""
        while self.is_connected:
            try:
                await self.heartbeat()
                
                # Check WebSocket connections
                for stream_id, session_info in list(self.ws_sessions.items()):
                    ws = session_info.get('ws')
                    last_msg_time = session_info.get('last_msg_time', 0)
                    
                    # If no message received for 30 seconds, reconnect
                    if time.time() - last_msg_time > 30:
                        self.logger.warning(f"No messages received for stream {stream_id} in 30s, reconnecting...")
                        task = session_info.get('task')
                        if task and not task.done():
                            task.cancel()
                        
                        # Re-establish subscription
                        callbacks = self.callbacks.get(stream_id, {})
                        symbol, data_type = stream_id.split(':')
                        asyncio.create_task(self.subscribe_market_data(symbol, [data_type], callbacks))
                
                await asyncio.sleep(15)  # Heartbeat every 15 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(5)  # Shorter delay on error
    
    async def _order_status_monitor(self):
        """Monitor open orders and update their status."""
        while self.is_connected:
            try:
                # Only run if we have API credentials
                if self.api_key and self.api_secret:
                    # Get all open orders
                    open_orders = await self.fetch_open_orders()
                    
                    # Publish order updates
                    for order in open_orders:
                        await self.event_bus.publish('order_update', {
                            'exchange': self.exchange_id,
                            'order': order,
                            'timestamp': time.time()
                        })
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in order status monitor: {str(e)}")
                await asyncio.sleep(30)  # Shorter delay on error
    
    async def fetch_exchange_info(self) -> Dict[str, Any]:
        """
        Fetch exchange information including trading rules.
        
        Returns:
            Exchange information
        """
        endpoint = "/api/v3/exchangeInfo"
        
        async with self.market_data_rate_limiter:
            response = await self._make_request("GET", endpoint)
            
        return response
    
    @async_retry_with_backoff(retries=3, delay=1, backoff=2)
    async def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None, 
                          data: Dict[str, Any] = None, signed: bool = False) -> Dict[str, Any]:
        """
        Make a request to Binance REST API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            signed: Whether the request needs signature
            
        Returns:
            Response data
        """
        if not self.rest_session:
            raise ConnectionError("REST session not initialized")
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        # Sign the request if needed
        if signed:
            params = await self.sign_request(endpoint, params, method)
        
        try:
            if method == "GET":
                async with self.rest_session.get(url, params=params) as response:
                    response_data = await response.json()
            elif method == "POST":
                if data:
                    async with self.rest_session.post(url, params=params, json=data) as response:
                        response_data = await response.json()
                else:
                    async with self.rest_session.post(url, params=params) as response:
                        response_data = await response.json()
            elif method == "DELETE":
                async with self.rest_session.delete(url, params=params) as response:
                    response_data = await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Handle error responses
            if response.status >= 400:
                error_msg = response_data.get('msg', str(response_data))
                error_code = response_data.get('code', 0)
                
                if response.status == 401:
                    raise AuthenticationError(f"Authentication error: {error_msg}")
                elif response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    raise RateLimitError(f"Rate limit exceeded: {error_msg}. Retry after {retry_after}s")
                else:
                    # Map Binance error codes to specific exceptions
                    if error_code == -2010:  # Insufficient balance
                        raise InsufficientBalanceError(f"Insufficient balance: {error_msg}")
                    elif error_code == -2013:  # Order does not exist
                        raise OrderNotFoundError(f"Order not found: {error_msg}")
                    else:
                        raise ExchangeError(f"Binance error {error_code}: {error_msg}")
            
            # Update metrics
            self.metrics.record_successful_request(endpoint)
            
            return response_data
            
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.metrics.record_failed_request(endpoint)
            raise ConnectionError(f"Connection error: {str(e)}")
        except json.JSONDecodeError:
            self.metrics.record_failed_request(endpoint)
            raise ExchangeError("Invalid JSON response")
    
    async def fetch_market_data(self, symbol: str, data_type: str, **kwargs) -> Any:
        """
        Fetch specific market data from Binance.
        
        Args:
            symbol: Trading symbol
            data_type: Type of data ('ticker', 'orderbook', 'trades', 'candles')
            **kwargs: Additional parameters
            
        Returns:
            Market data
        """
        try:
            # Normalize symbol (Binance uses uppercase)
            symbol = symbol.upper()
            
            if data_type == 'ticker':
                return await self._fetch_ticker(symbol)
            elif data_type == 'orderbook':
                limit = kwargs.get('limit', 100)
                return await self._fetch_orderbook(symbol, limit)
            elif data_type == 'trades':
                limit = kwargs.get('limit', 100)
                return await self._fetch_trades(symbol, limit)
            elif data_type == 'candles':
                interval = kwargs.get('interval', '1m')
                limit = kwargs.get('limit', 100)
                start_time = kwargs.get('start_time')
                end_time = kwargs.get('end_time')
                return await self._fetch_candles(symbol, interval, limit, start_time, end_time)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
                
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
        endpoint = "/api/v3/ticker/24hr"
        params = {"symbol": symbol}
        
        async with self.market_data_rate_limiter:
            response = await self._make_request("GET", endpoint, params)
            
        # Convert to standard model
        ticker = Ticker(
            symbol=symbol,
            exchange="binance",
            last_price=float(response['lastPrice']),
            bid_price=float(response['bidPrice']),
            ask_price=float(response['askPrice']),
            volume=float(response['volume']),
            quote_volume=float(response['quoteVolume']),
            timestamp=int(time.time() * 1000),
            high_24h=float(response['highPrice']),
            low_24h=float(response['lowPrice']),
            price_change_24h=float(response['priceChange']),
            price_change_percent_24h=float(response['priceChangePercent']),
            raw=response
        )
        
        # Cache ticker data
        self.ticker_cache[symbol] = ticker
        
        return ticker
    
    async def _fetch_orderbook(self, symbol: str, limit: int = 100) -> OrderBook:
        """Fetch orderbook data for a symbol."""
        endpoint = "/api/v3/depth"
        params = {"symbol": symbol, "limit": limit}
        
        async with self.market_data_rate_limiter:
            response = await self._make_request("GET", endpoint, params)
            
        # Convert to standard model
        orderbook = OrderBook(
            symbol=symbol,
            exchange="binance",
            bids=[[float(price), float(amount)] for price, amount in response['bids']],
            asks=[[float(price), float(amount)] for price, amount in response['asks']],
            timestamp=int(time.time() * 1000),
            last_update_id=response['lastUpdateId'],
            raw=response
        )
        
        # Cache orderbook data
        self.orderbook_cache[symbol] = orderbook
        self.last_update_id[symbol] = response['lastUpdateId']
        
        return orderbook
    
    async def _fetch_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Fetch recent trades for a symbol."""
        endpoint = "/api/v3/trades"
        params = {"symbol": symbol, "limit": limit}
        
        async with self.market_data_rate_limiter:
            response = await self._make_request("GET", endpoint, params)
            
        # Convert to standard model
        trades = [
            Trade(
                symbol=symbol,
                exchange="binance",
                id=str(trade['id']),
                price=float(trade['price']),
                amount=float(trade['qty']),
                cost=float(trade['price']) * float(trade['qty']),
                timestamp=trade['time'],
                side="sell" if trade['isBuyerMaker'] else "buy",
                raw=trade
            )
            for trade in response
        ]
        
        return trades
    
    async def _fetch_candles(self, symbol: str, interval: str = '1m', limit: int = 100,
                           start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Candle]:
        """Fetch candlestick data for a symbol."""
        endpoint = "/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        async with self.market_data_rate_limiter:
            response = await self._make_request("GET", endpoint, params)
            
        # Convert to standard model
        candles = [
            Candle(
                symbol=symbol,
                exchange="binance",
                timestamp=candle[0],
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=float(candle[5]),
                close_time=candle[6],
                quote_volume=float(candle[7]),
                trades=int(candle[8]),
                taker_buy_base_volume=float(candle[9]),
                taker_buy_quote_volume=float(candle[10]),
                interval=interval,
                raw=candle
            )
            for candle in response
        ]
        
        return candles
    
    async def subscribe_market_data(self, symbol: str, data_types: List[str], callbacks: Dict[str, Callable]) -> bool:
        """
        Subscribe to real-time market data via WebSocket.
        
        Args:
            symbol: Trading symbol
            data_types: Types of data to subscribe to
            callbacks: Callback functions for each data type
            
        Returns:
            Subscription success status
        """
        try:
            # Normalize symbol (Binance uses lowercase for websocket streams)
            symbol_lower = symbol.lower()
            
            for data_type in data_types:
                stream_id = f"{symbol}:{data_type}"
                
                # Save callbacks
                self.callbacks[stream_id] = callbacks
                
                # Check if already subscribed
                if stream_id in self.active_subscriptions:
                    self.logger.debug(f"Already subscribed to {stream_id}")
                    continue
                
                # Determine stream name
                if data_type == 'ticker':
                    stream_name = f"{symbol_lower}@ticker"
                elif data_type == 'orderbook':
                    stream_name = f"{symbol_lower}@depth@100ms"
                elif data_type == 'trades':
                    stream_name = f"{symbol_lower}@trade"
                elif data_type == 'candles':
                    interval = callbacks.get('interval', '1m')
                    stream_name = f"{symbol_lower}@kline_{interval}"
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
                
                # Start WebSocket connection
                ws_url = f"{self.ws_base_url}/ws/{stream_name}"
                
                task = asyncio.create_task(self._handle_websocket(stream_id, ws_url, data_type))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
                
                # Mark as subscribed
                self.active_subscriptions[stream_id] = stream_name
                
                self.logger.info(f"Subscribed to {stream_id} ({stream_name})")
            
            return True
            
        except Exception as e:
            await self.handle_error(e, context={
                "action": "subscribe_market_data",
                "symbol": symbol,
                "data_types": data_types
            })
            return False
    
    async def _handle_websocket(self, stream_id: str, ws_url: str, data_type: str):
        """
        Handle WebSocket connection and messages for a specific stream.
        
        Args:
            stream_id: Stream identifier (symbol:data_type)
            ws_url: WebSocket URL
            data_type: Type of data
        """
        retry_count = 0
        max_retries = 10
        
        while self.is_connected and retry_count < max_retries:
            try:
                async with self.ws_connect_lock:
                    self.logger.debug(f"Connecting to WebSocket {ws_url}")
                    async with websockets.connect(ws_url) as ws:
                        # Store WebSocket connection
                        self.ws_sessions[stream_id] = {
                            'ws': ws,
                            'task': asyncio.current_task(),
                            'last_msg_time': time.time()
                        }
                        
                        retry_count = 0  # Reset retry counter on successful connection
                        
                        # Process messages
                        async for message in ws:
                            # Update last message time
                            self.ws_sessions[stream_id]['last_msg_time'] = time.time()
                            
                            # Parse and process message
                            try:
                                data = json.loads(message)
                                await self._process_websocket_message(stream_id, data_type, data)
                            except json.JSONDecodeError:
                                self.logger.error(f"Invalid JSON in WebSocket message: {message[:100]}...")
                            except Exception as e:
                                self.logger.error(f"Error processing WebSocket message: {str(e)}")
                                self.logger.debug(traceback.format_exc())
            
            except (websockets.exceptions.ConnectionClosed, ConnectionError) as e:
                retry_count += 1
                backoff = min(
                    self.max_reconnect_delay,
                    self.reconnect_delay * (2 ** retry_count)
                )
                self.logger.warning(f"WebSocket connection closed for {stream_id}: {str(e)}. Reconnecting in {backoff:.2f}s")
                await asyncio.sleep(backoff)
                
            except asyncio.CancelledError:
                self.logger.info(f"WebSocket handler for {stream_id} cancelled")
                break
                
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Error in WebSocket handler for {stream_id}: {str(e)}")
                self.logger.debug(traceback.format_exc())
                await asyncio.sleep(5)
        
        if retry_count >= max_retries:
            self.logger.error(f"Max retries reached for WebSocket {stream_id}, giving up")
            # Remove from active subscriptions
            if stream_id in self.active_subscriptions:
                del self.active_subscriptions[stream_id]
    
    async def _process_websocket_message(self, stream_id: str, data_type: str, data: Dict[str, Any]):
        """
        Process a WebSocket message.
        
        Args:
            stream_id: Stream identifier
            data_type: Type of data
            data: Message data
        """
        symbol, _ = stream_id.split(':')
        callbacks = self.callbacks.get(stream_id, {})
        callback = callbacks.get(data_type)
        
        # Process based on data type
        if data_type == 'ticker':
            # Convert to standard model
            ticker = Ticker(
                symbol=symbol,
                exchange="binance",
                last_price=float(data['c']),
                bid_price=float(data['b']),
                ask_price=float(data['a']),
                volume=float(data['v']),
                quote_volume=float(data['q']),
                timestamp=data['E'],
                high_24h=float(data['h']),
                low_24h=float(data['l']),
                price_change_24h=float(data['p']),
                price_change_percent_24h=float(data['P']),
                raw=data
            )
            
            # Cache ticker data
            self.ticker_cache[symbol] = ticker
            
            # Call callback if provided
            if callback:
                await callback(ticker)
            
            # Publish event
            await self.event_bus.publish('market_data_ticker', {
                'exchange': self.exchange_id,
                'symbol': symbol,
                'ticker': ticker.to_dict(),
                'timestamp': time.time()
            })
            
        elif data_type == 'orderbook':
            # Check if we have the initial snapshot
            if symbol not in self.orderbook_cache:
                # Fetch initial snapshot
                await self._fetch_orderbook(symbol)
            
            # Process incremental update
            if 'u' in data and 'b' in data and 'a' in data:
                last_update_id = data['u']
                first_update_id = data['U']
                
                # Check if update is valid
                cached_update_id = self.last_update_id.get(symbol, 0)
                if first_update_id <= cached_update_id + 1 and last_update_id >= cached_update_id + 1:
                    # Valid update, apply it to cached orderbook
                    orderbook = self.orderbook_cache.get(symbol)
                    if orderbook:
                        # Update bids
                        for bid in data['b']:
                            price, amount = float(bid[0]), float(bid[1])
                            if amount == 0:
                                # Remove price level
                                orderbook.bids = [b for b in orderbook.bids if b[0] != price]
                            else:
                                # Update price level
                                exists = False
                                for i, b in enumerate(orderbook.bids):
                                    if b[0] == price:
                                        orderbook.bids[i] = [price, amount]
                                        exists = True
                                        break
                                if not exists:
                                    orderbook.bids.append([price, amount])
                                    # Sort bids in descending order
                                    orderbook.bids.sort(key=lambda x: x[0], reverse=True)
                        
                        # Update asks
                        for ask in data['a']:
                            price, amount = float(ask[0]), float(ask[1])
                            if amount == 0:
                                # Remove price level
                                orderbook.asks = [a for a in orderbook.asks if a[0] != price]
                            else:
                                # Update price level
                                exists = False
                                for i, a in enumerate(orderbook.asks):
                                    if a[0] == price:
                                        orderbook.asks[i] = [price, amount]
                                        exists = True
                                        break
                                if not exists:
                                    orderbook.asks.append([price, amount])
                                    # Sort asks in ascending order
                                    orderbook.asks.sort(key=lambda x: x[0])
                        
                        # Update timestamp and last update ID
                        orderbook.timestamp = data['E']
                        orderbook.last_update_id = last_update_id
                        self.last_update_id[symbol] = last_update_id
                        
                        # Call callback if provided
                        if callback:
                            await callback(orderbook)
                        
                        # Publish event
                        await self.event_bus.publish('market_data_orderbook', {
                            'exchange': self.exchange_id,
                            'symbol': symbol,
                            'orderbook': orderbook.to_dict(),
                            'timestamp': time.time()
                        })
                else:
                    self.logger.warning(f"Out of sequence orderbook update for {symbol}, re-fetching snapshot")
                    await self._fetch_orderbook(symbol)
            
        elif data_type == 'trades':
            # Convert to standard model
            trade = Trade(
                symbol=symbol,
                exchange="binance",
                id=str(data['t']),
                price=float(data['p']),
                amount=float(data['q']),
                cost=float(data['p']) * float(data['q']),
                timestamp=data['T'],
                side="sell" if data['m'] else "buy",
                raw=data
            )
            
            # Call callback if provided
            if callback:
                await callback(trade)
            
            # Publish event
            await self.event_bus.publish('market_data_trade', {
                'exchange': self.exchange_id,
                'symbol': symbol,
                'trade': trade.to_dict(),
                'timestamp': time.time()
            })
            
        elif data_type == 'candles':
            k = data['k']
            
            # Convert to standard model
            candle = Candle(
                symbol=symbol,
                exchange="binance",
                timestamp=k['t'],
                open=float(k['o']),
                high=float(k['h']),
                low=float(k['l']),
                close=float(k['c']),
                volume=float(k['v']),
                close_time=k['T'],
                quote_volume=float(k['q']),
                trades=int(k['n']),
                taker_buy_base_volume=float(k['V']),
                taker_buy_quote_volume=float(k['Q']),
                interval=k['i'],
                is_closed=k['x'],
                raw=data
            )
            
            # Call callback if provided
            if callback:
                await callback(candle)
            
            # Publish event
            await self.event_bus.publish('market_data_candle', {
                'exchange': self.exchange_id,
                'symbol': symbol,
                'candle': candle.to_dict(),
                'interval': k['i'],
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
            for data_type in data_types:
                stream_id = f"{symbol}:{data_type}"
                
                # Check if subscribed
                if stream_id not in self.active_subscriptions:
                    self.logger.debug(f"Not subscribed to {stream_id}")
                    continue
                
                # Cancel WebSocket task
                session_info = self.ws_sessions.get(stream_id)
                if session_info:
                    task = session_info.get('task')
                    if task and not task.done():
                        task.cancel()
                    
                    ws = session_info.get('ws')
                    if ws:
                        await ws.close()
                    
                    del self.ws_sessions[stream_id]
                
                # Remove from active subscriptions and callbacks
                del self.active_subscriptions[stream_id]
                if stream_id in self.callbacks:
                    del self.callbacks[stream_id]
                
                self.logger.info(f"Unsubscribed from {stream_id}")
            
            return True
            
        except Exception as e:
            await self.handle_error(e, context={
                "action": "unsubscribe_market_data",
                "symbol": symbol,
                "data_types": data_types
            })
            return False
    
    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, 
                         price: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Create a new order on Binance.
        
        Args:
            symbol: Trading symbol
            order_type: Order type (MARKET, LIMIT, etc.)
            side: Order side (BUY or SELL)
            amount: Order quantity
            price: Order price (for LIMIT orders)
            **kwargs: Additional parameters
            
        Returns:
            Order details
        """
        if not self.api_key or not self.api_secret:
            raise AuthenticationError("API key and secret required for trading")
        
        try:
            # Generate client order ID
            client_order_id = kwargs.get('client_order_id')
            if not client_order_id:
                local_id = await self.generate_local_order_id()
                client_order_id = f"{self.client_order_id_prefix}{int(time.time() * 1000)}"
                
                # Map local ID to client order ID
                await self.map_order_id(local_id, client_order_id)
            
            # Prepare parameters
            endpoint = "/api/v3/order"
            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": amount,
                "newClientOrderId": client_order_id
            }
            
            # Add price for LIMIT orders
            if order_type.upper() == "LIMIT":
                if not price:
                    raise ValueError("Price is required for LIMIT orders")
                params["price"] = price
                params["timeInForce"] = kwargs.get("time_in_force", "GTC")
            
            # Add optional parameters
            if "time_in_force" in kwargs:
                params["timeInForce"] = kwargs["time_in_force"]
            if "iceberg_qty" in kwargs:
                params["icebergQty"] = kwargs["iceberg_qty"]
            if "stop_price" in kwargs:
                params["stopPrice"] = kwargs["stop_price"]
            
            # Execute order with rate limiting
            async with self.order_rate_limiter:
                response = await self._make_request("POST", endpoint, params=params, signed=True)
            
            # Publish order event
            await self.event_bus.publish('order_created', {
                'exchange': self.exchange_id,
                'order': response,
                'local_id': local_id if 'local_id' in locals() else None,
                'timestamp': time.time()
            })
            
            return response
            
        except Exception as e:
            await self.handle_error(e, context={
                "action": "create_order",
                "symbol": symbol,
                "order_type": order_type,
                "side": side,
                "amount": amount,
                "price": price
            })
            raise
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an existing order on Binance.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID or client order ID
            
        Returns:
            Cancellation success status
        """
        if not self.api_key or not self.api_secret:
            raise AuthenticationError("API key and secret required for trading")
        
        try:
            # Prepare parameters
            endpoint = "/api/v3/order"
            params = {"symbol": symbol.upper()}
            
            # Check if order_id is a client order ID or exchange order ID
            if order_id.startswith(self.client_order_id_prefix):
                params["origClientOrderId"] = order_id
            else:
                params["orderId"] = order_id
            
            # Execute cancellation with rate limiting
            async with self.order_rate_limiter:
                response = await self._make_request("DELETE", endpoint, params=params, signed=True)
            
            # Publish order event
            await self.event_bus.publish('order_cancelled', {
                'exchange': self.exchange_id,
                'order': response,
                'timestamp': time.time()
            })
            
            return True
            
        except OrderNotFoundError:
            self.logger.warning(f"Order {order_id} not found for cancellation")
            return False
        except Exception as e:
            await self.handle_error(e, context={
                "action": "cancel_order",
                "symbol": symbol,
                "order_id": order_id
            })
            return False
    
    async def fetch_balance(self) -> Dict[str, Dict[str, float]]:
        """
        Fetch account balance from Binance.
        
        Returns:
            Account balance information
        """
        if not self.api_key or not self.api_secret:
            raise AuthenticationError("API key and secret required for fetching balance")
        
        try:
            endpoint = "/api/v3/account"
            
            response = await self._make_request("GET", endpoint, params={}, signed=True)
            
            # Process balances
            balances = {}
            for asset_data in response['balances']:
                asset = asset_data['asset']
                free = float(asset_data['free'])
                locked = float(asset_data['locked'])
                
                if free > 0 or locked > 0:
                    balances[asset] = {
                        'free': free,
                        'used': locked,
                        'total': free + locked
                    }
            
            return balances
            
        except Exception as e:
            await self.handle_error(e, context={"action": "fetch_balance"})
            raise
    
    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch open orders from Binance.
        
        Args:
            symbol: Optional trading symbol
            
        Returns:
            List of open orders
        """
        if not self.api_key or not self.api_secret:
            raise AuthenticationError("API key and secret required for fetching orders")
        
        try:
            endpoint = "/api/v3/openOrders"
            params = {}
            
            if symbol:
                params["symbol"] = symbol.upper()
            
            response = await self._make_request("GET", endpoint, params=params, signed=True)
            
            return response
            
        except Exception as e:
            await self.handle_error(e, context={
                "action": "fetch_open_orders",
                "symbol": symbol
            })
            raise
    
    async def fetch_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch order details from Binance.
        
        Args:
            order_id: Order ID or client order ID
            symbol: Trading symbol (required by Binance)
            
        Returns:
            Order details
        """
        if not self.api_key or not self.api_secret:
            raise AuthenticationError("API key and secret required for fetching order")
        
        if not symbol:
            raise ValueError("Symbol is required for fetching order on Binance")
        
        try:
            endpoint = "/api/v3/order"
            params = {"symbol": symbol.upper()}
            
            # Check if order_id is a client order ID or exchange order ID
            if order_id.startswith(self.client_order_id_prefix):
                params["origClientOrderId"] = order_id
            else:
                params["orderId"] = order_id
            
            response = await self._make_request("GET", endpoint, params=params, signed=True)
            
            return response
            
        except Exception as e:
            await self.handle_error(e, context={
                "action": "fetch_order",
                "order_id": order_id,
                "symbol": symbol
            })
            raise
