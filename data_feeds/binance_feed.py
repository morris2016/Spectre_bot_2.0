#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Binance Feed Implementation

This module implements the Binance exchange data feed, providing
real-time market data from Binance via REST and WebSocket APIs.
"""

import time
import json
import hmac
import hashlib
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from data_feeds.base_feed import BaseFeed
from common.exceptions import (
    FeedError, FeedConnectionError, FeedDisconnectedError,
    FeedRateLimitError, SecurityError
)
from common.async_utils import run_with_timeout


class BinanceFeed(BaseFeed):
    """Feed implementation for Binance exchange."""
    
    def __init__(self, config, loop=None, redis_client=None):
        """Initialize the Binance feed."""
        super().__init__(config, loop, redis_client)
        
        # Binance specific configuration
        self.api_key = self.config.get("api_key", "")
        self.api_secret = self.config.get("api_secret", "")
        self.testnet = self.config.get("testnet", True)
        
        # API endpoints
        self.base_url = "https://testnet.binance.vision/api" if self.testnet else "https://api.binance.com/api"
        self.base_ws_url = "wss://testnet.binance.vision/ws" if self.testnet else "wss://stream.binance.com:9443/ws"
        
        # HTTP session and WebSocket connections
        self.session = None
        self.ws_connections = {}
        
        # Data structures for market data
        self.order_books = {}
        self.last_update_ids = {}
        self.klines = {}
        self.trades = {}
        
        # Rate limiting
        self.request_weights = {}
        self.last_request_time = 0
        self.rate_limit_buffer = self.config.get("rate_limit_buffer", 0.8)  # Use 80% of rate limit by default
    
    async def start(self):
        """Start the Binance feed."""
        self.logger.info(f"Starting Binance feed (testnet: {self.testnet})")
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        try:
            # Test connection
            await self._test_connection()
            
            # Start WebSocket connections for configured assets
            assets = self.config.get("assets", [])
            for asset in assets:
                symbol = asset.replace("/", "").lower()  # Convert BTC/USDT to btcusdt
                self.logger.info(f"Setting up data streams for {asset} ({symbol})")
                
                # Subscribe to ticker stream
                if self.config.get("streams", {}).get("ticker", True):
                    await self._subscribe_ticker(symbol)
                
                # Subscribe to kline (candlestick) streams
                if self.config.get("streams", {}).get("klines", True):
                    timeframes = self.config.get("timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])
                    for timeframe in timeframes:
                        await self._subscribe_klines(symbol, timeframe)
                
                # Subscribe to trade stream
                if self.config.get("streams", {}).get("trades", True):
                    await self._subscribe_trades(symbol)
                
                # Subscribe to order book stream
                if self.config.get("streams", {}).get("depth", True):
                    await self._subscribe_depth(symbol)
            
            # Mark feed as running
            self.running = True
            self.update_connection_state(True)
            self.logger.info("Binance feed started successfully")
            
        except Exception as e:
            # Clean up on failure
            if self.session:
                await self.session.close()
                self.session = None
            
            self.update_connection_state(False)
            self.logger.error(f"Failed to start Binance feed: {str(e)}")
            raise FeedConnectionError(f"Failed to start Binance feed: {str(e)}")
    
    async def stop(self):
        """Stop the Binance feed."""
        self.logger.info("Stopping Binance feed")
        self.shutting_down = True
        
        # Close all WebSocket connections
        for channel, ws_task in list(self.ws_connections.items()):
            if not ws_task.done():
                self.logger.info(f"Closing WebSocket connection: {channel}")
                ws_task.cancel()
                try:
                    await ws_task
                except asyncio.CancelledError:
                    pass
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
        
        self.running = False
        self.update_connection_state(False)
        self.logger.info("Binance feed stopped successfully")
    
    async def _test_connection(self):
        """Test the connection to Binance API."""
        try:
            # Call the ping endpoint
            url = f"{self.base_url}/v3/ping"
            async with self.session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise FeedConnectionError(f"Failed to connect to Binance API: {error_text}")
                
                # Get exchange info
                await self._get_exchange_info()
                
        except aiohttp.ClientConnectionError as e:
            raise FeedConnectionError(f"Failed to connect to Binance API: {str(e)}")
        except Exception as e:
            raise FeedConnectionError(f"Error testing Binance connection: {str(e)}")
    
    async def _get_exchange_info(self):
        """Get exchange information from Binance."""
        url = f"{self.base_url}/v3/exchangeInfo"
        async with self.session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise FeedError(f"Failed to get exchange info: {error_text}")
            
            info = await response.json()
            
            # Log exchange status
            if info.get("serverTime"):
                server_time = datetime.fromtimestamp(info["serverTime"] / 1000)
                local_time = datetime.now()
                time_diff = abs((local_time - server_time).total_seconds())
                
                self.logger.info(f"Binance server time: {server_time}, Time difference: {time_diff:.2f} seconds")
                if time_diff > 10:
                    self.logger.warning(f"Large time difference between local and server time: {time_diff:.2f} seconds")
            
            # Store rate limits
            rate_limits = info.get("rateLimits", [])
            for limit in rate_limits:
                if limit["rateLimitType"] == "REQUEST_WEIGHT":
                    self.logger.info(f"Request weight limit: {limit['limit']} per {limit['intervalNum']} {limit['interval']}")
            
            return info
    
    def _generate_signature(self, query_string):
        """Generate HMAC-SHA256 signature for Binance API authentication."""
        if not self.api_secret:
            raise SecurityError("API secret is required for authenticated endpoints")
            
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    async def _authenticated_request(self, method, endpoint, params=None):
        """Make an authenticated request to the Binance API."""
        if not self.api_key:
            raise SecurityError("API key is required for authenticated endpoints")
            
        url = f"{self.base_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self.api_key}
        
        # Add timestamp and signature for authentication
        if params is None:
            params = {}
        params['timestamp'] = int(time.time() * 1000)
        
        # Create query string
        query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        
        # Generate signature
        signature = self._generate_signature(query_string)
        params['signature'] = signature
        
        # Make the request
        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                if response.status == 429:
                    self.logger.warning("Rate limit exceeded for Binance API")
                    raise FeedRateLimitError("Rate limit exceeded for Binance API")
                    
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Binance API error: {error_text}")
                    raise FeedError(f"Binance API error: {error_text}")
                    
                return await response.json()
                
        except aiohttp.ClientConnectionError as e:
            raise FeedConnectionError(f"Connection error to Binance API: {str(e)}")
        except Exception as e:
            raise FeedError(f"Error in Binance API request: {str(e)}")
    
    async def _subscribe_ticker(self, symbol):
        """Subscribe to ticker updates for a symbol."""
        self.logger.info(f"Subscribing to ticker updates for {symbol}")
        
        # Create WebSocket URL
        ws_url = f"{self.base_ws_url}/{symbol}@ticker"
        
        # Create and store the WebSocket task
        ws_task = asyncio.create_task(self._handle_ticker_websocket(ws_url, symbol))
        self.ws_connections[f"{symbol}_ticker"] = ws_task
    
    async def _handle_ticker_websocket(self, url, symbol):
        """Handle the ticker WebSocket connection."""
        reconnect_delay = self.config.get("reconnect_delay", 5)
        
        while not self.shutting_down:
            try:
                self.logger.info(f"Connecting to ticker WebSocket for {symbol}: {url}")
                
                async with self.session.ws_connect(url) as ws:
                    self.logger.info(f"Connected to ticker WebSocket for {symbol}")
                    
                    # Process messages
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._process_ticker_message(symbol, msg.data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            self.logger.error(f"WebSocket error for {symbol} ticker: {msg.data}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            self.logger.warning(f"WebSocket closed for {symbol} ticker")
                            break
                
            except asyncio.CancelledError:
                self.logger.info(f"Ticker WebSocket for {symbol} cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in ticker WebSocket for {symbol}: {str(e)}")
                
                if not self.shutting_down:
                    self.logger.info(f"Reconnecting ticker WebSocket for {symbol} in {reconnect_delay} seconds")
                    await asyncio.sleep(reconnect_delay)
                else:
                    break
    
    async def _process_ticker_message(self, symbol, message_data):
        """Process a ticker WebSocket message."""
        try:
            # Parse the message
            message = json.loads(message_data)
            
            # Create ticker data
            ticker_data = {
                "type": "ticker",
                "exchange": "binance",
                "symbol": symbol.upper(),  # Convert btcusdt to BTCUSDT
                "timestamp": int(message.get("E", time.time() * 1000)) / 1000,  # Convert from ms to seconds
                "ticker": {
                    "price": float(message.get("c", 0)),
                    "price_change": float(message.get("p", 0)),
                    "price_change_percent": float(message.get("P", 0)),
                    "volume": float(message.get("v", 0)),
                    "quote_volume": float(message.get("q", 0)),
                    "high": float(message.get("h", 0)),
                    "low": float(message.get("l", 0)),
                    "open": float(message.get("o", 0)),
                    "count": int(message.get("n", 0))
                },
                "raw": message
            }
            
            # Publish the data
            await self._publish_data(ticker_data, f"ticker.binance.{symbol}")
            
        except Exception as e:
            self.logger.error(f"Error processing ticker message for {symbol}: {str(e)}")
            self.metrics.increment("ticker.error")
    
    async def _subscribe_klines(self, symbol, timeframe):
        """Subscribe to kline (candlestick) updates for a symbol and timeframe."""
        self.logger.info(f"Subscribing to {timeframe} klines for {symbol}")
        
        # Create WebSocket URL
        ws_url = f"{self.base_ws_url}/{symbol}@kline_{timeframe}"
        
        # Create and store the WebSocket task
        ws_task = asyncio.create_task(self._handle_klines_websocket(ws_url, symbol, timeframe))
        self.ws_connections[f"{symbol}_kline_{timeframe}"] = ws_task
    
    async def _handle_klines_websocket(self, url, symbol, timeframe):
        """Handle the klines WebSocket connection."""
        reconnect_delay = self.config.get("reconnect_delay", 5)
        
        while not self.shutting_down:
            try:
                self.logger.info(f"Connecting to {timeframe} klines WebSocket for {symbol}: {url}")
                
                async with self.session.ws_connect(url) as ws:
                    self.logger.info(f"Connected to {timeframe} klines WebSocket for {symbol}")
                    
                    # Process messages
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._process_klines_message(symbol, timeframe, msg.data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            self.logger.error(f"WebSocket error for {symbol} {timeframe} klines: {msg.data}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            self.logger.warning(f"WebSocket closed for {symbol} {timeframe} klines")
                            break
                
            except asyncio.CancelledError:
                self.logger.info(f"Klines WebSocket for {symbol} {timeframe} cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in klines WebSocket for {symbol} {timeframe}: {str(e)}")
                
                if not self.shutting_down:
                    self.logger.info(f"Reconnecting klines WebSocket for {symbol} {timeframe} in {reconnect_delay} seconds")
                    await asyncio.sleep(reconnect_delay)
                else:
                    break
    
    async def _process_klines_message(self, symbol, timeframe, message_data):
        """Process a klines WebSocket message."""
        try:
            # Parse the message
            message = json.loads(message_data)
            
            # Extract the kline data
            kline = message.get("k", {})
            
            # Only process if the candle is closed or this is the first update
            is_closed = kline.get("x", False)
            
            # Create OHLCV data
            ohlcv_data = {
                "type": "ohlcv",
                "exchange": "binance",
                "symbol": symbol.upper(),
                "timestamp": int(kline.get("t", time.time() * 1000)) / 1000,
                "timeframe": timeframe,
                "ohlcv": {
                    "open": float(kline.get("o", 0)),
                    "high": float(kline.get("h", 0)),
                    "low": float(kline.get("l", 0)),
                    "close": float(kline.get("c", 0)),
                    "volume": float(kline.get("v", 0)),
                    "close_time": int(kline.get("T", 0)) / 1000,
                    "quote_volume": float(kline.get("q", 0)),
                    "count": int(kline.get("n", 0)),
                    "is_closed": is_closed
                },
                "raw": kline
            }
            
            # Publish the data
            await self._publish_data(ohlcv_data, f"ohlcv.binance.{symbol}.{timeframe}")
            
            # Store last kline for the symbol and timeframe
            if symbol not in self.klines:
                self.klines[symbol] = {}
            self.klines[symbol][timeframe] = ohlcv_data
            
        except Exception as e:
            self.logger.error(f"Error processing klines message for {symbol} {timeframe}: {str(e)}")
            self.metrics.increment("klines.error")
    
    async def _subscribe_trades(self, symbol):
        """Subscribe to trade updates for a symbol."""
        self.logger.info(f"Subscribing to trades for {symbol}")
        
        # Create WebSocket URL
        ws_url = f"{self.base_ws_url}/{symbol}@trade"
        
        # Create and store the WebSocket task
        ws_task = asyncio.create_task(self._handle_trades_websocket(ws_url, symbol))
        self.ws_connections[f"{symbol}_trades"] = ws_task
    
    async def _handle_trades_websocket(self, url, symbol):
        """Handle the trades WebSocket connection."""
        reconnect_delay = self.config.get("reconnect_delay", 5)
        
        while not self.shutting_down:
            try:
                self.logger.info(f"Connecting to trades WebSocket for {symbol}: {url}")
                
                async with self.session.ws_connect(url) as ws:
                    self.logger.info(f"Connected to trades WebSocket for {symbol}")
                    
                    # Process messages
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._process_trade_message(symbol, msg.data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            self.logger.error(f"WebSocket error for {symbol} trades: {msg.data}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            self.logger.warning(f"WebSocket closed for {symbol} trades")
                            break
                
            except asyncio.CancelledError:
                self.logger.info(f"Trades WebSocket for {symbol} cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in trades WebSocket for {symbol}: {str(e)}")
                
                if not self.shutting_down:
                    self.logger.info(f"Reconnecting trades WebSocket for {symbol} in {reconnect_delay} seconds")
                    await asyncio.sleep(reconnect_delay)
                else:
                    break
    
    async def _process_trade_message(self, symbol, message_data):
        """Process a trade WebSocket message."""
        try:
            # Parse the message
            message = json.loads(message_data)
            
            # Create trade data
            trade_data = {
                "type": "trade",
                "exchange": "binance",
                "symbol": symbol.upper(),
                "timestamp": int(message.get("E", time.time() * 1000)) / 1000,
                "trade": {
                    "id": str(message.get("t", "")),
                    "price": float(message.get("p", 0)),
                    "amount": float(message.get("q", 0)),
                    "side": "buy" if message.get("m", False) else "sell",  # m=true means buyer is market maker (sell)
                    "time": int(message.get("T", 0)) / 1000
                },
                "raw": message
            }
            
            # Publish the data
            await self._publish_data(trade_data, f"trade.binance.{symbol}")
            
            # Store last few trades for the symbol
            if symbol not in self.trades:
                self.trades[symbol] = []
            
            self.trades[symbol].append(trade_data)
            
            # Keep only the last 100 trades
            if len(self.trades[symbol]) > 100:
                self.trades[symbol] = self.trades[symbol][-100:]
            
        except Exception as e:
            self.logger.error(f"Error processing trade message for {symbol}: {str(e)}")
            self.metrics.increment("trade.error")
    
    async def _subscribe_depth(self, symbol):
        """Subscribe to order book updates for a symbol."""
        self.logger.info(f"Subscribing to order book for {symbol}")
        
        # First get a snapshot of the order book
        await self._get_order_book_snapshot(symbol)
        
        # Create WebSocket URL
        ws_url = f"{self.base_ws_url}/{symbol}@depth"
        
        # Create and store the WebSocket task
        ws_task = asyncio.create_task(self._handle_depth_websocket(ws_url, symbol))
        self.ws_connections[f"{symbol}_depth"] = ws_task
    
    async def _get_order_book_snapshot(self, symbol):
        """Get a snapshot of the order book for a symbol."""
        try:
            url = f"{self.base_url}/v3/depth"
            params = {"symbol": symbol.upper(), "limit": 1000}
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Failed to get order book snapshot for {symbol}: {error_text}")
                    return
                
                data = await response.json()
                
                # Store the last update ID
                last_update_id = data.get("lastUpdateId", 0)
                self.last_update_ids[symbol] = last_update_id
                
                # Create and store the order book
                self.order_books[symbol] = {
                    "bids": [[float(price), float(qty)] for price, qty in data.get("bids", [])],
                    "asks": [[float(price), float(qty)] for price, qty in data.get("asks", [])],
                    "last_update_id": last_update_id,
                    "timestamp": time.time()
                }
                
                self.logger.info(f"Got order book snapshot for {symbol}: {len(self.order_books[symbol]['bids'])} bids, {len(self.order_books[symbol]['asks'])} asks")
                
        except Exception as e:
            self.logger.error(f"Error getting order book snapshot for {symbol}: {str(e)}")
    
    async def _handle_depth_websocket(self, url, symbol):
        """Handle the depth WebSocket connection."""
        reconnect_delay = self.config.get("reconnect_delay", 5)
        
        while not self.shutting_down:
            try:
                self.logger.info(f"Connecting to depth WebSocket for {symbol}: {url}")
                
                async with self.session.ws_connect(url) as ws:
                    self.logger.info(f"Connected to depth WebSocket for {symbol}")
                    
                    # Process messages
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._process_depth_message(symbol, msg.data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            self.logger.error(f"WebSocket error for {symbol} depth: {msg.data}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            self.logger.warning(f"WebSocket closed for {symbol} depth")
                            break
                
            except asyncio.CancelledError:
                self.logger.info(f"Depth WebSocket for {symbol} cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in depth WebSocket for {symbol}: {str(e)}")
                
                if not self.shutting_down:
                    self.logger.info(f"Reconnecting depth WebSocket for {symbol} in {reconnect_delay} seconds")
                    await asyncio.sleep(reconnect_delay)
                else:
                    break
    
    async def _process_depth_message(self, symbol, message_data):
        """Process a depth WebSocket message."""
        try:
            # Parse the message
            message = json.loads(message_data)
            
            # Get update ID
            first_update_id = message.get("U", 0)
            final_update_id = message.get("u", 0)
            
            # Check if we have the order book for this symbol
            if symbol not in self.order_books:
                self.logger.warning(f"No order book snapshot for {symbol}, getting one now")
                await self._get_order_book_snapshot(symbol)
                return
            
            # Check if the message is in sequence with our last update
            last_update_id = self.last_update_ids.get(symbol, 0)
            
            if final_update_id <= last_update_id:
                # We already have this update, ignore it
                return
            
            if first_update_id > last_update_id + 1:
                # We've missed some updates, get a new snapshot
                self.logger.warning(f"Missed updates for {symbol} order book, getting new snapshot")
                await self._get_order_book_snapshot(symbol)
                return
            
            # Update the order book
            order_book = self.order_books[symbol]
            
            # Update bids
            for bid_update in message.get("b", []):
                price = float(bid_update[0])
                qty = float(bid_update[1])
                
                # Find the price level in the order book
                found = False
                for i, bid in enumerate(order_book["bids"]):
                    if bid[0] == price:
                        if qty == 0:
                            # Remove the price level
                            order_book["bids"].pop(i)
                        else:
                            # Update the quantity
                            order_book["bids"][i][1] = qty
                        found = True
                        break
                
                # Add new price level if not found and quantity is not zero
                if not found and qty > 0:
                    order_book["bids"].append([price, qty])
                    # Re-sort bids (highest first)
                    order_book["bids"].sort(key=lambda x: x[0], reverse=True)
            
            # Update asks
            for ask_update in message.get("a", []):
                price = float(ask_update[0])
                qty = float(ask_update[1])
                
                # Find the price level in the order book
                found = False
                for i, ask in enumerate(order_book["asks"]):
                    if ask[0] == price:
                        if qty == 0:
                            # Remove the price level
                            order_book["asks"].pop(i)
                        else:
                            # Update the quantity
                            order_book["asks"][i][1] = qty
                        found = True
                        break
                
                # Add new price level if not found and quantity is not zero
                if not found and qty > 0:
                    order_book["asks"].append([price, qty])
                    # Re-sort asks (lowest first)
                    order_book["asks"].sort(key=lambda x: x[0])
            
            # Update the last update ID
            self.last_update_ids[symbol] = final_update_id
            order_book["last_update_id"] = final_update_id
            order_book["timestamp"] = time.time()
            
            # Create order book data for publishing
            order_book_data = {
                "type": "order_book",
                "exchange": "binance",
                "symbol": symbol.upper(),
                "timestamp": time.time(),
                "bids": order_book["bids"][:10],  # Send top 10 levels
                "asks": order_book["asks"][:10],  # Send top 10 levels
                "last_update_id": final_update_id
            }
            
            # Publish the data
            await self._publish_data(order_book_data, f"order_book.binance.{symbol}")
            
        except Exception as e:
            self.logger.error(f"Error processing depth message for {symbol}: {str(e)}")
            self.metrics.increment("depth.error")
