#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Trading Service

This module provides trading-related functionality for the API Gateway,
including market data access, order management, position tracking, and
strategy execution.
"""

import json
import time
import asyncio
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import logging
from enum import Enum

from pydantic import BaseModel, Field, validator

from config import Config
from common.logger import get_logger
from common.metrics import MetricsCollector
from common.redis_client import RedisClient
from common.db_client import DatabaseClient
from common.exceptions import (
    NotFoundError, ValidationError, ServiceUnavailableError
)
from api_gateway.services.base_service import BaseService

# Initialize logger
logger = get_logger(__name__)

# Enums
class Platform(str, Enum):
    """Trading platforms."""
    BINANCE = "binance"
    DERIV = "deriv"

class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class TimeInForce(str, Enum):
    """Time in force options."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill

class OrderStatus(str, Enum):
    """Order statuses."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

# Models
class OrderRequest(BaseModel):
    """Model for order requests."""
    platform: Platform
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Optional[TimeInForce] = None
    client_order_id: Optional[str] = None
    reduce_only: bool = False
    close_position: bool = False
    
    @validator("price")
    def price_required_for_limit(cls, v, values):
        if values.get("type") in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError("Price is required for limit orders")
        return v
    
    @validator("stop_price")
    def stop_price_required_for_stops(cls, v, values):
        if values.get("type") in [OrderType.STOP, OrderType.STOP_LIMIT] and v is None:
            raise ValueError("Stop price is required for stop orders")
        return v

class TradingService(BaseService):
    """Trading service for market data and order management."""
    
    def __init__(self, config: Config, redis_client: RedisClient, db_client: DatabaseClient):
        """Initialize the trading service."""
        super().__init__("trading_service", redis_client, db_client)
        self.config = config
        self.start_time = time.time()
        self.platform_statuses = {}
        self.active_symbols = {}
    
    async def start(self):
        """Start the trading service."""
        await super().start()
        
        # Subscribe to trading-related messages
        await self.subscribe("market:*", self._handle_market_message)
        await self.subscribe("order:*", self._handle_order_message)
        await self.subscribe("strategy:*", self._handle_strategy_message)
        await self.subscribe("execution:*", self._handle_execution_message)
    
    async def get_market_data(self, platform: str, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Get market data for a symbol and timeframe.
        
        Args:
            platform: Trading platform
            symbol: Trading symbol
            timeframe: Chart timeframe
            
        Returns:
            Market data including OHLCV and indicators
            
        Raises:
            NotFoundError: If symbol or timeframe not found
            ServiceUnavailableError: If market data service fails
        """
        try:
            # Validate platform
            if platform not in [p.value for p in Platform]:
                raise ValidationError(f"Invalid platform: {platform}")
            
            # Check if symbol is available
            symbols = await self.get_available_symbols(platform)
            if symbol not in [s["symbol"] for s in symbols]:
                raise NotFoundError(f"Symbol not found: {symbol}")
            
            # Validate timeframe
            valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
            if timeframe not in valid_timeframes:
                raise ValidationError(f"Invalid timeframe: {timeframe}")
            
            # Get cached chart data
            cache_key = f"chart:{platform}:{symbol}:{timeframe}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            # If not in cache, request it
            request_id = f"req_market_{time.time_ns()}"
            request_key = f"request:{request_id}"
            response_key = f"response:{request_id}"
            
            # Publish request
            await self.publish("market:data:request", {
                "platform": platform,
                "symbol": symbol,
                "timeframe": timeframe,
                "request_id": request_id
            })
            
            # Wait for response with timeout
            start_time = time.time()
            max_wait = 10  # seconds
            
            while time.time() - start_time < max_wait:
                response = await self.redis_client.get(response_key)
                if response:
                    # Clean up
                    await self.redis_client.delete(response_key)
                    
                    # Parse and return
                    data = json.loads(response)
                    return data
                
                # Wait and try again
                await asyncio.sleep(0.1)
            
            # Timeout
            logger.error(f"Timeout waiting for market data: {platform}/{symbol}/{timeframe}")
            raise ServiceUnavailableError("Market data service timed out")
            
        except NotFoundError:
            raise
        except ValidationError as e:
            logger.warning(f"Market data validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Market data error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("Failed to get market data") from e
    
    async def get_available_symbols(self, platform: str) -> List[Dict[str, Any]]:
        """
        Get available trading symbols for a platform.
        
        Args:
            platform: Trading platform
            
        Returns:
            List of available symbols with metadata
            
        Raises:
            ValidationError: If platform is invalid
            ServiceUnavailableError: If symbol service fails
        """
        try:
            # Validate platform
            if platform not in [p.value for p in Platform]:
                raise ValidationError(f"Invalid platform: {platform}")
            
            # Check if we have cached data
            if platform in self.active_symbols:
                return self.active_symbols[platform]
            
            # Get from Redis
            cache_key = f"symbols:{platform}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                symbols = json.loads(cached_data)
                self.active_symbols[platform] = symbols
                return symbols
            
            # If not in cache, request it
            request_id = f"req_symbols_{time.time_ns()}"
            response_key = f"response:{request_id}"
            
            # Publish request
            await self.publish("market:symbols:request", {
                "platform": platform,
                "request_id": request_id
            })
            
            # Wait for response with timeout
            start_time = time.time()
            max_wait = 10  # seconds
            
            while time.time() - start_time < max_wait:
                response = await self.redis_client.get(response_key)
                if response:
                    # Clean up
                    await self.redis_client.delete(response_key)
                    
                    # Parse and store
                    symbols = json.loads(response)
                    self.active_symbols[platform] = symbols
                    return symbols
                
                # Wait and try again
                await asyncio.sleep(0.1)
            
            # Timeout - return empty list as fallback
            logger.error(f"Timeout waiting for symbols: {platform}")
            return []
            
        except ValidationError as e:
            logger.warning(f"Symbol validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Symbol service error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("Failed to get symbols") from e
    
    async def get_platform_status(self, platform: str) -> Dict[str, Any]:
        """
        Get platform status.
        
        Args:
            platform: Trading platform
            
        Returns:
            Platform status information
            
        Raises:
            ValidationError: If platform is invalid
            ServiceUnavailableError: If status service fails
        """
        try:
            # Validate platform
            if platform not in [p.value for p in Platform]:
                raise ValidationError(f"Invalid platform: {platform}")
            
            # Check if we have cached data
            if platform in self.platform_statuses:
                return self.platform_statuses[platform]
            
            # Get from Redis
            cache_key = f"status:{platform}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                status = json.loads(cached_data)
                self.platform_statuses[platform] = status
                return status
            
            # If not in cache, request it
            request_id = f"req_status_{time.time_ns()}"
            response_key = f"response:{request_id}"
            
            # Publish request
            await self.publish("market:status:request", {
                "platform": platform,
                "request_id": request_id
            })
            
            # Wait for response with timeout
            start_time = time.time()
            max_wait = 5  # seconds
            
            while time.time() - start_time < max_wait:
                response = await self.redis_client.get(response_key)
                if response:
                    # Clean up
                    await self.redis_client.delete(response_key)
                    
                    # Parse and store
                    status = json.loads(response)
                    self.platform_statuses[platform] = status
                    return status
                
                # Wait and try again
                await asyncio.sleep(0.1)
            
            # Timeout - return default status as fallback
            logger.error(f"Timeout waiting for status: {platform}")
            default_status = {
                "platform": platform,
                "status": "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            return default_status
            
        except ValidationError as e:
            logger.warning(f"Status validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Status service error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("Failed to get platform status") from e
    
    async def place_order(self, user_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place a trading order.
        
        Args:
            user_id: User ID
            order_data: Order parameters
            
        Returns:
            Order confirmation
            
        Raises:
            ValidationError: If order data is invalid
            ServiceUnavailableError: If order placement fails
        """
        try:
            # Validate order data
            order_request = OrderRequest(**order_data)
            
            # Check platform status
            platform_status = await self.get_platform_status(order_request.platform.value)
            if platform_status.get("status") != "online":
                raise ValidationError(f"Platform {order_request.platform.value} is not available")
            
            # Prepare order with user ID and timestamp
            order = order_request.dict()
            order["user_id"] = user_id
            order["created_at"] = datetime.now(timezone.utc).isoformat()
            order["status"] = OrderStatus.NEW.value
            
            # Generate unique client order ID if not provided
            if not order.get("client_order_id"):
                order["client_order_id"] = f"order_{user_id}_{time.time_ns()}"
            
            # Get unique order ID for tracking
            order_id = f"order_{time.time_ns()}"
            order["order_id"] = order_id
            
            # Store order in pending state
            await self.db_client.insert_one("orders", order)
            
            # Publish order placement request
            await self.publish(f"order:place:{order_request.platform.value}", order)
            
            # Track metrics
            self.metrics.increment("orders_placed_total", tags={
                "platform": order_request.platform.value,
                "symbol": order_request.symbol,
                "side": order_request.side.value,
                "type": order_request.type.value
            })
            
            logger.info(f"Order placed: id={order_id} user={user_id} " +
                       f"platform={order_request.platform.value} " +
                       f"symbol={order_request.symbol} " +
                       f"side={order_request.side.value} " +
                       f"type={order_request.type.value}")
            
            # Return order confirmation
            return {
                "order_id": order_id,
                "client_order_id": order["client_order_id"],
                "platform": order_request.platform.value,
                "symbol": order_request.symbol,
                "side": order_request.side.value,
                "type": order_request.type.value,
                "status": OrderStatus.NEW.value,
                "created_at": order["created_at"]
            }
            
        except ValidationError as e:
            logger.warning(f"Order validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Order placement error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("Failed to place order") from e
    
    async def cancel_order(self, user_id: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel a trading order.
        
        Args:
            user_id: User ID
            order_id: Order ID
            
        Returns:
            Cancellation confirmation
            
        Raises:
            NotFoundError: If order not found
            ValidationError: If cancellation not allowed
            ServiceUnavailableError: If cancellation fails
        """
        try:
            # Get order
            order = await self.db_client.find_one("orders", {"order_id": order_id})
            if not order:
                raise NotFoundError(f"Order not found: {order_id}")
            
            # Check ownership
            if order["user_id"] != user_id:
                raise ValidationError("Not authorized to cancel this order")
            
            # Check if order can be cancelled
            if order["status"] in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value]:
                raise ValidationError(f"Order cannot be cancelled: status is {order['status']}")
            
            # Publish cancellation request
            await self.publish(f"order:cancel:{order['platform']}", {
                "order_id": order_id,
                "client_order_id": order.get("client_order_id"),
                "platform": order["platform"],
                "symbol": order["symbol"],
                "user_id": user_id
            })
            
            # Track metrics
            self.metrics.increment("orders_cancelled_total", tags={
                "platform": order["platform"],
                "symbol": order["symbol"]
            })
            
            logger.info(f"Order cancellation requested: id={order_id} user={user_id}")
            
            # Return confirmation
            return {
                "order_id": order_id,
                "status": "cancellation_requested",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except NotFoundError:
            raise
        except ValidationError as e:
            logger.warning(f"Order cancellation validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Order cancellation error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("Failed to cancel order") from e
    
    async def get_order(self, user_id: str, order_id: str) -> Dict[str, Any]:
        """
        Get order details.
        
        Args:
            user_id: User ID
            order_id: Order ID
            
        Returns:
            Order details
            
        Raises:
            NotFoundError: If order not found
            ValidationError: If not authorized
        """
        try:
            # Get order
            order = await self.db_client.find_one("orders", {"order_id": order_id})
            if not order:
                raise NotFoundError(f"Order not found: {order_id}")
            
            # Check ownership
            if order["user_id"] != user_id:
                raise ValidationError("Not authorized to view this order")
            
            # Clean MongoDB-specific fields
            order_info = {k: v for k, v in order.items() if not k.startswith("_")}
            
            return order_info
            
        except NotFoundError:
            raise
        except ValidationError as e:
            logger.warning(f"Order access validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Order retrieval error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("Failed to retrieve order") from e
    
    async def get_user_orders(self, user_id: str, platform: Optional[str] = None, 
                              symbol: Optional[str] = None, status: Optional[str] = None, 
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get user orders with filtering.
        
        Args:
            user_id: User ID
            platform: Filter by platform
            symbol: Filter by symbol
            status: Filter by status
            limit: Maximum number of orders to return
            
        Returns:
            List of orders
        """
        try:
            # Build query
            query = {"user_id": user_id}
            
            if platform:
                if platform not in [p.value for p in Platform]:
                    raise ValidationError(f"Invalid platform: {platform}")
                query["platform"] = platform
            
            if symbol:
                query["symbol"] = symbol
            
            if status:
                if status not in [s.value for s in OrderStatus]:
                    raise ValidationError(f"Invalid status: {status}")
                query["status"] = status
            
            # Get orders
            orders = await self.db_client.find(
                "orders",
                query,
                sort=[("created_at", -1)],
                limit=limit
            )
            
            # Clean MongoDB-specific fields
            orders_list = []
            for order in orders:
                order_info = {k: v for k, v in order.items() if not k.startswith("_")}
                orders_list.append(order_info)
            
            return orders_list
            
        except ValidationError as e:
            logger.warning(f"Orders query validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Orders query error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("Failed to retrieve orders") from e
    
    async def get_user_positions(self, user_id: str, platform: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get user positions.
        
        Args:
            user_id: User ID
            platform: Filter by platform
            
        Returns:
            List of positions
        """
        try:
            # Build query
            query = {"user_id": user_id}
            
            if platform:
                if platform not in [p.value for p in Platform]:
                    raise ValidationError(f"Invalid platform: {platform}")
                query["platform"] = platform
            
            # Get positions
            positions = await self.db_client.find(
                "positions",
                query,
                sort=[("updated_at", -1)]
            )
            
            # Clean MongoDB-specific fields
            positions_list = []
            for position in positions:
                position_info = {k: v for k, v in position.items() if not k.startswith("_")}
                positions_list.append(position_info)
            
            # If no positions in database, request current positions
            if not positions_list and platform:
                # Request positions from platform service
                request_id = f"req_positions_{time.time_ns()}"
                response_key = f"response:{request_id}"
                
                # Publish request
                await self.publish(f"position:get:{platform}", {
                    "user_id": user_id,
                    "request_id": request_id
                })
                
                # Wait for response with timeout
                start_time = time.time()
                max_wait = 5  # seconds
                
                while time.time() - start_time < max_wait:
                    response = await self.redis_client.get(response_key)
                    if response:
                        # Clean up
                        await self.redis_client.delete(response_key)
                        
                        # Parse and return
                        return json.loads(response)
                    
                    # Wait and try again
                    await asyncio.sleep(0.1)
            
            return positions_list
            
        except ValidationError as e:
            logger.warning(f"Positions query validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Positions query error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("Failed to retrieve positions") from e
    
    async def _handle_market_message(self, channel: str, message: Dict[str, Any]):
        """
        Handle market-related Redis messages.
        
        Args:
            channel: Message channel
            message: Message data
        """
        try:
            # Handle different message types
            if channel.startswith("market:data:"):
                platform = message.get("platform")
                symbol = message.get("symbol")
                
                # Update symbol cache if it's a tick update
                if channel == "market:data:tick" and platform and symbol:
                    await self._update_symbol_tick(platform, symbol, message)
                
                # Update chart cache if it's a candle update
                elif channel == "market:data:candle" and platform and symbol:
                    await self._update_candle_data(platform, symbol, message)
            
            # Handle symbol information updates
            elif channel.startswith("market:symbols:"):
                platform = message.get("platform")
                if platform and "symbols" in message:
                    # Update symbols cache
                    self.active_symbols[platform] = message["symbols"]
                    
                    # Update Redis cache
                    await self.redis_client.set(
                        f"symbols:{platform}", 
                        json.dumps(message["symbols"]),
                        ex=3600  # 1 hour
                    )
            
            # Handle platform status updates
            elif channel.startswith("market:status:"):
                platform = message.get("platform")
                if platform and "status" in message:
                    # Update status cache
                    self.platform_statuses[platform] = message
                    
                    # Update Redis cache
                    await self.redis_client.set(
                        f"status:{platform}", 
                        json.dumps(message),
                        ex=300  # 5 minutes
                    )
        except Exception as e:
            logger.error(f"Error handling market message on channel {channel}: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _handle_order_message(self, channel: str, message: Dict[str, Any]):
        """
        Handle order-related Redis messages.
        
        Args:
            channel: Message channel
            message: Message data
        """
        try:
            # Handle order updates
            if channel.startswith("order:update:"):
                order_id = message.get("order_id")
                if not order_id:
                    return
                
                # Get existing order
                order = await self.db_client.find_one("orders", {"order_id": order_id})
                if not order:
                    logger.warning(f"Order update for unknown order: {order_id}")
                    return
                
                # Update order
                updates = {
                    "status": message.get("status", order.get("status")),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                
                # Add other fields if provided
                for field in ["executed_quantity", "average_price", "fills", "error"]:
                    if field in message:
                        updates[field] = message[field]
                
                # Update in database
                await self.db_client.update_one(
                    "orders",
                    {"order_id": order_id},
                    {"$set": updates}
                )
                
                # Track metrics
                if "status" in updates and updates["status"] != order.get("status"):
                    self.metrics.increment("order_status_changes_total", tags={
                        "platform": order.get("platform"),
                        "symbol": order.get("symbol"),
                        "status": updates["status"]
                    })
                
                # Notify user through WebSocket
                user_id = order.get("user_id")
                if user_id:
                    # Full order with updates
                    full_order = {**order, **updates}
                    full_order = {k: v for k, v in full_order.items() if not k.startswith("_")}
                    
                    await self.publish(f"websocket:user:{user_id}", {
                        "type": "order_update",
                        "order": full_order
                    })
            
            # Handle order errors
            elif channel.startswith("order:error:"):
                order_id = message.get("order_id")
                if not order_id:
                    return
                
                error = message.get("error", "Unknown error")
                
                # Update order status
                await self.db_client.update_one(
                    "orders",
                    {"order_id": order_id},
                    {"$set": {
                        "status": OrderStatus.REJECTED.value,
                        "error": error,
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }}
                )
                
                # Track metrics
                self.metrics.increment("order_errors_total", tags={
                    "platform": message.get("platform", "unknown"),
                    "error_type": message.get("error_type", "unknown")
                })
                
                # Get user ID for notification
                order = await self.db_client.find_one("orders", {"order_id": order_id})
                if order:
                    user_id = order.get("user_id")
                    if user_id:
                        # Send notification to user
                        await self.publish(f"websocket:user:{user_id}", {
                            "type": "order_error",
                            "order_id": order_id,
                            "error": error
                        })
        except Exception as e:
            logger.error(f"Error handling order message on channel {channel}: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _handle_strategy_message(self, channel: str, message: Dict[str, Any]):
        """
        Handle strategy-related Redis messages.
        
        Args:
            channel: Message channel
            message: Message data
        """
        try:
            # Handle strategy signals
            if channel == "strategy:signal":
                strategy_id = message.get("strategy_id")
                symbol = message.get("symbol")
                platform = message.get("platform")
                signal_type = message.get("signal_type")
                confidence = message.get("confidence", 0)
                
                if not all([strategy_id, symbol, platform, signal_type]):
                    logger.warning("Incomplete strategy signal message")
                    return
                
                # Store signal in database
                signal_data = {
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "platform": platform,
                    "signal_type": signal_type,
                    "confidence": confidence,
                    "price": message.get("price"),
                    "timestamp": message.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                    "indicators": message.get("indicators", {}),
                    "context": message.get("context", {})
                }
                
                await self.db_client.insert_one("signals", signal_data)
                
                # Broadcast signal to relevant topic
                await self.publish(f"websocket:topic:signals:{platform}:{symbol}", {
                    "type": "strategy_signal",
                    "signal": signal_data
                })
                
                # Track metrics
                self.metrics.increment("strategy_signals_total", tags={
                    "platform": platform,
                    "symbol": symbol,
                    "signal_type": signal_type,
                    "confidence_range": f"{int(confidence * 10) / 10:.1f}"
                })
                
                logger.info(f"Strategy signal: id={strategy_id} platform={platform} " +
                           f"symbol={symbol} type={signal_type} confidence={confidence:.2f}")
            
            # Handle strategy status updates
            elif channel.startswith("strategy:status:"):
                strategy_id = message.get("strategy_id")
                status = message.get("status")
                
                if strategy_id and status:
                    # Store or update in Redis
                    await self.redis_client.hset(
                        "strategy:status",
                        strategy_id,
                        json.dumps(message)
                    )
                    
                    # Also broadcast to subscribers
                    await self.publish(f"websocket:topic:strategy:{strategy_id}", {
                        "type": "strategy_status",
                        "status": message
                    })
        except Exception as e:
            logger.error(f"Error handling strategy message on channel {channel}: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _handle_execution_message(self, channel: str, message: Dict[str, Any]):
        """
        Handle execution-related Redis messages.
        
        Args:
            channel: Message channel
            message: Message data
        """
        try:
            # Handle position updates
            if channel.startswith("execution:position:"):
                user_id = message.get("user_id")
                platform = message.get("platform")
                symbol = message.get("symbol")
                
                if not all([user_id, platform, symbol]):
                    logger.warning("Incomplete position update message")
                    return
                
                # Check if position exists
                position = await self.db_client.find_one(
                    "positions",
                    {
                        "user_id": user_id,
                        "platform": platform,
                        "symbol": symbol
                    }
                )
                
                # Prepare position data
                position_data = {
                    "user_id": user_id,
                    "platform": platform,
                    "symbol": symbol,
                    "size": message.get("size", 0),
                    "entry_price": message.get("entry_price"),
                    "liquidation_price": message.get("liquidation_price"),
                    "unrealized_pnl": message.get("unrealized_pnl"),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                
                if position:
                    # Update existing position
                    await self.db_client.update_one(
                        "positions",
                        {"_id": position["_id"]},
                        {"$set": position_data}
                    )
                else:
                    # Create new position
                    position_data["created_at"] = position_data["updated_at"]
                    await self.db_client.insert_one("positions", position_data)
                
                # Notify user through WebSocket
                await self.publish(f"websocket:user:{user_id}", {
                    "type": "position_update",
                    "position": position_data
                })
        except Exception as e:
            logger.error(f"Error handling execution message on channel {channel}: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _update_symbol_tick(self, platform: str, symbol: str, data: Dict[str, Any]):
        """
        Update symbol tick data in cache.
        
        Args:
            platform: Trading platform
            symbol: Trading symbol
            data: Tick data
        """
        try:
            # Store latest tick in Redis
            cache_key = f"tick:{platform}:{symbol}"
            await self.redis_client.set(
                cache_key, 
                json.dumps(data),
                ex=60  # 1 minute
            )
            
            # Also update in time series if price is available
            if "price" in data:
                ts_key = f"ts:price:{platform}:{symbol}"
                timestamp = int(time.time())
                price = float(data["price"])
                await self.redis_client.ts_add(ts_key, timestamp, price)
        except Exception as e:
            logger.error(f"Error updating tick data: {str(e)}")
    
    async def _update_candle_data(self, platform: str, symbol: str, data: Dict[str, Any]):
        """
        Update candle data in cache.
        
        Args:
            platform: Trading platform
            symbol: Trading symbol
            data: Candle data
        """
        try:
            # Check if we have timeframe
            timeframe = data.get("timeframe")
            if not timeframe:
                return
            
            # Store candle data in Redis
            cache_key = f"chart:{platform}:{symbol}:{timeframe}"
            await self.redis_client.set(
                cache_key, 
                json.dumps(data),
                ex=300  # 5 minutes
            )
        except Exception as e:
            logger.error(f"Error updating candle data: {str(e)}")
    
    async def _get_health_info(self) -> Dict[str, Any]:
        """Get service-specific health information."""
        try:
            # Count active symbols by platform
            symbol_counts = {}
            for platform, symbols in self.active_symbols.items():
                symbol_counts[platform] = len(symbols)
            
            # Get platform statuses
            statuses = {}
            for platform, status in self.platform_statuses.items():
                statuses[platform] = status.get("status", "unknown")
            
            return {
                "active_symbols_count": symbol_counts,
                "platform_statuses": statuses
            }
        except Exception as e:
            logger.error(f"Error getting health info: {str(e)}")
            return {}
