#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Order Manager Module

Advanced order management for executing trades with optimal execution on both Binance and Deriv platforms.
Features intelligent order routing, order type selection, and execution optimization.
"""

import asyncio
import time
import uuid
import logging
import json
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from decimal import Decimal
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd

from common.logger import get_logger
from common.exceptions import (
    OrderExecutionError, InsufficientFundsError, InvalidOrderError,
    RateLimitError, NetworkError, OrderCancellationError, SlippageExceededError
)
from common.constants import (
    OrderType as ORDER_TYPE, OrderSide as ORDER_SIDE, OrderStatus as ORDER_STATUS, TIMEFRAMES,
    Exchange, SUPPORTED_PLATFORMS as PLATFORMS,
    BINANCE_ORDER_TYPES, DERIV_ORDER_TYPES,
    BINANCE_ORDER_STATUS_MAP, DERIV_ORDER_STATUS_MAP,
    MAX_SLIPPAGE_PERCENT, MAX_RETRY_ATTEMPTS
)
from common.utils import (
    calculate_price_precision, calculate_quantity_precision,
    round_to_precision, convert_timeframe, calculate_order_cost,
    calculate_order_risk, normalize_price, normalize_quantity
)
from common.metrics import MetricsCollector
from common.redis_client import RedisClient

from data_feeds.binance_feed import BinanceFeed
from data_feeds.deriv_feed import DerivFeed
from data_storage.market_data import MarketDataRepository

logger = get_logger("execution_engine.order_manager")


class OrderManager:
    """
    Advanced Order Manager that handles order creation, submission, tracking, and lifecycle management.
    
    Features:
    - Smart order routing based on execution probability and cost
    - Dynamic order type selection based on market conditions
    - Intelligent order sizing and risk management
    - Advanced execution algorithms for minimal slippage
    - Order execution tracking and reconciliation
    - Retry mechanisms with exponential backoff
    - Order execution analytics and optimization
    """
    
    def __init__(
        self,
        platform_apis: Dict[str, Any],
        market_data: MarketDataRepository,
        redis_client: RedisClient,
        metrics_collector: MetricsCollector,
        config: Dict[str, Any]
    ):
        """
        Initialize the OrderManager with required dependencies.
        
        Args:
            platform_apis: Dict of platform API clients for Binance and Deriv
            market_data: Market data instance for checking current market conditions
            redis_client: Redis client for order cache and real-time updates
            metrics_collector: Metrics collector for tracking order execution performance
            config: Configuration dictionary with execution parameters
        """
        self.platform_apis = platform_apis
        self.market_data = market_data
        self.redis_client = redis_client
        self.metrics_collector = metrics_collector
        self.config = config
        
        # Initialize order tracking containers
        self.active_orders = {}  # order_id -> order_details
        self.order_history = deque(maxlen=1000)  # Limited history for memory management
        self.pending_submissions = {}  # submission_id -> order_details
        
        # Load platform-specific modules and configurations
        self.platform_handlers = {
            PLATFORMS.BINANCE: self._create_binance_order,
            PLATFORMS.DERIV: self._create_deriv_order
        }
        self.platform_cancelation_handlers = {
            PLATFORMS.BINANCE: self._cancel_binance_order,
            PLATFORMS.DERIV: self._cancel_deriv_order
        }
        
        # Set up execution optimization parameters
        self.max_slippage = config.get("max_slippage", MAX_SLIPPAGE_PERCENT)
        self.retry_attempts = config.get("retry_attempts", MAX_RETRY_ATTEMPTS)
        self.retry_delay_base = config.get("retry_delay_base", 0.5)
        
        # Load order execution algorithms
        self.execution_algorithms = {
            "twap": self._twap_execution,
            "vwap": self._vwap_execution,
            "iceberg": self._iceberg_execution,
            "pegged": self._pegged_execution,
            "adaptive": self._adaptive_execution,
            "market_aware": self._market_aware_execution,
            "liquidity_seeking": self._liquidity_seeking_execution,
            "immediate": self._immediate_execution
        }
        
        # Initialize performance tracking
        self._init_performance_tracking()
        
        logger.info("OrderManager initialized with %s execution algorithms and %s platform handlers", 
                   len(self.execution_algorithms), len(self.platform_handlers))
        
        # Start background tasks
        self._start_background_tasks()
    
    def _init_performance_tracking(self):
        """Initialize performance tracking metrics for order execution optimization."""
        self.execution_metrics = {
            "slippage_by_algorithm": {algo: [] for algo in self.execution_algorithms},
            "execution_time_by_algorithm": {algo: [] for algo in self.execution_algorithms},
            "success_rate_by_algorithm": {algo: {"success": 0, "total": 0} for algo in self.execution_algorithms},
            "cost_by_algorithm": {algo: [] for algo in self.execution_algorithms}
        }
        
        # Initialize platform-specific metrics
        self.platform_metrics = {platform: {
            "latency": [],
            "success_rate": {"success": 0, "total": 0},
            "retry_count": [],
            "rate_limit_hits": 0
        } for platform in self.platform_handlers}

    def _start_background_tasks(self):
        """Start background tasks for order tracking and optimization."""
        self.bg_tasks = []
        self.stop_signal = asyncio.Event()
        
        # Order tracking and status update task
        self.bg_tasks.append(asyncio.create_task(self._track_active_orders()))
        
        # Execution optimizer task - periodically analyzes performance to improve execution
        self.bg_tasks.append(asyncio.create_task(self._execution_optimizer()))
        
        # Order reconciliation task - ensures all orders are properly accounted for
        self.bg_tasks.append(asyncio.create_task(self._order_reconciliation()))
        
        logger.info("Started %s background tasks for order management", len(self.bg_tasks))

    async def create_order(
        self,
        platform: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Union[float, Decimal],
        price: Optional[Union[float, Decimal]] = None,
        order_params: Optional[Dict[str, Any]] = None,
        execution_algorithm: str = "adaptive",
        time_in_force: str = "GTC",
        execution_timeout: int = 60,
        max_slippage: Optional[float] = None,
        custom_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create and execute an order with advanced execution capabilities.
        
        Args:
            platform: Trading platform (e.g., BINANCE, DERIV)
            symbol: Trading symbol/instrument
            side: Order side (BUY, SELL)
            order_type: Order type (MARKET, LIMIT, etc.)
            quantity: Order quantity
            price: Order price (required for limit orders)
            order_params: Additional platform-specific parameters
            execution_algorithm: Algorithm to use for optimal execution
            time_in_force: Time in force for the order
            execution_timeout: Maximum time allowed for execution in seconds
            max_slippage: Maximum allowed slippage for this order (overrides global)
            custom_id: Custom client order ID
            
        Returns:
            Dictionary with order details and execution status
        
        Raises:
            OrderExecutionError: If order execution fails
            InsufficientFundsError: If insufficient funds for order
            InvalidOrderError: If order parameters are invalid
            RateLimitError: If platform rate limit is exceeded
            NetworkError: If network issues prevent order execution
        """
        start_time = time.time()
        order_params = order_params or {}
        submission_id = str(uuid.uuid4())
        
        # Normalize and validate inputs
        normalized_quantity = normalize_quantity(quantity, symbol, platform)
        normalized_price = normalize_price(price, symbol, platform) if price else None
        
        # Validate if the order is executable
        await self._validate_order(platform, symbol, side, order_type, normalized_quantity, normalized_price)
        
        # Generate client order ID
        client_order_id = custom_id or f"QS_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Prepare base order information
        order_info = {
            "submission_id": submission_id,
            "client_order_id": client_order_id,
            "platform": platform,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": normalized_quantity,
            "price": normalized_price,
            "params": order_params,
            "time_in_force": time_in_force,
            "submission_time": datetime.utcnow().isoformat(),
            "status": ORDER_STATUS.PENDING,
            "execution_algorithm": execution_algorithm,
            "execution_start": None,
            "execution_end": None,
            "fills": [],
            "average_price": None,
            "executed_quantity": Decimal("0"),
            "remaining_quantity": normalized_quantity,
            "execution_logs": [],
            "metadata": {
                "slippage": None,
                "execution_latency": None,
                "retry_count": 0
            }
        }
        
        # Store in pending submissions
        self.pending_submissions[submission_id] = order_info
        
        try:
            # Log order creation attempt
            logger.info(
                "Creating %s %s order for %s %s on %s using %s algorithm",
                side, order_type, normalized_quantity, symbol, platform, execution_algorithm
            )
            
            # Execute using the specified algorithm
            if execution_algorithm in self.execution_algorithms:
                execution_func = self.execution_algorithms[execution_algorithm]
                execution_start_time = time.time()
                order_info["execution_start"] = datetime.utcnow().isoformat()
                
                result = await execution_func(
                    order_info,
                    max_slippage or self.max_slippage,
                    execution_timeout
                )
                
                order_info["execution_end"] = datetime.utcnow().isoformat()
                execution_time = time.time() - execution_start_time
                
                # Update metrics
                self._update_execution_metrics(
                    execution_algorithm, 
                    result.get("slippage", 0),
                    execution_time,
                    result.get("success", False),
                    result.get("cost", 0)
                )
            else:
                # Fallback to immediate execution if algorithm not found
                logger.warning("Execution algorithm %s not found, using immediate execution", execution_algorithm)
                result = await self._immediate_execution(
                    order_info,
                    max_slippage or self.max_slippage,
                    execution_timeout
                )
            
            # Process result
            if result.get("success", False):
                order_info.update(result.get("order_info", {}))
                
                # If order is active, add to tracking
                if order_info["status"] in [ORDER_STATUS.OPEN, ORDER_STATUS.PARTIALLY_FILLED]:
                    self.active_orders[order_info["order_id"]] = order_info
                    
                    # Set up Redis pub/sub for real-time order updates
                    order_key = f"order:{platform}:{order_info['order_id']}"
                    await self.redis_client.set_dict(order_key, order_info)
                    
                # Remove from pending
                self.pending_submissions.pop(submission_id, None)
                
                # Add to history
                self.order_history.append(order_info)
                
                logger.info(
                    "Successfully created order: %s (%s) with status: %s", 
                    order_info.get("order_id", "Unknown"), 
                    client_order_id,
                    order_info["status"]
                )
                
                # Update global metrics
                self.metrics_collector.increment_counter("orders_created_success")
                self.metrics_collector.observe_latency(
                    "order_creation_latency", 
                    (time.time() - start_time) * 1000
                )
                
                return order_info
            else:
                raise OrderExecutionError(result.get("error", "Unknown execution error"))
                
        except InsufficientFundsError as e:
            logger.error("Insufficient funds for order: %s", str(e))
            order_info["status"] = ORDER_STATUS.REJECTED
            order_info["error"] = str(e)
            self.metrics_collector.increment_counter("orders_rejected_insufficient_funds")
            raise
            
        except InvalidOrderError as e:
            logger.error("Invalid order parameters: %s", str(e))
            order_info["status"] = ORDER_STATUS.REJECTED
            order_info["error"] = str(e)
            self.metrics_collector.increment_counter("orders_rejected_invalid_params")
            raise
            
        except RateLimitError as e:
            logger.error("Rate limit exceeded: %s", str(e))
            order_info["status"] = ORDER_STATUS.REJECTED
            order_info["error"] = str(e)
            self.metrics_collector.increment_counter("orders_rejected_rate_limit")
            self.platform_metrics[platform]["rate_limit_hits"] += 1
            raise
            
        except NetworkError as e:
            logger.error("Network error during order creation: %s", str(e))
            order_info["status"] = ORDER_STATUS.UNKNOWN
            order_info["error"] = str(e)
            self.metrics_collector.increment_counter("orders_failed_network")
            raise
            
        except Exception as e:
            logger.exception("Unexpected error creating order: %s", str(e))
            order_info["status"] = ORDER_STATUS.REJECTED
            order_info["error"] = str(e)
            self.metrics_collector.increment_counter("orders_failed_unexpected")
            raise OrderExecutionError(f"Unexpected error: {str(e)}")
            
        finally:
            # Always store the attempt in history for auditing
            if order_info["status"] in [ORDER_STATUS.REJECTED, ORDER_STATUS.UNKNOWN]:
                self.order_history.append(order_info)
                self.pending_submissions.pop(submission_id, None)

    async def cancel_order(self, platform: str, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Cancel an active order.
        
        Args:
            platform: Trading platform
            symbol: Trading symbol
            order_id: Order ID to cancel
            
        Returns:
            Dictionary with cancellation status and details
            
        Raises:
            OrderCancellationError: If cancellation fails
            InvalidOrderError: If order doesn't exist or can't be cancelled
        """
        start_time = time.time()
        
        # Check if order exists and is active
        if order_id not in self.active_orders:
            # Try to find in pending submissions
            for sub_id, order in self.pending_submissions.items():
                if order.get("order_id") == order_id:
                    logger.warning("Attempting to cancel order that is still being submitted: %s", order_id)
                    break
            else:
                logger.warning("Attempting to cancel unknown order: %s", order_id)
                raise InvalidOrderError(f"Order {order_id} not found in active orders")
        
        order_info = self.active_orders.get(order_id, {})
        logger.info("Attempting to cancel order %s for %s on %s", order_id, symbol, platform)
        
        # Select the appropriate platform handler
        if platform not in self.platform_cancelation_handlers:
            raise InvalidOrderError(f"Unsupported platform: {platform}")
        
        cancel_handler = self.platform_cancelation_handlers[platform]
        
        try:
            # Execute cancellation with retry logic
            retry_count = 0
            while retry_count < self.retry_attempts:
                try:
                    result = await cancel_handler(symbol, order_id)
                    
                    # Update order status
                    if result.get("success", False):
                        # Remove from active orders
                        if order_id in self.active_orders:
                            order_info = self.active_orders.pop(order_id)
                            order_info["status"] = ORDER_STATUS.CANCELLED
                            order_info["cancellation_time"] = datetime.utcnow().isoformat()
                            
                            # Update in Redis
                            order_key = f"order:{platform}:{order_id}"
                            await self.redis_client.set_dict(order_key, order_info)
                            
                            # Add to history
                            self.order_history.append(order_info)
                        
                        logger.info("Successfully cancelled order %s", order_id)
                        
                        # Update metrics
                        self.metrics_collector.increment_counter("orders_cancelled_success")
                        self.metrics_collector.observe_latency(
                            "order_cancellation_latency", 
                            (time.time() - start_time) * 1000
                        )
                        
                        return {
                            "success": True,
                            "order_id": order_id,
                            "status": ORDER_STATUS.CANCELLED,
                            "message": "Order successfully cancelled",
                            "latency_ms": (time.time() - start_time) * 1000
                        }
                    else:
                        raise OrderCancellationError(result.get("error", "Unknown cancellation error"))
                        
                except (NetworkError, RateLimitError) as e:
                    # These errors are retryable
                    retry_count += 1
                    if retry_count >= self.retry_attempts:
                        logger.error("Failed to cancel order after %s attempts: %s", retry_count, str(e))
                        raise
                    
                    # Exponential backoff
                    await asyncio.sleep(self.retry_delay_base * (2 ** retry_count))
                    logger.warning("Retrying order cancellation (attempt %s): %s", retry_count + 1, str(e))
                    
                except Exception as e:
                    # Non-retryable errors
                    logger.exception("Error cancelling order: %s", str(e))
                    raise
        
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, str(e))
            self.metrics_collector.increment_counter("orders_cancelled_failed")
            
            raise OrderCancellationError(f"Failed to cancel order: {str(e)}")

    async def get_order_status(self, platform: str, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Get the current status of an order.
        
        Args:
            platform: Trading platform
            symbol: Trading symbol
            order_id: Order ID
            
        Returns:
            Dictionary with order status and details
        """
        # First check local cache
        if order_id in self.active_orders:
            return self.active_orders[order_id]
        
        # Check Redis for recent updates
        order_key = f"order:{platform}:{order_id}"
        cached_order = await self.redis_client.get_dict(order_key)
        if cached_order:
            return cached_order
        
        # Query platform API
        try:
            if platform == PLATFORMS.BINANCE:
                response = await self.platform_apis[PLATFORMS.BINANCE].get_order(symbol=symbol, orderId=order_id)
                status = BINANCE_ORDER_STATUS_MAP.get(response.get("status"), ORDER_STATUS.UNKNOWN)
                
                order_info = {
                    "order_id": str(response.get("orderId")),
                    "client_order_id": response.get("clientOrderId"),
                    "platform": PLATFORMS.BINANCE,
                    "symbol": response.get("symbol"),
                    "side": response.get("side"),
                    "order_type": response.get("type"),
                    "quantity": Decimal(str(response.get("origQty"))),
                    "price": Decimal(str(response.get("price"))) if float(response.get("price", 0)) > 0 else None,
                    "status": status,
                    "executed_quantity": Decimal(str(response.get("executedQty", 0))),
                    "average_price": Decimal(str(response.get("avgPrice", 0))) if float(response.get("avgPrice", 0)) > 0 else None,
                    "time_in_force": response.get("timeInForce"),
                    "creation_time": datetime.fromtimestamp(response.get("time", 0) / 1000).isoformat() if response.get("time") else None,
                    "update_time": datetime.fromtimestamp(response.get("updateTime", 0) / 1000).isoformat() if response.get("updateTime") else None
                }
                
            elif platform == PLATFORMS.DERIV:
                response = await self.platform_apis[PLATFORMS.DERIV].get_contract(contract_id=order_id)
                status = DERIV_ORDER_STATUS_MAP.get(response.get("status"), ORDER_STATUS.UNKNOWN)
                
                order_info = {
                    "order_id": response.get("contract_id"),
                    "client_order_id": response.get("app_id"),
                    "platform": PLATFORMS.DERIV,
                    "symbol": response.get("underlying"),
                    "side": "BUY" if response.get("contract_type", "").startswith("CALL") else "SELL",
                    "order_type": response.get("contract_type"),
                    "quantity": Decimal(str(response.get("amount"))),
                    "price": Decimal(str(response.get("entry_spot"))) if response.get("entry_spot") else None,
                    "status": status,
                    "executed_quantity": Decimal(str(response.get("amount"))) if status in [ORDER_STATUS.FILLED, ORDER_STATUS.PARTIALLY_FILLED] else Decimal("0"),
                    "average_price": Decimal(str(response.get("entry_spot"))) if response.get("entry_spot") else None,
                    "creation_time": datetime.fromtimestamp(response.get("date_start", 0)).isoformat() if response.get("date_start") else None,
                    "update_time": datetime.fromtimestamp(response.get("date_settlement", 0)).isoformat() if response.get("date_settlement") else None
                }
            else:
                raise InvalidOrderError(f"Unsupported platform: {platform}")
            
            # Cache result in Redis with expiration
            await self.redis_client.set_dict(order_key, order_info, expire=3600)  # 1 hour expiration
            
            # Update active orders if needed
            if order_info["status"] in [ORDER_STATUS.OPEN, ORDER_STATUS.PARTIALLY_FILLED]:
                self.active_orders[order_id] = order_info
            
            return order_info
            
        except Exception as e:
            logger.error("Error fetching order status for %s on %s: %s", order_id, platform, str(e))
            raise InvalidOrderError(f"Failed to fetch order status: {str(e)}")

    async def get_open_orders(self, platform: str, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders for a platform and optionally filtered by symbol.
        
        Args:
            platform: Trading platform
            symbol: Optional symbol to filter orders
            
        Returns:
            List of open orders
        """
        try:
            # First check local cache for faster response
            cached_orders = [
                order for order_id, order in self.active_orders.items()
                if order["platform"] == platform and (symbol is None or order["symbol"] == symbol)
            ]
            
            # Get from platform API to ensure we have the latest data
            if platform == PLATFORMS.BINANCE:
                params = {"symbol": symbol} if symbol else {}
                response = await self.platform_apis[PLATFORMS.BINANCE].get_open_orders(**params)
                
                api_orders = []
                for order_data in response:
                    order_info = {
                        "order_id": str(order_data.get("orderId")),
                        "client_order_id": order_data.get("clientOrderId"),
                        "platform": PLATFORMS.BINANCE,
                        "symbol": order_data.get("symbol"),
                        "side": order_data.get("side"),
                        "order_type": order_data.get("type"),
                        "quantity": Decimal(str(order_data.get("origQty"))),
                        "price": Decimal(str(order_data.get("price"))) if float(order_data.get("price", 0)) > 0 else None,
                        "status": BINANCE_ORDER_STATUS_MAP.get(order_data.get("status"), ORDER_STATUS.UNKNOWN),
                        "executed_quantity": Decimal(str(order_data.get("executedQty", 0))),
                        "average_price": Decimal(str(order_data.get("avgPrice", 0))) if float(order_data.get("avgPrice", 0)) > 0 else None,
                        "time_in_force": order_data.get("timeInForce"),
                        "creation_time": datetime.fromtimestamp(order_data.get("time", 0) / 1000).isoformat() if order_data.get("time") else None,
                        "update_time": datetime.fromtimestamp(order_data.get("updateTime", 0) / 1000).isoformat() if order_data.get("updateTime") else None
                    }
                    api_orders.append(order_info)
                    
                    # Update local cache
                    self.active_orders[order_info["order_id"]] = order_info
                
                # Sync api_orders with our cache (remove orders that are no longer active)
                for order in cached_orders:
                    if not any(api_order["order_id"] == order["order_id"] for api_order in api_orders):
                        self.active_orders.pop(order["order_id"], None)
                
                return api_orders
                
            elif platform == PLATFORMS.DERIV:
                params = {"underlying": symbol} if symbol else {}
                response = await self.platform_apis[PLATFORMS.DERIV].get_open_contracts(**params)
                
                api_orders = []
                for contract in response.get("open_contracts", []):
                    order_info = {
                        "order_id": contract.get("contract_id"),
                        "client_order_id": contract.get("app_id"),
                        "platform": PLATFORMS.DERIV,
                        "symbol": contract.get("underlying"),
                        "side": "BUY" if contract.get("contract_type", "").startswith("CALL") else "SELL",
                        "order_type": contract.get("contract_type"),
                        "quantity": Decimal(str(contract.get("amount"))),
                        "price": Decimal(str(contract.get("entry_spot"))) if contract.get("entry_spot") else None,
                        "status": DERIV_ORDER_STATUS_MAP.get(contract.get("status"), ORDER_STATUS.UNKNOWN),
                        "executed_quantity": Decimal(str(contract.get("amount"))) if contract.get("status") in ["open", "partially_filled"] else Decimal("0"),
                        "average_price": Decimal(str(contract.get("entry_spot"))) if contract.get("entry_spot") else None,
                        "creation_time": datetime.fromtimestamp(contract.get("date_start", 0)).isoformat() if contract.get("date_start") else None,
                        "update_time": datetime.fromtimestamp(contract.get("date_settlement", 0)).isoformat() if contract.get("date_settlement") else None
                    }
                    api_orders.append(order_info)
                    
                    # Update local cache
                    self.active_orders[order_info["order_id"]] = order_info
                
                # Sync api_orders with our cache
                for order in cached_orders:
                    if not any(api_order["order_id"] == order["order_id"] for api_order in api_orders):
                        self.active_orders.pop(order["order_id"], None)
                
                return api_orders
            else:
                raise InvalidOrderError(f"Unsupported platform: {platform}")
                
        except Exception as e:
            logger.error("Error fetching open orders for %s: %s", platform, str(e))
            # Fall back to cached orders if API call fails
            return cached_orders if 'cached_orders' in locals() else []

    async def _validate_order(
        self, 
        platform: str, 
        symbol: str, 
        side: str, 
        order_type: str, 
        quantity: Decimal,
        price: Optional[Decimal]
    ):
        """Validate order parameters before submission."""
        # Check if platform is supported
        if platform not in self.platform_handlers:
            raise InvalidOrderError(f"Unsupported platform: {platform}")
        
        # Check if we have symbol data
        symbol_info = await self.market_data.get_symbol_info(platform, symbol)
        if not symbol_info:
            raise InvalidOrderError(f"Invalid symbol: {symbol} for platform: {platform}")
        
        # Check if quantity meets minimum and maximum requirements
        min_qty = symbol_info.get("min_qty", Decimal("0"))
        max_qty = symbol_info.get("max_qty", Decimal("9999999999"))
        
        if quantity < min_qty:
            raise InvalidOrderError(f"Quantity {quantity} below minimum {min_qty} for {symbol}")
        
        if quantity > max_qty:
            raise InvalidOrderError(f"Quantity {quantity} above maximum {max_qty} for {symbol}")
        
        # Check if price is provided for limit orders
        if order_type in ["LIMIT", "STOP_LIMIT", "LIMIT_MAKER"] and price is None:
            raise InvalidOrderError(f"Price required for {order_type} orders")
        
        # Check if price is within allowed range
        if price is not None:
            min_price = symbol_info.get("min_price", Decimal("0"))
            max_price = symbol_info.get("max_price", Decimal("9999999999"))
            
            if price < min_price:
                raise InvalidOrderError(f"Price {price} below minimum {min_price} for {symbol}")
            
            if price > max_price:
                raise InvalidOrderError(f"Price {price} above maximum {max_price} for {symbol}")
        
        # Check for sufficient balance (approximate check - will be verified by platform)
        current_price = price or await self.market_data.get_last_price(platform, symbol)
        if current_price is None:
            raise InvalidOrderError(f"Unable to get current price for {symbol}")
        
        order_cost = quantity * current_price
        
        # Get user balance
        try:
            balance = await self._get_available_balance(platform, symbol, side)
            
            if side == ORDER_SIDE.BUY and balance < order_cost:
                raise InsufficientFundsError(
                    f"Insufficient funds for {side} order. Required: {order_cost}, Available: {balance}"
                )
            elif side == ORDER_SIDE.SELL:
                # For sell orders, check if we have enough of the asset
                asset = symbol.split('/')[0] if '/' in symbol else symbol[:-3] if symbol.endswith('USD') else symbol
                asset_balance = await self._get_asset_balance(platform, asset)
                
                if asset_balance < quantity:
                    raise InsufficientFundsError(
                        f"Insufficient {asset} for {side} order. Required: {quantity}, Available: {asset_balance}"
                    )
        except Exception as e:
            logger.warning("Balance check failed: %s. Proceeding with order submission.", str(e))
            # Don't block order if balance check fails - platform will validate
        
        return True

    async def _get_available_balance(self, platform: str, symbol: str, side: str) -> Decimal:
        """Get available balance for a trade."""
        try:
            if platform == PLATFORMS.BINANCE:
                # Determine quote currency from symbol
                quote_currency = symbol.split('/')[1] if '/' in symbol else symbol[3:] if symbol.endswith('USD') else 'USDT'
                account_info = await self.platform_apis[PLATFORMS.BINANCE].get_account()
                
                for balance in account_info.get("balances", []):
                    if balance["asset"] == quote_currency:
                        return Decimal(str(balance["free"]))
                
                return Decimal("0")
                
            elif platform == PLATFORMS.DERIV:
                account_info = await self.platform_apis[PLATFORMS.DERIV].get_account_balance()
                return Decimal(str(account_info.get("balance", 0)))
            else:
                raise InvalidOrderError(f"Unsupported platform: {platform}")
        except Exception as e:
            logger.error("Error getting available balance: %s", str(e))
            raise

    async def _get_asset_balance(self, platform: str, asset: str) -> Decimal:
        """Get available balance for a specific asset."""
        try:
            if platform == PLATFORMS.BINANCE:
                account_info = await self.platform_apis[PLATFORMS.BINANCE].get_account()
                
                for balance in account_info.get("balances", []):
                    if balance["asset"] == asset:
                        return Decimal(str(balance["free"]))
                
                return Decimal("0")
                
            elif platform == PLATFORMS.DERIV:
                # For Deriv, asset balance is not directly applicable
                # Return a large number to bypass the check for now
                return Decimal("9999999999")
            else:
                raise InvalidOrderError(f"Unsupported platform: {platform}")
        except Exception as e:
            logger.error("Error getting asset balance: %s", str(e))
            raise

    # Platform-specific order creation methods
    async def _create_binance_order(self, order_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create an order on Binance platform."""
        platform_api = self.platform_apis[PLATFORMS.BINANCE]
        symbol = order_info["symbol"]
        side = order_info["side"]
        order_type = order_info["order_type"]
        quantity = order_info["quantity"]
        price = order_info["price"]
        client_order_id = order_info["client_order_id"]
        params = order_info["params"].copy()
        time_in_force = order_info["time_in_force"]
        
        # Add required parameters
        params.update({
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": float(quantity),
            "newClientOrderId": client_order_id,
            "timestamp": int(time.time() * 1000)
        })
        
        # Add price for limit orders
        if order_type in ["LIMIT", "STOP_LIMIT", "LIMIT_MAKER"]:
            params["price"] = float(price)
            
            # Add time in force for limit orders
            if order_type != "LIMIT_MAKER":  # LIMIT_MAKER doesn't use timeInForce
                params["timeInForce"] = time_in_force
        
        try:
            # Log request
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "order_request",
                "params": params
            }
            order_info["execution_logs"].append(log_entry)
            
            # Execute order
            start_time = time.time()
            response = await platform_api.create_order(**params)
            latency = (time.time() - start_time) * 1000  # ms
            
            # Update platform metrics
            self.platform_metrics[PLATFORMS.BINANCE]["latency"].append(latency)
            self.platform_metrics[PLATFORMS.BINANCE]["success_rate"]["success"] += 1
            self.platform_metrics[PLATFORMS.BINANCE]["success_rate"]["total"] += 1
            
            # Log response
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "order_response",
                "response": response,
                "latency_ms": latency
            }
            order_info["execution_logs"].append(log_entry)
            
            # Process response
            order_id = str(response.get("orderId"))
            status = BINANCE_ORDER_STATUS_MAP.get(response.get("status"), ORDER_STATUS.UNKNOWN)
            
            # Update order info
            result = {
                "success": True,
                "order_info": {
                    "order_id": order_id,
                    "status": status,
                    "transact_time": datetime.fromtimestamp(response.get("transactTime", time.time() * 1000) / 1000).isoformat(),
                    "fills": response.get("fills", []),
                }
            }
            
            # Calculate fill information if any
            if response.get("fills"):
                executed_qty = Decimal("0")
                weighted_price = Decimal("0")
                
                for fill in response["fills"]:
                    fill_qty = Decimal(str(fill["qty"]))
                    fill_price = Decimal(str(fill["price"]))
                    executed_qty += fill_qty
                    weighted_price += fill_qty * fill_price
                
                avg_price = weighted_price / executed_qty if executed_qty > 0 else None
                result["order_info"]["executed_quantity"] = executed_qty
                result["order_info"]["remaining_quantity"] = quantity - executed_qty
                result["order_info"]["average_price"] = avg_price
                
                # Calculate slippage
                if price and avg_price:
                    if side == ORDER_SIDE.BUY:
                        slippage = ((avg_price - price) / price) * 100 if price > 0 else 0
                    else:  # SELL
                        slippage = ((price - avg_price) / price) * 100 if price > 0 else 0
                    
                    result["slippage"] = slippage
                    result["order_info"]["metadata"] = {
                        "slippage": slippage,
                        "execution_latency": latency,
                        "retry_count": order_info["metadata"]["retry_count"]
                    }
            
            return result
            
        except Exception as e:
            logger.error("Error creating Binance order: %s", str(e))
            
            # Update platform metrics
            self.platform_metrics[PLATFORMS.BINANCE]["success_rate"]["total"] += 1
            
            # Determine error type
            if "insufficient balance" in str(e).lower():
                raise InsufficientFundsError(f"Insufficient balance: {str(e)}")
            elif "invalid" in str(e).lower():
                raise InvalidOrderError(f"Invalid order parameters: {str(e)}")
            elif "rate limit" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded: {str(e)}")
            elif "network" in str(e).lower() or "timeout" in str(e).lower():
                raise NetworkError(f"Network error: {str(e)}")
            else:
                raise OrderExecutionError(f"Order execution error: {str(e)}")

    async def _create_deriv_order(self, order_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create an order on Deriv platform."""
        platform_api = self.platform_apis[PLATFORMS.DERIV]
        symbol = order_info["symbol"]
        side = order_info["side"]
        order_type = order_info["order_type"]
        quantity = order_info["quantity"]
        price = order_info["price"]
        client_order_id = order_info["client_order_id"]
        params = order_info["params"].copy()
        
        # Map to Deriv contract parameters
        contract_params = {
            "app_id": client_order_id,
            "proposal": 1,
            "amount": float(quantity),
            "basis": "stake",
            "contract_type": "CALL" if side == ORDER_SIDE.BUY else "PUT",
            "currency": "USD",
            "duration": params.get("duration", 5),
            "duration_unit": params.get("duration_unit", "m"),
            "symbol": symbol
        }
        
        # Handle different order types
        if order_type == "MARKET":
            contract_params["trading_type"] = "standard"
        elif order_type == "LIMIT":
            contract_params["trading_type"] = "forward_starting"
            contract_params["date_start"] = params.get("date_start", int(time.time() + 60))  # Default 1 minute from now
            contract_params["barrier"] = float(price)
        
        try:
            # Log request
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "order_request",
                "params": contract_params
            }
            order_info["execution_logs"].append(log_entry)
            
            # Execute order
            start_time = time.time()
            response = await platform_api.buy_contract(**contract_params)
            latency = (time.time() - start_time) * 1000  # ms
            
            # Update platform metrics
            self.platform_metrics[PLATFORMS.DERIV]["latency"].append(latency)
            self.platform_metrics[PLATFORMS.DERIV]["success_rate"]["success"] += 1
            self.platform_metrics[PLATFORMS.DERIV]["success_rate"]["total"] += 1
            
            # Log response
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "order_response",
                "response": response,
                "latency_ms": latency
            }
            order_info["execution_logs"].append(log_entry)
            
            # Process response
            contract = response.get("buy")
            if not contract:
                raise OrderExecutionError(f"No contract in response: {response}")
                
            order_id = contract.get("contract_id")
            status = DERIV_ORDER_STATUS_MAP.get(contract.get("status", "unknown"), ORDER_STATUS.UNKNOWN)
            
            # Update order info
            result = {
                "success": True,
                "order_info": {
                    "order_id": order_id,
                    "status": status,
                    "transact_time": datetime.utcnow().isoformat(),
                    "contract_info": contract,
                }
            }
            
            # If contract is already active, update quantities
            if status in [ORDER_STATUS.OPEN, ORDER_STATUS.PARTIALLY_FILLED, ORDER_STATUS.FILLED]:
                entry_spot = Decimal(str(contract.get("entry_spot", 0))) if contract.get("entry_spot") else None
                
                # Deriv contracts are fully executed upon creation
                result["order_info"]["executed_quantity"] = quantity
                result["order_info"]["remaining_quantity"] = Decimal("0")
                result["order_info"]["average_price"] = entry_spot
                
                # Calculate slippage if we have both expected and actual price
                if price and entry_spot:
                    if side == ORDER_SIDE.BUY:
                        slippage = ((entry_spot - price) / price) * 100 if price > 0 else 0
                    else:  # SELL
                        slippage = ((price - entry_spot) / price) * 100 if price > 0 else 0
                    
                    result["slippage"] = slippage
                    result["order_info"]["metadata"] = {
                        "slippage": slippage,
                        "execution_latency": latency,
                        "retry_count": order_info["metadata"]["retry_count"]
                    }
            
            return result
            
        except Exception as e:
            logger.error("Error creating Deriv order: %s", str(e))
            
            # Update platform metrics
            self.platform_metrics[PLATFORMS.DERIV]["success_rate"]["total"] += 1
            
            # Determine error type
            if "balance" in str(e).lower():
                raise InsufficientFundsError(f"Insufficient balance: {str(e)}")
            elif "invalid" in str(e).lower():
                raise InvalidOrderError(f"Invalid order parameters: {str(e)}")
            elif "rate limit" in str(e).lower():
                raise RateLimitError(f"Rate limit exceeded: {str(e)}")
            elif "network" in str(e).lower() or "timeout" in str(e).lower():
                raise NetworkError(f"Network error: {str(e)}")
            else:
                raise OrderExecutionError(f"Order execution error: {str(e)}")

    async def _cancel_binance_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an order on Binance platform."""
        try:
            platform_api = self.platform_apis[PLATFORMS.BINANCE]
            
            response = await platform_api.cancel_order(
                symbol=symbol,
                orderId=order_id,
                timestamp=int(time.time() * 1000)
            )
            
            return {
                "success": True,
                "order_id": str(response.get("orderId")),
                "status": BINANCE_ORDER_STATUS_MAP.get(response.get("status"), ORDER_STATUS.UNKNOWN),
                "message": "Order successfully cancelled"
            }
        except Exception as e:
            logger.error("Error cancelling Binance order: %s", str(e))
            
            # Determine if order already closed or cancelled
            if "does not exist" in str(e).lower() or "not found" in str(e).lower():
                # Check if order was already filled
                try:
                    order_status = await self.get_order_status(PLATFORMS.BINANCE, symbol, order_id)
                    if order_status["status"] in [ORDER_STATUS.FILLED, ORDER_STATUS.CANCELLED, ORDER_STATUS.REJECTED]:
                        return {
                            "success": True,
                            "order_id": order_id,
                            "status": order_status["status"],
                            "message": f"Order already in final state: {order_status['status']}"
                        }
                except Exception:
                    pass
            
            return {
                "success": False,
                "error": f"Failed to cancel order: {str(e)}"
            }

    async def _cancel_deriv_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an order on Deriv platform."""
        try:
            platform_api = self.platform_apis[PLATFORMS.DERIV]
            
            response = await platform_api.cancel_contract(contract_id=order_id)
            
            if response.get("cancel"):
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": ORDER_STATUS.CANCELLED,
                    "message": "Order successfully cancelled"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to cancel order: {response.get('error', 'Unknown error')}"
                }
        except Exception as e:
            logger.error("Error cancelling Deriv order: %s", str(e))
            
            # Determine if order already closed or cancelled
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                # Check if order was already filled
                try:
                    order_status = await self.get_order_status(PLATFORMS.DERIV, symbol, order_id)
                    if order_status["status"] in [ORDER_STATUS.FILLED, ORDER_STATUS.CANCELLED, ORDER_STATUS.REJECTED]:
                        return {
                            "success": True,
                            "order_id": order_id,
                            "status": order_status["status"],
                            "message": f"Order already in final state: {order_status['status']}"
                        }
                except Exception:
                    pass
            
            return {
                "success": False,
                "error": f"Failed to cancel order: {str(e)}"
            }

    # Execution algorithms
    async def _immediate_execution(
        self, 
        order_info: Dict[str, Any], 
        max_slippage: float,
        execution_timeout: int
    ) -> Dict[str, Any]:
        """
        Execute order immediately using standard platform API.
        
        This is the simplest execution algorithm with no special handling.
        """
        platform = order_info["platform"]
        
        # Get platform-specific handler
        if platform not in self.platform_handlers:
            return {
                "success": False,
                "error": f"Unsupported platform: {platform}"
            }
        
        handler = self.platform_handlers[platform]
        
        # Execute with retry logic
        max_retries = self.retry_attempts
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Update retry count in metadata
                order_info["metadata"]["retry_count"] = retry_count
                
                # Execute order
                result = await handler(order_info)
                
                # Add execution cost information
                if result.get("success", False):
                    avg_price = result.get("order_info", {}).get("average_price")
                    executed_qty = result.get("order_info", {}).get("executed_quantity")
                    
                    if avg_price and executed_qty:
                        cost = float(avg_price) * float(executed_qty)
                        result["cost"] = cost
                
                return result
                
            except (NetworkError, RateLimitError) as e:
                # These errors are retryable
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error("Max retries reached for order execution: %s", str(e))
                    return {
                        "success": False,
                        "error": f"Failed after {retry_count} attempts: {str(e)}"
                    }
                
                # Exponential backoff
                backoff_time = self.retry_delay_base * (2 ** retry_count)
                logger.warning(
                    "Retrying order execution (attempt %s/%s) after %.2f seconds: %s", 
                    retry_count + 1, max_retries, backoff_time, str(e)
                )
                await asyncio.sleep(backoff_time)
                
            except SlippageExceededError as e:
                # Special handling for slippage errors
                logger.warning("Slippage exceeded maximum allowed: %s", str(e))
                return {
                    "success": False,
                    "error": f"Slippage exceeded maximum allowed: {str(e)}"
                }
                
            except (InsufficientFundsError, InvalidOrderError) as e:
                # These errors are not retryable
                logger.error("Non-retryable error during order execution: %s", str(e))
                return {
                    "success": False,
                    "error": str(e)
                }
                
            except Exception as e:
                # Unexpected errors
                logger.exception("Unexpected error during order execution: %s", str(e))
                return {
                    "success": False,
                    "error": f"Unexpected error: {str(e)}"
                }
    
    async def _twap_execution(
        self, 
        order_info: Dict[str, Any], 
        max_slippage: float,
        execution_timeout: int
    ) -> Dict[str, Any]:
        """
        Time-Weighted Average Price execution algorithm.
        
        Splits the order into smaller chunks and executes them over time to minimize market impact.
        """
        platform = order_info["platform"]
        symbol = order_info["symbol"]
        side = order_info["side"]
        order_type = order_info["order_type"]
        quantity = order_info["quantity"]
        price = order_info["price"]
        client_order_id = order_info["client_order_id"]
        
        # Only applicable for limit orders on platforms that support it
        if order_type != "LIMIT" or platform not in [PLATFORMS.BINANCE]:
            logger.warning("TWAP execution not supported for %s %s orders, falling back to immediate execution", 
                         platform, order_type)
            return await self._immediate_execution(order_info, max_slippage, execution_timeout)
        
        # TWAP parameters
        num_slices = min(int(quantity) // 10 + 1, 5)  # Max 5 slices
        slice_size = quantity / Decimal(num_slices)
        interval = execution_timeout / num_slices
        
        logger.info("Executing TWAP with %s slices of %s %s over %s seconds", 
                   num_slices, slice_size, symbol, execution_timeout)
        
        # Track all executions
        executions = []
        total_executed = Decimal("0")
        total_cost = Decimal("0")
        all_fills = []
        overall_success = True
        
        # Start execution
        execution_start = time.time()
        time_spent = 0
        
        for i in range(num_slices):
            # Check if we should continue
            remaining_time = execution_timeout - (time.time() - execution_start)
            if remaining_time <= 0:
                logger.warning("TWAP execution timeout reached after %s/%s slices", i, num_slices)
                break
            
            # Adjust last slice to ensure we use the full quantity
            if i == num_slices - 1:
                current_slice = quantity - total_executed
            else:
                current_slice = slice_size
            
            # Skip if slice is too small
            if current_slice <= Decimal("0"):
                continue
            
            # Create sub-order
            slice_order_info = order_info.copy()
            slice_order_info["client_order_id"] = f"{client_order_id}_slice_{i+1}"
            slice_order_info["quantity"] = current_slice
            
            # Execute slice
            slice_result = await self._immediate_execution(slice_order_info, max_slippage, int(remaining_time))
            
            if slice_result.get("success", False):
                # Track execution details
                slice_order_info = slice_result.get("order_info", {})
                executed_qty = slice_order_info.get("executed_quantity", Decimal("0"))
                avg_price = slice_order_info.get("average_price")
                
                if executed_qty > Decimal("0") and avg_price:
                    total_executed += executed_qty
                    total_cost += executed_qty * avg_price
                    executions.append(slice_order_info)
                    all_fills.extend(slice_order_info.get("fills", []))
                    
                    logger.info(
                        "TWAP slice %s/%s executed: %s %s at avg price %s", 
                        i+1, num_slices, executed_qty, symbol, avg_price
                    )
            else:
                logger.error("TWAP slice %s/%s failed: %s", i+1, num_slices, slice_result.get("error"))
                overall_success = False
            
            # Wait for next interval if not the last slice
            if i < num_slices - 1:
                sleep_time = max(0, interval - (time.time() - execution_start - time_spent))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                time_spent = time.time() - execution_start
        
        # Calculate results
        avg_execution_price = total_cost / total_executed if total_executed > 0 else None
        execution_time = time.time() - execution_start
        
        # Calculate slippage
        slippage = None
        if price and avg_execution_price:
            if side == ORDER_SIDE.BUY:
                slippage = ((avg_execution_price - price) / price) * 100 if price > 0 else 0
            else:  # SELL
                slippage = ((price - avg_execution_price) / price) * 100 if price > 0 else 0
        
        # Update order info
        result = {
            "success": overall_success,
            "order_info": {
                "execution_algorithm": "twap",
                "executed_quantity": total_executed,
                "remaining_quantity": quantity - total_executed,
                "average_price": avg_execution_price,
                "fills": all_fills,
                "execution_time": execution_time,
                "sub_orders": executions,
                "status": ORDER_STATUS.FILLED if total_executed >= quantity else 
                         ORDER_STATUS.PARTIALLY_FILLED if total_executed > 0 else 
                         ORDER_STATUS.REJECTED,
                "metadata": {
                    "slippage": slippage,
                    "execution_latency": execution_time * 1000,
                    "num_slices": num_slices,
                    "slice_interval": interval
                }
            },
            "slippage": slippage,
            "cost": float(total_cost) if total_cost > 0 else 0
        }
        
        if not overall_success and total_executed == Decimal("0"):
            result["error"] = "All TWAP slices failed to execute"
        
        return result

    async def _vwap_execution(
        self, 
        order_info: Dict[str, Any], 
        max_slippage: float,
        execution_timeout: int
    ) -> Dict[str, Any]:
        """
        Volume-Weighted Average Price execution algorithm.
        
        Executes the order in proportion to expected volume distribution.
        """
        platform = order_info["platform"]
        symbol = order_info["symbol"]
        side = order_info["side"]
        order_type = order_info["order_type"]
        quantity = order_info["quantity"]
        price = order_info["price"]
        client_order_id = order_info["client_order_id"]
        
        # Only applicable for limit orders on platforms that support it
        if order_type != "LIMIT" or platform not in [PLATFORMS.BINANCE]:
            logger.warning("VWAP execution not supported for %s %s orders, falling back to immediate execution", 
                         platform, order_type)
            return await self._immediate_execution(order_info, max_slippage, execution_timeout)
        
        # Get historical volume profile for optimal distribution
        try:
            # Get volume distribution from market data
            volume_profile = await self.market_data.get_volume_profile(platform, symbol, TIMEFRAME.HOUR, 24)
            
            if not volume_profile or len(volume_profile) < 4:
                logger.warning("Insufficient volume profile data for VWAP, falling back to TWAP")
                return await self._twap_execution(order_info, max_slippage, execution_timeout)
            
            # Determine current position in the profile
            current_hour = datetime.utcnow().hour
            
            # Create a normalized distribution for the execution period
            total_volume = sum(vol for hour, vol in volume_profile)
            normalized_profile = {hour: vol/total_volume for hour, vol in volume_profile}
            
            # Calculate slice sizes based on remaining time and volume profile
            remaining_hours = [(h % 24) for h in range(current_hour, current_hour + 24)]
            num_slices = min(8, execution_timeout // 60)  # Max 8 slices, minimum 1 minute per slice
            
            # Calculate time distribution
            time_dist = []
            vol_dist = []
            for i in range(num_slices):
                pos = (current_hour + (i * 24 // num_slices)) % 24
                time_dist.append(pos)
                vol_dist.append(normalized_profile.get(pos, 1/24))
            
            # Normalize to sum to 1
            total_dist = sum(vol_dist)
            vol_dist = [v/total_dist for v in vol_dist]
            
            # Calculate quantity distribution
            qty_dist = [quantity * Decimal(str(v)) for v in vol_dist]
            
            logger.info("Executing VWAP with %s slices over %s seconds with volume distribution: %s", 
                       num_slices, execution_timeout, vol_dist)
            
        except Exception as e:
            logger.error("Error calculating VWAP distribution: %s, falling back to TWAP", str(e))
            return await self._twap_execution(order_info, max_slippage, execution_timeout)
        
        # Track all executions
        executions = []
        total_executed = Decimal("0")
        total_cost = Decimal("0")
        all_fills = []
        overall_success = True
        
        # Start execution
        execution_start = time.time()
        time_spent = 0
        
        for i in range(num_slices):
            # Check if we should continue
            remaining_time = execution_timeout - (time.time() - execution_start)
            if remaining_time <= 0:
                logger.warning("VWAP execution timeout reached after %s/%s slices", i, num_slices)
                break
            
            # Get slice size from distribution
            current_slice = qty_dist[i]
            
            # Adjust last slice to ensure we use the full quantity
            if i == num_slices - 1:
                current_slice = quantity - total_executed
            
            # Skip if slice is too small
            if current_slice <= Decimal("0"):
                continue
            
            # Create sub-order
            slice_order_info = order_info.copy()
            slice_order_info["client_order_id"] = f"{client_order_id}_slice_{i+1}"
            slice_order_info["quantity"] = current_slice
            
            # Execute slice
            slice_interval = execution_timeout / num_slices
            slice_result = await self._immediate_execution(slice_order_info, max_slippage, int(remaining_time))
            
            if slice_result.get("success", False):
                # Track execution details
                slice_order_info = slice_result.get("order_info", {})
                executed_qty = slice_order_info.get("executed_quantity", Decimal("0"))
                avg_price = slice_order_info.get("average_price")
                
                if executed_qty > Decimal("0") and avg_price:
                    total_executed += executed_qty
                    total_cost += executed_qty * avg_price
                    executions.append(slice_order_info)
                    all_fills.extend(slice_order_info.get("fills", []))
                    
                    logger.info(
                        "VWAP slice %s/%s executed: %s %s at avg price %s", 
                        i+1, num_slices, executed_qty, symbol, avg_price
                    )
            else:
                logger.error("VWAP slice %s/%s failed: %s", i+1, num_slices, slice_result.get("error"))
                overall_success = False
            
            # Wait for next interval if not the last slice
            if i < num_slices - 1:
                sleep_time = max(0, slice_interval - (time.time() - execution_start - time_spent))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                time_spent = time.time() - execution_start
        
        # Calculate results
        avg_execution_price = total_cost / total_executed if total_executed > 0 else None
        execution_time = time.time() - execution_start
        
        # Calculate slippage
        slippage = None
        if price and avg_execution_price:
            if side == ORDER_SIDE.BUY:
                slippage = ((avg_execution_price - price) / price) * 100 if price > 0 else 0
            else:  # SELL
                slippage = ((price - avg_execution_price) / price) * 100 if price > 0 else 0
        
        # Update order info
        result = {
            "success": overall_success,
            "order_info": {
                "execution_algorithm": "vwap",
                "executed_quantity": total_executed,
                "remaining_quantity": quantity - total_executed,
                "average_price": avg_execution_price,
                "fills": all_fills,
                "execution_time": execution_time,
                "sub_orders": executions,
                "status": ORDER_STATUS.FILLED if total_executed >= quantity else 
                         ORDER_STATUS.PARTIALLY_FILLED if total_executed > 0 else 
                         ORDER_STATUS.REJECTED,
                "metadata": {
                    "slippage": slippage,
                    "execution_latency": execution_time * 1000,
                    "num_slices": num_slices,
                    "volume_distribution": vol_dist
                }
            },
            "slippage": slippage,
            "cost": float(total_cost) if total_cost > 0 else 0
        }
        
        if not overall_success and total_executed == Decimal("0"):
            result["error"] = "All VWAP slices failed to execute"
        
        return result

    async def _iceberg_execution(
        self, 
        order_info: Dict[str, Any], 
        max_slippage: float,
        execution_timeout: int
    ) -> Dict[str, Any]:
        """
        Iceberg execution algorithm.
        
        Shows only a small portion of the order at a time to hide the full size.
        """
        platform = order_info["platform"]
        symbol = order_info["symbol"]
        side = order_info["side"]
        order_type = order_info["order_type"]
        quantity = order_info["quantity"]
        price = order_info["price"]
        client_order_id = order_info["client_order_id"]
        
        # Only applicable for limit orders on platforms that support it
        if order_type != "LIMIT" or platform not in [PLATFORMS.BINANCE]:
            logger.warning("Iceberg execution not supported for %s %s orders, falling back to immediate execution", 
                         platform, order_type)
            return await self._immediate_execution(order_info, max_slippage, execution_timeout)
        
        # Get market depth to determine optimal iceberg size
        try:
            depth = await self.market_data.get_order_book(platform, symbol)
            
            if side == ORDER_SIDE.BUY:
                asks = depth.get("asks", [])
                if len(asks) > 0:
                    # Calculate average size of top 5 asks
                    top_asks = asks[:5]
                    avg_size = sum(Decimal(str(size)) for _, size in top_asks) / len(top_asks)
                    # Set iceberg size to blend in with average order size
                    visible_size = min(avg_size * Decimal("1.5"), quantity * Decimal("0.1"))
                else:
                    visible_size = quantity * Decimal("0.1")  # Default to 10% of total
            else:  # SELL
                bids = depth.get("bids", [])
                if len(bids) > 0:
                    # Calculate average size of top 5 bids
                    top_bids = bids[:5]
                    avg_size = sum(Decimal(str(size)) for _, size in top_bids) / len(top_bids)
                    # Set iceberg size to blend in with average order size
                    visible_size = min(avg_size * Decimal("1.5"), quantity * Decimal("0.1"))
                else:
                    visible_size = quantity * Decimal("0.1")  # Default to 10% of total
            
            # Ensure visible size is at least 1% of total
            visible_size = max(visible_size, quantity * Decimal("0.01"))
            
            logger.info("Executing Iceberg with visible size of %s out of total %s %s", 
                       visible_size, quantity, symbol)
            
        except Exception as e:
            logger.error("Error calculating iceberg size: %s, using default", str(e))
            visible_size = quantity * Decimal("0.1")  # Default to 10% of total
        
        # Track all executions
        executions = []
        total_executed = Decimal("0")
        total_cost = Decimal("0")
        all_fills = []
        overall_success = True
        
        # Start execution
        execution_start = time.time()
        
        # For Binance, we can use the icebergQty parameter
        if platform == PLATFORMS.BINANCE:
            # Create order with iceberg parameters
            iceberg_order_info = order_info.copy()
            iceberg_order_info["params"] = order_info["params"].copy()
            iceberg_order_info["params"]["icebergQty"] = float(visible_size)
            
            # Execute with iceberg
            result = await self._immediate_execution(iceberg_order_info, max_slippage, execution_timeout)
            
            # No need to modify the result - the platform handles the iceberg logic
            result["order_info"]["execution_algorithm"] = "iceberg"
            result["order_info"]["metadata"]["visible_size"] = float(visible_size)
            
            return result
        else:
            # For other platforms, we need to manually implement iceberg logic
            # by repeatedly placing smaller orders
            
            while total_executed < quantity:
                # Check if we should continue
                remaining_time = execution_timeout - (time.time() - execution_start)
                if remaining_time <= 0:
                    logger.warning("Iceberg execution timeout reached after executing %s/%s", 
                                 total_executed, quantity)
                    break
                
                # Calculate current chunk size
                remaining = quantity - total_executed
                current_chunk = min(visible_size, remaining)
                
                # Create sub-order
                chunk_order_info = order_info.copy()
                chunk_order_info["client_order_id"] = f"{client_order_id}_chunk_{len(executions)+1}"
                chunk_order_info["quantity"] = current_chunk
                
                # Execute chunk
                chunk_result = await self._immediate_execution(chunk_order_info, max_slippage, int(remaining_time))
                
                if chunk_result.get("success", False):
                    # Track execution details
                    chunk_order_info = chunk_result.get("order_info", {})
                    executed_qty = chunk_order_info.get("executed_quantity", Decimal("0"))
                    avg_price = chunk_order_info.get("average_price")
                    
                    if executed_qty > Decimal("0") and avg_price:
                        total_executed += executed_qty
                        total_cost += executed_qty * avg_price
                        executions.append(chunk_order_info)
                        all_fills.extend(chunk_order_info.get("fills", []))
                        
                        logger.info(
                            "Iceberg chunk %s executed: %s/%s %s at avg price %s", 
                            len(executions), total_executed, quantity, symbol, avg_price
                        )
                else:
                    logger.error("Iceberg chunk %s failed: %s", len(executions)+1, chunk_result.get("error"))
                    overall_success = False
                    break
            
            # Calculate results
            avg_execution_price = total_cost / total_executed if total_executed > 0 else None
            execution_time = time.time() - execution_start
            
            # Calculate slippage
            slippage = None
            if price and avg_execution_price:
                if side == ORDER_SIDE.BUY:
                    slippage = ((avg_execution_price - price) / price) * 100 if price > 0 else 0
                else:  # SELL
                    slippage = ((price - avg_execution_price) / price) * 100 if price > 0 else 0
            
            # Update order info
            result = {
                "success": overall_success,
                "order_info": {
                    "execution_algorithm": "iceberg",
                    "executed_quantity": total_executed,
                    "remaining_quantity": quantity - total_executed,
                    "average_price": avg_execution_price,
                    "fills": all_fills,
                    "execution_time": execution_time,
                    "sub_orders": executions,
                    "status": ORDER_STATUS.FILLED if total_executed >= quantity else 
                             ORDER_STATUS.PARTIALLY_FILLED if total_executed > 0 else 
                             ORDER_STATUS.REJECTED,
                    "metadata": {
                        "slippage": slippage,
                        "execution_latency": execution_time * 1000,
                        "visible_size": float(visible_size),
                        "num_chunks": len(executions)
                    }
                },
                "slippage": slippage,
                "cost": float(total_cost) if total_cost > 0 else 0
            }
            
            if not overall_success and total_executed == Decimal("0"):
                result["error"] = "All iceberg chunks failed to execute"
            
            return result

    async def _pegged_execution(
        self, 
        order_info: Dict[str, Any], 
        max_slippage: float,
        execution_timeout: int
    ) -> Dict[str, Any]:
        """
        Pegged execution algorithm.
        
        Continuously adjusts order price to track the best bid/ask.
        """
        platform = order_info["platform"]
        symbol = order_info["symbol"]
        side = order_info["side"]
        order_type = order_info["order_type"]
        quantity = order_info["quantity"]
        price = order_info["price"]
        client_order_id = order_info["client_order_id"]
        
        # Only applicable for limit orders on platforms that support it
        if order_type != "LIMIT" or platform not in [PLATFORMS.BINANCE]:
            logger.warning("Pegged execution not supported for %s %s orders, falling back to immediate execution", 
                         platform, order_type)
            return await self._immediate_execution(order_info, max_slippage, execution_timeout)
        
        # Track execution details
        executions = []
        total_executed = Decimal("0")
        total_cost = Decimal("0")
        all_fills = []
        
        # Start execution
        execution_start = time.time()
        
        # Parameters for price pegging
        check_interval = 1.0  # seconds
        offset_ticks = 1  # number of ticks from best bid/ask
        current_order_id = None
        
        # Get tick size for the symbol
        symbol_info = await self.market_data.get_symbol_info(platform, symbol)
        tick_size = symbol_info.get("tick_size", Decimal("0.01"))
        
        while total_executed < quantity and (time.time() - execution_start) < execution_timeout:
            # Get current order book
            try:
                order_book = await self.market_data.get_order_book(platform, symbol)
                
                # Determine target price based on side
                if side == ORDER_SIDE.BUY:
                    best_bid = Decimal(str(order_book["bids"][0][0])) if order_book["bids"] else None
                    target_price = best_bid + (tick_size * offset_ticks) if best_bid else price
                else:  # SELL
                    best_ask = Decimal(str(order_book["asks"][0][0])) if order_book["asks"] else None
                    target_price = best_ask - (tick_size * offset_ticks) if best_ask else price
                
                # Ensure price is within allowed range
                if price:
                    # For buys, don't go above limit price
                    if side == ORDER_SIDE.BUY:
                        target_price = min(target_price, price)
                    # For sells, don't go below limit price
                    else:
                        target_price = max(target_price, price)
                
                logger.debug("Pegged execution - Current best %s: %s, Target price: %s", 
                            "bid" if side == ORDER_SIDE.BUY else "ask",
                            best_bid if side == ORDER_SIDE.BUY else best_ask,
                            target_price)
                
                # If we have an active order, check if we need to update it
                if current_order_id:
                    current_order = await self.get_order_status(platform, symbol, current_order_id)
                    
                    # If order is filled or partially filled, update tracking
                    if current_order["status"] in [ORDER_STATUS.FILLED, ORDER_STATUS.PARTIALLY_FILLED]:
                        executed_qty = current_order.get("executed_quantity", Decimal("0"))
                        avg_price = current_order.get("average_price")
                        
                        if executed_qty > Decimal("0") and avg_price:
                            newly_executed = executed_qty - total_executed
                            if newly_executed > Decimal("0"):
                                total_executed = executed_qty
                                total_cost += newly_executed * avg_price
                                
                                logger.info(
                                    "Pegged order partially executed: %s/%s %s at avg price %s", 
                                    total_executed, quantity, symbol, avg_price
                                )
                    
                    # If order is filled or we need to update the price, cancel and replace
                    if (current_order["status"] == ORDER_STATUS.FILLED or 
                        (current_order["status"] == ORDER_STATUS.OPEN and 
                         abs(Decimal(str(current_order["price"])) - target_price) > tick_size)):
                        
                        # Only cancel if order is open
                        if current_order["status"] == ORDER_STATUS.OPEN:
                            try:
                                await self.cancel_order(platform, symbol, current_order_id)
                                logger.debug("Cancelled pegged order %s to update price", current_order_id)
                            except Exception as e:
                                logger.warning("Error cancelling pegged order: %s", str(e))
                        
                        # Add to executions list if we had any fills
                        if current_order.get("executed_quantity", Decimal("0")) > Decimal("0"):
                            executions.append(current_order)
                            all_fills.extend(current_order.get("fills", []))
                        
                        current_order_id = None
                
                # If no active order and we still have quantity to execute, create a new one
                if not current_order_id and total_executed < quantity:
                    remaining_qty = quantity - total_executed
                    
                    # Create a new pegged order
                    new_order_info = order_info.copy()
                    new_order_info["client_order_id"] = f"{client_order_id}_peg_{len(executions)+1}"
                    new_order_info["quantity"] = remaining_qty
                    new_order_info["price"] = target_price
                    
                    # Create the order
                    result = await self._immediate_execution(new_order_info, max_slippage, 5)  # Short timeout for quick replacement
                    
                    if result.get("success", False):
                        current_order_id = result["order_info"]["order_id"]
                        logger.debug("Created new pegged order %s at price %s", current_order_id, target_price)
                    else:
                        logger.warning("Failed to create pegged order: %s", result.get("error"))
                
                # Wait before checking again
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error("Error in pegged execution loop: %s", str(e))
                await asyncio.sleep(check_interval)
        
        # Execution complete or timed out - cancel any remaining order
        if current_order_id:
            try:
                # Get final status before cancelling
                final_order = await self.get_order_status(platform, symbol, current_order_id)
                
                # Update execution tracking
                executed_qty = final_order.get("executed_quantity", Decimal("0"))
                avg_price = final_order.get("average_price")
                
                if executed_qty > Decimal("0") and avg_price:
                    newly_executed = executed_qty - total_executed
                    if newly_executed > Decimal("0"):
                        total_executed = executed_qty
                        total_cost += newly_executed * avg_price
                
                # Add to executions if it had any fills
                if executed_qty > Decimal("0"):
                    executions.append(final_order)
                    all_fills.extend(final_order.get("fills", []))
                
                # Cancel if still open
                if final_order["status"] == ORDER_STATUS.OPEN:
                    await self.cancel_order(platform, symbol, current_order_id)
                    logger.info("Cancelled final pegged order %s at end of execution", current_order_id)
            except Exception as e:
                logger.error("Error finalizing pegged execution: %s", str(e))
        
        # Calculate results
        avg_execution_price = total_cost / total_executed if total_executed > 0 else None
        execution_time = time.time() - execution_start
        
        # Calculate slippage
        slippage = None
        if price and avg_execution_price:
            if side == ORDER_SIDE.BUY:
                slippage = ((avg_execution_price - price) / price) * 100 if price > 0 else 0
            else:  # SELL
                slippage = ((price - avg_execution_price) / price) * 100 if price > 0 else 0
        
        # Determine overall success
        overall_success = total_executed > Decimal("0")
        
        # Update order info
        result = {
            "success": overall_success,
            "order_info": {
                "execution_algorithm": "pegged",
                "executed_quantity": total_executed,
                "remaining_quantity": quantity - total_executed,
                "average_price": avg_execution_price,
                "fills": all_fills,
                "execution_time": execution_time,
                "sub_orders": executions,
                "status": ORDER_STATUS.FILLED if total_executed >= quantity else 
                         ORDER_STATUS.PARTIALLY_FILLED if total_executed > 0 else 
                         ORDER_STATUS.REJECTED,
                "metadata": {
                    "slippage": slippage,
                    "execution_latency": execution_time * 1000,
                    "check_interval": check_interval,
                    "offset_ticks": offset_ticks,
                    "num_replacements": len(executions)
                }
            },
            "slippage": slippage,
            "cost": float(total_cost) if total_cost > 0 else 0
        }
        
        if not overall_success:
            result["error"] = "Failed to execute any quantity with pegged algorithm"
        
        return result

    async def _adaptive_execution(
        self, 
        order_info: Dict[str, Any], 
        max_slippage: float,
        execution_timeout: int
    ) -> Dict[str, Any]:
        """
        Adaptive execution algorithm.
        
        Analyzes market conditions and selects the most appropriate execution strategy.
        """
        platform = order_info["platform"]
        symbol = order_info["symbol"]
        side = order_info["side"]
        order_type = order_info["order_type"]
        quantity = order_info["quantity"]
        
        # Only proceed with adaptive execution for limit orders on supported platforms
        if order_type != "LIMIT" or platform not in [PLATFORMS.BINANCE]:
            logger.warning("Adaptive execution not fully supported for %s %s orders, using basic adaptation", 
                         platform, order_type)
        
        # Analyze market conditions to select best strategy
        try:
            # Get volatility
            volatility = await self.market_data.get_recent_volatility(platform, symbol)
            
            # Get volume
            volume = await self.market_data.get_recent_volume(platform, symbol)
            
            # Get depth
            depth = await self.market_data.get_order_book_depth(platform, symbol)
            
            # Get spread
            spread = await self.market_data.get_spread(platform, symbol)
            
            # Analyze conditions and select strategy
            if volatility < 0.5 and volume > 1000000 and depth > 10000:
                # High liquidity, low volatility - use TWAP for gradual execution
                logger.info("Adaptive execution selected TWAP strategy for %s %s order", side, symbol)
                selected_strategy = self._twap_execution
            elif volatility > 1.5 or spread > 0.1:
                # High volatility or wide spread - use Pegged to adapt to rapid price changes
                logger.info("Adaptive execution selected Pegged strategy for %s %s order", side, symbol)
                selected_strategy = self._pegged_execution
            elif quantity > 1000 and depth < 5000:
                # Large order in thin market - use Iceberg to hide size
                logger.info("Adaptive execution selected Iceberg strategy for %s %s order", side, symbol)
                selected_strategy = self._iceberg_execution
            elif volume > 5000000:
                # Very liquid market - use VWAP for optimal execution
                logger.info("Adaptive execution selected VWAP strategy for %s %s order", side, symbol)
                selected_strategy = self._vwap_execution
            else:
                # Default to immediate execution for most cases
                logger.info("Adaptive execution selected Immediate strategy for %s %s order", side, symbol)
                selected_strategy = self._immediate_execution
                
        except Exception as e:
            logger.error("Error analyzing market conditions for adaptive execution: %s, using immediate", str(e))
            selected_strategy = self._immediate_execution
        
        # Execute with selected strategy
        result = await selected_strategy(order_info, max_slippage, execution_timeout)
        
        # Update with adaptive metadata
        if "order_info" in result:
            original_algo = result["order_info"].get("execution_algorithm", "unknown")
            result["order_info"]["execution_algorithm"] = f"adaptive_{original_algo}"
            
            # Add adaptive selection metadata
            if "metadata" in result["order_info"]:
                result["order_info"]["metadata"]["adaptive_selection"] = {
                    "volatility": volatility if 'volatility' in locals() else None,
                    "volume": volume if 'volume' in locals() else None,
                    "depth": depth if 'depth' in locals() else None,
                    "spread": spread if 'spread' in locals() else None,
                    "selected_strategy": original_algo
                }
        
        return result

    async def _market_aware_execution(
        self, 
        order_info: Dict[str, Any], 
        max_slippage: float,
        execution_timeout: int
    ) -> Dict[str, Any]:
        """
        Market-aware execution algorithm.
        
        Monitors and adapts to real-time market conditions during execution.
        """
        platform = order_info["platform"]
        symbol = order_info["symbol"]
        side = order_info["side"]
        order_type = order_info["order_type"]
        quantity = order_info["quantity"]
        price = order_info["price"]
        client_order_id = order_info["client_order_id"]
        
        # Start with an initial strategy based on current conditions
        initial_result = await self._adaptive_execution(order_info, max_slippage, int(execution_timeout * 0.3))
        
        # If initial execution completed the order, return result
        if initial_result.get("success", False):
            initial_info = initial_result.get("order_info", {})
            if initial_info.get("status") == ORDER_STATUS.FILLED:
                # Update algorithm name
                if "order_info" in initial_result:
                    initial_result["order_info"]["execution_algorithm"] = "market_aware_full"
                return initial_result
        
        # If we got here, we need to continue execution
        executed_so_far = initial_result.get("order_info", {}).get("executed_quantity", Decimal("0"))
        remaining_qty = quantity - executed_so_far
        
        if remaining_qty <= Decimal("0"):
            # No remaining quantity, return initial result
            if "order_info" in initial_result:
                initial_result["order_info"]["execution_algorithm"] = "market_aware_complete"
            return initial_result
        
        # Track executions
        executions = [initial_result.get("order_info", {})] if executed_so_far > Decimal("0") else []
        total_executed = executed_so_far
        total_cost = (
            executed_so_far * initial_result.get("order_info", {}).get("average_price", Decimal("0"))
            if executed_so_far > Decimal("0") else Decimal("0")
        )
        all_fills = initial_result.get("order_info", {}).get("fills", [])
        
        # Calculate remaining time
        remaining_time = execution_timeout - (
            initial_result.get("order_info", {}).get("execution_time", 0)
            if "order_info" in initial_result else 0
        )
        
        if remaining_time <= 0:
            # No time left, return initial result
            if "order_info" in initial_result:
                initial_result["order_info"]["execution_algorithm"] = "market_aware_timeout"
            return initial_result
        
        # Determine next strategy based on updated market conditions and execution so far
        try:
            # Analyze new market conditions
            volatility = await self.market_data.get_recent_volatility(platform, symbol)
            market_price = await self.market_data.get_last_price(platform, symbol)
            
            # Determine price deviation from expectation
            price_deviation = abs((market_price - price) / price) * 100 if price else 0
            
            # Select strategy based on conditions
            if volatility > 2.0 or price_deviation > 1.0:
                # Market conditions have changed significantly, use aggressive execution
                logger.info("Market-aware execution detected high volatility (%.2f) or price deviation (%.2f%%), using immediate execution for remainder", 
                           volatility, price_deviation)
                next_strategy = self._immediate_execution
            elif executed_so_far < (quantity * Decimal("0.3")):
                # Initial execution was slow, try a different approach
                logger.info("Market-aware execution detected slow initial execution (%.2f%%), switching to iceberg", 
                           float(executed_so_far / quantity * 100))
                next_strategy = self._iceberg_execution
            else:
                # Continue with pegged execution for remainder
                logger.info("Market-aware execution continuing with pegged execution for remainder (%.2f%%)", 
                           float(remaining_qty / quantity * 100))
                next_strategy = self._pegged_execution
                
        except Exception as e:
            logger.error("Error in market-aware condition analysis: %s, using immediate for remainder", str(e))
            next_strategy = self._immediate_execution
        
        # Execute remainder with selected strategy
        remainder_order_info = order_info.copy()
        remainder_order_info["client_order_id"] = f"{client_order_id}_remainder"
        remainder_order_info["quantity"] = remaining_qty
        
        remainder_result = await next_strategy(remainder_order_info, max_slippage, int(remaining_time))
        
        # Merge results
        remainder_info = remainder_result.get("order_info", {})
        remainder_executed = remainder_info.get("executed_quantity", Decimal("0"))
        remainder_avg_price = remainder_info.get("average_price")
        
        if remainder_executed > Decimal("0") and remainder_avg_price:
            total_executed += remainder_executed
            total_cost += remainder_executed * remainder_avg_price
            executions.append(remainder_info)
            all_fills.extend(remainder_info.get("fills", []))
        
        # Calculate overall results
        avg_execution_price = total_cost / total_executed if total_executed > Decimal("0") else None
        
        # Calculate slippage
        slippage = None
        if price and avg_execution_price:
            if side == ORDER_SIDE.BUY:
                slippage = ((avg_execution_price - price) / price) * 100 if price > 0 else 0
            else:  # SELL
                slippage = ((price - avg_execution_price) / price) * 100 if price > 0 else 0
        
        # Determine overall success
        overall_success = total_executed > Decimal("0")
        
        # Update order info
        result = {
            "success": overall_success,
            "order_info": {
                "execution_algorithm": "market_aware",
                "executed_quantity": total_executed,
                "remaining_quantity": quantity - total_executed,
                "average_price": avg_execution_price,
                "fills": all_fills,
                "execution_time": (
                    (initial_result.get("order_info", {}).get("execution_time", 0) if "order_info" in initial_result else 0) +
                    (remainder_result.get("order_info", {}).get("execution_time", 0) if "order_info" in remainder_result else 0)
                ),
                "sub_orders": executions,
                "status": ORDER_STATUS.FILLED if total_executed >= quantity else 
                         ORDER_STATUS.PARTIALLY_FILLED if total_executed > 0 else 
                         ORDER_STATUS.REJECTED,
                "metadata": {
                    "slippage": slippage,
                    "initial_strategy": initial_result.get("order_info", {}).get("execution_algorithm", "unknown") if "order_info" in initial_result else "unknown",
                    "remainder_strategy": remainder_result.get("order_info", {}).get("execution_algorithm", "unknown") if "order_info" in remainder_result else "unknown",
                    "initial_executed_pct": float(executed_so_far / quantity * 100) if quantity > 0 else 0,
                    "volatility": volatility if 'volatility' in locals() else None,
                    "price_deviation": price_deviation if 'price_deviation' in locals() else None
                }
            },
            "slippage": slippage,
            "cost": float(total_cost) if total_cost > 0 else 0
        }
        
        if not overall_success:
            result["error"] = "Failed to execute any quantity with market-aware algorithm"
        
        return result

    async def _liquidity_seeking_execution(
        self, 
        order_info: Dict[str, Any], 
        max_slippage: float,
        execution_timeout: int
    ) -> Dict[str, Any]:
        """
        Liquidity-seeking execution algorithm.
        
        Actively searches for and targets available liquidity across the order book.
        """
        platform = order_info["platform"]
        symbol = order_info["symbol"]
        side = order_info["side"]
        order_type = order_info["order_type"]
        quantity = order_info["quantity"]
        price = order_info["price"]
        client_order_id = order_info["client_order_id"]
        
        # Only applicable for limit orders on platforms that support it
        if order_type != "LIMIT" or platform not in [PLATFORMS.BINANCE]:
            logger.warning("Liquidity-seeking execution not supported for %s %s orders, falling back to immediate execution", 
                         platform, order_type)
            return await self._immediate_execution(order_info, max_slippage, execution_timeout)
        
        # Start execution timer
        execution_start = time.time()
        
        # Track executions
        executions = []
        total_executed = Decimal("0")
        total_cost = Decimal("0")
        all_fills = []
        
        # Get symbol info for price precision
        symbol_info = await self.market_data.get_symbol_info(platform, symbol)
        tick_size = symbol_info.get("tick_size", Decimal("0.01"))
        
        # Main execution loop
        while total_executed < quantity and (time.time() - execution_start) < execution_timeout:
            # Identify liquidity pockets
            try:
                # Get detailed order book
                order_book = await self.market_data.get_order_book(platform, symbol, 20)  # Deeper depth
                
                if side == ORDER_SIDE.BUY:
                    # For buy orders, analyze ask side
                    asks = order_book.get("asks", [])
                    
                    if not asks:
                        logger.warning("No asks found in order book, waiting and retrying")
                        await asyncio.sleep(1)
                        continue
                    
                    # Find liquidity pockets (larger sizes, or multiple levels with similar prices)
                    liquidity_pockets = []
                    
                    # Simple approach: find levels with above average size
                    avg_size = sum(Decimal(str(size)) for _, size in asks) / len(asks)
                    threshold = avg_size * Decimal("1.5")
                    
                    for i, (level_price, level_size) in enumerate(asks):
                        level_price = Decimal(str(level_price))
                        level_size = Decimal(str(level_size))
                        
                        # Skip if price is above our limit
                        if price and level_price > price:
                            continue
                            
                        # Check if this level has above-average size
                        if level_size > threshold:
                            liquidity_pockets.append((level_price, level_size))
                            continue
                            
                        # Check if this level and next few levels have similar prices (liquidity wall)
                        if i < len(asks) - 2:
                            next_price = Decimal(str(asks[i+1][0]))
                            next_next_price = Decimal(str(asks[i+2][0]))
                            
                            # If prices are close together, consider it a liquidity pocket
                            if (next_price - level_price) < tick_size * 3 and (next_next_price - level_price) < tick_size * 5:
                                combined_size = level_size + Decimal(str(asks[i+1][1])) + Decimal(str(asks[i+2][1]))
                                liquidity_pockets.append((level_price, combined_size))
                    
                    # Sort by price (lowest first for buys)
                    liquidity_pockets.sort(key=lambda x: x[0])
                    
                else:  # SELL
                    # For sell orders, analyze bid side
                    bids = order_book.get("bids", [])
                    
                    if not bids:
                        logger.warning("No bids found in order book, waiting and retrying")
                        await asyncio.sleep(1)
                        continue
                    
                    # Find liquidity pockets (larger sizes, or multiple levels with similar prices)
                    liquidity_pockets = []
                    
                    # Simple approach: find levels with above average size
                    avg_size = sum(Decimal(str(size)) for _, size in bids) / len(bids)
                    threshold = avg_size * Decimal("1.5")
                    
                    for i, (level_price, level_size) in enumerate(bids):
                        level_price = Decimal(str(level_price))
                        level_size = Decimal(str(level_size))
                        
                        # Skip if price is below our limit
                        if price and level_price < price:
                            continue
                            
                        # Check if this level has above-average size
                        if level_size > threshold:
                            liquidity_pockets.append((level_price, level_size))
                            continue
                            
                        # Check if this level and next few levels have similar prices (liquidity wall)
                        if i < len(bids) - 2:
                            next_price = Decimal(str(bids[i+1][0]))
                            next_next_price = Decimal(str(bids[i+2][0]))
                            
                            # If prices are close together, consider it a liquidity pocket
                            if (level_price - next_price) < tick_size * 3 and (level_price - next_next_price) < tick_size * 5:
                                combined_size = level_size + Decimal(str(bids[i+1][1])) + Decimal(str(bids[i+2][1]))
                                liquidity_pockets.append((level_price, combined_size))
                    
                    # Sort by price (highest first for sells)
                    liquidity_pockets.sort(key=lambda x: x[0], reverse=True)
                
                # If no liquidity pockets found, use standard approach
                if not liquidity_pockets:
                    logger.info("No significant liquidity pockets found, using standard price")
                    target_price = price or (
                        Decimal(str(asks[0][0])) if side == ORDER_SIDE.BUY else Decimal(str(bids[0][0]))
                    )
                    target_size = min(quantity - total_executed, Decimal("10"))  # Small size to test market
                else:
                    # Select the best liquidity pocket to target
                    target_price, available_size = liquidity_pockets[0]
                    target_size = min(quantity - total_executed, available_size)
                    logger.info("Targeting liquidity pocket at price %s with estimated size %s", 
                               target_price, available_size)
                
                # Create order targeting identified liquidity
                pocket_order_info = order_info.copy()
                pocket_order_info["client_order_id"] = f"{client_order_id}_liquid_{len(executions)+1}"
                pocket_order_info["quantity"] = target_size
                pocket_order_info["price"] = target_price
                
                # Execute the order
                remaining_time = execution_timeout - (time.time() - execution_start)
                result = await self._immediate_execution(pocket_order_info, max_slippage, int(remaining_time))
                
                if result.get("success", False):
                    # Track execution details
                    pocket_order_info = result.get("order_info", {})
                    executed_qty = pocket_order_info.get("executed_quantity", Decimal("0"))
                    avg_price = pocket_order_info.get("average_price")
                    
                    if executed_qty > Decimal("0") and avg_price:
                        total_executed += executed_qty
                        total_cost += executed_qty * avg_price
                        executions.append(pocket_order_info)
                        all_fills.extend(pocket_order_info.get("fills", []))
                        
                        logger.info(
                            "Liquidity pocket order executed: %s/%s %s at avg price %s", 
                            total_executed, quantity, symbol, avg_price
                        )
                    
                    # If order not fully filled, wait for a bit before retrying
                    if pocket_order_info.get("status") != ORDER_STATUS.FILLED:
                        await asyncio.sleep(2)
                else:
                    logger.error("Liquidity pocket order failed: %s", result.get("error"))
                    # Wait before trying again
                    await asyncio.sleep(2)
                
            except Exception as e:
                logger.error("Error in liquidity-seeking execution: %s", str(e))
                await asyncio.sleep(2)
        
        # Calculate results
        avg_execution_price = total_cost / total_executed if total_executed > Decimal("0") else None
        execution_time = time.time() - execution_start
        
        # Calculate slippage
        slippage = None
        if price and avg_execution_price:
            if side == ORDER_SIDE.BUY:
                slippage = ((avg_execution_price - price) / price) * 100 if price > 0 else 0
            else:  # SELL
                slippage = ((price - avg_execution_price) / price) * 100 if price > 0 else 0
        
        # Determine overall success
        overall_success = total_executed > Decimal("0")
        
        # Update order info
        result = {
            "success": overall_success,
            "order_info": {
                "execution_algorithm": "liquidity_seeking",
                "executed_quantity": total_executed,
                "remaining_quantity": quantity - total_executed,
                "average_price": avg_execution_price,
                "fills": all_fills,
                "execution_time": execution_time,
                "sub_orders": executions,
                "status": ORDER_STATUS.FILLED if total_executed >= quantity else 
                         ORDER_STATUS.PARTIALLY_FILLED if total_executed > 0 else 
                         ORDER_STATUS.REJECTED,
                "metadata": {
                    "slippage": slippage,
                    "execution_latency": execution_time * 1000,
                    "num_orders": len(executions)
                }
            },
            "slippage": slippage,
            "cost": float(total_cost) if total_cost > 0 else 0
        }
        
        if not overall_success:
            result["error"] = "Failed to execute any quantity with liquidity-seeking algorithm"
        
        return result

    # Helper methods for execution
    def _update_execution_metrics(self, algorithm, slippage, execution_time, success, cost):
        """Update metrics for algorithm performance tracking."""
        if slippage is not None:
            self.execution_metrics["slippage_by_algorithm"][algorithm].append(slippage)
            
        if execution_time is not None:
            self.execution_metrics["execution_time_by_algorithm"][algorithm].append(execution_time)
            
        if success is not None:
            self.execution_metrics["success_rate_by_algorithm"][algorithm]["total"] += 1
            if success:
                self.execution_metrics["success_rate_by_algorithm"][algorithm]["success"] += 1
                
        if cost is not None:
            self.execution_metrics["cost_by_algorithm"][algorithm].append(cost)

    # Background tasks
    async def _track_active_orders(self):
        """Background task to track and update status of active orders."""
        while not self.stop_signal.is_set():
            try:
                # Get all active order IDs
                order_ids = list(self.active_orders.keys())
                
                for order_id in order_ids:
                    order = self.active_orders.get(order_id)
                    if not order:
                        continue
                        
                    platform = order["platform"]
                    symbol = order["symbol"]
                    
                    # Check if enough time has passed since last update
                    last_update = order.get("last_status_check")
                    now = time.time()
                    
                    # Only check orders every 5 seconds to avoid API rate limits
                    if last_update and now - last_update < 5:
                        continue
                        
                    try:
                        # Update order status
                        updated_order = await self.get_order_status(platform, symbol, order_id)
                        order["last_status_check"] = now
                        
                        # If order is no longer active, move to history
                        if updated_order["status"] not in [ORDER_STATUS.OPEN, ORDER_STATUS.PARTIALLY_FILLED]:
                            self.active_orders.pop(order_id, None)
                            self.order_history.append(updated_order)
                            
                            logger.info("Order %s final status: %s", order_id, updated_order["status"])
                    except Exception as e:
                        logger.error("Error updating order %s: %s", order_id, str(e))
                
                # Sleep to avoid excessive CPU usage
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.exception("Error in order tracking task: %s", str(e))
                await asyncio.sleep(5)

    async def _execution_optimizer(self):
        """Background task to analyze and optimize execution performance."""
        while not self.stop_signal.is_set():
            try:
                # Run optimization every 10 minutes
                await asyncio.sleep(600)
                
                # Analyze algorithm performance
                for algorithm, metrics in self.execution_metrics["slippage_by_algorithm"].items():
                    if len(metrics) >= 5:  # Only analyze with sufficient data
                        avg_slippage = sum(metrics) / len(metrics)
                        success_data = self.execution_metrics["success_rate_by_algorithm"][algorithm]
                        success_rate = (
                            success_data["success"] / success_data["total"] 
                            if success_data["total"] > 0 else 0
                        )
                        
                        logger.info(
                            "Execution algorithm %s performance: Avg slippage %.4f%%, Success rate %.2f%%",
                            algorithm, avg_slippage, success_rate * 100
                        )
                
                # Analyze platform performance
                for platform, metrics in self.platform_metrics.items():
                    if len(metrics["latency"]) >= 5:  # Only analyze with sufficient data
                        avg_latency = sum(metrics["latency"]) / len(metrics["latency"])
                        success_data = metrics["success_rate"]
                        success_rate = (
                            success_data["success"] / success_data["total"] 
                            if success_data["total"] > 0 else 0
                        )
                        
                        logger.info(
                            "Platform %s performance: Avg latency %.2fms, Success rate %.2f%%, Rate limit hits %d",
                            platform, avg_latency, success_rate * 100, metrics["rate_limit_hits"]
                        )
                
            except Exception as e:
                logger.exception("Error in execution optimizer task: %s", str(e))
                await asyncio.sleep(60)

    async def _order_reconciliation(self):
        """Background task to reconcile order status with platform."""
        while not self.stop_signal.is_set():
            try:
                # Run reconciliation every hour
                await asyncio.sleep(3600)
                
                # Get all open orders from platforms
                for platform in self.platform_handlers.keys():
                    try:
                        platform_orders = await self.get_open_orders(platform)
                        
                        # Find orders in platform not in our active_orders
                        platform_order_ids = {order["order_id"] for order in platform_orders}
                        our_order_ids = set(self.active_orders.keys())
                        
                        # Orders missing from our tracking
                        missing_orders = platform_order_ids - our_order_ids
                        
                        # Orders we think are active but platform doesn't
                        phantom_orders = our_order_ids - platform_order_ids
                        
                        # Add missing orders to our tracking
                        for order_id in missing_orders:
                            missing_order = next((o for o in platform_orders if o["order_id"] == order_id), None)
                            if missing_order:
                                symbol = missing_order["symbol"]
                                full_order = await self.get_order_status(platform, symbol, order_id)
                                self.active_orders[order_id] = full_order
                                logger.warning("Reconciliation: Added missing order %s to tracking", order_id)
                        
                        # Remove phantom orders from our tracking
                        for order_id in phantom_orders:
                            phantom_order = self.active_orders.get(order_id)
                            if phantom_order:
                                symbol = phantom_order["symbol"]
                                try:
                                    # Double-check order status before removing
                                    status = await self.get_order_status(platform, symbol, order_id)
                                    if status["status"] not in [ORDER_STATUS.OPEN, ORDER_STATUS.PARTIALLY_FILLED]:
                                        self.active_orders.pop(order_id, None)
                                        self.order_history.append(status)
                                        logger.warning("Reconciliation: Removed inactive order %s from tracking", order_id)
                                except Exception as e:
                                    # If we can't get the status, assume it's no longer active
                                    self.active_orders.pop(order_id, None)
                                    phantom_order["status"] = ORDER_STATUS.UNKNOWN
                                    self.order_history.append(phantom_order)
                                    logger.warning("Reconciliation: Removed phantom order %s (error: %s)", order_id, str(e))
                        
                        logger.info(
                            "Order reconciliation for %s: Found %d missing orders, %d phantom orders",
                            platform, len(missing_orders), len(phantom_orders)
                        )
                            
                    except Exception as e:
                        logger.error("Error reconciling orders for platform %s: %s", platform, str(e))
                
            except Exception as e:
                logger.exception("Error in order reconciliation task: %s", str(e))
                await asyncio.sleep(600)  # Retry sooner on error

    async def close(self):
        """Gracefully shut down the OrderManager."""
        logger.info("Shutting down OrderManager...")
        
        # Signal background tasks to stop
        self.stop_signal.set()
        
        # Wait for background tasks to complete
        if hasattr(self, 'bg_tasks') and self.bg_tasks:
            for task in self.bg_tasks:
                try:
                    task.cancel()
                except Exception:
                    pass
            
            # Wait for tasks to complete cancellation
            if self.bg_tasks:
                await asyncio.gather(*self.bg_tasks, return_exceptions=True)
        
        logger.info("OrderManager shutdown complete")
