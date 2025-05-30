#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Execution Engine Service

This module implements the Execution Engine Service, responsible for executing
trades on exchanges based on signals from the Brain Council Service.
"""

import os
import sys
import time
import uuid
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Set

from common.logger import get_logger
from common.metrics import MetricsCollector
from common.exceptions import (
    ConfigurationError, ServiceStartupError, ServiceShutdownError,
    ExecutionError, OrderError, OrderRejectedError, OrderTimeoutError,
    InsufficientFundsError, RiskLimitExceededError
)
from common.constants import (
    SIGNAL_TYPES, POSITION_DIRECTION, ORDER_TYPES, SUPPORTED_PLATFORMS
)
from execution_engine.order import Order
from execution_engine.position import Position
from execution_engine.exchange_client import ExchangeClient
from execution_engine.binance_client import BinanceClient
from execution_engine.deriv_client import DerivClient


class ExecutionEngineService:
    """
    Service for executing trades on exchanges.
    
    Handles order creation, submission, tracking, and management based on
    signals received from the Brain Council Service.
    """
    
    def __init__(self, config, loop=None, redis_client=None, db_client=None):
        """
        Initialize the Execution Engine Service.
        
        Args:
            config: System configuration
            loop: Event loop
            redis_client: Redis client for communication
            db_client: Database client
        """
        self.config = config
        self.loop = loop or asyncio.get_event_loop()
        self.redis_client = redis_client
        self.db_client = db_client
        self.logger = get_logger("ExecutionEngineService")
        self.metrics = MetricsCollector("execution_engine")
        
        self.running = False
        self.tasks = []
        self.exchange_clients = {}
        
        # Order and position tracking
        self.orders = {}  # order_id -> Order
        self.positions = {}  # exchange:symbol -> Position
        self.pending_signals = []
        self.signal_lock = asyncio.Lock()
        
        # Risk manager communication
        self.risk_check_queue = asyncio.Queue()
        self.risk_response_queues = {}
        
        # Trading enabled flag
        self.trading_enabled = self.config.get("trading.enabled", False)
        self.trading_mode = self.config.get("trading.mode", "paper")
        
        self.logger.info(f"Trading {'enabled' if self.trading_enabled else 'disabled'}, mode: {self.trading_mode}")
        
    async def start(self):
        """Start the Execution Engine Service."""
        self.logger.info("Starting Execution Engine Service")
        
        # Initialize exchange clients
        await self._initialize_exchange_clients()
        
        # Subscribe to signal channel
        await self._subscribe_to_signals()
        
        # Start order tracker
        self.tasks.append(asyncio.create_task(self._order_tracker()))
        
        # Start position monitor
        self.tasks.append(asyncio.create_task(self._position_monitor()))
        
        # Start signal processor
        self.tasks.append(asyncio.create_task(self._signal_processor()))
        
        # Start risk check processor
        self.tasks.append(asyncio.create_task(self._risk_check_processor()))
        
        self.running = True
        self.logger.info("Execution Engine Service started successfully")
        
    async def stop(self):
        """Stop the Execution Engine Service."""
        self.logger.info("Stopping Execution Engine Service")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        # Close exchange clients
        for exchange_name, client in self.exchange_clients.items():
            try:
                await client.close()
                self.logger.info(f"Closed {exchange_name} exchange client")
            except Exception as e:
                self.logger.error(f"Error closing {exchange_name} exchange client: {str(e)}")
                
        self.logger.info("Execution Engine Service stopped successfully")
        
    async def health_check(self) -> bool:
        """
        Perform a health check on the Execution Engine Service.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        if not self.running:
            return False
            
        # Check exchange clients
        for exchange_name, client in self.exchange_clients.items():
            if not await client.health_check():
                self.logger.warning(f"{exchange_name} exchange client health check failed")
                return False
                
        return True
        
    async def _initialize_exchange_clients(self):
        """Initialize exchange clients based on configuration."""
        exchange_configs = self.config.get("exchanges", {})
        
        for exchange_name, exchange_config in exchange_configs.items():
            if not exchange_config.get("enabled", False):
                continue
                
            try:
                self.logger.info(f"Initializing {exchange_name} exchange client")
                
                # Create exchange client
                if exchange_name == "binance":
                    client = BinanceClient(
                        config=exchange_config,
                        trading_mode=self.trading_mode,
                        redis_client=self.redis_client,
                        db_client=self.db_client,
                        loop=self.loop
                    )
                elif exchange_name == "deriv":
                    client = DerivClient(
                        config=exchange_config,
                        trading_mode=self.trading_mode,
                        redis_client=self.redis_client,
                        db_client=self.db_client,
                        loop=self.loop
                    )
                else:
                    self.logger.warning(f"Unsupported exchange: {exchange_name}")
                    continue
                    
                # Initialize the client
                await client.initialize()
                self.exchange_clients[exchange_name] = client
                
                # Load existing positions
                positions = await client.get_positions()
                for position in positions:
                    position_key = f"{exchange_name}:{position.symbol}"
                    self.positions[position_key] = position
                    self.logger.info(f"Loaded existing position: {position_key} ({position.direction})")
                    
                self.logger.info(f"{exchange_name} exchange client initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_name} exchange client: {str(e)}")
                
    async def _subscribe_to_signals(self):
        """Subscribe to the signals channel from Brain Council."""
        try:
            channel = "execution.signals"
            
            async def signal_callback(message):
                if self.running and message:
                    async with self.signal_lock:
                        self.pending_signals.append(message)
                        
            await self.redis_client.subscribe(channel, signal_callback)
            self.logger.info(f"Subscribed to signals on channel: {channel}")
            
        except Exception as e:
            self.logger.error(f"Error subscribing to signals: {str(e)}")
            raise ServiceStartupError(f"Failed to subscribe to signals: {str(e)}")
            
    async def _signal_processor(self):
        """Process signals from the Brain Council."""
        self.logger.info("Starting signal processor")
        
        while self.running:
            try:
                # Get signals to process
                signals_to_process = []
                async with self.signal_lock:
                    if self.pending_signals:
                        signals_to_process = self.pending_signals.copy()
                        self.pending_signals = []
                        
                # Process each signal
                for signal in signals_to_process:
                    await self._process_signal(signal)
                    
                # Sleep briefly
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                self.logger.info("Signal processor cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in signal processor: {str(e)}")
                await asyncio.sleep(1)
                
    async def _process_signal(self, signal: Dict[str, Any]):
        """
        Process a trading signal.
        
        Args:
            signal: Signal dictionary
        """
        try:
            # Validate signal
            if not self._validate_signal(signal):
                self.logger.warning(f"Invalid signal: {signal}")
                return
                
            # Extract signal details
            exchange_name = signal["exchange"]
            symbol = signal["symbol"]
            signal_type = signal["type"]
            direction = signal["direction"]
            
            # Check if exchange is supported
            if exchange_name not in self.exchange_clients:
                self.logger.warning(f"Unsupported exchange: {exchange_name}")
                return
                
            # Get exchange client
            exchange_client = self.exchange_clients[exchange_name]
            
            # Check if trading is enabled
            if not self.trading_enabled:
                self.logger.info(f"Trading disabled, signal would have been: {signal_type} {direction} {symbol} on {exchange_name}")
                return
                
            # Handle signal based on type
            if signal_type == SIGNAL_TYPES["ENTRY"]:
                await self._handle_entry_signal(signal, exchange_client)
            elif signal_type == SIGNAL_TYPES["EXIT"]:
                await self._handle_exit_signal(signal, exchange_client)
            elif signal_type == SIGNAL_TYPES["MODIFY"]:
                await self._handle_modify_signal(signal, exchange_client)
            else:
                self.logger.warning(f"Unknown signal type: {signal_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
            self.metrics.increment("signal_processing_errors")
            
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate a trading signal.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Check required fields
        required_fields = ["exchange", "symbol", "type", "direction"]
        for field in required_fields:
            if field not in signal:
                self.logger.warning(f"Signal missing required field: {field}")
                return False
                
        # Validate signal type
        if signal["type"] not in SIGNAL_TYPES.values():
            self.logger.warning(f"Invalid signal type: {signal['type']}")
            return False
            
        # Validate direction
        if signal["direction"] not in POSITION_DIRECTION.values():
            self.logger.warning(f"Invalid signal direction: {signal['direction']}")
            return False
            
        # Validate exchange
        if signal["exchange"] not in SUPPORTED_PLATFORMS:
            self.logger.warning(f"Unsupported exchange: {signal['exchange']}")
            return False
            
        return True
        
    async def _handle_entry_signal(self, signal: Dict[str, Any], exchange_client: ExchangeClient):
        """
        Handle an entry signal.
        
        Args:
            signal: Entry signal
            exchange_client: Exchange client
        """
        exchange_name = signal["exchange"]
        symbol = signal["symbol"]
        direction = signal["direction"]
        position_key = f"{exchange_name}:{symbol}"
        
        self.logger.info(f"Processing entry signal: {direction} {symbol} on {exchange_name}")
        
        try:
            # Check for existing position
            existing_position = self.positions.get(position_key)
            if existing_position:
                # If same direction, ignore
                if existing_position.direction == direction:
                    self.logger.info(f"Already have {direction} position for {symbol}, ignoring entry signal")
                    return
                    
                # If opposite direction, close existing and open new
                self.logger.info(f"Closing existing {existing_position.direction} position before opening {direction} position")
                await self._close_position(existing_position, exchange_client)
                
            # Determine position size
            position_size = await self._calculate_position_size(signal, exchange_client)
            if position_size <= 0:
                self.logger.warning(f"Calculated position size is {position_size}, skipping entry")
                return
                
            # Perform risk check
            risk_approved, risk_message = await self._perform_risk_check(signal, position_size)
            if not risk_approved:
                self.logger.warning(f"Risk check failed: {risk_message}")
                return
                
            # Create entry order
            order_id = str(uuid.uuid4())
            order = Order(
                id=order_id,
                exchange=exchange_name,
                symbol=symbol,
                order_type=ORDER_TYPES["MARKET"],  # Default to market order
                direction=direction,
                quantity=position_size,
                created_at=time.time(),
                status="pending",
                signal_id=signal.get("id")
            )
            
            # Submit the order
            result = await exchange_client.create_order(order)
            
            if result["success"]:
                # Update order
                order.status = "open"
                order.exchange_id = result.get("exchange_id")
                order.filled_quantity = result.get("filled_quantity", 0)
                order.fill_price = result.get("fill_price", 0)
                order.fees = result.get("fees", 0)
                order.updated_at = time.time()
                
                # Track order
                self.orders[order_id] = order
                
                # Create position
                position = Position(
                    exchange=exchange_name,
                    symbol=symbol,
                    direction=direction,
                    quantity=position_size,
                    entry_price=result.get("fill_price", 0),
                    entry_time=time.time(),
                    order_ids=[order_id]
                )
                
                # Set stop loss and take profit if configured
                if self.config.get("trading.stop_loss.enabled", True):
                    position.stop_loss = await self._calculate_stop_loss(position, signal)
                    
                if self.config.get("trading.take_profit.enabled", True):
                    position.take_profit = await self._calculate_take_profit(position, signal)
                    
                # Track position
                self.positions[position_key] = position
                
                # Update metrics
                self.metrics.increment("positions.opened")
                self.metrics.increment(f"positions.{direction}")
                
                # Publish position update
                await self._publish_position_update(position)
                
                self.logger.info(
                    f"Opened {direction} position for {symbol} on {exchange_name}: "
                    f"size={position_size}, price={position.entry_price}"
                )
                
            else:
                # Order failed
                order.status = "failed"
                order.error = result.get("error", "Unknown error")
                order.updated_at = time.time()
                
                self.logger.error(f"Failed to create order: {order.error}")
                self.metrics.increment("orders.failed")
                
        except InsufficientFundsError as e:
            self.logger.error(f"Insufficient funds: {str(e)}")
            self.metrics.increment("errors.insufficient_funds")
        except RiskLimitExceededError as e:
            self.logger.warning(f"Risk limit exceeded: {str(e)}")
            self.metrics.increment("errors.risk_limit_exceeded")
        except Exception as e:
            self.logger.error(f"Error handling entry signal: {str(e)}")
            self.metrics.increment("errors.entry_signal")
            
    async def _handle_exit_signal(self, signal: Dict[str, Any], exchange_client: ExchangeClient):
        """
        Handle an exit signal.
        
        Args:
            signal: Exit signal
            exchange_client: Exchange client
        """
        exchange_name = signal["exchange"]
        symbol = signal["symbol"]
        position_key = f"{exchange_name}:{symbol}"
        
        self.logger.info(f"Processing exit signal for {symbol} on {exchange_name}")
        
        try:
            # Check for existing position
            position = self.positions.get(position_key)
            if not position:
                self.logger.info(f"No position found for {symbol}, ignoring exit signal")
                return
                
            # Close the position
            await self._close_position(position, exchange_client)
            
        except Exception as e:
            self.logger.error(f"Error handling exit signal: {str(e)}")
            self.metrics.increment("errors.exit_signal")
            
    async def _handle_modify_signal(self, signal: Dict[str, Any], exchange_client: ExchangeClient):
        """
        Handle a modify signal (adjust stops, take profits, etc).
        
        Args:
            signal: Modify signal
            exchange_client: Exchange client
        """
        exchange_name = signal["exchange"]
        symbol = signal["symbol"]
        position_key = f"{exchange_name}:{symbol}"
        
        self.logger.info(f"Processing modify signal for {symbol} on {exchange_name}")
        
        try:
            # Check for existing position
            position = self.positions.get(position_key)
            if not position:
                self.logger.info(f"No position found for {symbol}, ignoring modify signal")
                return
                
            # Extract modification details
            modifications = signal.get("modifications", {})
            
            # Apply modifications
            modified = False
            
            # Update stop loss
            if "stop_loss" in modifications:
                new_stop = modifications["stop_loss"]
                old_stop = position.stop_loss
                
                # Update stop loss order if it exists
                if position.stop_loss_order_id:
                    order = self.orders.get(position.stop_loss_order_id)
                    if order and order.status == "open":
                        # Modify the order
                        result = await exchange_client.modify_order(
                            order_id=position.stop_loss_order_id,
                            new_price=new_stop
                        )
                        
                        if result["success"]:
                            position.stop_loss = new_stop
                            modified = True
                            self.logger.info(f"Modified stop loss from {old_stop} to {new_stop}")
                        else:
                            self.logger.error(f"Failed to modify stop loss order: {result.get('error')}")
                else:
                    # Just update the position
                    position.stop_loss = new_stop
                    modified = True
                    self.logger.info(f"Updated stop loss from {old_stop} to {new_stop}")
                    
            # Update take profit
            if "take_profit" in modifications:
                new_tp = modifications["take_profit"]
                old_tp = position.take_profit
                
                # Update take profit order if it exists
                if position.take_profit_order_id:
                    order = self.orders.get(position.take_profit_order_id)
                    if order and order.status == "open":
                        # Modify the order
                        result = await exchange_client.modify_order(
                            order_id=position.take_profit_order_id,
                            new_price=new_tp
                        )
                        
                        if result["success"]:
                            position.take_profit = new_tp
                            modified = True
                            self.logger.info(f"Modified take profit from {old_tp} to {new_tp}")
                        else:
                            self.logger.error(f"Failed to modify take profit order: {result.get('error')}")
                else:
                    # Just update the position
                    position.take_profit = new_tp
                    modified = True
                    self.logger.info(f"Updated take profit from {old_tp} to {new_tp}")
                    
            # Publish position update if modified
            if modified:
                await self._publish_position_update(position)
                
        except Exception as e:
            self.logger.error(f"Error handling modify signal: {str(e)}")
            self.metrics.increment("errors.modify_signal")
            
    async def _close_position(self, position: Position, exchange_client: ExchangeClient):
        """
        Close an existing position.
        
        Args:
            position: Position to close
            exchange_client: Exchange client
        """
        try:
            # Create exit order
            order_id = str(uuid.uuid4())
            exit_direction = POSITION_DIRECTION["SHORT"] if position.direction == POSITION_DIRECTION["LONG"] else POSITION_DIRECTION["LONG"]
            
            order = Order(
                id=order_id,
                exchange=position.exchange,
                symbol=position.symbol,
                order_type=ORDER_TYPES["MARKET"],
                direction=exit_direction,
                quantity=position.quantity,
                created_at=time.time(),
                status="pending"
            )
            
            # Submit the order
            result = await exchange_client.create_order(order, is_close=True)
            
            if result["success"]:
                # Update order
                order.status = "filled"
                order.exchange_id = result.get("exchange_id")
                order.filled_quantity = result.get("filled_quantity", 0)
                order.fill_price = result.get("fill_price", 0)
                order.fees = result.get("fees", 0)
                order.updated_at = time.time()
                
                # Track order
                self.orders[order_id] = order
                
                # Update position
                position.exit_price = result.get("fill_price", 0)
                position.exit_time = time.time()
                position.status = "closed"
                position.order_ids.append(order_id)
                
                # Calculate profit/loss
                if position.direction == POSITION_DIRECTION["LONG"]:
                    profit = (position.exit_price - position.entry_price) * position.quantity
                else:
                    profit = (position.entry_price - position.exit_price) * position.quantity
                    
                position.profit = profit
                
                # Remove position from tracking
                position_key = f"{position.exchange}:{position.symbol}"
                if position_key in self.positions:
                    del self.positions[position_key]
                    
                # Update metrics
                self.metrics.increment("positions.closed")
                self.metrics.observe("position.profit", profit)
                
                # Publish position update
                await self._publish_position_update(position)
                
                # Store position in database
                if self.db_client:
                    await self._store_position_history(position)
                    
                self.logger.info(
                    f"Closed {position.direction} position for {position.symbol} on {position.exchange}: "
                    f"profit={profit:.2f}, exit_price={position.exit_price}"
                )
                
            else:
                # Order failed
                order.status = "failed"
                order.error = result.get("error", "Unknown error")
                order.updated_at = time.time()
                
                self.logger.error(f"Failed to close position: {order.error}")
                self.metrics.increment("orders.failed")
                
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            self.metrics.increment("errors.close_position")
            
    async def _calculate_position_size(self, signal: Dict[str, Any], exchange_client: ExchangeClient) -> float:
        """
        Calculate position size based on configuration and available balance.
        
        Args:
            signal: Trading signal
            exchange_client: Exchange client
            
        Returns:
            float: Position size
        """
        try:
            exchange = signal["exchange"]
            symbol = signal["symbol"]
            
            # Get account balance
            balance = await exchange_client.get_balance()
            
            # Get position sizing config
            default_size = self.config.get("trading.default_position_size", 0.01)  # 1% of account
            max_size = self.config.get("trading.max_position_size", 0.1)  # 10% of account
            
            # Calculate base position size
            position_size = balance * default_size
            
            # If signal has confidence, adjust size
            if "confidence" in signal:
                confidence = signal["confidence"]
                # Scale position size by confidence
                position_size *= confidence
                
            # Apply maximum position size limit
            max_position_size = balance * max_size
            position_size = min(position_size, max_position_size)
            
            # Get minimum trade size for the symbol
            min_size = await exchange_client.get_min_trade_size(symbol)
            
            # Ensure position size is at least the minimum
            position_size = max(position_size, min_size)
            
            # Round to appropriate precision
            precision = await exchange_client.get_asset_precision(symbol)
            position_size = round(position_size, precision)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0
            
    async def _calculate_stop_loss(self, position: Position, signal: Dict[str, Any]) -> Optional[float]:
        """
        Calculate stop loss price for a position.
        
        Args:
            position: Position object
            signal: Signal that triggered the position
            
        Returns:
            float: Stop loss price or None if not applicable
        """
        try:
            # Get stop loss configuration
            stop_loss_enabled = self.config.get("trading.stop_loss.enabled", True)
            if not stop_loss_enabled:
                return None
                
            default_sl_percent = self.config.get("trading.stop_loss.default", 0.02)  # 2%
            
            # Use signal's stop loss if provided
            if "stop_loss" in signal:
                return signal["stop_loss"]
                
            # Calculate based on direction
            if position.direction == POSITION_DIRECTION["LONG"]:
                stop_loss = position.entry_price * (1 - default_sl_percent)
            else:
                stop_loss = position.entry_price * (1 + default_sl_percent)
                
            # Round to appropriate precision
            exchange_client = self.exchange_clients.get(position.exchange)
            if exchange_client:
                precision = await exchange_client.get_price_precision(position.symbol)
                stop_loss = round(stop_loss, precision)
                
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return None
            
    async def _calculate_take_profit(self, position: Position, signal: Dict[str, Any]) -> Optional[float]:
        """
        Calculate take profit price for a position.
        
        Args:
            position: Position object
            signal: Signal that triggered the position
            
        Returns:
            float: Take profit price or None if not applicable
        """
        try:
            # Get take profit configuration
            take_profit_enabled = self.config.get("trading.take_profit.enabled", True)
            if not take_profit_enabled:
                return None
                
            default_tp_percent = self.config.get("trading.take_profit.default", 0.03)  # 3%
            
            # Use signal's take profit if provided
            if "take_profit" in signal:
                return signal["take_profit"]
                
            # Calculate based on direction
            if position.direction == POSITION_DIRECTION["LONG"]:
                take_profit = position.entry_price * (1 + default_tp_percent)
            else:
                take_profit = position.entry_price * (1 - default_tp_percent)
                
            # Round to appropriate precision
            exchange_client = self.exchange_clients.get(position.exchange)
            if exchange_client:
                precision = await exchange_client.get_price_precision(position.symbol)
                take_profit = round(take_profit, precision)
                
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return None
            
    async def _perform_risk_check(self, signal: Dict[str, Any], position_size: float) -> Tuple[bool, str]:
        """
        Perform a risk check with the Risk Manager Service.
        
        Args:
            signal: Trading signal
            position_size: Calculated position size
            
        Returns:
            tuple: (approved, message)
        """
        try:
            # Create risk check request
            request_id = str(uuid.uuid4())
            response_queue = asyncio.Queue()
            
            self.risk_response_queues[request_id] = response_queue
            
            # Build request
            request = {
                "request_id": request_id,
                "timestamp": time.time(),
                "exchange": signal["exchange"],
                "symbol": signal["symbol"],
                "direction": signal["direction"],
                "position_size": position_size,
                "signal": signal
            }
            
            # Send request to risk manager
            await self.risk_check_queue.put(request)
            
            # Wait for response
            try:
                response = await asyncio.wait_for(response_queue.get(), timeout=5.0)
                
                # Clean up
                if request_id in self.risk_response_queues:
                    del self.risk_response_queues[request_id]
                    
                return response["approved"], response.get("message", "")
                
            except asyncio.TimeoutError:
                if request_id in self.risk_response_queues:
                    del self.risk_response_queues[request_id]
                return False, "Risk check timed out"
                
        except Exception as e:
            self.logger.error(f"Error performing risk check: {str(e)}")
            return False, f"Risk check error: {str(e)}"
            
    async def _risk_check_processor(self):
        """Process risk check requests and responses."""
        self.logger.info("Starting risk check processor")
        
        while self.running:
            try:
                # Check if we have any requests
                if self.risk_check_queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                    
                # Get the next request
                request = await self.risk_check_queue.get()
                
                # Get request details
                request_id = request["request_id"]
                exchange = request["exchange"]
                symbol = request["symbol"]
                direction = request["direction"]
                position_size = request["position_size"]
                
                # In this implementation, perform basic risk checks
                # In a real system, this would communicate with the Risk Manager Service
                
                # Check if we're already in a position for this symbol
                position_key = f"{exchange}:{symbol}"
                if position_key in self.positions:
                    # Only allow if it's in the opposite direction (to close)
                    existing_position = self.positions[position_key]
                    if existing_position.direction == direction:
                        response = {
                            "approved": False,
                            "message": f"Already have {direction} position for {symbol}"
                        }
                    else:
                        response = {"approved": True, "message": ""}
                else:
                    # Check total position count
                    max_positions = self.config.get("trading.risk_management.max_concurrent_trades", 3)
                    if len(self.positions) >= max_positions:
                        response = {
                            "approved": False,
                            "message": f"Maximum number of positions ({max_positions}) reached"
                        }
                    else:
                        response = {"approved": True, "message": ""}
                    
                # Send response
                if request_id in self.risk_response_queues:
                    await self.risk_response_queues[request_id].put(response)
                    
            except asyncio.CancelledError:
                self.logger.info("Risk check processor cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in risk check processor: {str(e)}")
                await asyncio.sleep(1)
                
    async def _order_tracker(self):
        """Track and update order status."""
        self.logger.info("Starting order tracker")
        
        while self.running:
            try:
                # Get list of open orders
                open_orders = [order for order in self.orders.values() if order.status in ["pending", "open"]]
                
                if not open_orders:
                    await asyncio.sleep(1)
                    continue
                    
                # Group orders by exchange
                exchange_orders = {}
                for order in open_orders:
                    if order.exchange not in exchange_orders:
                        exchange_orders[order.exchange] = []
                    exchange_orders[order.exchange].append(order)
                    
                # Update order status for each exchange
                for exchange_name, orders in exchange_orders.items():
                    if exchange_name not in self.exchange_clients:
                        continue
                        
                    exchange_client = self.exchange_clients[exchange_name]
                    
                    for order in orders:
                        try:
                            # Skip orders without exchange ID
                            if not order.exchange_id:
                                continue
                                
                            # Get order status
                            order_info = await exchange_client.get_order_status(order.id, order.exchange_id)
                            
                            # Update order
                            if order_info["status"] != order.status:
                                order.status = order_info["status"]
                                order.updated_at = time.time()
                                
                                if order.status == "filled":
                                    order.filled_quantity = order_info.get("filled_quantity", 0)
                                    order.fill_price = order_info.get("fill_price", 0)
                                    order.fees = order_info.get("fees", 0)
                                    
                                    self.logger.info(
                                        f"Order {order.id} filled: {order.symbol} {order.direction} "
                                        f"quantity={order.filled_quantity} price={order.fill_price}"
                                    )
                                    
                                elif order.status == "canceled":
                                    self.logger.info(f"Order {order.id} canceled: {order.symbol} {order.direction}")
                                elif order.status == "rejected":
                                    order.error = order_info.get("error", "Unknown rejection reason")
                                    self.logger.warning(f"Order {order.id} rejected: {order.error}")
                                    
                        except Exception as e:
                            self.logger.error(f"Error updating order {order.id} status: {str(e)}")
                            
                # Sleep before next update
                await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                self.logger.info("Order tracker cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in order tracker: {str(e)}")
                await asyncio.sleep(5)
                
    async def _position_monitor(self):
        """Monitor positions for stop loss/take profit and update metrics."""
        self.logger.info("Starting position monitor")
        
        while self.running:
            try:
                if not self.positions:
                    await asyncio.sleep(1)
                    continue
                    
                # Group positions by exchange
                exchange_positions = {}
                for position_key, position in self.positions.items():
                    if position.exchange not in exchange_positions:
                        exchange_positions[position.exchange] = []
                    exchange_positions[position.exchange].append(position)
                    
                # Update each position
                for exchange_name, positions in exchange_positions.items():
                    if exchange_name not in self.exchange_clients:
                        continue
                        
                    exchange_client = self.exchange_clients[exchange_name]
                    
                    # Get current prices
                    symbols = [position.symbol for position in positions]
                    prices = await exchange_client.get_current_prices(symbols)
                    
                    for position in positions:
                        if position.symbol not in prices:
                            continue
                            
                        current_price = prices[position.symbol]
                        
                        # Update unrealized profit/loss
                        if position.direction == POSITION_DIRECTION["LONG"]:
                            unrealized_profit = (current_price - position.entry_price) * position.quantity
                        else:
                            unrealized_profit = (position.entry_price - current_price) * position.quantity
                            
                        position.unrealized_profit = unrealized_profit
                        position.current_price = current_price
                        
                        # Update metrics
                        position_key = f"{position.exchange}:{position.symbol}"
                        self.metrics.set(f"position.unrealized_profit.{position_key}", unrealized_profit)
                        
                        # Check stop loss
                        if position.stop_loss is not None:
                            if (position.direction == POSITION_DIRECTION["LONG"] and current_price <= position.stop_loss) or \
                               (position.direction == POSITION_DIRECTION["SHORT"] and current_price >= position.stop_loss):
                                # Hit stop loss
                                self.logger.info(
                                    f"Stop loss hit for {position.symbol} {position.direction} position: "
                                    f"price={current_price}, stop={position.stop_loss}"
                                )
                                
                                # Close position
                                await self._close_position(position, exchange_client)
                                continue  # Skip rest of checks for this position
                                
                        # Check take profit
                        if position.take_profit is not None:
                            if (position.direction == POSITION_DIRECTION["LONG"] and current_price >= position.take_profit) or \
                               (position.direction == POSITION_DIRECTION["SHORT"] and current_price <= position.take_profit):
                                # Hit take profit
                                self.logger.info(
                                    f"Take profit hit for {position.symbol} {position.direction} position: "
                                    f"price={current_price}, take_profit={position.take_profit}"
                                )
                                
                                # Close position
                                await self._close_position(position, exchange_client)
                                continue  # Skip rest of checks for this position
                                
                        # Check trailing stop if enabled
                        if self.config.get("trading.stop_loss.trailing.enabled", True):
                            await self._update_trailing_stop(position, current_price, exchange_client)
                            
                # Sleep before next update
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                self.logger.info("Position monitor cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in position monitor: {str(e)}")
                await asyncio.sleep(5)
                
    async def _update_trailing_stop(self, position: Position, current_price: float, exchange_client: ExchangeClient):
        """
        Update trailing stop for a position.
        
        Args:
            position: Position to update
            current_price: Current market price
            exchange_client: Exchange client
        """
        try:
            # Get trailing stop configuration
            activation_percent = self.config.get("trading.stop_loss.trailing.activation", 0.01)  # 1%
            trailing_distance = self.config.get("trading.stop_loss.trailing.distance", 0.005)  # 0.5%
            
            # Check if we need to activate or update trailing stop
            if position.direction == POSITION_DIRECTION["LONG"]:
                # For long positions
                profit_percent = (current_price - position.entry_price) / position.entry_price
                
                # Check if profit meets activation threshold
                if profit_percent >= activation_percent:
                    # Calculate new stop loss level
                    new_stop = current_price * (1 - trailing_distance)
                    
                    # Only update if new stop is higher than current stop
                    if position.stop_loss is None or new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                        self.logger.info(
                            f"Updated trailing stop for {position.symbol} LONG position: {new_stop}"
                        )
                        
                        # Publish position update
                        await self._publish_position_update(position)
                        
            elif position.direction == POSITION_DIRECTION["SHORT"]:
                # For short positions
                profit_percent = (position.entry_price - current_price) / position.entry_price
                
                # Check if profit meets activation threshold
                if profit_percent >= activation_percent:
                    # Calculate new stop loss level
                    new_stop = current_price * (1 + trailing_distance)
                    
                    # Only update if new stop is lower than current stop
                    if position.stop_loss is None or new_stop < position.stop_loss:
                        position.stop_loss = new_stop
                        self.logger.info(
                            f"Updated trailing stop for {position.symbol} SHORT position: {new_stop}"
                        )
                        
                        # Publish position update
                        await self._publish_position_update(position)
                        
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {str(e)}")
            
    async def _publish_position_update(self, position: Position):
        """
        Publish position update to Redis.
        
        Args:
            position: Position to publish
        """
        try:
            # Create position update message
            update = {
                "exchange": position.exchange,
                "symbol": position.symbol,
                "direction": position.direction,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "entry_time": position.entry_time,
                "current_price": position.current_price,
                "unrealized_profit": position.unrealized_profit,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "status": position.status,
                "timestamp": time.time()
            }
            
            if position.status == "closed":
                update["exit_price"] = position.exit_price
                update["exit_time"] = position.exit_time
                update["profit"] = position.profit
                
            # Publish update
            channel = "execution.position_update"
            await self.redis_client.publish(channel, update)
            
        except Exception as e:
            self.logger.error(f"Error publishing position update: {str(e)}")
            
    async def _store_position_history(self, position: Position):
        """
        Store closed position in database.
        
        Args:
            position: Closed position
        """
        if not self.db_client or position.status != "closed":
            return
            
        try:
            # Store position history
            query = """
            INSERT INTO position_history (
                exchange, symbol, direction, quantity, entry_price, entry_time,
                exit_price, exit_time, profit, status
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
            )
            """
            
            await self.db_client.execute(
                query,
                position.exchange,
                position.symbol,
                position.direction,
                position.quantity,
                position.entry_price,
                position.entry_time,
                position.exit_price,
                position.exit_time,
                position.profit,
                position.status
            )
            
            self.logger.info(f"Stored position history for {position.symbol}")
            
            # Update signal results if we have a signal ID
            for order_id in position.order_ids:
                order = self.orders.get(order_id)
                if order and order.signal_id:
                    await self._update_signal_result(order.signal_id, position)
                    
        except Exception as e:
            self.logger.error(f"Error storing position history: {str(e)}")
            
    async def _update_signal_result(self, signal_id: str, position: Position):
        """
        Update signal result in database.
        
        Args:
            signal_id: Signal ID
            position: Position resulting from the signal
        """
        if not self.db_client:
            return
            
        try:
            # Get profit percentage
            if position.entry_price > 0:
                profit_percent = (position.exit_price - position.entry_price) / position.entry_price
                if position.direction == POSITION_DIRECTION["SHORT"]:
                    profit_percent = -profit_percent
            else:
                profit_percent = 0
                
            # Determine result
            result = "success" if position.profit > 0 else "failure"
            
            # Update signal result
            query = """
            INSERT INTO signal_results (
                signal_id, timestamp, exchange, symbol, direction,
                profit, profit_percent, entry_price, exit_price, result
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
            )
            """
            
            await self.db_client.execute(
                query,
                signal_id,
                time.time(),
                position.exchange,
                position.symbol,
                position.direction,
                position.profit,
                profit_percent,
                position.entry_price,
                position.exit_price,
                result
            )
            
            self.logger.info(f"Updated result for signal {signal_id}: {result}, profit={position.profit:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating signal result: {str(e)}")


async def create_app(config: Dict[str, Any]) -> ExecutionEngineService:
    """
    Create and configure an Execution Engine Service instance.

    Args:
        config: Application configuration

    Returns:
        Configured ExecutionEngineService instance
    """
    from common.redis_client import RedisClient
    from common.db_client import get_db_client

    # Initialize Redis client
    redis_client = RedisClient(
        host=config.get("redis.host", "localhost"),
        port=config.get("redis.port", 6379),
        db=config.get("redis.db", 0),
        password=config.get("redis.password", None)
    )

    # Initialize database client
    db_client = await get_db_client(
        db_type=config.get("database.type", "postgresql"),
        host=config.get("database.host", "localhost"),
        port=config.get("database.port", 5432),
        username=config.get("database.username", "postgres"),
        password=config.get("database.password", ""),
        database=config.get("database.database", "quantumspectre")
    )

    # Create and return service
    return ExecutionEngineService(
        config=config,
        redis_client=redis_client,
        db_client=db_client
    )
