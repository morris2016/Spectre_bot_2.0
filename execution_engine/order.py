#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Order Management Module

This module provides order management functionality for the QuantumSpectre Elite Trading System.
It handles order creation, validation, submission, and tracking.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable

from common.constants import OrderType, OrderSide, OrderStatus, TimeInForce
from common.exceptions import OrderValidationError, OrderExecutionError
from common.utils import TradingMode


@dataclass
class Order:
    """
    Represents a trading order in the system.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    exchange: str = ""
    symbol: str = ""
    order_type: str = OrderType.MARKET.value
    side: str = OrderSide.BUY.value
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = TimeInForce.GTC.value
    status: str = OrderStatus.NEW.value
    client_order_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    fees: float = 0.0
    fee_currency: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if the order is still active."""
        return self.status in [
            OrderStatus.NEW.value,
            OrderStatus.PARTIALLY_FILLED.value
        ]
    
    def is_complete(self) -> bool:
        """Check if the order is complete."""
        return self.status == OrderStatus.FILLED.value
    
    def is_canceled(self) -> bool:
        """Check if the order was canceled."""
        return self.status == OrderStatus.CANCELED.value
    
    def is_rejected(self) -> bool:
        """Check if the order was rejected."""
        return self.status == OrderStatus.REJECTED.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary representation."""
        return {
            "id": self.id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "order_type": self.order_type,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "status": self.status,
            "client_order_id": self.client_order_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "fees": self.fees,
            "fee_currency": self.fee_currency,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create order from dictionary representation."""
        return cls(**data)


class OrderManager:
    """
    Manages orders across different exchanges.
    """
    
    def __init__(self):
        """Initialize the order manager."""
        self.orders = {}
        self.order_callbacks = {}
        self.trading_mode = TradingMode()
    
    async def create_order(self, 
                          exchange: str,
                          symbol: str,
                          order_type: str,
                          side: str,
                          quantity: float,
                          price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          time_in_force: str = TimeInForce.GTC.value,
                          client_order_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Order:
        """
        Create a new order.
        
        Args:
            exchange: Exchange to place the order on
            symbol: Trading symbol
            order_type: Type of order (market, limit, etc.)
            side: Order side (buy or sell)
            quantity: Order quantity
            price: Order price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            client_order_id: Optional client-specified order ID
            metadata: Additional order metadata
            
        Returns:
            The created order
        """
        # Validate order parameters
        self._validate_order_params(
            exchange, symbol, order_type, side, quantity, price, stop_price
        )
        
        # Create order object
        order = Order(
            exchange=exchange,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            metadata=metadata or {}
        )
        
        # Store order
        self.orders[order.id] = order
        
        return order
    
    def _validate_order_params(self,
                              exchange: str,
                              symbol: str,
                              order_type: str,
                              side: str,
                              quantity: float,
                              price: Optional[float],
                              stop_price: Optional[float]) -> None:
        """
        Validate order parameters.
        
        Args:
            exchange: Exchange to place the order on
            symbol: Trading symbol
            order_type: Type of order (market, limit, etc.)
            side: Order side (buy or sell)
            quantity: Order quantity
            price: Order price
            stop_price: Stop price
            
        Raises:
            OrderValidationError: If order parameters are invalid
        """
        # Check exchange
        if not exchange:
            raise OrderValidationError("Exchange is required")
        
        # Check symbol
        if not symbol:
            raise OrderValidationError("Symbol is required")
        
        # Check order type
        valid_order_types = [ot.value for ot in OrderType]
        if order_type not in valid_order_types:
            raise OrderValidationError(f"Invalid order type: {order_type}")
        
        # Check side
        valid_sides = [side.value for side in OrderSide]
        if side not in valid_sides:
            raise OrderValidationError(f"Invalid order side: {side}")
        
        # Check quantity
        if quantity <= 0:
            raise OrderValidationError(f"Invalid quantity: {quantity}")
        
        # Check price for limit orders
        if order_type == OrderType.LIMIT.value and price is None:
            raise OrderValidationError("Price is required for limit orders")
        
        # Check stop price for stop orders
        if order_type in [OrderType.STOP_MARKET.value, OrderType.STOP_LIMIT.value] and stop_price is None:
            raise OrderValidationError("Stop price is required for stop orders")
    
    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the exchange.
        
        Args:
            order: The order to submit
            
        Returns:
            The updated order
        """
        # In paper trading mode, simulate order execution
        if self.trading_mode.get_mode() == TradingMode.PAPER:
            return await self._simulate_order_execution(order)
        
        # In live mode, submit to exchange
        # This would be implemented with actual exchange API calls
        raise NotImplementedError("Live order submission not implemented")
    
    async def _simulate_order_execution(self, order: Order) -> Order:
        """
        Simulate order execution for paper trading.
        
        Args:
            order: The order to simulate
            
        Returns:
            The updated order
        """
        # Simulate a delay
        await asyncio.sleep(0.5)
        
        # Update order status
        order.status = OrderStatus.FILLED.value
        order.filled_quantity = order.quantity
        
        # For market orders, use the current price
        if order.order_type == OrderType.MARKET.value:
            # In a real implementation, this would fetch the current market price
            order.average_fill_price = 0.0  # Placeholder
        else:
            order.average_fill_price = order.price
        
        order.updated_at = time.time()
        
        # Trigger callbacks
        await self._trigger_order_callbacks(order)
        
        return order
    
    async def cancel_order(self, order_id: str) -> Order:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            The updated order
            
        Raises:
            OrderExecutionError: If the order cannot be canceled
        """
        if order_id not in self.orders:
            raise OrderExecutionError(f"Order not found: {order_id}")
        
        order = self.orders[order_id]
        
        # Check if order can be canceled
        if not order.is_active():
            raise OrderExecutionError(f"Cannot cancel order with status: {order.status}")
        
        # In paper trading mode, simulate cancellation
        if self.trading_mode.get_mode() == TradingMode.PAPER:
            order.status = OrderStatus.CANCELED.value
            order.updated_at = time.time()
            await self._trigger_order_callbacks(order)
            return order
        
        # In live mode, submit cancellation to exchange
        # This would be implemented with actual exchange API calls
        raise NotImplementedError("Live order cancellation not implemented")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.
        
        Args:
            order_id: ID of the order to get
            
        Returns:
            The order, or None if not found
        """
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all active orders.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of active orders
        """
        active_orders = [
            order for order in self.orders.values() 
            if order.is_active()
        ]
        
        if symbol:
            active_orders = [
                order for order in active_orders 
                if order.symbol == symbol
            ]
        
        return active_orders
    
    def register_order_callback(self, order_id: str, callback: Callable[[Order], None]) -> None:
        """
        Register a callback for order updates.
        
        Args:
            order_id: ID of the order to register callback for
            callback: Callback function to call when order is updated
        """
        if order_id not in self.order_callbacks:
            self.order_callbacks[order_id] = []
        
        self.order_callbacks[order_id].append(callback)
    
    async def _trigger_order_callbacks(self, order: Order) -> None:
        """
        Trigger callbacks for an order.
        
        Args:
            order: The updated order
        """
        callbacks = self.order_callbacks.get(order.id, [])
        
        for callback in callbacks:
            try:
                callback(order)
            except Exception as e:
                # Log error but continue with other callbacks
                print(f"Error in order callback: {e}")