#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Position Management Module

This module provides position management functionality for the QuantumSpectre Elite Trading System.
It handles position creation, tracking, and management.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable

from common.constants import PositionSide, PositionStatus
from common.exceptions import PositionError


@dataclass
class Position:
    """
    Represents a trading position in the system.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    exchange: str = ""
    symbol: str = ""
    side: str = PositionSide.LONG.value
    entry_price: float = 0.0
    current_price: float = 0.0
    quantity: float = 0.0
    status: str = PositionStatus.OPEN.value
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees: float = 0.0
    fee_currency: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_open(self) -> bool:
        """Check if the position is open."""
        return self.status == PositionStatus.OPEN.value
    
    def is_closed(self) -> bool:
        """Check if the position is closed."""
        return self.status == PositionStatus.CLOSED.value
    
    def is_partially_closed(self) -> bool:
        """Check if the position is partially closed."""
        return self.status == PositionStatus.PARTIALLY_CLOSED.value
    
    def update_price(self, price: float) -> None:
        """
        Update the current price and unrealized PnL.
        
        Args:
            price: The new current price
        """
        self.current_price = price
        self.updated_at = time.time()
        
        # Calculate unrealized PnL
        if self.side == PositionSide.LONG.value:
            self.unrealized_pnl = (price - self.entry_price) * self.quantity
        else:  # Short position
            self.unrealized_pnl = (self.entry_price - price) * self.quantity
    
    def close(self, price: float, quantity: Optional[float] = None) -> float:
        """
        Close the position or a portion of it.
        
        Args:
            price: The closing price
            quantity: The quantity to close (None for full close)
            
        Returns:
            The realized PnL
        """
        close_quantity = quantity if quantity is not None else self.quantity
        
        if close_quantity > self.quantity:
            raise PositionError(f"Cannot close more than position size: {close_quantity} > {self.quantity}")
        
        # Calculate realized PnL
        if self.side == PositionSide.LONG.value:
            realized_pnl = (price - self.entry_price) * close_quantity
        else:  # Short position
            realized_pnl = (self.entry_price - price) * close_quantity
        
        # Update position
        if quantity is None or close_quantity >= self.quantity:
            # Full close
            self.quantity = 0
            self.status = PositionStatus.CLOSED.value
            self.closed_at = time.time()
        else:
            # Partial close
            self.quantity -= close_quantity
            self.status = PositionStatus.PARTIALLY_CLOSED.value
        
        self.realized_pnl += realized_pnl
        self.updated_at = time.time()
        
        return realized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary representation."""
        return {
            "id": self.id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "quantity": self.quantity,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "fees": self.fees,
            "fee_currency": self.fee_currency,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create position from dictionary representation."""
        return cls(**data)


class PositionManager:
    """
    Manages positions across different exchanges.
    """
    
    def __init__(self):
        """Initialize the position manager."""
        self.positions = {}
        self.position_callbacks = {}
    
    async def create_position(self, 
                             exchange: str,
                             symbol: str,
                             side: str,
                             entry_price: float,
                             quantity: float,
                             stop_loss: Optional[float] = None,
                             take_profit: Optional[float] = None,
                             trailing_stop: Optional[float] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> Position:
        """
        Create a new position.
        
        Args:
            exchange: Exchange where the position is held
            symbol: Trading symbol
            side: Position side (long or short)
            entry_price: Entry price
            quantity: Position size
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            trailing_stop: Optional trailing stop distance
            metadata: Additional position metadata
            
        Returns:
            The created position
        """
        # Validate position parameters
        self._validate_position_params(
            exchange, symbol, side, entry_price, quantity
        )
        
        # Create position object
        position = Position(
            exchange=exchange,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            metadata=metadata or {}
        )
        
        # Store position
        self.positions[position.id] = position
        
        return position
    
    def _validate_position_params(self,
                                 exchange: str,
                                 symbol: str,
                                 side: str,
                                 entry_price: float,
                                 quantity: float) -> None:
        """
        Validate position parameters.
        
        Args:
            exchange: Exchange where the position is held
            symbol: Trading symbol
            side: Position side (long or short)
            entry_price: Entry price
            quantity: Position size
            
        Raises:
            PositionError: If position parameters are invalid
        """
        # Check exchange
        if not exchange:
            raise PositionError("Exchange is required")
        
        # Check symbol
        if not symbol:
            raise PositionError("Symbol is required")
        
        # Check side
        valid_sides = [side.value for side in PositionSide]
        if side not in valid_sides:
            raise PositionError(f"Invalid position side: {side}")
        
        # Check entry price
        if entry_price <= 0:
            raise PositionError(f"Invalid entry price: {entry_price}")
        
        # Check quantity
        if quantity <= 0:
            raise PositionError(f"Invalid quantity: {quantity}")
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Get a position by ID.
        
        Args:
            position_id: ID of the position to get
            
        Returns:
            The position, or None if not found
        """
        return self.positions.get(position_id)
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get all open positions.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of open positions
        """
        open_positions = [
            position for position in self.positions.values() 
            if position.is_open() or position.is_partially_closed()
        ]
        
        if symbol:
            open_positions = [
                position for position in open_positions 
                if position.symbol == symbol
            ]
        
        return open_positions
    
    def get_closed_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """
        Get all closed positions.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of closed positions
        """
        closed_positions = [
            position for position in self.positions.values() 
            if position.is_closed()
        ]
        
        if symbol:
            closed_positions = [
                position for position in closed_positions 
                if position.symbol == symbol
            ]
        
        return closed_positions
    
    def register_position_callback(self, position_id: str, callback: Callable[[Position], None]) -> None:
        """
        Register a callback for position updates.
        
        Args:
            position_id: ID of the position to register callback for
            callback: Callback function to call when position is updated
        """
        if position_id not in self.position_callbacks:
            self.position_callbacks[position_id] = []
        
        self.position_callbacks[position_id].append(callback)
    
    async def _trigger_position_callbacks(self, position: Position) -> None:
        """
        Trigger callbacks for a position.
        
        Args:
            position: The updated position
        """
        callbacks = self.position_callbacks.get(position.id, [])
        
        for callback in callbacks:
            try:
                callback(position)
            except Exception as e:
                # Log error but continue with other callbacks
                print(f"Error in position callback: {e}")
    
    async def update_positions(self, prices: Dict[str, float]) -> None:
        """
        Update positions with new prices.
        
        Args:
            prices: Dictionary of symbol -> price
        """
        for position in self.positions.values():
            if position.is_open() or position.is_partially_closed():
                if position.symbol in prices:
                    price = prices[position.symbol]
                    position.update_price(price)
                    
                    # Check stop loss
                    if position.stop_loss is not None:
                        if (position.side == PositionSide.LONG.value and price <= position.stop_loss) or \
                           (position.side == PositionSide.SHORT.value and price >= position.stop_loss):
                            await self.close_position(position.id, price)
                            continue
                    
                    # Check take profit
                    if position.take_profit is not None:
                        if (position.side == PositionSide.LONG.value and price >= position.take_profit) or \
                           (position.side == PositionSide.SHORT.value and price <= position.take_profit):
                            await self.close_position(position.id, price)
                            continue
                    
                    # Update trailing stop if needed
                    if position.trailing_stop is not None:
                        # Implementation would depend on trailing stop logic
                        pass
                    
                    await self._trigger_position_callbacks(position)
    
    async def close_position(self, position_id: str, price: float, quantity: Optional[float] = None) -> float:
        """
        Close a position or a portion of it.
        
        Args:
            position_id: ID of the position to close
            price: The closing price
            quantity: The quantity to close (None for full close)
            
        Returns:
            The realized PnL
            
        Raises:
            PositionError: If the position cannot be closed
        """
        if position_id not in self.positions:
            raise PositionError(f"Position not found: {position_id}")
        
        position = self.positions[position_id]
        
        if position.is_closed():
            raise PositionError(f"Position already closed: {position_id}")
        
        realized_pnl = position.close(price, quantity)
        
        await self._trigger_position_callbacks(position)
        
        return realized_pnl