#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Position Manager

This module handles all aspects of position management including:
- Position tracking and state management
- Entry/exit execution with partial positions
- Position sizing based on risk parameters
- Position performance monitoring
- Advanced position management strategies
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import uuid
import time
import logging

# Internal imports
from common.logger import get_logger
from common.utils import (
    round_to_tick_size,
    dict_to_namedtuple,
    calculate_liquidation_price,
)
from common.constants import (
    PositionStatus, OrderType, PositionSide, TimeInForce,
    DEFAULT_STOP_LOSS_MULTIPLIER, DEFAULT_TAKE_PROFIT_MULTIPLIER,
    DEFAULT_MAX_RISK_PER_TRADE, PARTIAL_CLOSE_LEVELS
)
from common.metrics import MetricsCollector
from common.exceptions import (
    PositionError, PositionExecutionError, PositionSizingError,
    InvalidPositionStateError, MaxDrawdownExceededError,
    MarginCallError, PositionLiquidationError
)
from data_storage.models.strategy_data import PositionModel
from execution_engine.order_manager import OrderManager


@dataclass
class Position:
    """Data class representing a trading position with all relevant metadata."""
    position_id: str
    symbol: str
    side: PositionSide
    entry_price: float = 0.0
    current_price: float = 0.0
    quantity: float = 0.0
    remaining_quantity: float = 0.0
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    status: PositionStatus = PositionStatus.PENDING
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    liquidation_price: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    fees_paid: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    strategy_id: Optional[str] = None
    brain_id: Optional[str] = None
    entry_orders: List[Dict] = field(default_factory=list)
    exit_orders: List[Dict] = field(default_factory=list)
    partial_exits: List[Dict] = field(default_factory=list)
    risk_amount: float = 0.0
    platform: str = ""
    leverage: float = 1.0
    margin_type: str = "ISOLATED"
    collateral: float = 0.0
    
    def update_unrealized_pnl(self) -> float:
        """Calculate and update the current unrealized PnL."""
        if self.status not in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]:
            return 0.0
            
        if self.remaining_quantity <= 0 or self.entry_price <= 0:
            return 0.0
            
        multiplier = 1 if self.side == PositionSide.LONG else -1
        price_diff = (self.current_price - self.entry_price) * multiplier
        
        # Apply leverage for margin trading
        self.unrealized_pnl = price_diff * self.remaining_quantity * self.leverage
        
        # Update max profit/drawdown metrics
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        
        drawdown = self.max_profit - self.unrealized_pnl
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            
        return self.unrealized_pnl
        
    def get_roi_percentage(self) -> float:
        """Calculate ROI as a percentage of initial risk."""
        if self.risk_amount <= 0:
            return 0.0
            
        total_pnl = self.realized_pnl + self.unrealized_pnl
        return (total_pnl / self.risk_amount) * 100
        
    def get_r_multiple(self) -> float:
        """Calculate the R-multiple (profit/loss as a multiple of initial risk)."""
        if self.risk_amount <= 0:
            return 0.0
            
        total_pnl = self.realized_pnl + self.unrealized_pnl
        return total_pnl / self.risk_amount

    def get_duration(self) -> Optional[timedelta]:
        """Calculate the position duration."""
        if self.entry_time is None:
            return None
            
        end_time = self.exit_time if self.exit_time else datetime.utcnow()
        return end_time - self.entry_time
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'quantity': self.quantity,
            'remaining_quantity': self.remaining_quantity,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'status': self.status.value,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'liquidation_price': self.liquidation_price,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'max_drawdown': self.max_drawdown,
            'max_profit': self.max_profit,
            'fees_paid': self.fees_paid,
            'metadata': self.metadata,
            'strategy_id': self.strategy_id,
            'brain_id': self.brain_id,
            'entry_orders': self.entry_orders,
            'exit_orders': self.exit_orders,
            'partial_exits': self.partial_exits,
            'risk_amount': self.risk_amount,
            'platform': self.platform,
            'leverage': self.leverage,
            'margin_type': self.margin_type,
            'collateral': self.collateral,
            'r_multiple': self.get_r_multiple(),
            'roi_percentage': self.get_roi_percentage(),
            'duration': str(self.get_duration()) if self.get_duration() else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create a Position object from a dictionary."""
        # Handle enum conversions
        if 'side' in data and isinstance(data['side'], str):
            data['side'] = PositionSide(data['side'])
            
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = PositionStatus(data['status'])
            
        # Handle datetime conversions
        if 'entry_time' in data and data['entry_time'] and isinstance(data['entry_time'], str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
            
        if 'exit_time' in data and data['exit_time'] and isinstance(data['exit_time'], str):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
            
        # Remove calculated fields that aren't in the constructor
        data.pop('r_multiple', None)
        data.pop('roi_percentage', None)
        data.pop('duration', None)
        
        return cls(**data)


class PositionManager:
    """
    Manages all aspects of position creation, tracking, and lifecycle management.
    
    This class handles:
    1. Position sizing based on risk parameters
    2. Position entry and exit execution
    3. Stop loss and take profit management
    4. Position state tracking and updates
    5. Partial position management (scaling in/out)
    6. Position performance metrics
    """
    
    def __init__(self, order_manager: OrderManager, metrics_collector: MetricsCollector, 
                 max_positions: int = 10, max_correlated_exposure: float = 0.5):
        """
        Initialize the Position Manager.
        
        Args:
            order_manager: OrderManager instance for executing orders
            metrics_collector: MetricsCollector for tracking performance
            max_positions: Maximum number of concurrent positions allowed
            max_correlated_exposure: Maximum exposure to correlated assets (0.0-1.0)
        """
        self.logger = get_logger("PositionManager")
        self.order_manager = order_manager
        self.metrics_collector = metrics_collector
        self.max_positions = max_positions
        self.max_correlated_exposure = max_correlated_exposure
        
        # Active positions by ID
        self.positions: Dict[str, Position] = {}
        
        # Performance tracking
        self.total_realized_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        
        # Correlation tracking
        self.position_correlations = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        self.logger.info("Position Manager initialized with max_positions=%d, max_correlated_exposure=%.2f", 
                         max_positions, max_correlated_exposure)
    
    async def calculate_position_size(
        self, 
        symbol: str, 
        side: PositionSide,
        account_balance: float,
        risk_percentage: float,
        entry_price: float,
        stop_loss_price: float,
        tick_size: float,
        min_qty: float,
        max_qty: float,
        leverage: float = 1.0,
        platform: str = "binance",
        kelly_fraction: float = 0.5,
        win_rate: float = 0.5,
        recent_volatility: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            symbol: The trading pair symbol
            side: Long or Short position
            account_balance: Current account balance
            risk_percentage: Percentage of account to risk (0-100)
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            tick_size: Minimum price movement
            min_qty: Minimum quantity
            max_qty: Maximum quantity
            leverage: Trading leverage (1 for spot, >1 for margin/futures)
            platform: Trading platform (binance, deriv, etc.)
            kelly_fraction: Fraction of kelly criterion to use (0.0-1.0)
            win_rate: Expected win rate for this strategy (0.0-1.0)
            recent_volatility: Recent volatility for this asset (optional)
            
        Returns:
            Tuple of (quantity, risk_amount)
        """
        async with self._lock:
            # Validate inputs
            if account_balance <= 0:
                raise PositionSizingError("Invalid account balance")
                
            if risk_percentage <= 0 or risk_percentage > 100:
                raise PositionSizingError("Risk percentage must be between 0 and 100")
                
            if entry_price <= 0:
                raise PositionSizingError("Entry price must be positive")
                
            if stop_loss_price <= 0:
                raise PositionSizingError("Stop loss price must be positive")
                
            # For long positions, stop must be below entry; for shorts, stop must be above entry
            if (side == PositionSide.LONG and stop_loss_price >= entry_price) or \
               (side == PositionSide.SHORT and stop_loss_price <= entry_price):
                raise PositionSizingError(f"Invalid stop loss price for {side.name} position")
            
            # Calculate risk amount in account currency
            risk_amount = account_balance * (risk_percentage / 100)
            
            # Calculate the price distance to stop loss
            if side == PositionSide.LONG:
                price_risk = entry_price - stop_loss_price
            else:
                price_risk = stop_loss_price - entry_price
                
            if price_risk <= 0:
                raise PositionSizingError("Invalid price risk calculation")
            
            # Calculate position size based on risk
            base_position_size = risk_amount / price_risk
            
            # Adjust for leverage
            position_size = base_position_size * leverage
            
            # Apply Kelly criterion adjustment if win rate is provided
            if win_rate > 0:
                # Kelly formula: f* = (bp - q) / b
                # where p = win probability, q = loss probability, b = odds received on wager
                loss_rate = 1.0 - win_rate
                
                # Calculate average win/loss ratio from historical data
                avg_win_loss_ratio = await self._get_average_win_loss_ratio(symbol, side)
                
                # Calculate Kelly percentage
                kelly_percentage = (win_rate * avg_win_loss_ratio - loss_rate) / avg_win_loss_ratio
                
                # Apply Kelly fraction to avoid over-betting
                kelly_percentage = max(0, kelly_percentage) * kelly_fraction
                
                # Adjust position size by Kelly criterion
                position_size = position_size * kelly_percentage
                
            # Adjust for volatility if provided
            if recent_volatility is not None and recent_volatility > 0:
                # Reduce position size as volatility increases
                volatility_adjustment = 1.0 / (1.0 + recent_volatility)
                position_size = position_size * volatility_adjustment
                
            # Adjust for correlation with existing positions
            correlation_adjustment = await self._calculate_correlation_adjustment(symbol)
            position_size = position_size * correlation_adjustment
            
            # Check for max positions limit
            if len(self.get_open_positions()) >= self.max_positions:
                raise PositionSizingError(f"Maximum positions limit reached ({self.max_positions})")
                
            # Round to appropriate precision for the platform
            position_size = round_to_tick_size(position_size, tick_size)
            
            # Ensure position size is within limits
            position_size = max(min_qty, min(position_size, max_qty))
            
            # Re-calculate actual risk based on final position size
            actual_risk = position_size * price_risk / leverage
            
            self.logger.info(
                "Calculated position size for %s %s: %.6f (risk: %.2f %%, amount: %.2f)",
                symbol, side.name, position_size, risk_percentage, actual_risk
            )
            
            return position_size, actual_risk
    
    async def _get_average_win_loss_ratio(self, symbol: str, side: PositionSide) -> float:
        """Calculate the average win/loss ratio from historical trades."""
        # Default to 1.0 if not enough historical data
        default_ratio = 1.0
        
        # Get recent positions for this symbol and side
        recent_positions = await self._get_recent_positions(symbol, side, limit=20)
        
        if not recent_positions:
            return default_ratio
            
        wins = [p for p in recent_positions if p.realized_pnl > 0]
        losses = [p for p in recent_positions if p.realized_pnl < 0]
        
        if not wins or not losses:
            return default_ratio
            
        avg_win = sum(p.realized_pnl for p in wins) / len(wins)
        avg_loss = abs(sum(p.realized_pnl for p in losses) / len(losses))
        
        if avg_loss == 0:
            return default_ratio
            
        win_loss_ratio = avg_win / avg_loss
        
        # Sanity check - limit to reasonable range
        win_loss_ratio = max(0.1, min(win_loss_ratio, 10.0))
        
        return win_loss_ratio
        
    async def _get_recent_positions(self, symbol: str, side: PositionSide, limit: int = 20) -> List[Position]:
        """Get recent closed positions for a symbol and side."""
        # In a real implementation, this would query the database
        # Here we'll use in-memory positions as a simplified example
        closed_positions = [
            p for p in self.positions.values() 
            if p.symbol == symbol and p.side == side and p.status == PositionStatus.CLOSED
        ]
        
        # Sort by exit time, most recent first
        closed_positions.sort(key=lambda p: p.exit_time if p.exit_time else datetime.min, reverse=True)
        
        return closed_positions[:limit]
    
    async def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """
        Calculate position size adjustment based on correlation with existing positions.
        
        Returns a factor between 0.1 and 1.0 where:
        - 1.0 means no reduction (no correlation)
        - 0.1 means 90% reduction (high correlation)
        """
        open_positions = self.get_open_positions()
        
        if not open_positions:
            return 1.0  # No open positions, no adjustment needed
            
        # Get correlation data
        correlations = await self._get_symbol_correlations(symbol)
        
        if not correlations:
            return 0.8  # No correlation data, use conservative adjustment
            
        # Calculate weighted correlation based on position sizes
        total_exposure = sum(p.remaining_quantity * p.entry_price for p in open_positions)
        
        if total_exposure <= 0:
            return 1.0
            
        weighted_correlation = 0.0
        
        for position in open_positions:
            position_weight = (position.remaining_quantity * position.entry_price) / total_exposure
            correlation_value = correlations.get(position.symbol, 0.0)
            weighted_correlation += position_weight * abs(correlation_value)
            
        # Calculate adjustment factor (higher correlation = lower adjustment)
        # Map correlation from [0,1] to [0.1,1.0]
        adjustment = 1.0 - (weighted_correlation * 0.9)
        
        # Ensure reasonable bounds
        adjustment = max(0.1, min(adjustment, 1.0))
        
        self.logger.debug(
            "Correlation adjustment for %s: %.2f (weighted correlation: %.2f)", 
            symbol, adjustment, weighted_correlation
        )
        
        return adjustment
    
    async def _get_symbol_correlations(self, symbol: str) -> Dict[str, float]:
        """
        Get correlation coefficients between this symbol and other symbols.
        
        Returns a dictionary mapping symbol -> correlation coefficient (-1.0 to 1.0)
        """
        # This would typically fetch from a correlation service or calculate on-the-fly
        # For simplicity, we'll return a mock result
        # In the full implementation, this would use price data to calculate actual correlations
        
        # Check if we have cached correlations
        if symbol in self.position_correlations:
            return self.position_correlations[symbol]
            
        # Mock correlation data for illustration
        # In production, this would be calculated from price data
        correlations = {}
        
        # Get all symbols from open positions
        open_symbols = {p.symbol for p in self.get_open_positions()}
        
        for other_symbol in open_symbols:
            if other_symbol == symbol:
                correlations[other_symbol] = 1.0  # Self-correlation is always 1.0
            else:
                # Calculate or retrieve correlation between symbol and other_symbol
                # This is a simplified mock implementation
                if symbol.startswith("BTC") and other_symbol.startswith("BTC"):
                    correlations[other_symbol] = 0.9  # High correlation between BTC pairs
                elif symbol.startswith("ETH") and other_symbol.startswith("ETH"):
                    correlations[other_symbol] = 0.85  # High correlation between ETH pairs
                elif (symbol.startswith("BTC") and other_symbol.startswith("ETH")) or \
                     (symbol.startswith("ETH") and other_symbol.startswith("BTC")):
                    correlations[other_symbol] = 0.7  # Moderate correlation between BTC and ETH
                else:
                    correlations[other_symbol] = 0.3  # Default moderate-low correlation
        
        # Cache the results
        self.position_correlations[symbol] = correlations
        
        return correlations
    
    async def create_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        entry_price: float = 0.0,  # 0 means market order
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        risk_amount: float = 0.0,
        strategy_id: Optional[str] = None,
        brain_id: Optional[str] = None,
        leverage: float = 1.0,
        platform: str = "binance",
        margin_type: str = "ISOLATED",
        metadata: Dict[str, Any] = None,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> Position:
        """
        Create a new trading position.
        
        Args:
            symbol: Trading pair symbol
            side: Long or Short
            quantity: Position size
            entry_price: Limit price (0 for market orders)
            stop_loss: Initial stop loss price
            take_profit: Initial take profit price
            risk_amount: Amount risked in account currency
            strategy_id: ID of the strategy that generated this trade
            brain_id: ID of the brain that generated this trade
            leverage: Trading leverage
            platform: Trading platform (binance, deriv, etc.)
            margin_type: Margin type (ISOLATED or CROSS)
            metadata: Additional metadata for the position
            order_type: Type of entry order
            time_in_force: Time in force for limit orders
            
        Returns:
            The created Position object
        """
        async with self._lock:
            # Generate unique position ID
            position_id = str(uuid.uuid4())
            
            # Create position object
            position = Position(
                position_id=position_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                remaining_quantity=quantity,
                risk_amount=risk_amount,
                strategy_id=strategy_id,
                brain_id=brain_id,
                platform=platform,
                leverage=leverage,
                margin_type=margin_type,
                metadata=metadata or {},
                status=PositionStatus.PENDING
            )
            
            # Store position
            self.positions[position_id] = position
            
            # Log position creation
            self.logger.info(
                "Created new position %s: %s %s %.6f @ %s",
                position_id, side.name, symbol, quantity,
                f"{entry_price}" if entry_price > 0 else "MARKET"
            )
            
            # Place entry order
            try:
                # Set leverage if needed
                if leverage > 1.0:
                    await self.order_manager.set_leverage(
                        symbol=symbol,
                        leverage=leverage,
                        margin_type=margin_type,
                        platform=platform
                    )
                
                # Create entry order
                entry_order = await self.order_manager.place_order(
                    symbol=symbol,
                    side="BUY" if side == PositionSide.LONG else "SELL",
                    quantity=quantity,
                    price=entry_price if order_type == OrderType.LIMIT else 0,
                    order_type=order_type,
                    time_in_force=time_in_force,
                    platform=platform,
                    position_id=position_id
                )
                
                # Update position with entry order details
                position.entry_orders.append(entry_order)
                
                # If market order, update position details immediately
                if order_type == OrderType.MARKET:
                    fill_price = float(entry_order.get('price', 0)) or float(entry_order.get('avgPrice', 0))
                    
                    if fill_price > 0:
                        position.entry_price = fill_price
                        position.current_price = fill_price
                        position.entry_time = datetime.utcnow()
                        position.status = PositionStatus.OPEN
                        
                        # Set stop loss and take profit if provided
                        if stop_loss:
                            position.stop_loss = stop_loss
                            await self._place_stop_loss_order(position)
                            
                        if take_profit:
                            position.take_profit = take_profit
                            await self._place_take_profit_order(position)
                            
                        # Calculate liquidation price for leveraged trades
                        if leverage > 1.0:
                            position.liquidation_price = calculate_liquidation_price(
                                side=side,
                                entry_price=fill_price,
                                leverage=leverage,
                                maintenance_margin=0.005  # Typical value, would get from exchange info
                            )
                        
                        # Log position opened
                        self.logger.info(
                            "Position %s opened: %s %s %.6f @ %.6f",
                            position_id, side.name, symbol, quantity, fill_price
                        )
                        
                        # Track metrics
                        self.metrics_collector.record_position_opened(position.to_dict())
                
                return position
                
            except Exception as e:
                # Set position as failed
                position.status = PositionStatus.FAILED
                self.logger.error("Failed to create position %s: %s", position_id, str(e))
                raise PositionExecutionError(f"Failed to create position: {str(e)}") from e
    
    async def _place_stop_loss_order(self, position: Position) -> Dict:
        """Place a stop loss order for the position."""
        if not position.stop_loss or position.status != PositionStatus.OPEN:
            return {}
            
        stop_side = "SELL" if position.side == PositionSide.LONG else "BUY"
        
        stop_order = await self.order_manager.place_order(
            symbol=position.symbol,
            side=stop_side,
            quantity=position.remaining_quantity,
            price=position.stop_loss,
            order_type=OrderType.STOP_LOSS,
            time_in_force=TimeInForce.GTC,
            platform=position.platform,
            position_id=position.position_id
        )
        
        position.exit_orders.append(stop_order)
        
        self.logger.info(
            "Stop loss placed for position %s: %s @ %.6f",
            position.position_id, position.symbol, position.stop_loss
        )
        
        return stop_order
    
    async def _place_take_profit_order(self, position: Position) -> Dict:
        """Place a take profit order for the position."""
        if not position.take_profit or position.status != PositionStatus.OPEN:
            return {}
            
        tp_side = "SELL" if position.side == PositionSide.LONG else "BUY"
        
        tp_order = await self.order_manager.place_order(
            symbol=position.symbol,
            side=tp_side,
            quantity=position.remaining_quantity,
            price=position.take_profit,
            order_type=OrderType.TAKE_PROFIT,
            time_in_force=TimeInForce.GTC,
            platform=position.platform,
            position_id=position.position_id
        )
        
        position.exit_orders.append(tp_order)
        
        self.logger.info(
            "Take profit placed for position %s: %s @ %.6f",
            position.position_id, position.symbol, position.take_profit
        )
        
        return tp_order
        
    async def update_position_price(self, position_id: str, current_price: float) -> Position:
        """
        Update a position with the current market price.
        
        Args:
            position_id: ID of the position to update
            current_price: Current market price
            
        Returns:
            Updated Position object
        """
        position = self.get_position(position_id)
        
        if not position:
            raise PositionError(f"Position not found: {position_id}")
            
        if position.status not in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]:
            return position
            
        position.current_price = current_price
        position.update_unrealized_pnl()
        
        # Check for stop loss or take profit hits (for non-exchange SL/TP)
        if position.status == PositionStatus.OPEN:
            # Check stop loss
            if position.stop_loss and (
                (position.side == PositionSide.LONG and current_price <= position.stop_loss) or
                (position.side == PositionSide.SHORT and current_price >= position.stop_loss)
            ):
                await self.close_position(position_id, "Stop loss triggered")
                
            # Check take profit
            elif position.take_profit and (
                (position.side == PositionSide.LONG and current_price >= position.take_profit) or
                (position.side == PositionSide.SHORT and current_price <= position.take_profit)
            ):
                await self.close_position(position_id, "Take profit triggered")
                
        return position
    
    async def update_stop_loss(self, position_id: str, new_stop_loss: float, 
                              trail_percent: float = 0.0) -> Position:
        """
        Update a position's stop loss level.
        
        Args:
            position_id: ID of the position
            new_stop_loss: New stop loss price
            trail_percent: If > 0, enables a trailing stop with this percentage
            
        Returns:
            Updated Position object
        """
        position = self.get_position(position_id)
        
        if not position:
            raise PositionError(f"Position not found: {position_id}")
            
        if position.status not in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]:
            raise InvalidPositionStateError(
                f"Cannot update stop loss for position with status {position.status.name}"
            )
            
        # Validate stop loss
        if (position.side == PositionSide.LONG and new_stop_loss >= position.current_price) or \
           (position.side == PositionSide.SHORT and new_stop_loss <= position.current_price):
            raise PositionError(f"Invalid stop loss price for {position.side.name} position")
            
        # Cancel existing stop loss orders
        for order in position.exit_orders:
            if order.get('type') == OrderType.STOP_LOSS.value:
                await self.order_manager.cancel_order(
                    symbol=position.symbol,
                    order_id=order.get('orderId'),
                    platform=position.platform
                )
                
        # Update position
        position.stop_loss = new_stop_loss
        
        # Set trailing stop if requested
        if trail_percent > 0:
            position.metadata['trailing_stop'] = True
            position.metadata['trail_percent'] = trail_percent
            
        # Place new stop loss order
        await self._place_stop_loss_order(position)
        
        self.logger.info(
            "Updated stop loss for position %s: %.6f %s",
            position_id, new_stop_loss,
            f"(trailing {trail_percent}%)" if trail_percent > 0 else ""
        )
        
        return position
    
    async def update_take_profit(self, position_id: str, new_take_profit: float) -> Position:
        """
        Update a position's take profit level.
        
        Args:
            position_id: ID of the position
            new_take_profit: New take profit price
            
        Returns:
            Updated Position object
        """
        position = self.get_position(position_id)
        
        if not position:
            raise PositionError(f"Position not found: {position_id}")
            
        if position.status not in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]:
            raise InvalidPositionStateError(
                f"Cannot update take profit for position with status {position.status.name}"
            )
            
        # Validate take profit
        if (position.side == PositionSide.LONG and new_take_profit <= position.current_price) or \
           (position.side == PositionSide.SHORT and new_take_profit >= position.current_price):
            raise PositionError(f"Invalid take profit price for {position.side.name} position")
            
        # Cancel existing take profit orders
        for order in position.exit_orders:
            if order.get('type') == OrderType.TAKE_PROFIT.value:
                await self.order_manager.cancel_order(
                    symbol=position.symbol,
                    order_id=order.get('orderId'),
                    platform=position.platform
                )
                
        # Update position
        position.take_profit = new_take_profit
        
        # Place new take profit order
        await self._place_take_profit_order(position)
        
        self.logger.info(
            "Updated take profit for position %s: %.6f",
            position_id, new_take_profit
        )
        
        return position
    
    async def close_position(self, position_id: str, reason: str = "") -> Position:
        """
        Close a position completely.
        
        Args:
            position_id: ID of the position to close
            reason: Reason for closing the position
            
        Returns:
            Updated Position object
        """
        position = self.get_position(position_id)
        
        if not position:
            raise PositionError(f"Position not found: {position_id}")
            
        if position.status not in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]:
            self.logger.warning(
                "Cannot close position %s with status %s",
                position_id, position.status.name
            )
            return position
            
        if position.remaining_quantity <= 0:
            position.status = PositionStatus.CLOSED
            position.exit_time = datetime.utcnow()
            return position
            
        # Cancel all existing exit orders
        for order in position.exit_orders:
            try:
                await self.order_manager.cancel_order(
                    symbol=position.symbol,
                    order_id=order.get('orderId'),
                    platform=position.platform
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to cancel exit order for position %s: %s",
                    position_id, str(e)
                )
        
        # Place market close order
        close_side = "SELL" if position.side == PositionSide.LONG else "BUY"
        
        try:
            close_order = await self.order_manager.place_order(
                symbol=position.symbol,
                side=close_side,
                quantity=position.remaining_quantity,
                price=0,  # Market order
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                platform=position.platform,
                position_id=position.position_id
            )
            
            # Update position
            fill_price = float(close_order.get('price', 0)) or float(close_order.get('avgPrice', 0))
            
            if fill_price > 0:
                # Calculate realized PnL
                price_diff = 0
                if position.side == PositionSide.LONG:
                    price_diff = fill_price - position.entry_price
                else:
                    price_diff = position.entry_price - fill_price
                
                position.realized_pnl += price_diff * position.remaining_quantity * position.leverage
                position.total_realized_pnl += position.realized_pnl
                
                # Update fees paid
                fees = float(close_order.get('fee', 0))
                position.fees_paid += fees
                
                # Update status
                position.status = PositionStatus.CLOSED
                position.exit_time = datetime.utcnow()
                position.current_price = fill_price
                position.remaining_quantity = 0
                position.unrealized_pnl = 0
                
                # Record trade result
                if position.realized_pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                    
                self.total_trades += 1
                
                # Log position closed
                self.logger.info(
                    "Position %s closed: %s %s @ %.6f, PnL: %.2f, Reason: %s",
                    position_id, position.symbol, position.side.name, 
                    fill_price, position.realized_pnl, reason
                )
                
                # Track metrics
                self.metrics_collector.record_position_closed(position.to_dict())
                
                # Clear correlation cache
                if position.symbol in self.position_correlations:
                    del self.position_correlations[position.symbol]
                
        except Exception as e:
            self.logger.error("Failed to close position %s: %s", position_id, str(e))
            raise PositionExecutionError(f"Failed to close position: {str(e)}") from e
            
        return position
    
    async def close_partial_position(self, position_id: str, 
                                    close_percent: float, reason: str = "") -> Position:
        """
        Close a portion of a position.
        
        Args:
            position_id: ID of the position
            close_percent: Percentage of the position to close (0-100)
            reason: Reason for partial closure
            
        Returns:
            Updated Position object
        """
        position = self.get_position(position_id)
        
        if not position:
            raise PositionError(f"Position not found: {position_id}")
            
        if position.status not in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]:
            raise InvalidPositionStateError(
                f"Cannot partially close position with status {position.status.name}"
            )
            
        if close_percent <= 0 or close_percent >= 100:
            raise PositionError("Close percentage must be between 0 and 100")
            
        # Calculate quantity to close
        close_qty = position.remaining_quantity * (close_percent / 100)
        
        # Ensure minimum quantity
        min_qty = position.metadata.get('min_qty', 0.001)  # Exchange dependent
        if close_qty < min_qty:
            close_qty = min_qty
            
        # Ensure we don't close more than remaining
        close_qty = min(close_qty, position.remaining_quantity)
        
        # Place market close order
        close_side = "SELL" if position.side == PositionSide.LONG else "BUY"
        
        try:
            close_order = await self.order_manager.place_order(
                symbol=position.symbol,
                side=close_side,
                quantity=close_qty,
                price=0,  # Market order
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                platform=position.platform,
                position_id=position.position_id
            )
            
            # Update position
            fill_price = float(close_order.get('price', 0)) or float(close_order.get('avgPrice', 0))
            
            if fill_price > 0:
                # Calculate realized PnL for this partial close
                price_diff = 0
                if position.side == PositionSide.LONG:
                    price_diff = fill_price - position.entry_price
                else:
                    price_diff = position.entry_price - fill_price
                
                partial_realized_pnl = price_diff * close_qty * position.leverage
                position.realized_pnl += partial_realized_pnl
                
                # Update fees paid
                fees = float(close_order.get('fee', 0))
                position.fees_paid += fees
                
                # Update remaining quantity
                position.remaining_quantity -= close_qty
                
                # Update status
                position.status = PositionStatus.PARTIALLY_CLOSED
                position.current_price = fill_price
                
                # Record partial exit
                position.partial_exits.append({
                    'time': datetime.utcnow().isoformat(),
                    'price': fill_price,
                    'quantity': close_qty,
                    'realized_pnl': partial_realized_pnl,
                    'reason': reason
                })
                
                # If remaining quantity is very small, consider it fully closed
                if position.remaining_quantity <= min_qty:
                    position.status = PositionStatus.CLOSED
                    position.exit_time = datetime.utcnow()
                    position.remaining_quantity = 0
                    position.unrealized_pnl = 0
                    
                    # Record trade result
                    if position.realized_pnl > 0:
                        self.win_count += 1
                    else:
                        self.loss_count += 1
                        
                    self.total_trades += 1
                
                # Log position partially closed
                self.logger.info(
                    "Position %s partially closed: %s %s %.6f @ %.6f, Partial PnL: %.2f, Reason: %s",
                    position_id, position.symbol, position.side.name, 
                    close_qty, fill_price, partial_realized_pnl, reason
                )
                
                # Track metrics
                self.metrics_collector.record_position_updated(position.to_dict())
                
        except Exception as e:
            self.logger.error("Failed to partially close position %s: %s", position_id, str(e))
            raise PositionExecutionError(f"Failed to partially close position: {str(e)}") from e
            
        return position
    
    async def add_to_position(self, position_id: str, additional_quantity: float, 
                             entry_price: float = 0) -> Position:
        """
        Add to an existing position.
        
        Args:
            position_id: ID of the position
            additional_quantity: Additional quantity to add
            entry_price: Limit price (0 for market orders)
            
        Returns:
            Updated Position object
        """
        position = self.get_position(position_id)
        
        if not position:
            raise PositionError(f"Position not found: {position_id}")
            
        if position.status not in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]:
            raise InvalidPositionStateError(
                f"Cannot add to position with status {position.status.name}"
            )
            
        if additional_quantity <= 0:
            raise PositionError("Additional quantity must be positive")
            
        # Place order to add to position
        order_side = "BUY" if position.side == PositionSide.LONG else "SELL"
        order_type = OrderType.LIMIT if entry_price > 0 else OrderType.MARKET
        
        try:
            add_order = await self.order_manager.place_order(
                symbol=position.symbol,
                side=order_side,
                quantity=additional_quantity,
                price=entry_price,
                order_type=order_type,
                time_in_force=TimeInForce.GTC,
                platform=position.platform,
                position_id=position.position_id
            )
            
            # Update position for market orders
            if order_type == OrderType.MARKET:
                fill_price = float(add_order.get('price', 0)) or float(add_order.get('avgPrice', 0))
                
                if fill_price > 0:
                    # Calculate new average entry price
                    total_cost = (position.entry_price * position.quantity) + (fill_price * additional_quantity)
                    new_total_qty = position.quantity + additional_quantity
                    new_avg_price = total_cost / new_total_qty
                    
                    # Update position
                    position.entry_price = new_avg_price
                    position.current_price = fill_price
                    position.quantity += additional_quantity
                    position.remaining_quantity += additional_quantity
                    
                    # Update fees paid
                    fees = float(add_order.get('fee', 0))
                    position.fees_paid += fees
                    
                    # Log position update
                    self.logger.info(
                        "Added to position %s: %s %s +%.6f @ %.6f, New avg price: %.6f",
                        position_id, position.symbol, position.side.name, 
                        additional_quantity, fill_price, new_avg_price
                    )
                    
                    # Track metrics
                    self.metrics_collector.record_position_updated(position.to_dict())
            
            # For limit orders, just store the order
            position.entry_orders.append(add_order)
                
        except Exception as e:
            self.logger.error("Failed to add to position %s: %s", position_id, str(e))
            raise PositionExecutionError(f"Failed to add to position: {str(e)}") from e
            
        return position
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by ID."""
        return self.positions.get(position_id)
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a symbol."""
        return [p for p in self.positions.values() if p.symbol == symbol]
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self.positions.values() 
                if p.status in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]]
    
    def get_closed_positions(self, limit: int = 100) -> List[Position]:
        """Get closed positions, most recent first."""
        closed = [p for p in self.positions.values() if p.status == PositionStatus.CLOSED]
        closed.sort(key=lambda p: p.exit_time if p.exit_time else datetime.min, reverse=True)
        return closed[:limit]
    
    def get_positions_by_strategy(self, strategy_id: str) -> List[Position]:
        """Get all positions for a strategy."""
        return [p for p in self.positions.values() if p.strategy_id == strategy_id]
    
    def get_positions_by_brain(self, brain_id: str) -> List[Position]:
        """Get all positions for a brain."""
        return [p for p in self.positions.values() if p.brain_id == brain_id]
    
    async def update_trailing_stops(self) -> None:
        """Update trailing stops for all positions with trailing stops enabled."""
        open_positions = self.get_open_positions()
        
        for position in open_positions:
            if position.metadata.get('trailing_stop') and position.metadata.get('trail_percent'):
                trail_percent = float(position.metadata.get('trail_percent', 0))
                
                if trail_percent <= 0:
                    continue
                    
                # Calculate the ideal stop loss based on current price
                if position.side == PositionSide.LONG:
                    ideal_stop = position.current_price * (1 - trail_percent / 100)
                    # Only move the stop loss up, never down
                    if position.stop_loss and ideal_stop > position.stop_loss:
                        await self.update_stop_loss(
                            position_id=position.position_id,
                            new_stop_loss=ideal_stop,
                            trail_percent=trail_percent
                        )
                else:  # SHORT
                    ideal_stop = position.current_price * (1 + trail_percent / 100)
                    # Only move the stop loss down, never up
                    if position.stop_loss and ideal_stop < position.stop_loss:
                        await self.update_stop_loss(
                            position_id=position.position_id,
                            new_stop_loss=ideal_stop,
                            trail_percent=trail_percent
                        )
    
    async def check_for_partial_exits(self) -> None:
        """Check all positions for partial exit conditions."""
        open_positions = self.get_open_positions()
        
        for position in open_positions:
            # Skip if not set up for partial exits
            if not position.take_profit or 'partial_exits' not in position.metadata:
                continue
                
            partial_exits = position.metadata.get('partial_exits', [])
            
            for exit_level in partial_exits:
                price_level = exit_level.get('price')
                percentage = exit_level.get('percentage', 0)
                executed = exit_level.get('executed', False)
                
                if executed or not price_level or percentage <= 0:
                    continue
                    
                # Check if price level is reached
                if ((position.side == PositionSide.LONG and position.current_price >= price_level) or
                    (position.side == PositionSide.SHORT and position.current_price <= price_level)):
                    
                    # Mark as executed to prevent repeated execution
                    exit_level['executed'] = True
                    
                    # Execute partial exit
                    await self.close_partial_position(
                        position_id=position.position_id,
                        close_percent=percentage,
                        reason=f"Partial exit at {price_level}"
                    )
    
    async def process_order_update(self, order_update: Dict[str, Any]) -> None:
        """Process an order update event from the exchange."""
        # Extract order details
        order_id = order_update.get('orderId')
        symbol = order_update.get('symbol')
        position_id = order_update.get('position_id')
        status = order_update.get('status')
        
        if not order_id or not position_id:
            self.logger.warning("Received order update without orderId or position_id")
            return
            
        position = self.get_position(position_id)
        
        if not position:
            self.logger.warning("Received order update for unknown position: %s", position_id)
            return
            
        self.logger.debug("Processing order update for position %s: %s", position_id, order_update)
        
        # Handle different status updates
        if status == 'FILLED':
            # Find if this is an entry or exit order
            is_entry = any(o.get('orderId') == order_id for o in position.entry_orders)
            is_exit = any(o.get('orderId') == order_id for o in position.exit_orders)
            
            if is_entry:
                # Process entry order fill
                await self._process_entry_fill(position, order_update)
            elif is_exit:
                # Process exit order fill
                await self._process_exit_fill(position, order_update)
                
        elif status == 'CANCELED':
            # Remove from entry/exit orders lists
            position.entry_orders = [o for o in position.entry_orders if o.get('orderId') != order_id]
            position.exit_orders = [o for o in position.exit_orders if o.get('orderId') != order_id]
            
        elif status == 'REJECTED':
            # Log the rejection
            self.logger.warning("Order rejected for position %s: %s", position_id, order_update)
            
            # If entry order was rejected and no other entry orders exist, mark position as failed
            if position.status == PositionStatus.PENDING and not position.entry_orders:
                position.status = PositionStatus.FAILED
                self.logger.error("Position %s failed due to rejected entry order", position_id)
    
    async def _process_entry_fill(self, position: Position, order_update: Dict[str, Any]) -> None:
        """Process a filled entry order."""
        # Only process if position is in appropriate state
        if position.status not in [PositionStatus.PENDING, PositionStatus.OPEN]:
            return
            
        # Extract fill details
        fill_price = float(order_update.get('price', 0))
        fill_qty = float(order_update.get('executedQty', 0))
        
        if fill_price <= 0 or fill_qty <= 0:
            return
            
        # If this is first fill, set entry price and status
        if position.status == PositionStatus.PENDING:
            position.entry_price = fill_price
            position.current_price = fill_price
            position.entry_time = datetime.utcnow()
            position.status = PositionStatus.OPEN
            
            # Calculate liquidation price for leveraged trades
            if position.leverage > 1.0:
                position.liquidation_price = calculate_liquidation_price(
                    side=position.side,
                    entry_price=fill_price,
                    leverage=position.leverage,
                    maintenance_margin=0.005  # Typical value, would get from exchange info
                )
            
            # Place stop loss and take profit orders if set
            if position.stop_loss:
                await self._place_stop_loss_order(position)
                
            if position.take_profit:
                await self._place_take_profit_order(position)
                
            # Log position opened
            self.logger.info(
                "Position %s opened: %s %s %.6f @ %.6f",
                position.position_id, position.side.name, position.symbol, 
                position.quantity, fill_price
            )
            
            # Track metrics
            self.metrics_collector.record_position_opened(position.to_dict())
            
        else:  # Adding to existing position
            # Calculate new average entry price
            total_cost = (position.entry_price * position.quantity) + (fill_price * fill_qty)
            new_total_qty = position.quantity + fill_qty
            new_avg_price = total_cost / new_total_qty
            
            # Update position
            position.entry_price = new_avg_price
            position.current_price = fill_price
            position.quantity += fill_qty
            position.remaining_quantity += fill_qty
            
            # Log position update
            self.logger.info(
                "Added to position %s: %s %s +%.6f @ %.6f, New avg price: %.6f",
                position.position_id, position.symbol, position.side.name, 
                fill_qty, fill_price, new_avg_price
            )
            
            # Track metrics
            self.metrics_collector.record_position_updated(position.to_dict())
    
    async def _process_exit_fill(self, position: Position, order_update: Dict[str, Any]) -> None:
        """Process a filled exit order."""
        # Only process if position is in appropriate state
        if position.status not in [PositionStatus.OPEN, PositionStatus.PARTIALLY_CLOSED]:
            return
            
        # Extract fill details
        fill_price = float(order_update.get('price', 0))
        fill_qty = float(order_update.get('executedQty', 0))
        
        if fill_price <= 0 or fill_qty <= 0:
            return
            
        # Calculate realized PnL for this exit
        price_diff = 0
        if position.side == PositionSide.LONG:
            price_diff = fill_price - position.entry_price
        else:
            price_diff = position.entry_price - fill_price
        
        partial_realized_pnl = price_diff * fill_qty * position.leverage
        position.realized_pnl += partial_realized_pnl
        
        # Update fees paid
        fees = float(order_update.get('fee', 0))
        position.fees_paid += fees
        
        # Update remaining quantity
        position.remaining_quantity -= fill_qty
        
        # Update position status
        if position.remaining_quantity <= 0:
            position.status = PositionStatus.CLOSED
            position.exit_time = datetime.utcnow()
            position.current_price = fill_price
            position.unrealized_pnl = 0
            
            # Cancel any remaining exit orders
            for order in position.exit_orders:
                if order.get('orderId') != order_update.get('orderId'):
                    try:
                        await self.order_manager.cancel_order(
                            symbol=position.symbol,
                            order_id=order.get('orderId'),
                            platform=position.platform
                        )
                    except Exception as e:
                        self.logger.warning(
                            "Failed to cancel exit order for position %s: %s",
                            position.position_id, str(e)
                        )
            
            # Record trade result
            if position.realized_pnl > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
                
            self.total_trades += 1
            
            # Log position closed
            self.logger.info(
                "Position %s closed: %s %s @ %.6f, PnL: %.2f",
                position.position_id, position.symbol, position.side.name, 
                fill_price, position.realized_pnl
            )
            
            # Track metrics
            self.metrics_collector.record_position_closed(position.to_dict())
            
        else:
            position.status = PositionStatus.PARTIALLY_CLOSED
            position.current_price = fill_price
            
            # Record partial exit
            position.partial_exits.append({
                'time': datetime.utcnow().isoformat(),
                'price': fill_price,
                'quantity': fill_qty,
                'realized_pnl': partial_realized_pnl,
                'order_id': order_update.get('orderId')
            })
            
            # Log partial closure
            self.logger.info(
                "Position %s partially closed: %s %s %.6f @ %.6f, Partial PnL: %.2f",
                position.position_id, position.symbol, position.side.name, 
                fill_qty, fill_price, partial_realized_pnl
            )
            
            # Track metrics
            self.metrics_collector.record_position_updated(position.to_dict())
    
    async def save_positions(self) -> None:
        """Save all positions to persistent storage."""
        # In a real implementation this would save to a database
        # For this implementation we'll just log the current state
        open_count = len(self.get_open_positions())
        closed_count = len(self.get_closed_positions())
        
        self.logger.info(
            "Position state snapshot: %d open, %d closed, %.2f realized PnL, %d wins, %d losses", 
            open_count, closed_count, self.total_realized_pnl, self.win_count, self.loss_count
        )
    
    async def load_positions(self) -> None:
        """Load positions from persistent storage."""
        # In a real implementation this would load from a database
        # For this implementation we'll just log that we would do this
        self.logger.info("Would load positions from database here")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all positions."""
        total_positions = self.win_count + self.loss_count
        win_rate = self.win_count / total_positions if total_positions > 0 else 0
        
        return {
            'total_positions': total_positions,
            'open_positions': len(self.get_open_positions()),
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'total_realized_pnl': self.total_realized_pnl,
            'positions_by_symbol': self._get_positions_by_symbol_count(),
            'avg_r_multiple': self._calculate_avg_r_multiple(),
            'largest_winner': self._get_largest_winner(),
            'largest_loser': self._get_largest_loser(),
            'avg_duration': self._calculate_avg_duration()
        }
    
    def _get_positions_by_symbol_count(self) -> Dict[str, int]:
        """Get count of positions by symbol."""
        result = {}
        for position in self.positions.values():
            if position.symbol not in result:
                result[position.symbol] = 0
            result[position.symbol] += 1
        return result
    
    def _calculate_avg_r_multiple(self) -> float:
        """Calculate average R multiple across all closed positions."""
        closed_positions = self.get_closed_positions(limit=1000)
        
        if not closed_positions:
            return 0.0
            
        r_values = [p.get_r_multiple() for p in closed_positions if p.risk_amount > 0]
        
        if not r_values:
            return 0.0
            
        return sum(r_values) / len(r_values)
    
    def _get_largest_winner(self) -> Dict[str, Any]:
        """Get details of the largest winning position."""
        closed_positions = self.get_closed_positions(limit=1000)
        
        if not closed_positions:
            return {}
            
        winners = [p for p in closed_positions if p.realized_pnl > 0]
        
        if not winners:
            return {}
            
        largest = max(winners, key=lambda p: p.realized_pnl)
        
        return {
            'position_id': largest.position_id,
            'symbol': largest.symbol,
            'side': largest.side.value,
            'pnl': largest.realized_pnl,
            'r_multiple': largest.get_r_multiple(),
            'entry_time': largest.entry_time.isoformat() if largest.entry_time else None,
            'exit_time': largest.exit_time.isoformat() if largest.exit_time else None
        }
    
    def _get_largest_loser(self) -> Dict[str, Any]:
        """Get details of the largest losing position."""
        closed_positions = self.get_closed_positions(limit=1000)
        
        if not closed_positions:
            return {}
            
        losers = [p for p in closed_positions if p.realized_pnl < 0]
        
        if not losers:
            return {}
            
        largest = min(losers, key=lambda p: p.realized_pnl)
        
        return {
            'position_id': largest.position_id,
            'symbol': largest.symbol,
            'side': largest.side.value,
            'pnl': largest.realized_pnl,
            'r_multiple': largest.get_r_multiple(),
            'entry_time': largest.entry_time.isoformat() if largest.entry_time else None,
            'exit_time': largest.exit_time.isoformat() if largest.exit_time else None
        }
    
    def _calculate_avg_duration(self) -> str:
        """Calculate average position duration."""
        closed_positions = self.get_closed_positions(limit=1000)
        
        if not closed_positions:
            return "0:00:00"
            
        durations = [p.get_duration() for p in closed_positions if p.get_duration() is not None]
        
        if not durations:
            return "0:00:00"
            
        total_seconds = sum(d.total_seconds() for d in durations)
        avg_seconds = total_seconds / len(durations)

        hours, remainder = divmod(avg_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"


class RiskManager:
    """Minimal placeholder risk manager."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    async def validate_trade(self, *args: Any, **kwargs: Any) -> bool:
        """Validate a trade request. Always returns True for now."""
        return True

    """Minimal risk manager placeholder."""

    def __init__(self, *args, **kwargs) -> None:
        self.active_positions: Dict[str, Position] = {}

    def evaluate_trade(self, *args, **kwargs) -> bool:
        """Evaluate whether a trade can be taken."""
        return True

    """Basic risk management helper."""

    def __init__(self, max_risk_per_trade: float = DEFAULT_MAX_RISK_PER_TRADE):
        self.max_risk_per_trade = max_risk_per_trade

    def check_risk(self, equity: float, risk_amount: float) -> bool:
        """Return True if the risk amount does not exceed allowed percentage."""
        if equity <= 0:
            return False
        return risk_amount <= equity * self.max_risk_per_trade
