

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Take Profit Management Module

This module provides sophisticated take profit management for the trading system,
including dynamic take profit levels, partial profit taking, and trailing profit mechanisms.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Type


class BaseTakeProfitStrategy:
    """Base class for take profit strategies."""

    registry: Dict[str, Type["BaseTakeProfitStrategy"]] = {}

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        key = name or cls.__name__
        BaseTakeProfitStrategy.registry[key] = cls

    def calculate_target(self, *args, **kwargs):
        raise NotImplementedError

import logging
import numpy as np
from decimal import Decimal

from common.constants import OrderSide, OrderType, TimeInForce, PLATFORMS
from common.utils import calculate_risk_reward_ratio, get_asset_precision
from common.async_utils import run_in_threadpool
from data_feeds.base_feed import MarketData
from feature_service.features.market_structure import MarketStructureAnalyzer
from feature_service.features.volatility import VolatilityAnalyzer

logger = logging.getLogger(__name__)


class TakeProfitManager(BaseTakeProfitStrategy):
    """
    Advanced take profit management system with dynamic profit targets based on market conditions,
    volatility, and pattern completion projections.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the TakeProfitManager with configuration.
        
        Args:
            config: Configuration dictionary for take profit settings
        """
        self.config = config or {}
        self.default_profit_factor = self.config.get('default_profit_factor', 1.5)
        self.min_reward_risk_ratio = self.config.get('min_reward_risk_ratio', 1.2)
        self.trailing_activation_threshold = self.config.get('trailing_activation_threshold', 0.7)
        self.volatility_adjuster = self.config.get('volatility_adjuster', True)
        self.pattern_projection_enabled = self.config.get('pattern_projection_enabled', True)
        self.partial_take_profit_levels = self.config.get('partial_take_profit_levels', [0.3, 0.5, 0.7, 0.85])
        self.partial_take_profit_portions = self.config.get('partial_take_profit_portions', [0.2, 0.2, 0.3, 0.3])
        
        # Initialize analyzers
        self.market_structure_analyzer = MarketStructureAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        
        # Maintain state of take profit levels for active positions
        self.active_take_profits = {}
        
        logger.info(f"TakeProfitManager initialized with config: {self.config}")
    
    async def calculate_take_profit_level(
        self, 
        symbol: str, 
        entry_price: Decimal, 
        stop_loss: Decimal, 
        side: OrderSide, 
        market_data: MarketData,
        platform: str
    ) -> Decimal:
        """
        Calculate the optimal take profit level based on market conditions, volatility,
        and pattern projections.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for the position
            stop_loss: Stop loss level for the position
            side: Order side (BUY/SELL)
            market_data: Current market data
            platform: Trading platform (BINANCE/DERIV)
            
        Returns:
            Calculated take profit price level
        """
        # Convert inputs to float for calculations
        entry_price_float = float(entry_price)
        stop_loss_float = float(stop_loss)
        
        # Calculate base take profit using risk-reward ratio
        risk = abs(entry_price_float - stop_loss_float)
        base_reward = risk * self.default_profit_factor
        
        # Adjust based on volatility if enabled
        volatility_multiplier = 1.0
        if self.volatility_adjuster:
            atr = await run_in_threadpool(
                self.volatility_analyzer.calculate_atr,
                market_data.get_candles(symbol, '1h', 14)
            )
            volatility_ratio = atr / entry_price_float
            
            # Adaptive volatility scaling
            if volatility_ratio > 0.03:  # High volatility
                volatility_multiplier = 1.3
            elif volatility_ratio < 0.01:  # Low volatility
                volatility_multiplier = 0.8
        
        # Pattern-based projection if enabled
        pattern_projection = None
        if self.pattern_projection_enabled:
            pattern_projection = await run_in_threadpool(
                self.market_structure_analyzer.project_pattern_completion,
                symbol, market_data, side
            )
        
        # Calculate take profit based on all factors
        if side == OrderSide.BUY:
            if pattern_projection and pattern_projection > entry_price_float:
                # Use pattern projection if available and reasonable
                take_profit = min(
                    entry_price_float + (base_reward * volatility_multiplier),
                    pattern_projection
                )
            else:
                take_profit = entry_price_float + (base_reward * volatility_multiplier)
        else:  # SELL
            if pattern_projection and pattern_projection < entry_price_float:
                # Use pattern projection if available and reasonable
                take_profit = max(
                    entry_price_float - (base_reward * volatility_multiplier),
                    pattern_projection
                )
            else:
                take_profit = entry_price_float - (base_reward * volatility_multiplier)
        
        # Ensure minimum reward-to-risk ratio
        actual_risk = abs(entry_price_float - stop_loss_float)
        actual_reward = abs(entry_price_float - take_profit)
        reward_risk_ratio = actual_reward / actual_risk if actual_risk != 0 else self.min_reward_risk_ratio
        
        if reward_risk_ratio < self.min_reward_risk_ratio:
            # Adjust take profit to meet minimum reward-risk ratio
            if side == OrderSide.BUY:
                take_profit = entry_price_float + (actual_risk * self.min_reward_risk_ratio)
            else:
                take_profit = entry_price_float - (actual_risk * self.min_reward_risk_ratio)
        
        # Adjust to asset precision
        precision = get_asset_precision(symbol, platform)
        take_profit_decimal = Decimal(str(round(take_profit, precision)))
        
        logger.info(f"Calculated take profit for {symbol}: {take_profit_decimal} "
                   f"(Entry: {entry_price}, Stop: {stop_loss}, Side: {side}, R/R: {reward_risk_ratio:.2f})")
        
        return take_profit_decimal
    
    async def generate_partial_take_profit_levels(
        self, 
        position_id: str,
        symbol: str, 
        entry_price: Decimal, 
        final_take_profit: Decimal, 
        side: OrderSide,
        quantity: Decimal,
        platform: str
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple partial take profit levels for a given position.
        
        Args:
            position_id: Unique identifier for the position
            symbol: Trading symbol
            entry_price: Entry price for the position
            final_take_profit: Final take profit target
            side: Order side (BUY/SELL)
            quantity: Position quantity
            platform: Trading platform (BINANCE/DERIV)
            
        Returns:
            List of take profit orders with price and quantity
        """
        entry_price_float = float(entry_price)
        final_take_profit_float = float(final_take_profit)
        quantity_float = float(quantity)
        
        # Calculate price range
        price_range = abs(final_take_profit_float - entry_price_float)
        take_profit_orders = []
        
        precision = get_asset_precision(symbol, platform)
        quantity_precision = 8 if platform == PLATFORMS.BINANCE else 2
        
        for level, portion in zip(self.partial_take_profit_levels, self.partial_take_profit_portions):
            # Calculate price level
            if side == OrderSide.BUY:
                price = entry_price_float + (price_range * level)
            else:
                price = entry_price_float - (price_range * level)
            
            # Calculate quantity for this level
            level_quantity = quantity_float * portion
            
            # Create order specification
            take_profit_order = {
                "position_id": position_id,
                "symbol": symbol,
                "side": OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
                "type": OrderType.LIMIT,
                "price": Decimal(str(round(price, precision))),
                "quantity": Decimal(str(round(level_quantity, quantity_precision))),
                "time_in_force": TimeInForce.GTC,
                "is_reduce_only": True,
                "platform": platform,
                "level": level
            }
            
            take_profit_orders.append(take_profit_order)
        
        # Store active take profits for position
        self.active_take_profits[position_id] = take_profit_orders
        
        logger.info(f"Generated {len(take_profit_orders)} partial take profit levels for position {position_id}")
        return take_profit_orders
    
    async def update_trailing_take_profit(
        self, 
        position_id: str, 
        symbol: str, 
        current_price: Decimal, 
        side: OrderSide,
        platform: str
    ) -> Optional[Decimal]:
        """
        Update trailing take profit based on price movement.
        
        Args:
            position_id: Unique identifier for the position
            symbol: Trading symbol
            current_price: Current market price
            side: Order side (BUY/SELL)
            platform: Trading platform (BINANCE/DERIV)
            
        Returns:
            New trailing take profit level or None if no update needed
        """
        if position_id not in self.active_take_profits or not self.active_take_profits[position_id]:
            logger.warning(f"No active take profits found for position {position_id}")
            return None
        
        # Get remaining take profit levels
        remaining_levels = self.active_take_profits[position_id]
        if not remaining_levels:
            return None
        
        # Check if we've reached the trailing activation threshold
        highest_level = max([level["level"] for level in remaining_levels])
        if highest_level < self.trailing_activation_threshold:
            # Not yet at trailing activation threshold
            return None
        
        # Calculate new trailing take profit
        current_price_float = float(current_price)
        last_tp = remaining_levels[-1]
        last_tp_price = float(last_tp["price"])
        
        # Only update if price has moved in favorable direction
        if (side == OrderSide.BUY and current_price_float > last_tp_price) or \
           (side == OrderSide.SELL and current_price_float < last_tp_price):
            
            # Calculate trailing distance (adaptive based on volatility)
            atr = await run_in_threadpool(
                self.volatility_analyzer.calculate_atr,
                MarketData().get_candles(symbol, '1h', 14)
            )
            
            # Trailing distance is a factor of ATR
            trailing_distance = atr * 1.5
            
            # Calculate new take profit level
            if side == OrderSide.BUY:
                new_tp = current_price_float - trailing_distance
                # Only update if new TP is higher than current TP
                if new_tp <= last_tp_price:
                    return None
            else:
                new_tp = current_price_float + trailing_distance
                # Only update if new TP is lower than current TP
                if new_tp >= last_tp_price:
                    return None
            
            # Adjust to asset precision
            precision = get_asset_precision(symbol, platform)
            new_tp_decimal = Decimal(str(round(new_tp, precision)))
            
            logger.info(f"Updated trailing take profit for {position_id}: {new_tp_decimal} "
                       f"(Previous: {last_tp['price']}, Current price: {current_price})")
            
            # Update take profit level in active take profits
            last_tp["price"] = new_tp_decimal
            
            return new_tp_decimal
        
        return None
    
    async def handle_take_profit_reached(
        self, 
        position_id: str, 
        take_profit_level: int
    ) -> Dict[str, Any]:
        """
        Handle the event when a take profit level is reached.
        
        Args:
            position_id: Unique identifier for the position
            take_profit_level: The take profit level that was reached (index)
            
        Returns:
            Updated state information
        """
        if position_id not in self.active_take_profits:
            logger.warning(f"Take profit reached for unknown position {position_id}")
            return {"success": False, "reason": "Position not found"}
        
        # Remove the reached take profit level
        if 0 <= take_profit_level < len(self.active_take_profits[position_id]):
            reached_tp = self.active_take_profits[position_id].pop(take_profit_level)
            
            # Check if this was the last take profit level
            remaining_levels = len(self.active_take_profits[position_id])
            
            logger.info(f"Take profit reached for position {position_id} at level {reached_tp['level']}. "
                       f"Remaining levels: {remaining_levels}")
            
            return {
                "success": True,
                "position_id": position_id,
                "remaining_levels": remaining_levels,
                "reached_level": reached_tp
            }
        
        return {"success": False, "reason": "Invalid take profit level"}
    
    async def adjust_take_profit_to_market_conditions(
        self, 
        position_id: str, 
        symbol: str, 
        market_data: MarketData,
        platform: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Adjust take profit levels based on changing market conditions.
        
        Args:
            position_id: Unique identifier for the position
            symbol: Trading symbol
            market_data: Current market data
            platform: Trading platform (BINANCE/DERIV)
            
        Returns:
            Updated take profit orders or None if no update needed
        """
        if position_id not in self.active_take_profits or not self.active_take_profits[position_id]:
            return None
        
        # Get current market conditions
        current_volatility = await run_in_threadpool(
            self.volatility_analyzer.calculate_volatility_percentile,
            market_data.get_candles(symbol, '1h', 100)
        )
        
        # Only adjust if there's a significant change in volatility
        if 0.3 <= current_volatility <= 0.7:
            # Market conditions are moderate, no need to adjust
            return None
        
        remaining_levels = self.active_take_profits[position_id]
        side = remaining_levels[0]["side"]
        
        # Reverse side for take profit (if position is BUY, take profit is SELL)
        position_side = OrderSide.BUY if side == OrderSide.SELL else OrderSide.SELL
        
        # Get the first and last take profit levels to calculate the range
        first_level = remaining_levels[0]["price"]
        last_level = remaining_levels[-1]["price"]
        
        # Adjust take profit range based on volatility
        adjustment_factor = 1.3 if current_volatility > 0.7 else 0.8
        
        # Recalculate levels
        updated_levels = []
        for tp in remaining_levels:
            level = tp["level"]
            original_price = float(tp["price"])
            
            # Adjust price based on volatility
            if position_side == OrderSide.BUY:
                # For long positions, high volatility = extend take profit
                new_price = float(first_level) + (float(last_level) - float(first_level)) * level * adjustment_factor
            else:
                # For short positions, high volatility = extend take profit
                new_price = float(first_level) - (float(first_level) - float(last_level)) * level * adjustment_factor
            
            # Only update if there's a significant change
            if abs(new_price - original_price) / original_price > 0.02:  # 2% threshold
                precision = get_asset_precision(symbol, platform)
                tp["price"] = Decimal(str(round(new_price, precision)))
                updated_levels.append(tp)
        
        if updated_levels:
            logger.info(f"Adjusted {len(updated_levels)} take profit levels for position {position_id} "
                       f"based on volatility: {current_volatility}")
            return updated_levels
        
        return None
    
    def clear_position_take_profits(self, position_id: str) -> bool:
        """
        Clear all take profit levels for a closed position.
        
        Args:
            position_id: Unique identifier for the position
            
        Returns:
            True if take profits were cleared, False if position not found
        """
        if position_id in self.active_take_profits:
            del self.active_take_profits[position_id]
            logger.info(f"Cleared take profit levels for position {position_id}")
            return True
        return False
    
    async def get_take_profit_status(self, position_id: str) -> Dict[str, Any]:
        """
        Get the current take profit status for a position.
        
        Args:
            position_id: Unique identifier for the position
            
        Returns:
            Dictionary with take profit status information
        """
        if position_id not in self.active_take_profits:
            return {"position_id": position_id, "status": "not_found", "levels": []}
        
        levels = self.active_take_profits[position_id]
        return {
            "position_id": position_id,
            "status": "active",
            "levels": levels,
            "count": len(levels),
            "highest_level": max([level["level"] for level in levels]) if levels else 0
        }


def get_take_profit_strategy(name: str, *args, **kwargs) -> BaseTakeProfitStrategy:
    """Instantiate a registered take-profit strategy by name."""
    cls = BaseTakeProfitStrategy.registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown take profit strategy: {name}")
    return cls(*args, **kwargs)

__all__ = ["BaseTakeProfitStrategy", "get_take_profit_strategy"]
