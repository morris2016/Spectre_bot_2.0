#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Stop Loss Management Module

This module contains sophisticated stop loss management strategies
for optimal trade risk management. Includes volatility-based, technical,
and adaptive stop loss calculation and management.
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Type


class BaseStopLossStrategy:
    """Base class for stop loss strategies."""

    registry: Dict[str, Type["BaseStopLossStrategy"]] = {}

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        key = name or cls.__name__
        BaseStopLossStrategy.registry[key] = cls

    def calculate_stop(self, *args, **kwargs):
        raise NotImplementedError

import numpy as np
import pandas as pd
import logging
import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import time
from datetime import datetime, timedelta

from common.utils import (
    safe_divide,
    calculate_risk_reward_ratio,
    pivot_points,
    normalize_value,
)
from feature_service.feature_extraction import atr, fibonacci_levels
from feature_service.features.market_structure import identify_swing_points
from common.constants import (
    DEFAULT_ATR_PERIODS, DEFAULT_ATR_MULTIPLIER,
    DEFAULT_FIXED_STOP_PERCENTAGE, DEFAULT_MIN_STOP_DISTANCE,
    DEFAULT_TRAILING_ACTIVATION_PERCENTAGE, DEFAULT_TRAILING_CALLBACK_RATE,
    MAX_STOP_LEVELS, DEFAULT_CHANDELIER_EXIT_MULTIPLIER
)
from common.exceptions import (
    StopLossError, InvalidParameterError, 
    MarketDataError, CalculationError
)
from common.logger import get_logger
from common.metrics import MetricsCollector
from data_storage.market_data import MarketDataRepository

logger = get_logger("risk_manager.stop_loss")


class StopLossManager(BaseStopLossStrategy):
    """
    Advanced stop loss management system with multiple strategies and dynamic adjustments.
    
    Features:
    - Volatility-based stops using ATR
    - Technical level stops using support/resistance
    - Fibonacci-based stops
    - Multiple timeframe analysis
    - Bracket orders with profit targets
    - Trailing stops with adaptive callback rates
    - Chandelier exits
    - Intelligent stop placement to avoid stop hunting zones
    - Dynamic adjustment based on trade progression
    """
    
    def __init__(
        self,
        market_data_repo: MarketDataRepository,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize the stop loss manager with market data access.
        
        Args:
            market_data_repo: Repository for accessing market data
            metrics_collector: Optional metrics collector for telemetry
        """
        self.market_data_repo = market_data_repo
        self.metrics_collector = metrics_collector
        
        # Internal state
        self._active_stops = {}  # Store active stop orders for tracking
        self._stop_history = {}  # Store history of stop adjustments
        
        # Register metrics
        if self.metrics_collector:
            self.metrics_collector.register_gauge(
                "stop_loss.count", 
                "Number of active stop loss orders"
            )
            self.metrics_collector.register_histogram(
                "stop_loss.distances", 
                "Stop loss distances as percentage of price"
            )
        
        logger.info("Stop Loss Manager initialized")
    
    async def calculate_optimal_stop_loss(
        self,
        asset: str,
        entry_price: float,
        direction: str,
        strategy_type: str = "adaptive",
        risk_percentage: Optional[float] = None,
        timeframe: str = "1h",
        lookback_periods: int = 20,
        consider_liquidity: bool = True,
        stop_hunt_protection: bool = True,
        reference_price: Optional[float] = None,
        volatility_multiplier: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate the optimal stop loss price based on various factors.
        
        Args:
            asset: Trading asset symbol
            entry_price: Entry price for the position
            direction: Trade direction ("long" or "short")
            strategy_type: Type of stop loss strategy to use
            risk_percentage: Target risk percentage (optional)
            timeframe: Timeframe to analyze
            lookback_periods: Number of periods to look back
            consider_liquidity: Whether to consider liquidity zones
            stop_hunt_protection: Whether to avoid common stop hunting zones
            reference_price: Optional reference price (current market price)
            volatility_multiplier: Custom volatility multiplier
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Dict with stop loss details and calculations
        """
        try:
            # Validate inputs
            if entry_price <= 0:
                raise InvalidParameterError("Entry price must be positive")
            
            direction = direction.lower()
            if direction not in ["long", "short"]:
                raise InvalidParameterError("Direction must be 'long' or 'short'")
            
            is_long = direction == "long"
            
            # Convert to Decimal for precise calculations
            entry_price = Decimal(str(entry_price))
            
            # Get reference price if not provided (current market price)
            if reference_price is None:
                reference_price = await self._get_current_price(asset)
            else:
                reference_price = Decimal(str(reference_price))
            
            # Validate reference price
            if reference_price <= 0:
                raise InvalidParameterError("Reference price must be positive")
            
            # Select stop loss calculation strategy
            stop_strategies = {
                "fixed": self._fixed_percentage_stop,
                "volatility": self._volatility_based_stop,
                "technical": self._technical_level_stop,
                "fibonacci": self._fibonacci_stop,
                "swing": self._swing_point_stop,
                "chandelier": self._chandelier_exit_stop,
                "adaptive": self._adaptive_stop,
                "multi_timeframe": self._multi_timeframe_stop,
                "risk_based": self._risk_based_stop,
                "smart": self._smart_detection_stop,
                "ml": self._ml_predicted_stop
            }
            
            if strategy_type not in stop_strategies:
                raise InvalidParameterError(
                    f"Invalid strategy type: {strategy_type}. "
                    f"Available strategies: {', '.join(stop_strategies.keys())}"
                )
            
            # Get the selected stop calculation function
            stop_calculation = stop_strategies[strategy_type]
            
            # Calculate stop loss
            stop_result = await stop_calculation(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                timeframe=timeframe,
                lookback_periods=lookback_periods,
                risk_percentage=risk_percentage,
                volatility_multiplier=volatility_multiplier,
                **kwargs
            )
            
            stop_price = stop_result.get("stop_price")
            
            # Apply additional adjustments if requested
            if consider_liquidity:
                stop_price = await self._adjust_for_liquidity(
                    asset=asset,
                    stop_price=stop_price,
                    is_long=is_long,
                    timeframe=timeframe
                )
            
            if stop_hunt_protection:
                stop_price = await self._avoid_stop_hunting_zones(
                    asset=asset,
                    stop_price=stop_price,
                    is_long=is_long,
                    entry_price=entry_price,
                    timeframe=timeframe
                )
            
            # Calculate risk metrics
            distance = abs(entry_price - stop_price)
            distance_percentage = 100 * distance / entry_price
            
            # Estimate dollar risk if account size is known
            dollar_risk = None
            if "account_size" in kwargs:
                account_size = Decimal(str(kwargs["account_size"]))
                position_size = kwargs.get("position_size", None)
                
                if position_size:
                    position_size = Decimal(str(position_size))
                    dollar_risk = distance * position_size
                elif risk_percentage:
                    risk_pct = Decimal(str(risk_percentage))
                    dollar_risk = account_size * (risk_pct / Decimal('100.0'))
            
            # Validate stop price direction relative to entry
            if is_long and stop_price >= entry_price:
                logger.warning(f"Stop price {stop_price} is above entry price {entry_price} for long position")
                # Set stop to be slightly below entry
                stop_price = entry_price - (entry_price * Decimal('0.01'))
                
            if not is_long and stop_price <= entry_price:
                logger.warning(f"Stop price {stop_price} is below entry price {entry_price} for short position")
                # Set stop to be slightly above entry
                stop_price = entry_price + (entry_price * Decimal('0.01'))
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_observation(
                    "stop_loss.distances", 
                    float(distance_percentage),
                    {"asset": asset, "strategy": strategy_type, "direction": direction}
                )
            
            # Prepare result
            result = {
                "asset": asset,
                "entry_price": float(entry_price),
                "stop_price": float(stop_price),
                "reference_price": float(reference_price),
                "direction": direction,
                "strategy_type": strategy_type,
                "distance": float(distance),
                "distance_percentage": float(distance_percentage),
                "method": stop_result.get("method", strategy_type),
                "timeframe": timeframe,
                "calculation_time": datetime.utcnow().isoformat(),
                **{k: v for k, v in stop_result.items() if k not in ["stop_price", "method"]}
            }
            
            if dollar_risk is not None:
                result["dollar_risk"] = float(dollar_risk)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating optimal stop loss: {str(e)}", exc_info=True)
            raise StopLossError(f"Failed to calculate optimal stop loss: {str(e)}")
    
    async def _get_current_price(self, asset: str) -> Decimal:
        """
        Get the current market price for an asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Current price as Decimal
        """
        try:
            # Attempt to get the latest price from the market data repository
            latest_data = await self.market_data_repo.get_latest_price(asset)
            
            if latest_data is None or latest_data <= 0:
                raise MarketDataError(f"Failed to get valid price data for {asset}")
            
            return Decimal(str(latest_data))
            
        except Exception as e:
            logger.error(f"Error getting current price for {asset}: {str(e)}", exc_info=True)
            raise MarketDataError(f"Failed to get current price for {asset}: {str(e)}")
    
    async def _fixed_percentage_stop(
        self,
        asset: str,
        entry_price: Decimal,
        direction: str,
        is_long: bool,
        reference_price: Decimal,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate a fixed percentage stop loss.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            direction: Trade direction
            is_long: Whether position is long
            reference_price: Current market price
            **kwargs: Additional parameters
            
        Returns:
            Dict with stop loss details
        """
        # Get percentage parameter or use default
        percentage = kwargs.get("percentage", DEFAULT_FIXED_STOP_PERCENTAGE)
        percentage = Decimal(str(percentage)) / Decimal('100.0')  # Convert to decimal
        
        # Ensure minimum percentage
        min_percentage = Decimal('0.005')  # 0.5% minimum stop distance
        percentage = max(percentage, min_percentage)
        
        # Calculate stop price
        if is_long:
            stop_price = entry_price * (Decimal('1.0') - percentage)
        else:
            stop_price = entry_price * (Decimal('1.0') + percentage)
        
        # Ensure minimum stop distance from entry
        min_distance = entry_price * DEFAULT_MIN_STOP_DISTANCE
        if is_long and entry_price - stop_price < min_distance:
            stop_price = entry_price - min_distance
        elif not is_long and stop_price - entry_price < min_distance:
            stop_price = entry_price + min_distance
        
        return {
            "stop_price": stop_price,
            "method": "fixed_percentage",
            "percentage": float(percentage * 100)  # Convert back to percentage for display
        }
    
    async def _volatility_based_stop(
        self,
        asset: str,
        entry_price: Decimal,
        direction: str,
        is_long: bool,
        reference_price: Decimal,
        timeframe: str = "1h",
        lookback_periods: int = DEFAULT_ATR_PERIODS,
        volatility_multiplier: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate a volatility-based stop using ATR.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            direction: Trade direction
            is_long: Whether position is long
            reference_price: Current market price
            timeframe: Timeframe to use
            lookback_periods: ATR lookback periods
            volatility_multiplier: ATR multiplier
            **kwargs: Additional parameters
            
        Returns:
            Dict with stop loss details
        """
        try:
            # Use provided multiplier or default
            multiplier = volatility_multiplier or DEFAULT_ATR_MULTIPLIER
            multiplier = Decimal(str(multiplier))
            
            # Get historical data
            ohlc_data = await self.market_data_repo.get_ohlc_data(
                asset=asset,
                timeframe=timeframe,
                limit=lookback_periods + 10  # Extra data for calculation
            )
            
            if ohlc_data is None or len(ohlc_data) < lookback_periods:
                raise MarketDataError(
                    f"Insufficient data for {asset} on {timeframe} timeframe. "
                    f"Required: {lookback_periods}, Got: {len(ohlc_data) if ohlc_data else 0}"
                )
            
            # Calculate ATR
            atr_value = atr(
                high=ohlc_data['high'].values,
                low=ohlc_data['low'].values,
                close=ohlc_data['close'].values,
                length=lookback_periods
            )
            
            # Get the latest ATR value
            latest_atr = Decimal(str(atr_value[-1]))
            
            # Calculate stop distance
            stop_distance = latest_atr * multiplier
            
            # Calculate stop price
            if is_long:
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance
            
            # Ensure minimum stop distance from entry
            min_distance = entry_price * DEFAULT_MIN_STOP_DISTANCE
            if is_long and entry_price - stop_price < min_distance:
                stop_price = entry_price - min_distance
            elif not is_long and stop_price - entry_price < min_distance:
                stop_price = entry_price + min_distance
            
            return {
                "stop_price": stop_price,
                "method": "volatility_based",
                "atr": float(latest_atr),
                "atr_periods": lookback_periods,
                "multiplier": float(multiplier)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility-based stop: {str(e)}", exc_info=True)
            # Fallback to fixed percentage stop
            logger.info(f"Falling back to fixed percentage stop for {asset}")
            return await self._fixed_percentage_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                **kwargs
            )
    
    async def _technical_level_stop(
        self,
        asset: str,
        entry_price: Decimal,
        direction: str,
        is_long: bool,
        reference_price: Decimal,
        timeframe: str = "1h",
        lookback_periods: int = 100,
        buffer_percentage: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate a stop loss based on technical support/resistance levels.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            direction: Trade direction
            is_long: Whether position is long
            reference_price: Current market price
            timeframe: Timeframe to use
            lookback_periods: Lookback periods
            buffer_percentage: Buffer percentage beyond the level
            **kwargs: Additional parameters
            
        Returns:
            Dict with stop loss details
        """
        try:
            # Use provided buffer or default
            buffer = buffer_percentage or 0.5  # Default 0.5% buffer
            buffer = Decimal(str(buffer)) / Decimal('100.0')  # Convert to decimal fraction
            
            # Get historical data
            ohlc_data = await self.market_data_repo.get_ohlc_data(
                asset=asset,
                timeframe=timeframe,
                limit=lookback_periods + 10  # Extra data for calculation
            )
            
            if ohlc_data is None or len(ohlc_data) < lookback_periods:
                raise MarketDataError(
                    f"Insufficient data for {asset} on {timeframe} timeframe. "
                    f"Required: {lookback_periods}, Got: {len(ohlc_data) if ohlc_data else 0}"
                )
            
            # Find support/resistance levels
            levels = []
            
            # Method 1: Pivot points (recent highs and lows)
            pivot_high, pivot_low = pivot_points(
                ohlc_data['high'].values, 
                ohlc_data['low'].values,
                ohlc_data['close'].values,
                left_bars=2, 
                right_bars=2
            )
            
            # Add pivot highs as resistance and pivot lows as support
            for idx in pivot_high:
                if idx < len(ohlc_data):
                    levels.append({
                        'price': Decimal(str(ohlc_data['high'].iloc[idx])),
                        'type': 'resistance',
                        'strength': 1.0  # Base strength
                    })
            
            for idx in pivot_low:
                if idx < len(ohlc_data):
                    levels.append({
                        'price': Decimal(str(ohlc_data['low'].iloc[idx])),
                        'type': 'support',
                        'strength': 1.0  # Base strength
                    })
            
            # Method 2: Key swing points
            swings = identify_swing_points(
                ohlc_data['high'].values,
                ohlc_data['low'].values,
                ohlc_data['close'].values
            )
            
            for swing in swings:
                if swing['type'] == 'high':
                    levels.append({
                        'price': Decimal(str(swing['price'])),
                        'type': 'resistance',
                        'strength': swing.get('strength', 1.0)
                    })
                else:
                    levels.append({
                        'price': Decimal(str(swing['price'])),
                        'type': 'support',
                        'strength': swing.get('strength', 1.0)
                    })
            
            # Filter and sort levels
            if is_long:
                # For long positions, find support levels below entry
                valid_levels = [
                    level for level in levels 
                    if level['price'] < entry_price and level['type'] == 'support'
                ]
                # Sort by price descending (closest to entry first)
                valid_levels.sort(key=lambda x: x['price'], reverse=True)
            else:
                # For short positions, find resistance levels above entry
                valid_levels = [
                    level for level in levels 
                    if level['price'] > entry_price and level['type'] == 'resistance'
                ]
                # Sort by price ascending (closest to entry first)
                valid_levels.sort(key=lambda x: x['price'])
            
            # Apply further filtering to find the best level
            # Prioritize stronger levels that are not too far from entry
            filtered_levels = []
            max_distance_pct = Decimal('0.05')  # Max 5% away from entry
            
            for level in valid_levels:
                distance_pct = abs(level['price'] - entry_price) / entry_price
                if distance_pct <= max_distance_pct:
                    # Score based on strength and distance (closer is better)
                    level['score'] = level['strength'] * (Decimal('1.0') - (distance_pct / max_distance_pct))
                    filtered_levels.append(level)
            
            # Sort by score descending
            filtered_levels.sort(key=lambda x: x['score'], reverse=True)
            
            # Find the best level or fallback to fixed percentage
            if filtered_levels:
                best_level = filtered_levels[0]
                level_price = best_level['price']
                
                # Apply buffer beyond the level
                if is_long:
                    # Long position: stop below support
                    stop_price = level_price * (Decimal('1.0') - buffer)
                else:
                    # Short position: stop above resistance
                    stop_price = level_price * (Decimal('1.0') + buffer)
                
                # Check if stop is too far from entry
                max_risk_pct = kwargs.get('max_risk_percentage', 10.0)
                max_risk_pct = Decimal(str(max_risk_pct)) / Decimal('100.0')
                
                distance_pct = abs(stop_price - entry_price) / entry_price
                if distance_pct > max_risk_pct:
                    # Stop is too far, use a fixed percentage stop instead
                    logger.info(f"Technical level stop for {asset} is too far ({float(distance_pct * 100):.2f}%), using fixed percentage")
                    return await self._fixed_percentage_stop(
                        asset=asset,
                        entry_price=entry_price,
                        direction=direction,
                        is_long=is_long,
                        reference_price=reference_price,
                        percentage=float(max_risk_pct * 100),
                        **kwargs
                    )
                
                return {
                    "stop_price": stop_price,
                    "method": "technical_level",
                    "level_price": float(level_price),
                    "level_type": best_level['type'],
                    "buffer_percentage": float(buffer * 100),
                    "level_strength": float(best_level['strength']),
                    "level_score": float(best_level['score'])
                }
            else:
                # No suitable levels found, fall back to volatility-based stop
                logger.info(f"No suitable technical levels found for {asset}, falling back to volatility-based stop")
                return await self._volatility_based_stop(
                    asset=asset,
                    entry_price=entry_price,
                    direction=direction,
                    is_long=is_long,
                    reference_price=reference_price,
                    timeframe=timeframe,
                    **kwargs
                )
            
        except Exception as e:
            logger.error(f"Error calculating technical level stop: {str(e)}", exc_info=True)
            # Fallback to volatility-based stop
            logger.info(f"Falling back to volatility-based stop for {asset}")
            return await self._volatility_based_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                timeframe=timeframe,
                **kwargs
            )
    
    async def _fibonacci_stop(
        self,
        asset: str,
        entry_price: Decimal,
        direction: str,
        is_long: bool,
        reference_price: Decimal,
        timeframe: str = "1h",
        fibonacci_level: float = 0.618,
        swing_lookback_periods: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate a Fibonacci-based stop loss.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            direction: Trade direction
            is_long: Whether position is long
            reference_price: Current market price
            timeframe: Timeframe to use
            fibonacci_level: Fibonacci level to use
            swing_lookback_periods: Lookback periods for swing highs/lows
            **kwargs: Additional parameters
            
        Returns:
            Dict with stop loss details
        """
        try:
            # Validate Fibonacci level
            fib_level = Decimal(str(fibonacci_level))
            if fib_level <= 0 or fib_level >= 1:
                raise InvalidParameterError("Fibonacci level must be between 0 and 1")
            
            # Get historical data
            ohlc_data = await self.market_data_repo.get_ohlc_data(
                asset=asset,
                timeframe=timeframe,
                limit=swing_lookback_periods + 10  # Extra data for calculation
            )
            
            if ohlc_data is None or len(ohlc_data) < swing_lookback_periods:
                raise MarketDataError(
                    f"Insufficient data for {asset} on {timeframe} timeframe. "
                    f"Required: {swing_lookback_periods}, Got: {len(ohlc_data) if ohlc_data else 0}"
                )
            
            # Find recent swing high and low
            swings = identify_swing_points(
                ohlc_data['high'].values,
                ohlc_data['low'].values,
                ohlc_data['close'].values
            )
            
            # Filter out old swings
            recent_swings = sorted(swings, key=lambda x: x['index'], reverse=True)[:10]
            
            # For long positions, need a recent swing up move (low to high)
            # For short positions, need a recent swing down move (high to low)
            if is_long:
                # Find most recent swing up move
                swing_lows = [s for s in recent_swings if s['type'] == 'low']
                swing_highs = [s for s in recent_swings if s['type'] == 'high']
                
                if not swing_lows or not swing_highs:
                    raise MarketDataError("Insufficient swing points found")
                
                # Get the most recent swing low
                swing_low = min(swing_lows, key=lambda x: abs(x['index'] - len(ohlc_data)))
                
                # Find a swing high that comes after this low
                valid_highs = [h for h in swing_highs if h['index'] > swing_low['index']]
                
                if not valid_highs:
                    raise MarketDataError("No valid swing high found after swing low")
                
                swing_high = min(valid_highs, key=lambda x: abs(x['index'] - len(ohlc_data)))
                
                # Calculate Fibonacci retracement level
                swing_range = Decimal(str(swing_high['price'])) - Decimal(str(swing_low['price']))
                retracement = Decimal(str(swing_low['price'])) + (swing_range * fib_level)
                
                # Set stop at the retracement level
                stop_price = retracement
                
                # Ensure stop is below entry for long position
                if stop_price >= entry_price:
                    # If retracement is above entry, fall back to swing low
                    stop_price = Decimal(str(swing_low['price']))
                    
                    # If still above entry, use fixed percentage
                    if stop_price >= entry_price:
                        logger.warning(f"Fibonacci stop for long position is above entry price, using fixed percentage")
                        return await self._fixed_percentage_stop(
                            asset=asset,
                            entry_price=entry_price,
                            direction=direction,
                            is_long=is_long,
                            reference_price=reference_price,
                            **kwargs
                        )
                
                swing_data = {
                    "swing_high": float(swing_high['price']),
                    "swing_low": float(swing_low['price']),
                    "swing_high_index": swing_high['index'],
                    "swing_low_index": swing_low['index']
                }
                
            else:  # Short position
                # Find most recent swing down move
                swing_lows = [s for s in recent_swings if s['type'] == 'low']
                swing_highs = [s for s in recent_swings if s['type'] == 'high']
                
                if not swing_lows or not swing_highs:
                    raise MarketDataError("Insufficient swing points found")
                
                # Get the most recent swing high
                swing_high = min(swing_highs, key=lambda x: abs(x['index'] - len(ohlc_data)))
                
                # Find a swing low that comes after this high
                valid_lows = [l for l in swing_lows if l['index'] > swing_high['index']]
                
                if not valid_lows:
                    raise MarketDataError("No valid swing low found after swing high")
                
                swing_low = min(valid_lows, key=lambda x: abs(x['index'] - len(ohlc_data)))
                
                # Calculate Fibonacci retracement level
                swing_range = Decimal(str(swing_high['price'])) - Decimal(str(swing_low['price']))
                retracement = Decimal(str(swing_high['price'])) - (swing_range * fib_level)
                
                # Set stop at the retracement level
                stop_price = retracement
                
                # Ensure stop is above entry for short position
                if stop_price <= entry_price:
                    # If retracement is below entry, fall back to swing high
                    stop_price = Decimal(str(swing_high['price']))
                    
                    # If still below entry, use fixed percentage
                    if stop_price <= entry_price:
                        logger.warning(f"Fibonacci stop for short position is below entry price, using fixed percentage")
                        return await self._fixed_percentage_stop(
                            asset=asset,
                            entry_price=entry_price,
                            direction=direction,
                            is_long=is_long,
                            reference_price=reference_price,
                            **kwargs
                        )
                
                swing_data = {
                    "swing_high": float(swing_high['price']),
                    "swing_low": float(swing_low['price']),
                    "swing_high_index": swing_high['index'],
                    "swing_low_index": swing_low['index']
                }
            
            return {
                "stop_price": stop_price,
                "method": "fibonacci_retracement",
                "fibonacci_level": float(fib_level),
                "retracement_price": float(retracement),
                **swing_data
            }
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci stop: {str(e)}", exc_info=True)
            # Fallback to technical level stop
            logger.info(f"Falling back to technical level stop for {asset}")
            return await self._technical_level_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                timeframe=timeframe,
                **kwargs
            )
    
    async def _swing_point_stop(
        self,
        asset: str,
        entry_price: Decimal,
        direction: str,
        is_long: bool,
        reference_price: Decimal,
        timeframe: str = "1h",
        lookback_periods: int = 20,
        buffer_percentage: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate a stop loss based on the most recent swing point.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            direction: Trade direction
            is_long: Whether position is long
            reference_price: Current market price
            timeframe: Timeframe to use
            lookback_periods: Lookback periods
            buffer_percentage: Buffer percentage beyond the swing point
            **kwargs: Additional parameters
            
        Returns:
            Dict with stop loss details
        """
        try:
            # Use provided buffer or default
            buffer = buffer_percentage or 0.5  # Default 0.5% buffer
            buffer = Decimal(str(buffer)) / Decimal('100.0')  # Convert to decimal fraction
            
            # Get historical data
            ohlc_data = await self.market_data_repo.get_ohlc_data(
                asset=asset,
                timeframe=timeframe,
                limit=lookback_periods + 10  # Extra data for calculation
            )
            
            if ohlc_data is None or len(ohlc_data) < lookback_periods:
                raise MarketDataError(
                    f"Insufficient data for {asset} on {timeframe} timeframe. "
                    f"Required: {lookback_periods}, Got: {len(ohlc_data) if ohlc_data else 0}"
                )
            
            # Find swing points
            swings = identify_swing_points(
                ohlc_data['high'].values,
                ohlc_data['low'].values,
                ohlc_data['close'].values
            )
            
            # Filter recent swings
            recent_swings = sorted(swings, key=lambda x: x['index'], reverse=True)[:5]
            
            if is_long:
                # For long positions, we want the most recent swing low
                swing_lows = [s for s in recent_swings if s['type'] == 'low']
                
                if not swing_lows:
                    raise MarketDataError("No recent swing lows found")
                
                # Get the most recent swing low
                swing_low = swing_lows[0]
                swing_price = Decimal(str(swing_low['price']))
                
                # Ensure it's below entry
                if swing_price >= entry_price:
                    # If swing low is above entry, fall back to volatility-based stop
                    logger.warning(f"Most recent swing low for {asset} is above entry price, using volatility-based stop")
                    return await self._volatility_based_stop(
                        asset=asset,
                        entry_price=entry_price,
                        direction=direction,
                        is_long=is_long,
                        reference_price=reference_price,
                        timeframe=timeframe,
                        **kwargs
                    )
                
                # Apply buffer below swing low
                stop_price = swing_price * (Decimal('1.0') - buffer)
                
                return {
                    "stop_price": stop_price,
                    "method": "swing_point",
                    "swing_price": float(swing_price),
                    "swing_type": "low",
                    "swing_strength": float(swing_low.get('strength', 1.0)),
                    "buffer_percentage": float(buffer * 100)
                }
                
            else:  # Short position
                # For short positions, we want the most recent swing high
                swing_highs = [s for s in recent_swings if s['type'] == 'high']
                
                if not swing_highs:
                    raise MarketDataError("No recent swing highs found")
                
                # Get the most recent swing high
                swing_high = swing_highs[0]
                swing_price = Decimal(str(swing_high['price']))
                
                # Ensure it's above entry
                if swing_price <= entry_price:
                    # If swing high is below entry, fall back to volatility-based stop
                    logger.warning(f"Most recent swing high for {asset} is below entry price, using volatility-based stop")
                    return await self._volatility_based_stop(
                        asset=asset,
                        entry_price=entry_price,
                        direction=direction,
                        is_long=is_long,
                        reference_price=reference_price,
                        timeframe=timeframe,
                        **kwargs
                    )
                
                # Apply buffer above swing high
                stop_price = swing_price * (Decimal('1.0') + buffer)
                
                return {
                    "stop_price": stop_price,
                    "method": "swing_point",
                    "swing_price": float(swing_price),
                    "swing_type": "high",
                    "swing_strength": float(swing_high.get('strength', 1.0)),
                    "buffer_percentage": float(buffer * 100)
                }
            
        except Exception as e:
            logger.error(f"Error calculating swing point stop: {str(e)}", exc_info=True)
            # Fallback to volatility-based stop
            logger.info(f"Falling back to volatility-based stop for {asset}")
            return await self._volatility_based_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                timeframe=timeframe,
                **kwargs
            )
    
    async def _chandelier_exit_stop(
        self,
        asset: str,
        entry_price: Decimal,
        direction: str,
        is_long: bool,
        reference_price: Decimal,
        timeframe: str = "1h",
        lookback_periods: int = 22,
        multiplier: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate a Chandelier Exit stop loss.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            direction: Trade direction
            is_long: Whether position is long
            reference_price: Current market price
            timeframe: Timeframe to use
            lookback_periods: ATR lookback periods
            multiplier: ATR multiplier for Chandelier Exit
            **kwargs: Additional parameters
            
        Returns:
            Dict with stop loss details
        """
        try:
            # Use provided multiplier or default
            ce_multiplier = multiplier or DEFAULT_CHANDELIER_EXIT_MULTIPLIER
            ce_multiplier = Decimal(str(ce_multiplier))
            
            # Get historical data
            ohlc_data = await self.market_data_repo.get_ohlc_data(
                asset=asset,
                timeframe=timeframe,
                limit=lookback_periods + 10  # Extra data for calculation
            )
            
            if ohlc_data is None or len(ohlc_data) < lookback_periods:
                raise MarketDataError(
                    f"Insufficient data for {asset} on {timeframe} timeframe. "
                    f"Required: {lookback_periods}, Got: {len(ohlc_data) if ohlc_data else 0}"
                )
            
            # Calculate ATR
            atr_value = atr(
                high=ohlc_data['high'].values,
                low=ohlc_data['low'].values,
                close=ohlc_data['close'].values,
                length=lookback_periods
            )
            
            # Get the latest ATR value
            latest_atr = Decimal(str(atr_value[-1]))
            
            # Get highest high or lowest low for the period
            if is_long:
                # For long positions, we need the highest high
                highest_high = Decimal(str(ohlc_data['high'].max()))
                
                # Calculate Chandelier Exit: Highest High - (ATR * multiplier)
                stop_price = highest_high - (latest_atr * ce_multiplier)
                
                # Ensure it's below entry price
                if stop_price >= entry_price:
                    stop_price = entry_price - (latest_atr * Decimal('1.0'))
                
            else:  # Short position
                # For short positions, we need the lowest low
                lowest_low = Decimal(str(ohlc_data['low'].min()))
                
                # Calculate Chandelier Exit: Lowest Low + (ATR * multiplier)
                stop_price = lowest_low + (latest_atr * ce_multiplier)
                
                # Ensure it's above entry price
                if stop_price <= entry_price:
                    stop_price = entry_price + (latest_atr * Decimal('1.0'))
            
            return {
                "stop_price": stop_price,
                "method": "chandelier_exit",
                "atr": float(latest_atr),
                "atr_multiplier": float(ce_multiplier),
                "reference_price": float(highest_high) if is_long else float(lowest_low),
                "reference_type": "highest_high" if is_long else "lowest_low"
            }
            
        except Exception as e:
            logger.error(f"Error calculating Chandelier Exit stop: {str(e)}", exc_info=True)
            # Fallback to volatility-based stop
            logger.info(f"Falling back to volatility-based stop for {asset}")
            return await self._volatility_based_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                timeframe=timeframe,
                **kwargs
            )
    
    async def _adaptive_stop(
        self,
        asset: str,
        entry_price: Decimal,
        direction: str,
        is_long: bool,
        reference_price: Decimal,
        timeframe: str = "1h",
        risk_percentage: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate an adaptive stop loss using multiple methods and selecting the best one.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            direction: Trade direction
            is_long: Whether position is long
            reference_price: Current market price
            timeframe: Timeframe to use
            risk_percentage: Target risk percentage
            **kwargs: Additional parameters
            
        Returns:
            Dict with stop loss details
        """
        try:
            # Calculate stops using different methods
            methods = []
            
            # 1. Volatility-based stop
            vol_stop = await self._volatility_based_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                timeframe=timeframe,
                **kwargs
            )
            methods.append(vol_stop)
            
            # 2. Technical level stop
            try:
                tech_stop = await self._technical_level_stop(
                    asset=asset,
                    entry_price=entry_price,
                    direction=direction,
                    is_long=is_long,
                    reference_price=reference_price,
                    timeframe=timeframe,
                    **kwargs
                )
                methods.append(tech_stop)
            except Exception as e:
                logger.warning(f"Technical level stop calculation failed: {str(e)}")
            
            # 3. Swing point stop
            try:
                swing_stop = await self._swing_point_stop(
                    asset=asset,
                    entry_price=entry_price,
                    direction=direction,
                    is_long=is_long,
                    reference_price=reference_price,
                    timeframe=timeframe,
                    **kwargs
                )
                methods.append(swing_stop)
            except Exception as e:
                logger.warning(f"Swing point stop calculation failed: {str(e)}")
            
            # 4. Risk-based stop (if risk percentage provided)
            if risk_percentage:
                try:
                    risk_stop = await self._risk_based_stop(
                        asset=asset,
                        entry_price=entry_price,
                        direction=direction,
                        is_long=is_long,
                        reference_price=reference_price,
                        risk_percentage=risk_percentage,
                        **kwargs
                    )
                    methods.append(risk_stop)
                except Exception as e:
                    logger.warning(f"Risk-based stop calculation failed: {str(e)}")
            
            # If no methods succeeded, fall back to fixed percentage
            if not methods:
                logger.warning(f"All stop calculation methods failed, using fixed percentage")
                return await self._fixed_percentage_stop(
                    asset=asset,
                    entry_price=entry_price,
                    direction=direction,
                    is_long=is_long,
                    reference_price=reference_price,
                    **kwargs
                )
            
            # Score each method based on various factors
            scored_methods = []
            
            for method in methods:
                score = Decimal('0')
                stop_price = Decimal(str(method["stop_price"]))
                
                # Factor 1: Distance from entry as percentage
                distance = abs(entry_price - stop_price)
                distance_percentage = distance / entry_price
                
                # Prefer stops that are not too close or too far
                if distance_percentage < Decimal('0.005'):  # Less than 0.5%
                    # Too close, penalize
                    score -= Decimal('10') * (Decimal('0.005') - distance_percentage) / Decimal('0.005')
                elif distance_percentage > Decimal('0.1'):  # More than 10%
                    # Too far, penalize
                    score -= Decimal('5') * (distance_percentage - Decimal('0.1')) / Decimal('0.1')
                else:
                    # Good distance, reward
                    optimal_distance = Decimal('0.02')  # 2% is ideal
                    score += Decimal('5') * (Decimal('1') - abs(distance_percentage - optimal_distance) / optimal_distance)
                
                # Factor 2: Method reliability
                method_reliability = {
                    "volatility_based": Decimal('8'),
                    "technical_level": Decimal('7'),
                    "swing_point": Decimal('6'),
                    "fibonacci_retracement": Decimal('5'),
                    "chandelier_exit": Decimal('4'),
                    "fixed_percentage": Decimal('3'),
                    "risk_based": Decimal('6')
                }
                
                method_type = method.get("method", "fixed_percentage")
                score += method_reliability.get(method_type, Decimal('3'))
                
                # Factor 3: For technical levels, consider strength
                if method_type == "technical_level" and "level_strength" in method:
                    level_strength = Decimal(str(method["level_strength"]))
                    score += level_strength * Decimal('3')
                
                # Factor 4: For swing points, consider strength
                if method_type == "swing_point" and "swing_strength" in method:
                    swing_strength = Decimal(str(method["swing_strength"]))
                    score += swing_strength * Decimal('2')
                
                # Factor 5: If risk-based, check how close it is to target risk
                if method_type == "risk_based" and risk_percentage:
                    target_risk = Decimal(str(risk_percentage))
                    actual_risk = Decimal(str(method.get("risk_percentage", 0))) / Decimal('100')
                    
                    # Reward being close to target risk
                    risk_diff = abs(actual_risk - target_risk) / target_risk
                    if risk_diff < Decimal('0.2'):  # Within 20% of target
                        score += Decimal('5') * (Decimal('1') - risk_diff / Decimal('0.2'))
                
                # Store the score
                scored_methods.append({
                    "method": method,
                    "score": score,
                    "distance_percentage": distance_percentage
                })
            
            # Sort by score descending
            scored_methods.sort(key=lambda x: x["score"], reverse=True)
            
            # Select the highest scoring method
            best_method = scored_methods[0]["method"]
            
            # Add scoring information
            best_method["adaptive_score"] = float(scored_methods[0]["score"])
            best_method["methods_considered"] = len(methods)
            
            # Additional diagnostics
            scored_summary = [
                {
                    "method": m["method"].get("method", "unknown"),
                    "score": float(m["score"]),
                    "distance_percentage": float(m["distance_percentage"] * 100)
                }
                for m in scored_methods
            ]
            best_method["method_scores"] = scored_summary
            
            logger.info(
                f"Adaptive stop for {asset} selected {best_method['method']} with score "
                f"{float(scored_methods[0]['score']):.2f} from {len(methods)} methods"
            )
            
            return best_method
            
        except Exception as e:
            logger.error(f"Error calculating adaptive stop: {str(e)}", exc_info=True)
            # Fallback to fixed percentage stop
            logger.info(f"Falling back to fixed percentage stop for {asset}")
            return await self._fixed_percentage_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                **kwargs
            )
    
    async def _multi_timeframe_stop(
        self,
        asset: str,
        entry_price: Decimal,
        direction: str,
        is_long: bool,
        reference_price: Decimal,
        timeframe: str = "1h",
        timeframes: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate a stop loss using multiple timeframes.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            direction: Trade direction
            is_long: Whether position is long
            reference_price: Current market price
            timeframe: Primary timeframe to use
            timeframes: List of timeframes to analyze
            **kwargs: Additional parameters
            
        Returns:
            Dict with stop loss details
        """
        try:
            # Use provided timeframes or default set
            if timeframes is None:
                if timeframe == "1m":
                    timeframes = ["1m", "5m", "15m"]
                elif timeframe == "5m":
                    timeframes = ["5m", "15m", "1h"]
                elif timeframe == "15m":
                    timeframes = ["15m", "1h", "4h"]
                elif timeframe == "1h":
                    timeframes = ["1h", "4h", "1d"]
                elif timeframe == "4h":
                    timeframes = ["4h", "1d", "1w"]
                else:
                    timeframes = ["1h", "4h", "1d"]
            
            # Ensure primary timeframe is included
            if timeframe not in timeframes:
                timeframes.insert(0, timeframe)
            
            # Calculate stops for each timeframe
            timeframe_stops = []
            
            for tf in timeframes:
                try:
                    # Use adaptive stop for each timeframe
                    stop = await self._adaptive_stop(
                        asset=asset,
                        entry_price=entry_price,
                        direction=direction,
                        is_long=is_long,
                        reference_price=reference_price,
                        timeframe=tf,
                        **kwargs
                    )
                    
                    # Add timeframe to result
                    stop["timeframe"] = tf
                    timeframe_stops.append(stop)
                    
                except Exception as e:
                    logger.warning(f"Stop calculation failed for timeframe {tf}: {str(e)}")
            
            # If no timeframes succeeded, fall back to fixed percentage
            if not timeframe_stops:
                logger.warning(f"All timeframe calculations failed, using fixed percentage")
                return await self._fixed_percentage_stop(
                    asset=asset,
                    entry_price=entry_price,
                    direction=direction,
                    is_long=is_long,
                    reference_price=reference_price,
                    **kwargs
                )
            
            # Calculate the consensus stop level
            # Weight higher timeframes more heavily
            total_weight = Decimal('0')
            weighted_stop = Decimal('0')
            
            # Define weights based on timeframe position
            timeframe_weights = {
                "1m": Decimal('1.0'),
                "5m": Decimal('1.5'),
                "15m": Decimal('2.0'),
                "30m": Decimal('2.5'),
                "1h": Decimal('3.0'),
                "4h": Decimal('4.0'),
                "1d": Decimal('5.0'),
                "1w": Decimal('6.0')
            }
            
            # Default weight for unknown timeframes
            default_weight = Decimal('2.0')
            
            for stop in timeframe_stops:
                tf = stop["timeframe"]
                stop_price = Decimal(str(stop["stop_price"]))
                
                # Get weight for this timeframe
                weight = timeframe_weights.get(tf, default_weight)
                
                # Add to weighted average
                weighted_stop += stop_price * weight
                total_weight += weight
            
            # Calculate weighted average stop price
            if total_weight > 0:
                consensus_stop = weighted_stop / total_weight
            else:
                # Fallback to first timeframe
                consensus_stop = Decimal(str(timeframe_stops[0]["stop_price"]))
            
            # Format stops from each timeframe for return
            timeframe_results = [
                {
                    "timeframe": stop["timeframe"],
                    "stop_price": stop["stop_price"],
                    "method": stop.get("method", "unknown"),
                    "weight": float(timeframe_weights.get(stop["timeframe"], default_weight))
                }
                for stop in timeframe_stops
            ]
            
            return {
                "stop_price": consensus_stop,
                "method": "multi_timeframe",
                "timeframes_analyzed": len(timeframe_stops),
                "timeframe_stops": timeframe_results,
                "primary_timeframe": timeframe
            }
            
        except Exception as e:
            logger.error(f"Error calculating multi-timeframe stop: {str(e)}", exc_info=True)
            # Fallback to adaptive stop on primary timeframe
            logger.info(f"Falling back to adaptive stop on primary timeframe for {asset}")
            return await self._adaptive_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                timeframe=timeframe,
                **kwargs
            )
    
    async def _risk_based_stop(
        self,
        asset: str,
        entry_price: Decimal,
        direction: str,
        is_long: bool,
        reference_price: Decimal,
        risk_percentage: float,
        position_size: Optional[float] = None,
        account_size: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate a stop loss based on a target risk percentage.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            direction: Trade direction
            is_long: Whether position is long
            reference_price: Current market price
            risk_percentage: Target risk percentage
            position_size: Position size in units
            account_size: Account size in currency
            **kwargs: Additional parameters
            
        Returns:
            Dict with stop loss details
        """
        try:
            # Validate inputs
            if risk_percentage <= 0 or risk_percentage > 20:
                raise InvalidParameterError("Risk percentage must be between 0 and 20")
            
            risk_pct = Decimal(str(risk_percentage)) / Decimal('100.0')  # Convert to decimal fraction
            
            # Both position_size and account_size are needed for risk calculation
            if position_size is None or account_size is None:
                raise InvalidParameterError("Both position_size and account_size are required for risk-based stop")
            
            position_size = Decimal(str(position_size))
            account_size = Decimal(str(account_size))
            
            # Calculate maximum dollar risk
            max_dollar_risk = account_size * risk_pct
            
            # Calculate stop distance in price
            stop_distance = max_dollar_risk / position_size
            
            # Calculate stop price
            if is_long:
                stop_price = entry_price - stop_distance
            else:
                stop_price = entry_price + stop_distance
            
            # Ensure minimum stop distance
            min_distance = entry_price * DEFAULT_MIN_STOP_DISTANCE
            
            if is_long and entry_price - stop_price < min_distance:
                stop_price = entry_price - min_distance
            elif not is_long and stop_price - entry_price < min_distance:
                stop_price = entry_price + min_distance
            
            # Recalculate actual risk after adjustments
            actual_distance = abs(entry_price - stop_price)
            actual_dollar_risk = actual_distance * position_size
            actual_risk_pct = (actual_dollar_risk / account_size) * Decimal('100.0')  # Convert to percentage
            
            return {
                "stop_price": stop_price,
                "method": "risk_based",
                "target_risk_percentage": float(risk_percentage),
                "actual_risk_percentage": float(actual_risk_pct),
                "dollar_risk": float(actual_dollar_risk),
                "account_size": float(account_size),
                "position_size": float(position_size)
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk-based stop: {str(e)}", exc_info=True)
            # Fallback to fixed percentage stop
            logger.info(f"Falling back to fixed percentage stop for {asset}")
            return await self._fixed_percentage_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                percentage=risk_percentage if risk_percentage <= 10 else 10,
                **kwargs
            )
    
    async def _smart_detection_stop(
        self,
        asset: str,
        entry_price: Decimal,
        direction: str,
        is_long: bool,
        reference_price: Decimal,
        timeframe: str = "1h",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate a stop loss using smart market structure detection.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            direction: Trade direction
            is_long: Whether position is long
            reference_price: Current market price
            timeframe: Timeframe to use
            **kwargs: Additional parameters
            
        Returns:
            Dict with stop loss details
        """
        try:
            # Get market data for analysis
            ohlc_data = await self.market_data_repo.get_ohlc_data(
                asset=asset,
                timeframe=timeframe,
                limit=100  # Get sufficient data for analysis
            )
            
            if ohlc_data is None or len(ohlc_data) < 50:
                raise MarketDataError(
                    f"Insufficient data for {asset} on {timeframe} timeframe. "
                    f"Required: 50, Got: {len(ohlc_data) if ohlc_data else 0}"
                )
            
            # Determine market structure type (trending, ranging, choppy)
            # This is a simplified version - in production, more sophisticated detection would be used
            
            # Calculate some basic indicators
            closes = ohlc_data['close'].values
            highs = ohlc_data['high'].values
            lows = ohlc_data['low'].values
            
            # Calculate recent volatility using ATR
            atr_value = atr(highs, lows, closes, 14)
            recent_atr = atr_value[-1]
            
            # Calculate average true range as percentage of price
            recent_close = closes[-1]
            atr_pct = recent_atr / recent_close
            
            # Calculate simple moving averages for trend detection
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else np.mean(closes)
            
            # Detect market structure
            if atr_pct > 0.03:  # High volatility (more than 3%)
                market_structure = "volatile"
            elif abs(sma_20 - sma_50) / sma_50 > 0.05:  # Strong trend
                market_structure = "trending"
                trend_direction = "up" if sma_20 > sma_50 else "down"
            else:  # Ranging market
                market_structure = "ranging"
            
            # Select stop strategy based on market structure
            if market_structure == "volatile":
                # In volatile markets, use a wider volatility-based stop
                volatility_multiplier = 3.0  # Use a wider multiplier
                return await self._volatility_based_stop(
                    asset=asset,
                    entry_price=entry_price,
                    direction=direction,
                    is_long=is_long,
                    reference_price=reference_price,
                    timeframe=timeframe,
                    volatility_multiplier=volatility_multiplier,
                    **kwargs
                )
            
            elif market_structure == "trending":
                # In trending markets, use swing point or chandelier exit
                if (is_long and trend_direction == "up") or (not is_long and trend_direction == "down"):
                    # Trading with the trend - use chandelier exit
                    return await self._chandelier_exit_stop(
                        asset=asset,
                        entry_price=entry_price,
                        direction=direction,
                        is_long=is_long,
                        reference_price=reference_price,
                        timeframe=timeframe,
                        **kwargs
                    )
                else:
                    # Trading against the trend - use tight swing point stop
                    return await self._swing_point_stop(
                        asset=asset,
                        entry_price=entry_price,
                        direction=direction,
                        is_long=is_long,
                        reference_price=reference_price,
                        timeframe=timeframe,
                        buffer_percentage=0.3,  # Tighter buffer
                        **kwargs
                    )
            
            else:  # Ranging market
                # In ranging markets, use technical level stops
                return await self._technical_level_stop(
                    asset=asset,
                    entry_price=entry_price,
                    direction=direction,
                    is_long=is_long,
                    reference_price=reference_price,
                    timeframe=timeframe,
                    **kwargs
                )
            
        except Exception as e:
            logger.error(f"Error calculating smart detection stop: {str(e)}", exc_info=True)
            # Fallback to adaptive stop
            logger.info(f"Falling back to adaptive stop for {asset}")
            return await self._adaptive_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                timeframe=timeframe,
                **kwargs
            )
    
    async def _ml_predicted_stop(
        self,
        asset: str,
        entry_price: Decimal,
        direction: str,
        is_long: bool,
        reference_price: Decimal,
        timeframe: str = "1h",
        model_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Use machine learning to predict optimal stop loss level.
        
        This is a placeholder implementation - in production, this would use
        actual ML models to predict optimal stop placement.
        
        Args:
            asset: Asset symbol
            entry_price: Entry price
            direction: Trade direction
            is_long: Whether position is long
            reference_price: Current market price
            timeframe: Timeframe to use
            model_id: ID of the ML model to use
            **kwargs: Additional parameters
            
        Returns:
            Dict with stop loss details
        """
        try:
            # This is a placeholder - in production, this would call an actual ML model
            logger.warning("ML predicted stop is not fully implemented, using adaptive stop instead")
            
            # Fall back to adaptive stop
            return await self._adaptive_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                timeframe=timeframe,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error calculating ML predicted stop: {str(e)}", exc_info=True)
            # Fallback to adaptive stop
            logger.info(f"Falling back to adaptive stop for {asset}")
            return await self._adaptive_stop(
                asset=asset,
                entry_price=entry_price,
                direction=direction,
                is_long=is_long,
                reference_price=reference_price,
                timeframe=timeframe,
                **kwargs
            )
    
    async def _adjust_for_liquidity(
        self,
        asset: str,
        stop_price: Decimal,
        is_long: bool,
        timeframe: str = "1h"
    ) -> Decimal:
        """
        Adjust stop price to account for liquidity zones.
        
        Args:
            asset: Asset symbol
            stop_price: Initial stop price
            is_long: Whether position is long
            timeframe: Timeframe to use
            
        Returns:
            Adjusted stop price
        """
        try:
            # Get historical data
            ohlc_data = await self.market_data_repo.get_ohlc_data(
                asset=asset,
                timeframe=timeframe,
                limit=50  # Recent data for liquidity analysis
            )
            
            if ohlc_data is None or len(ohlc_data) < 20:
                logger.warning(f"Insufficient data for liquidity analysis, using original stop price")
                return stop_price
            
            # Define a range around the stop price to check for liquidity
            price_range_pct = Decimal('0.02')  # 2% range
            range_min = stop_price * (Decimal('1.0') - price_range_pct)
            range_max = stop_price * (Decimal('1.0') + price_range_pct)
            
            # Look for price levels with high volume (liquidity zones)
            high_volume_levels = []
            mean_volume = np.mean(ohlc_data['volume'].values)
            
            for i, row in ohlc_data.iterrows():
                # Check if this bar has high volume
                if row['volume'] > mean_volume * 1.5:
                    high = Decimal(str(row['high']))
                    low = Decimal(str(row['low']))
                    
                    # Check if this bar's range intersects with our stop range
                    if (high >= range_min and high <= range_max) or (low >= range_min and low <= range_max):
                        # This is a high volume zone near our stop
                        high_volume_levels.append({
                            'high': high,
                            'low': low,
                            'volume': row['volume']
                        })
            
            if not high_volume_levels:
                # No high volume zones found, return original stop
                return stop_price
            
            # Adjust stop to avoid high volume zones
            if is_long:
                # For long positions, we want to place stop below liquidity
                # Sort by low price descending
                high_volume_levels.sort(key=lambda x: x['low'], reverse=True)
                
                for level in high_volume_levels:
                    if level['low'] < stop_price:
                        # Place stop just below this level
                        new_stop = level['low'] * Decimal('0.995')  # 0.5% below
                        logger.info(f"Adjusted long stop from {float(stop_price)} to {float(new_stop)} to avoid liquidity zone")
                        return new_stop
            else:
                # For short positions, we want to place stop above liquidity
                # Sort by high price ascending
                high_volume_levels.sort(key=lambda x: x['high'])
                
                for level in high_volume_levels:
                    if level['high'] > stop_price:
                        # Place stop just above this level
                        new_stop = level['high'] * Decimal('1.005')  # 0.5% above
                        logger.info(f"Adjusted short stop from {float(stop_price)} to {float(new_stop)} to avoid liquidity zone")
                        return new_stop
            
            # If no adjustment made, return original stop
            return stop_price
            
        except Exception as e:
            logger.error(f"Error adjusting for liquidity: {str(e)}", exc_info=True)
            # Return original stop price on error
            return stop_price
    
    async def _avoid_stop_hunting_zones(
        self,
        asset: str,
        stop_price: Decimal,
        is_long: bool,
        entry_price: Decimal,
        timeframe: str = "1h"
    ) -> Decimal:
        """
        Adjust stop price to avoid common stop hunting zones.
        
        Args:
            asset: Asset symbol
            stop_price: Initial stop price
            is_long: Whether position is long
            entry_price: Entry price
            timeframe: Timeframe to use
            
        Returns:
            Adjusted stop price
        """
        try:
            # Check for round numbers (common stop hunting zones)
            # Convert to float for easier manipulation
            stop_float = float(stop_price)
            
            # Get price precision
            precision = 0
            str_price = str(stop_float)
            if '.' in str_price:
                precision = len(str_price) - str_price.index('.') - 1
            
            # Identify round numbers based on price magnitude
            if stop_float < 0.1:
                round_levels = [0.001, 0.005, 0.01, 0.025, 0.05]
            elif stop_float < 1:
                round_levels = [0.01, 0.05, 0.1, 0.25, 0.5]
            elif stop_float < 10:
                round_levels = [0.1, 0.25, 0.5, 1, 2.5, 5]
            elif stop_float < 100:
                round_levels = [1, 2.5, 5, 10, 25, 50]
            elif stop_float < 1000:
                round_levels = [10, 25, 50, 100, 250, 500]
            else:
                round_levels = [100, 250, 500, 1000, 2500, 5000]
            
            # Check if stop is near a round number
            for level in round_levels:
                nearest_round = round(stop_float / level) * level
                distance_pct = abs(nearest_round - stop_float) / stop_float
                
                if distance_pct < 0.005:  # Within 0.5% of a round number
                    # Need to adjust stop away from round number
                    if is_long:
                        # For long positions, move stop lower
                        new_stop = Decimal(str(nearest_round)) * Decimal('0.99')  # 1% below round number
                        
                        # Ensure the stop isn't too far from entry
                        max_distance = entry_price * Decimal('0.1')  # Max 10% from entry
                        if entry_price - new_stop > max_distance:
                            new_stop = entry_price - max_distance
                        
                        logger.info(f"Adjusted long stop from {float(stop_price)} to {float(new_stop)} to avoid stop hunt zone")
                        return new_stop
                    else:
                        # For short positions, move stop higher
                        new_stop = Decimal(str(nearest_round)) * Decimal('1.01')  # 1% above round number
                        
                        # Ensure the stop isn't too far from entry
                        max_distance = entry_price * Decimal('0.1')  # Max 10% from entry
                        if new_stop - entry_price > max_distance:
                            new_stop = entry_price + max_distance
                        
                        logger.info(f"Adjusted short stop from {float(stop_price)} to {float(new_stop)} to avoid stop hunt zone")
                        return new_stop
            
            # If no adjustment needed, return original stop
            return stop_price
            
        except Exception as e:
            logger.error(f"Error avoiding stop hunting zones: {str(e)}", exc_info=True)
            # Return original stop price on error
            return stop_price
    
    async def calculate_take_profit_levels(
        self,
        asset: str,
        entry_price: float,
        stop_loss: float,
        direction: str,
        risk_reward_levels: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Calculate tiered take profit prices based on risk-reward ratios.

        Args:
            asset: Trading asset symbol.
            entry_price: Entry price of the position.
            stop_loss: Stop loss price for the position.
            direction: Trade direction ("long" or "short").
            risk_reward_levels: Optional list of reward:risk ratios
                for each take profit tier.

        Returns:
            Dictionary containing calculated take profit levels.
        """
        try:
            if risk_reward_levels is None:
                risk_reward_levels = [1.0, 2.0, 3.0]

            direction = direction.lower()
            if direction not in ["long", "short"]:
                raise InvalidParameterError("Direction must be 'long' or 'short'")

            is_long = direction == "long"

            entry_dec = Decimal(str(entry_price))
            stop_dec = Decimal(str(stop_loss))

            risk_value = (entry_dec - stop_dec) if is_long else (stop_dec - entry_dec)
            if risk_value <= 0:
                raise InvalidParameterError("Stop loss must be on the risk side of the entry price")

            levels = []
            for idx, rr in enumerate(risk_reward_levels):
                rr_dec = Decimal(str(rr))
                tp_price = entry_dec + risk_value * rr_dec if is_long else entry_dec - risk_value * rr_dec
                levels.append({
                    "level": idx + 1,
                    "ratio": float(rr_dec),
                    "price": float(tp_price)
                })

            return {
                "asset": asset,
                "entry_price": float(entry_dec),
                "stop_loss": float(stop_dec),
                "direction": direction,
                "risk": float(risk_value),
                "levels": levels,
                "calculation_time": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating take profit levels: {str(e)}", exc_info=True)
            raise CalculationError(f"Failed to calculate take profit levels: {str(e)}")


def get_stop_loss_strategy(name: str, *args, **kwargs) -> BaseStopLossStrategy:
    """Instantiate a registered stop-loss strategy by name."""
    cls = BaseStopLossStrategy.registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown stop loss strategy: {name}")
    return cls(*args, **kwargs)

__all__ = ["BaseStopLossStrategy", "get_stop_loss_strategy"]
