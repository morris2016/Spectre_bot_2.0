
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Position Sizing Module

This module contains advanced position sizing algorithms that dynamically adjust
position sizes based on confidence levels, account growth, market volatility,
expected value, and risk profile.
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Type


class BasePositionSizer:
    """Base class for position sizing strategies."""

    registry: Dict[str, Type["BasePositionSizer"]] = {}

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        key = name or cls.__name__
        BasePositionSizer.registry[key] = cls

    def size_position(self, *args, **kwargs):
        raise NotImplementedError

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import asyncio
from decimal import Decimal, ROUND_DOWN, ROUND_UP

from common.utils import (
    safe_divide, calculate_risk_reward_ratio,
    calculate_expected_value, normalize_value
)
from common.constants import (
    DEFAULT_MAX_RISK_PER_TRADE, DEFAULT_BASE_POSITION_SIZE,
    DEFAULT_KELLY_FRACTION, MAX_LEVERAGE_BINANCE, MAX_LEVERAGE_DERIV,
    POSITION_SIZE_PRECISION, DEFAULT_GROWTH_FACTOR
)
from common.exceptions import (
    PositionSizingError, InvalidParameterError, 
    InsufficientBalanceError, RiskExceededError
)
from common.logger import get_logger
from common.metrics import MetricsCollector
from common.redis_client import RedisClient
from data_storage.models.strategy_data import StrategyPerformance
from data_storage.models.system_data import SystemConfig

logger = get_logger("risk_manager.position_sizing")


class PositionSizing(BasePositionSizer):
    """
    Advanced position sizing system with multiple algorithms and risk controls.
    Dynamically adapts position sizes based on strategy performance, market conditions,
    and account growth phases.
    """
    
    def __init__(
        self, 
        account_balance: float,
        max_risk_per_trade: float = DEFAULT_MAX_RISK_PER_TRADE,
        platform: str = "binance",
        redis_client: Optional[RedisClient] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize the position sizing system with account balance and risk parameters.
        
        Args:
            account_balance: Current account balance
            max_risk_per_trade: Maximum percentage of account balance to risk per trade
            platform: Trading platform (binance or deriv)
            redis_client: Redis client for caching
            metrics_collector: Metrics collector for performance tracking
        """
        self.account_balance = Decimal(str(account_balance))
        self.max_risk_per_trade = Decimal(str(max_risk_per_trade))
        self.platform = platform.lower()
        self.redis_client = redis_client
        self.metrics_collector = metrics_collector
        
        # Platform-specific parameters
        self.max_leverage = MAX_LEVERAGE_BINANCE if self.platform == "binance" else MAX_LEVERAGE_DERIV
        
        # Cache for recent calculations
        self._calculation_cache = {}
        
        # State tracking for dynamic adjustments
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.initial_account_balance = self.account_balance
        self.peak_account_balance = self.account_balance
        
        # Performance tracking
        self.win_rate = Decimal('0.5')  # Start with neutral assumption
        self.avg_win_loss_ratio = Decimal('1.0')  # Start with neutral assumption
        self.recent_trades_performance = []
        
        # System state tracking
        self.in_drawdown = False
        self.drawdown_severity = Decimal('0.0')
        self.account_growth_phase = self._calculate_growth_phase()
        
        # Initialize metrics
        if self.metrics_collector:
            self.metrics_collector.register_gauge(
                "position_sizing.account_balance", 
                "Current account balance", 
                float(self.account_balance)
            )
            self.metrics_collector.register_gauge(
                "position_sizing.risk_per_trade", 
                "Current risk per trade (%)", 
                float(self.max_risk_per_trade * 100)
            )
            
        logger.info(f"Position sizing initialized with account balance: {self.account_balance}, "
                   f"max risk per trade: {self.max_risk_per_trade}, platform: {self.platform}")
    
    async def update_account_balance(self, new_balance: float) -> None:
        """
        Update the account balance and recalculate related metrics.
        
        Args:
            new_balance: New account balance value
        """
        old_balance = self.account_balance
        self.account_balance = Decimal(str(new_balance))
        
        # Update peak balance if new balance is higher
        if self.account_balance > self.peak_account_balance:
            self.peak_account_balance = self.account_balance
        
        # Calculate drawdown state
        if self.account_balance < self.peak_account_balance:
            self.in_drawdown = True
            self.drawdown_severity = (self.peak_account_balance - self.account_balance) / self.peak_account_balance
        else:
            self.in_drawdown = False
            self.drawdown_severity = Decimal('0.0')
        
        # Update growth phase
        self.account_growth_phase = self._calculate_growth_phase()
        
        # Update metrics
        if self.metrics_collector:
            self.metrics_collector.set_gauge_value(
                "position_sizing.account_balance", 
                float(self.account_balance)
            )
            self.metrics_collector.register_gauge(
                "position_sizing.drawdown_severity",
                "Current drawdown severity (%)",
                float(self.drawdown_severity * 100)
            )
        
        change_pct = ((self.account_balance - old_balance) / old_balance) * 100 if old_balance > 0 else 0
        logger.info(f"Account balance updated: {old_balance} → {self.account_balance} "
                   f"({change_pct:+.2f}%). Growth phase: {self.account_growth_phase}, "
                   f"Drawdown: {self.drawdown_severity:.2%}")
    
    async def update_performance_metrics(
        self, 
        win_rate: float, 
        avg_win_loss_ratio: float,
        recent_trades: List[Dict[str, Any]] = None
    ) -> None:
        """
        Update the performance metrics used for position sizing calculations.
        
        Args:
            win_rate: Current win rate (0.0 to 1.0)
            avg_win_loss_ratio: Average profit-to-loss ratio
            recent_trades: Recent trade data for more detailed analysis
        """
        self.win_rate = Decimal(str(win_rate))
        self.avg_win_loss_ratio = Decimal(str(avg_win_loss_ratio))
        
        if recent_trades:
            self.recent_trades_performance = recent_trades
            
            # Calculate consecutive wins/losses
            if recent_trades and len(recent_trades) > 0:
                self.consecutive_wins = 0
                self.consecutive_losses = 0
                
                for trade in reversed(recent_trades):
                    if trade.get('profit', 0) > 0:
                        if self.consecutive_losses > 0:
                            break
                        self.consecutive_wins += 1
                    else:
                        if self.consecutive_wins > 0:
                            break
                        self.consecutive_losses += 1
        
        logger.info(f"Performance metrics updated: Win rate: {self.win_rate:.2%}, "
                   f"Win/Loss ratio: {self.avg_win_loss_ratio:.2f}, "
                   f"Consecutive wins: {self.consecutive_wins}, "
                   f"Consecutive losses: {self.consecutive_losses}")
    
    async def _get_asset_specific_params(self, asset: str) -> Dict[str, Any]:
        """
        Get asset-specific parameters for position sizing.
        
        Args:
            asset: The trading asset symbol
            
        Returns:
            Dict containing asset-specific parameters
        """
        # Try to get from cache first
        cache_key = f"asset_params:{asset}:{self.platform}"
        if self.redis_client:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return cached_data
        
        # Default parameters if none found
        params = {
            'volatility': Decimal('0.02'),  # Default 2% daily volatility
            'tick_size': Decimal('0.01'),   # Default smallest price increment
            'min_quantity': Decimal('0.001'), # Default minimum order size
            'quantity_precision': 3,        # Default decimal places for quantity
            'max_leverage': self.max_leverage,
            'margin_requirement': Decimal('0.1') if self.platform == 'binance' else Decimal('0.05'), # 10% or 5%
            'fixed_parameters': False       # Flag to indicate if parameters are fixed or estimated
        }
        
        try:
            # Here we would query the database or API for asset-specific parameters
            # For demonstration, we'll use the default parameters
            
            # In practice, this would load from data_storage
            # asset_data = await AssetParameters.get_by_symbol(asset, self.platform)
            # if asset_data:
            #     params.update(asset_data)
            #     params['fixed_parameters'] = True
            
            # Cache the result
            if self.redis_client:
                await self.redis_client.set(cache_key, params, expires=3600)  # Cache for 1 hour
                
            return params
            
        except Exception as e:
            logger.warning(f"Failed to get asset parameters for {asset}: {str(e)}. Using defaults.")
            return params
    
    async def calculate_position_size(
        self,
        asset: str,
        entry_price: float,
        stop_loss_price: float,
        confidence: float = 0.7,
        signal_type: str = "standard",
        strategy_id: Optional[str] = None,
        take_profit_price: Optional[float] = None,
        market_volatility: Optional[float] = None,
        leverage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate the optimal position size based on various factors.
        
        Args:
            asset: Trading asset symbol
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price for the position
            confidence: Signal confidence level (0.0 to 1.0)
            signal_type: Type of trading signal (standard, scalp, swing, etc.)
            strategy_id: ID of the strategy generating the signal
            take_profit_price: Take profit price (optional)
            market_volatility: Current market volatility (optional)
            leverage: Desired leverage to use (optional)
            
        Returns:
            Dict with position size details and calculations
        """
        try:
            # Validate input parameters
            if entry_price <= 0 or stop_loss_price <= 0:
                raise InvalidParameterError("Entry and stop loss prices must be positive")
            
            if confidence < 0 or confidence > 1:
                raise InvalidParameterError("Confidence must be between 0 and 1")
            
            entry_price = Decimal(str(entry_price))
            stop_loss_price = Decimal(str(stop_loss_price))
            confidence = Decimal(str(confidence))
            
            # Get asset-specific parameters
            asset_params = await self._get_asset_specific_params(asset)
            
            # Calculate risk per pip
            is_long = entry_price > stop_loss_price
            risk_price_distance = abs(entry_price - stop_loss_price)
            
            # Risk percentage adjusted for confidence and current system state
            adjusted_risk_pct = await self._calculate_adjusted_risk_percentage(
                confidence, signal_type, strategy_id, market_volatility
            )
            
            # Dollar risk amount based on adjusted risk percentage
            dollar_risk = self.account_balance * adjusted_risk_pct
            
            # Calculate position size based on fixed fractional method first
            fixed_fractional_size = await self._fixed_fractional_position_size(
                dollar_risk, risk_price_distance, entry_price
            )
            
            # If take profit is provided, calculate expected value and adjust with Kelly
            if take_profit_price is not None:
                take_profit_price = Decimal(str(take_profit_price))
                reward_price_distance = abs(take_profit_price - entry_price)
                risk_reward_ratio = reward_price_distance / risk_price_distance if risk_price_distance > 0 else Decimal('1.0')
                
                # Calculate Kelly position size
                kelly_size = await self._kelly_criterion_position_size(
                    risk_reward_ratio, confidence
                )
                
                # Combine methods with weighting
                position_size = (fixed_fractional_size * Decimal('0.7')) + (kelly_size * Decimal('0.3'))
            else:
                # Without take profit, use fixed fractional with confidence adjustment
                position_size = fixed_fractional_size * (Decimal('0.5') + (confidence * Decimal('0.5')))
            
            # Apply asset-specific constraints
            position_size = self._apply_asset_constraints(position_size, asset_params)
            
            # Apply leverage if provided and allowed
            max_allowed_leverage = min(
                asset_params['max_leverage'],
                Decimal(str(leverage)) if leverage is not None else asset_params['max_leverage']
            )
            
            # Determine appropriate leverage based on risk profile
            if leverage is None:
                # Conservative leverage calculation if not specified
                calculated_leverage = Decimal('1.0')
                
                # Only increase leverage if confident and in good standing
                if confidence > Decimal('0.7') and self.drawdown_severity < Decimal('0.1'):
                    volatility_factor = Decimal('1.0')
                    if market_volatility is not None:
                        volatility_factor = Decimal('1.0') / (Decimal(str(market_volatility)) * Decimal('10.0'))
                        volatility_factor = max(min(volatility_factor, Decimal('2.0')), Decimal('0.5'))
                    
                    calculated_leverage = min(
                        Decimal('1.0') + (confidence * Decimal('3.0') * volatility_factor),
                        max_allowed_leverage
                    )
                
                leverage = calculated_leverage
            else:
                leverage = min(Decimal(str(leverage)), max_allowed_leverage)
            
            # Adjust position size for leverage
            leveraged_position_size = position_size * leverage
            
            # Calculate final quantity with proper rounding
            quantity_precision = asset_params['quantity_precision']
            quantity = self._round_to_precision(leveraged_position_size / entry_price, quantity_precision)
            
            # Ensure minimum quantity is met
            if quantity < asset_params['min_quantity']:
                if dollar_risk > 0 and self.account_balance >= asset_params['min_quantity'] * entry_price:
                    quantity = asset_params['min_quantity']
                else:
                    raise InsufficientBalanceError(
                        f"Account balance too small for minimum quantity on {asset}"
                    )
            
            # Calculate notional value and margin required
            notional_value = quantity * entry_price
            margin_required = notional_value / leverage
            
            # Verify margin requirement is met
            if margin_required > self.account_balance:
                max_possible_quantity = self._round_to_precision(
                    (self.account_balance * leverage) / entry_price,
                    quantity_precision
                )
                
                if max_possible_quantity >= asset_params['min_quantity']:
                    quantity = max_possible_quantity
                    notional_value = quantity * entry_price
                    margin_required = notional_value / leverage
                    logger.warning(f"Reduced position size due to insufficient balance for {asset}")
                else:
                    raise InsufficientBalanceError(
                        f"Insufficient balance for minimum position on {asset}"
                    )
            
            # Calculate potential loss at stop price
            potential_loss = (abs(entry_price - stop_loss_price) * quantity)
            loss_percentage = potential_loss / self.account_balance
            
            # Double-check the risk percentage doesn't exceed our maximum
            if loss_percentage > self.max_risk_per_trade * Decimal('1.1'):  # Allow 10% buffer
                # Scale down the position to match max risk
                quantity = self._round_to_precision(
                    (self.max_risk_per_trade * self.account_balance) / risk_price_distance,
                    quantity_precision
                )
                notional_value = quantity * entry_price
                margin_required = notional_value / leverage
                potential_loss = (abs(entry_price - stop_loss_price) * quantity)
                loss_percentage = potential_loss / self.account_balance
                
                logger.warning(f"Position size reduced to meet maximum risk constraint for {asset}")
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_observation(
                    "position_sizing.calculated_sizes", 
                    float(position_size),
                    {"asset": asset, "signal_type": signal_type}
                )
                self.metrics_collector.record_observation(
                    "position_sizing.risk_percentages", 
                    float(loss_percentage * 100),
                    {"asset": asset, "signal_type": signal_type}
                )
            
            # Prepare and return the result
            result = {
                "asset": asset,
                "position_size_base": float(position_size),
                "position_size_leveraged": float(leveraged_position_size),
                "quantity": float(quantity),
                "notional_value": float(notional_value),
                "margin_required": float(margin_required),
                "leverage": float(leverage),
                "entry_price": float(entry_price),
                "risk_price_distance": float(risk_price_distance),
                "risk_percentage": float(adjusted_risk_pct * 100),
                "dollar_risk": float(dollar_risk),
                "potential_loss": float(potential_loss),
                "loss_percentage": float(loss_percentage * 100),
                "direction": "long" if is_long else "short",
                "confidence": float(confidence),
                "calculated_at": datetime.utcnow().isoformat(),
                "platform": self.platform,
                "signal_type": signal_type
            }
            
            if take_profit_price is not None:
                result.update({
                    "take_profit_price": float(take_profit_price),
                    "reward_price_distance": float(reward_price_distance),
                    "risk_reward_ratio": float(risk_reward_ratio),
                    "potential_profit": float(reward_price_distance * quantity),
                    "expected_value": float(
                        (confidence * reward_price_distance * quantity) - 
                        ((1 - confidence) * risk_price_distance * quantity)
                    )
                })
            
            # Cache the calculation
            calc_key = f"{asset}:{float(entry_price)}:{float(stop_loss_price)}:{float(confidence)}"
            self._calculation_cache[calc_key] = result
            
            logger.info(f"Position size calculated for {asset}: quantity={float(quantity)}, "
                      f"leverage={float(leverage)}x, risk={float(loss_percentage * 100):.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}", exc_info=True)
            raise PositionSizingError(f"Failed to calculate position size: {str(e)}")
    
    async def _calculate_adjusted_risk_percentage(
        self,
        confidence: Decimal,
        signal_type: str,
        strategy_id: Optional[str],
        market_volatility: Optional[float]
    ) -> Decimal:
        """
        Calculate the adjusted risk percentage based on various factors.
        
        Args:
            confidence: Signal confidence
            signal_type: Type of trading signal
            strategy_id: Strategy ID
            market_volatility: Market volatility
            
        Returns:
            Adjusted risk percentage
        """
        # Base risk is the maximum risk per trade
        base_risk = self.max_risk_per_trade
        
        # Adjust for confidence level (higher confidence = higher risk tolerance)
        confidence_factor = Decimal('0.5') + (confidence * Decimal('0.5'))
        
        # Adjust for signal type
        signal_type_factors = {
            "scalp": Decimal('0.7'),    # Scalping uses less risk per trade
            "swing": Decimal('1.2'),    # Swing trading can use slightly more risk
            "trend": Decimal('1.3'),    # Trend following can use more risk
            "reversal": Decimal('0.8'), # Reversals are higher risk, use less
            "breakout": Decimal('1.1'), # Breakouts can use slightly more risk
            "standard": Decimal('1.0')  # Standard signals use the baseline risk
        }
        signal_factor = signal_type_factors.get(signal_type.lower(), Decimal('1.0'))
        
        # Adjust for consecutive wins/losses (martingale-like but with limits)
        streak_factor = Decimal('1.0')
        if self.consecutive_wins > 3:
            # Increase risk slightly after winning streak
            streak_factor = min(Decimal('1.0') + (Decimal('0.05') * (self.consecutive_wins - 3)), Decimal('1.25'))
        elif self.consecutive_losses > 1:
            # Decrease risk after losing streak
            streak_factor = max(Decimal('1.0') - (Decimal('0.15') * self.consecutive_losses), Decimal('0.4'))
        
        # Adjust for drawdown
        drawdown_factor = Decimal('1.0')
        if self.in_drawdown:
            # Reduce risk during drawdown periods
            drawdown_factor = max(Decimal('1.0') - (self.drawdown_severity * Decimal('2.0')), Decimal('0.3'))
        
        # Adjust for account growth phase
        growth_phase_factors = {
            "preservation": Decimal('0.5'),  # Capital preservation phase
            "conservative": Decimal('0.8'),  # Conservative growth
            "moderate": Decimal('1.0'),      # Normal growth
            "aggressive": Decimal('1.2'),    # Aggressive growth
            "recovery": Decimal('0.6')       # Recovery after drawdown
        }
        growth_factor = growth_phase_factors.get(self.account_growth_phase, Decimal('1.0'))
        
        # Adjust for market volatility if provided
        volatility_factor = Decimal('1.0')
        if market_volatility is not None:
            vol = Decimal(str(market_volatility))
            # In high volatility, reduce risk
            if vol > Decimal('0.02'):  # If volatility > 2%
                volatility_factor = Decimal('1.0') / (vol * Decimal('30.0'))
                volatility_factor = max(min(volatility_factor, Decimal('1.0')), Decimal('0.5'))
        
        # Adjust for strategy performance if strategy_id is provided
        strategy_factor = Decimal('1.0')
        if strategy_id:
            try:
                # This would fetch the strategy's performance metrics from storage
                # For demonstration, we use a neutral factor
                # In production, this would query the strategy performance database
                strategy_factor = Decimal('1.0')
            except Exception as e:
                logger.warning(f"Failed to get strategy performance for {strategy_id}: {str(e)}")
        
        # Combine all factors
        combined_factor = (
            confidence_factor * 
            signal_factor * 
            streak_factor * 
            drawdown_factor * 
            growth_factor * 
            volatility_factor *
            strategy_factor
        )
        
        # Apply the combined factor to the base risk
        adjusted_risk = base_risk * combined_factor
        
        # Ensure risk stays within reasonable bounds
        min_risk = base_risk * Decimal('0.2')  # Never go below 20% of base risk
        max_risk = base_risk * Decimal('1.5')  # Never exceed 150% of base risk
        
        final_risk = max(min(adjusted_risk, max_risk), min_risk)
        
        # Log the factors for debugging and analysis
        logger.debug(
            f"Risk adjustment factors: confidence={float(confidence_factor):.2f}, "
            f"signal={float(signal_factor):.2f}, streak={float(streak_factor):.2f}, "
            f"drawdown={float(drawdown_factor):.2f}, growth={float(growth_factor):.2f}, "
            f"volatility={float(volatility_factor):.2f}, strategy={float(strategy_factor):.2f} "
            f"→ Final risk: {float(final_risk * 100):.2f}%"
        )
        
        return final_risk
    
    async def _fixed_fractional_position_size(
        self,
        dollar_risk: Decimal,
        risk_price_distance: Decimal,
        entry_price: Decimal
    ) -> Decimal:
        """
        Calculate position size using the fixed fractional method.
        
        Args:
            dollar_risk: Amount in dollars to risk
            risk_price_distance: Price distance to stop loss
            entry_price: Entry price
            
        Returns:
            Position size in base currency
        """
        if risk_price_distance <= 0:
            return Decimal('0')
        
        # Risk per unit calculation
        risk_per_unit = risk_price_distance
        
        # Position size in base currency
        position_size = dollar_risk / risk_per_unit
        
        return position_size
    
    async def _kelly_criterion_position_size(
        self,
        risk_reward_ratio: Decimal,
        win_probability: Decimal
    ) -> Decimal:
        """
        Calculate optimal position size using the Kelly Criterion formula.
        
        Args:
            risk_reward_ratio: Potential reward divided by potential risk
            win_probability: Probability of winning the trade
            
        Returns:
            Kelly position size as a fraction of account
        """
        # Kelly formula: K = (bp - q) / b
        # where:
        # b = odds received on win (reward/risk)
        # p = probability of winning
        # q = probability of losing (1-p)
        
        b = risk_reward_ratio
        p = win_probability
        q = Decimal('1.0') - p
        
        # Calculate Kelly percentage
        kelly_pct = (b * p - q) / b
        
        # Limit kelly percentage to reasonable values
        kelly_pct = max(min(kelly_pct, Decimal('0.1')), Decimal('0'))
        
        # Apply a fraction of Kelly for more conservative sizing
        fractional_kelly = kelly_pct * DEFAULT_KELLY_FRACTION
        
        # Calculate dollar amount to risk
        kelly_position_dollars = self.account_balance * fractional_kelly
        
        return kelly_position_dollars
    
    def _apply_asset_constraints(
        self,
        position_size: Decimal,
        asset_params: Dict[str, Any]
    ) -> Decimal:
        """
        Apply asset-specific constraints to the position size.
        
        Args:
            position_size: Calculated position size
            asset_params: Asset parameters
            
        Returns:
            Adjusted position size
        """
        # Ensure position size is positive
        position_size = max(position_size, Decimal('0'))
        
        # Apply minimum position size if specified
        min_position = asset_params.get('min_position_size', Decimal('0'))
        if min_position > 0 and position_size > 0:
            position_size = max(position_size, min_position)
        
        # Apply maximum position size if specified
        max_position = asset_params.get('max_position_size')
        if max_position is not None:
            position_size = min(position_size, max_position)
        
        return position_size
    
    def _calculate_growth_phase(self) -> str:
        """
        Calculate the current account growth phase based on balance and performance.
        
        Returns:
            Growth phase identifier string
        """
        # Calculate growth rate from initial balance
        if self.initial_account_balance <= 0:
            growth_rate = Decimal('0')
        else:
            growth_rate = (self.account_balance / self.initial_account_balance) - Decimal('1.0')
        
        # Determine phase based on growth rate and drawdown
        if self.drawdown_severity > Decimal('0.2'):
            return "recovery"  # Deep drawdown, focus on recovery
        
        if self.drawdown_severity > Decimal('0.1'):
            return "preservation"  # Moderate drawdown, focus on preservation
        
        if growth_rate < Decimal('-0.05'):
            return "preservation"  # Account shrinking, preserve capital
        
        if growth_rate < Decimal('0.1'):
            return "conservative"  # Slow growth, be conservative
        
        if growth_rate < Decimal('0.5'):
            return "moderate"  # Good growth, normal operation
        
        return "aggressive"  # Excellent growth, can be more aggressive
    
    def _round_to_precision(self, value: Decimal, precision: int) -> Decimal:
        """
        Round a value to a specific decimal precision.
        
        Args:
            value: Value to round
            precision: Number of decimal places
            
        Returns:
            Rounded value
        """
        multiplier = Decimal(10) ** precision
        return (value * multiplier).quantize(Decimal('1'), rounding=ROUND_DOWN) / multiplier
    
    async def get_max_position_size(self, asset: str, leverage: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate the maximum possible position size for an asset.
        
        Args:
            asset: Asset symbol
            leverage: Optional leverage to use
            
        Returns:
            Dict with maximum position details
        """
        try:
            asset_params = await self._get_asset_specific_params(asset)
            
            # Determine leverage
            max_allowed_leverage = asset_params['max_leverage']
            if leverage is not None:
                max_allowed_leverage = min(Decimal(str(leverage)), max_allowed_leverage)
            
            # Calculate maximum position size
            available_margin = self.account_balance
            max_position_value = available_margin * max_allowed_leverage
            
            return {
                "asset": asset,
                "max_position_value": float(max_position_value),
                "available_margin": float(available_margin),
                "max_leverage": float(max_allowed_leverage),
                "platform": self.platform
            }
            
        except Exception as e:
            logger.error(f"Error calculating max position size: {str(e)}", exc_info=True)
            raise PositionSizingError(f"Failed to calculate max position size: {str(e)}")
    
    async def simulate_position_impact(
        self,
        quantity: float,
        asset: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: Optional[float] = None,
        leverage: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Simulate the impact of a position on the account.
        
        Args:
            quantity: Position quantity
            asset: Asset symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price (optional)
            leverage: Leverage to use (optional)
            
        Returns:
            Dict with simulation results
        """
        try:
            quantity = Decimal(str(quantity))
            entry_price = Decimal(str(entry_price))
            stop_loss_price = Decimal(str(stop_loss_price))
            
            asset_params = await self._get_asset_specific_params(asset)
            
            # Determine leverage
            max_allowed_leverage = asset_params['max_leverage']
            if leverage is None:
                leverage = Decimal('1.0')
            else:
                leverage = min(Decimal(str(leverage)), max_allowed_leverage)
            
            # Calculate position value and margin required
            position_value = quantity * entry_price
            margin_required = position_value / leverage
            
            # Check if we have enough margin
            if margin_required > self.account_balance:
                raise InsufficientBalanceError(
                    f"Insufficient balance for position: Required: {float(margin_required)}, "
                    f"Available: {float(self.account_balance)}"
                )
            
            # Calculate potential loss at stop price
            is_long = entry_price > stop_loss_price
            risk_price_distance = abs(entry_price - stop_loss_price)
            potential_loss = risk_price_distance * quantity
            loss_percentage = potential_loss / self.account_balance
            
            # Check if loss exceeds maximum risk
            if loss_percentage > self.max_risk_per_trade:
                logger.warning(
                    f"Position risk exceeds maximum: {float(loss_percentage * 100):.2f}% > "
                    f"{float(self.max_risk_per_trade * 100):.2f}%"
                )
            
            # Calculate potential profit if take profit is provided
            potential_profit = None
            reward_risk_ratio = None
            profit_percentage = None
            expected_value = None
            
            if take_profit_price is not None:
                take_profit_price = Decimal(str(take_profit_price))
                reward_price_distance = abs(take_profit_price - entry_price)
                potential_profit = reward_price_distance * quantity
                profit_percentage = potential_profit / self.account_balance
                reward_risk_ratio = reward_price_distance / risk_price_distance if risk_price_distance > 0 else Decimal('1.0')
                
                # Simple expected value calculation with 50% probability
                expected_value = (potential_profit * Decimal('0.5')) - (potential_loss * Decimal('0.5'))
            
            # Calculate margin utilization
            margin_utilization = margin_required / self.account_balance
            
            # Prepare results
            results = {
                "asset": asset,
                "quantity": float(quantity),
                "entry_price": float(entry_price),
                "stop_loss_price": float(stop_loss_price),
                "direction": "long" if is_long else "short",
                "position_value": float(position_value),
                "leverage": float(leverage),
                "margin_required": float(margin_required),
                "margin_utilization": float(margin_utilization * 100),  # as percentage
                "risk_price_distance": float(risk_price_distance),
                "potential_loss": float(potential_loss),
                "loss_percentage": float(loss_percentage * 100),  # as percentage
                "exceeds_max_risk": float(loss_percentage) > float(self.max_risk_per_trade),
                "platform": self.platform
            }
            
            if take_profit_price is not None:
                results.update({
                    "take_profit_price": float(take_profit_price),
                    "reward_price_distance": float(reward_price_distance),
                    "potential_profit": float(potential_profit),
                    "profit_percentage": float(profit_percentage * 100),  # as percentage
                    "reward_risk_ratio": float(reward_risk_ratio),
                    "expected_value": float(expected_value),
                    "expected_value_percentage": float(expected_value / self.account_balance * 100)  # as percentage
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error simulating position impact: {str(e)}", exc_info=True)
            raise PositionSizingError(f"Failed to simulate position impact: {str(e)}")
    
    async def update_max_risk_per_trade(self, max_risk: float) -> None:
        """
        Update the maximum risk per trade.
        
        Args:
            max_risk: New maximum risk per trade (as a decimal fraction)
        """
        if max_risk <= 0 or max_risk > 0.5:  # Don't allow more than 50% risk
            raise InvalidParameterError("Max risk must be between 0 and 0.5")
        
        self.max_risk_per_trade = Decimal(str(max_risk))
        
        # Update metrics
        if self.metrics_collector:
            self.metrics_collector.set_gauge_value(
                "position_sizing.risk_per_trade", 
                float(self.max_risk_per_trade * 100)
            )
        
        logger.info(f"Maximum risk per trade updated to {float(self.max_risk_per_trade * 100):.2f}%")
    
    async def calculate_pyramid_entries(
        self,
        asset: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        confidence: float = 0.7,
        num_entries: int = 3,
        price_spacing: Optional[float] = None,
        leverage: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate pyramid entry positions for a trade.
        
        Args:
            asset: Asset symbol
            entry_price: Initial entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            confidence: Signal confidence
            num_entries: Number of pyramid entries
            price_spacing: Spacing between entry prices (optional)
            leverage: Leverage to use (optional)
            
        Returns:
            Dict with pyramid entry details
        """
        try:
            # Convert to Decimal
            entry_price = Decimal(str(entry_price))
            stop_loss_price = Decimal(str(stop_loss_price))
            take_profit_price = Decimal(str(take_profit_price))
            confidence = Decimal(str(confidence))
            
            # Get asset parameters
            asset_params = await self._get_asset_specific_params(asset)
            
            # Determine direction
            is_long = entry_price < take_profit_price
            
            # Calculate price distance to target
            price_distance = abs(take_profit_price - entry_price)
            
            # Calculate default price spacing if not provided
            if price_spacing is None:
                # Default to spacing entries evenly across first half of the move
                default_spacing = price_distance / (Decimal(str(num_entries)) * Decimal('2.0'))
                # Adjust spacing based on asset volatility
                volatility_adjustment = asset_params.get('volatility', Decimal('0.02'))
                price_spacing = default_spacing * (Decimal('1.0') + volatility_adjustment * Decimal('10.0'))
            else:
                price_spacing = Decimal(str(price_spacing))
            
            # Calculate total risk allocation
            total_risk_allocation = Decimal('0.8')  # Use 80% of max risk for pyramid
            
            # Allocate risk to each entry (front-loaded)
            # First entry gets most risk, decreasing for later entries
            risk_weights = []
            remaining_weight = Decimal('1.0')
            decay_factor = Decimal('0.6')  # Controls how quickly risk decreases
            
            for i in range(num_entries):
                if i == num_entries - 1:
                    # Last entry gets remaining weight
                    weight = remaining_weight
                else:
                    weight = remaining_weight * (Decimal('1.0') - decay_factor)
                    remaining_weight -= weight
                
                risk_weights.append(weight)
            
            # Calculate entry prices
            entry_prices = []
            if is_long:
                # For long positions, entries are at increasing prices
                for i in range(num_entries):
                    # First entry is at the initial entry price
                    if i == 0:
                        entry_prices.append(entry_price)
                    else:
                        entry_prices.append(entry_price + (price_spacing * Decimal(str(i))))
            else:
                # For short positions, entries are at decreasing prices
                for i in range(num_entries):
                    # First entry is at the initial entry price
                    if i == 0:
                        entry_prices.append(entry_price)
                    else:
                        entry_prices.append(entry_price - (price_spacing * Decimal(str(i))))
            
            # Calculate position sizes for each entry
            entries = []
            total_quantity = Decimal('0')
            total_position_value = Decimal('0')
            total_margin_required = Decimal('0')
            
            for i in range(num_entries):
                # Adjust confidence for later entries
                entry_confidence = confidence * (Decimal('1.0') - (Decimal('0.1') * Decimal(str(i))))
                
                # Calculate risk amount for this entry
                entry_risk = self.max_risk_per_trade * risk_weights[i] * total_risk_allocation
                
                # Calculate dollar risk
                dollar_risk = self.account_balance * entry_risk
                
                # Calculate risk price distance (from entry to stop)
                risk_price_distance = abs(entry_prices[i] - stop_loss_price)
                
                # Calculate position size
                if risk_price_distance > 0:
                    position_size = dollar_risk / risk_price_distance
                else:
                    position_size = Decimal('0')
                
                # Apply asset constraints
                position_size = self._apply_asset_constraints(position_size, asset_params)
                
                # Apply leverage
                max_allowed_leverage = asset_params['max_leverage']
                if leverage is None:
                    # Auto-calculate conservative leverage
                    entry_leverage = Decimal('1.0')
                else:
                    entry_leverage = min(Decimal(str(leverage)), max_allowed_leverage)
                
                # Calculate quantity
                quantity = self._round_to_precision(
                    position_size / entry_prices[i],
                    asset_params['quantity_precision']
                )
                
                # Calculate position value and margin
                position_value = quantity * entry_prices[i]
                margin_required = position_value / entry_leverage
                
                # Add to totals
                total_quantity += quantity
                total_position_value += position_value
                total_margin_required += margin_required
                
                # Create entry object
                entry = {
                    "entry_number": i + 1,
                    "entry_price": float(entry_prices[i]),
                    "quantity": float(quantity),
                    "position_value": float(position_value),
                    "margin_required": float(margin_required),
                    "leverage": float(entry_leverage),
                    "risk_allocation": float(risk_weights[i]),
                    "dollar_risk": float(dollar_risk),
                    "confidence": float(entry_confidence)
                }
                
                entries.append(entry)
            
            # Calculate aggregate metrics
            avg_entry_price = sum(entry_prices[i] * entries[i]["quantity"] for i in range(num_entries)) / total_quantity if total_quantity > 0 else entry_price
            
            # Calculate potential loss at stop
            potential_loss = abs(avg_entry_price - stop_loss_price) * total_quantity
            loss_percentage = potential_loss / self.account_balance
            
            # Calculate potential profit at target
            potential_profit = abs(take_profit_price - avg_entry_price) * total_quantity
            profit_percentage = potential_profit / self.account_balance
            
            # Calculate reward-risk ratio
            reward_risk_ratio = potential_profit / potential_loss if potential_loss > 0 else Decimal('1.0')
            
            # Prepare final result
            result = {
                "asset": asset,
                "direction": "long" if is_long else "short",
                "stop_loss_price": float(stop_loss_price),
                "take_profit_price": float(take_profit_price),
                "entries": entries,
                "total_quantity": float(total_quantity),
                "total_position_value": float(total_position_value),
                "total_margin_required": float(total_margin_required),
                "average_entry_price": float(avg_entry_price),
                "potential_loss": float(potential_loss),
                "loss_percentage": float(loss_percentage * 100),
                "potential_profit": float(potential_profit),
                "profit_percentage": float(profit_percentage * 100),
                "reward_risk_ratio": float(reward_risk_ratio),
                "price_spacing": float(price_spacing),
                "margin_utilization": float(total_margin_required / self.account_balance * 100),
                "exceeds_max_risk": float(loss_percentage) > float(self.max_risk_per_trade),
                "platform": self.platform
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating pyramid entries: {str(e)}", exc_info=True)
            raise PositionSizingError(f"Failed to calculate pyramid entries: {str(e)}")
    
    async def get_optimal_position_size_distribution(
        self,
        assets: List[str],
        confidence_levels: List[float],
        total_allocation: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate the optimal distribution of position sizes across multiple assets.
        
        Args:
            assets: List of asset symbols
            confidence_levels: List of confidence levels for each asset
            total_allocation: Total allocation as a fraction of account balance
            
        Returns:
            Dict with optimal position distribution
        """
        try:
            if len(assets) != len(confidence_levels):
                raise InvalidParameterError("Assets and confidence levels must have the same length")
            
            if total_allocation <= 0 or total_allocation > 1:
                raise InvalidParameterError("Total allocation must be between 0 and 1")
            
            # Convert to Decimal
            total_allocation = Decimal(str(total_allocation))
            confidence_levels = [Decimal(str(conf)) for conf in confidence_levels]
            
            # Calculate confidence weights
            total_confidence = sum(confidence_levels)
            if total_confidence <= 0:
                # Equal weights if all confidences are zero
                weights = [Decimal('1.0') / Decimal(str(len(assets))) for _ in assets]
            else:
                weights = [conf / total_confidence for conf in confidence_levels]
            
            # Calculate dollar allocation for each asset
            dollar_allocations = [total_allocation * self.account_balance * weight for weight in weights]
            
            # Get asset parameters and calculate margins
            allocations = []
            for i, asset in enumerate(assets):
                asset_params = await self._get_asset_specific_params(asset)
                
                # Determine appropriate leverage based on confidence
                confidence = confidence_levels[i]
                leverage = min(
                    Decimal('1.0') + (confidence * Decimal('2.0')),
                    asset_params['max_leverage']
                )
                
                # Calculate max position value
                position_value = dollar_allocations[i] * leverage
                
                allocations.append({
                    "asset": asset,
                    "allocation_weight": float(weights[i]),
                    "dollar_allocation": float(dollar_allocations[i]),
                    "leverage": float(leverage),
                    "position_value": float(position_value),
                    "confidence": float(confidence_levels[i]),
                    "platform": self.platform
                })
            
            # Calculate total position value
            total_position_value = sum(alloc["position_value"] for alloc in allocations)
            
            # Calculate total dollar allocation
            total_dollar_allocation = sum(alloc["dollar_allocation"] for alloc in allocations)
            
            return {
                "allocations": allocations,
                "total_position_value": float(total_position_value),
                "total_dollar_allocation": float(total_dollar_allocation),
                "account_balance": float(self.account_balance),
                "allocation_percentage": float(total_allocation * 100),
                "allocation_utilization": float(total_dollar_allocation / self.account_balance * 100)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position size distribution: {str(e)}", exc_info=True)
            raise PositionSizingError(f"Failed to calculate position size distribution: {str(e)}")


def get_position_sizer(name: str, *args, **kwargs) -> BasePositionSizer:
    """Instantiate a registered position sizer by name."""
    cls = BasePositionSizer.registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown position sizer: {name}")
    return cls(*args, **kwargs)


class PositionSizer(PositionSizing):
    """Alias for backward compatibility."""
    pass

__all__ = ["BasePositionSizer", "get_position_sizer", "PositionSizer"]
