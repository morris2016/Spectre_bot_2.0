

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Capital Management Module

This module provides sophisticated capital management strategies to optimize 
trading performance while protecting account capital. It implements dynamic 
position sizing, portfolio allocation, and risk management strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import asyncio
import time
import math
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from collections import defaultdict, deque

from common.logger import get_logger
try:
    from common.constants import (
        POSITION_SIZE_LIMITS,
        MAX_DRAWDOWN_THRESHOLD,
        RECOVERY_FACTOR,
        KELLY_MODIFIER,
        RISK_REWARD_THRESHOLDS,
        DEFAULT_RISK_PER_TRADE,
        CAPITAL_ALLOCATION_LIMITS,
        MAX_LEVERAGE_SETTINGS,
    )
except Exception:  # pragma: no cover - provide sane defaults if constants missing
    POSITION_SIZE_LIMITS = {}
    MAX_DRAWDOWN_THRESHOLD = 0.3
    RECOVERY_FACTOR = 1.5
    KELLY_MODIFIER = 0.5
    RISK_REWARD_THRESHOLDS = {}
    DEFAULT_RISK_PER_TRADE = 0.01
    CAPITAL_ALLOCATION_LIMITS = {}
    MAX_LEVERAGE_SETTINGS = {}
from common.utils import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_drawdown
from common.exceptions import CapitalManagementError
from data_storage.models.user_data import UserPreferences, UserCapitalSettings

logger = get_logger("capital_manager")


@dataclass
class TradingState:
    """Data class representing the current trading state and performance metrics."""
    current_equity: float
    starting_equity: float
    peak_equity: float
    drawdown_percentage: float
    win_rate: float
    profit_factor: float
    recent_wins: int
    recent_losses: int
    avg_win_size: float
    avg_loss_size: float
    win_loss_ratio: float
    consecutive_wins: int
    consecutive_losses: int
    last_10_trades: List[float]  # Returns as percentages
    sharpe_ratio: float
    sortino_ratio: float


class CapitalManagement:
    """
    Capital Management System for the QuantumSpectre Elite Trading System.
    
    This class handles all aspects of capital allocation, position sizing,
    and risk management for trading operations.
    """
    
    def __init__(self,
                 initial_capital: float,
                 risk_per_trade: float = 0.02,
                 max_drawdown: float = 0.20,
                 position_sizing_method: str = "risk_based",
                 max_open_positions: int = 5,
                 max_risk_per_asset: float = 0.05,
                 max_risk_per_sector: float = 0.20,
                 leverage_limit: float = 3.0):
        """
        Initialize the Capital Management system.
        
        Args:
            initial_capital: Starting capital amount
            risk_per_trade: Percentage of capital to risk per trade (default: 2%)
            max_drawdown: Maximum allowed drawdown before reducing risk (default: 20%)
            position_sizing_method: Method for sizing positions (default: risk_based)
            max_open_positions: Maximum number of concurrent open positions
            max_risk_per_asset: Maximum risk allocation per asset
            max_risk_per_sector: Maximum risk allocation per sector
            leverage_limit: Maximum allowed leverage
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.position_sizing_method = position_sizing_method
        self.max_open_positions = max_open_positions
        self.max_risk_per_asset = max_risk_per_asset
        self.max_risk_per_sector = max_risk_per_sector
        self.leverage_limit = leverage_limit
        self.open_positions = {}
        self.position_history = []
        self.current_drawdown = 0.0
        self.in_recovery_mode = False
        
    async def calculate_position_size(self,
                                asset: str,
                                entry_price: float,
                                stop_loss: float,
                                risk_multiplier: float = 1.0) -> float:
        """
        Calculate the appropriate position size based on risk parameters.
        
        Args:
            asset: The asset to trade
            entry_price: Planned entry price
            stop_loss: Stop loss price
            risk_multiplier: Adjust risk up or down (default: 1.0)
            
        Returns:
            Position size in base units
        """
        # Calculate risk amount in account currency
        risk_amount = self.current_capital * self.risk_per_trade * risk_multiplier
        
        # Apply recovery mode if necessary
        if self.in_recovery_mode:
            risk_amount *= 0.5
            
        # Calculate risk per unit
        if entry_price <= stop_loss:  # Short position
            risk_per_unit = entry_price - stop_loss
        else:  # Long position
            risk_per_unit = stop_loss - entry_price
            
        # Avoid division by zero
        if abs(risk_per_unit) < 0.000001:
            return 0
            
        # Calculate position size
        position_size = risk_amount / abs(risk_per_unit)
        
        # Apply additional constraints
        position_size = self._apply_position_constraints(position_size, asset)
        
        return position_size
        
    def _apply_position_constraints(self, position_size: float, asset: str) -> float:
        """Apply additional constraints to the position size."""
        # Check if we're at max positions
        if len(self.open_positions) >= self.max_open_positions:
            return 0
            
        # Check asset-specific constraints
        # (Implementation would depend on asset-specific logic)
        
        return position_size
        
    async def update_capital(self, new_capital: float) -> None:
        """
        Update the current capital amount.
        
        Args:
            new_capital: The new capital amount
        """
        self.current_capital = new_capital
        
        # Update peak capital if we have a new high
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
            
        # Calculate current drawdown
        if self.peak_capital > 0:
            self.current_drawdown = 1 - (self.current_capital / self.peak_capital)
            
        # Check if we need to enter recovery mode
        if self.current_drawdown >= self.max_drawdown and not self.in_recovery_mode:
            self.in_recovery_mode = True
        
        # Check if we can exit recovery mode
        if self.in_recovery_mode and self.current_drawdown < (self.max_drawdown * 0.5):
            self.in_recovery_mode = False
    trade_frequency: float  # Trades per day
    recovery_mode: bool
    current_exposure: Dict[str, float]  # Symbol to position size mapping
    timestamp: float


class CapitalManager:
    """
    Sophisticated capital management system for optimizing position sizing and risk.
    
    This class provides advanced capital management capabilities including:
    - Dynamic position sizing based on trading performance and market conditions
    - Portfolio allocation across multiple assets
    - Risk management based on drawdown, win rate, and other performance metrics
    - Adaptive recovery strategies
    - Confidence-based position sizing
    """
    
    def __init__(self, user_id: str, initial_capital: float):
        """
        Initialize the CapitalManager.
        
        Args:
            user_id: User identifier
            initial_capital: Initial account capital
        """
        self.user_id = user_id
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.logger = logging.getLogger(f"capital_manager.{user_id}")
        
        # Load user preferences
        self.user_prefs = self._load_user_preferences()
        self.capital_settings = self._load_capital_settings()
        
        # Trading performance tracking
        self.trades_history = []  # Full trade history
        self.recent_trades = deque(maxlen=50)  # Most recent trades
        self.running_pnl = 0.0
        self.n_winning_trades = 0
        self.n_losing_trades = 0
        self.total_win_amount = 0.0
        self.total_loss_amount = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # Asset-specific tracking
        self.asset_performance = defaultdict(lambda: {
            'n_trades': 0,
            'win_rate': 0.0,
            'avg_return': 0.0,
            'profit_factor': 1.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 1.0,
            'sharpe': 0.0,
            'trades': []
        })
        
        # Portfolio allocation
        self.portfolio_allocation = {}  # Asset to allocation percentage mapping
        self.current_exposure = {}  # Asset to current exposure amount mapping
        
        # State
        self.recovery_mode = False
        self.drawdown_percentage = 0.0
        self.last_update_time = time.time()
        
        self.logger.info(f"CapitalManager initialized with {initial_capital} initial capital")

    def _load_user_preferences(self) -> UserPreferences:
        """
        Load user preferences from database.
        
        Returns:
            UserPreferences object
        """
        # In a production environment, this would load from database
        # Here we create defaults
        return UserPreferences(
            user_id=self.user_id,
            risk_profile="moderate",
            max_drawdown_threshold=20.0,  # 20% max drawdown
            risk_per_trade=1.0,  # 1% risk per trade
            leverage_preference=2.0,  # 2x leverage
            recovery_aggressiveness=0.5,  # Moderate recovery
            auto_compound=True,
            capital_allocation_strategy="dynamic",
            created_at=time.time(),
            updated_at=time.time()
        )

    def _load_capital_settings(self) -> UserCapitalSettings:
        """
        Load capital settings from database.
        
        Returns:
            UserCapitalSettings object
        """
        # In a production environment, this would load from database
        # Here we create defaults
        return UserCapitalSettings(
            user_id=self.user_id,
            max_position_size_percentage=10.0,  # Max 10% in any one position
            min_position_size=0.01,  # Minimum position size
            kelly_criterion_modifier=0.5,  # Half Kelly for safety
            max_correlated_exposure=25.0,  # Max 25% in correlated assets
            reserve_percentage=10.0,  # Keep 10% in reserve
            profit_distribution={
                'reinvest': 80.0,  # Reinvest 80% of profits
                'reserve': 20.0  # Add 20% to reserve
            },
            created_at=time.time(),
            updated_at=time.time()
        )

    def update_account_value(self, current_value: float) -> None:
        """
        Update the current account value.
        
        Args:
            current_value: Current total account value
        """
        self.current_capital = current_value
        
        # Update peak capital if new high
        if current_value > self.peak_capital:
            self.peak_capital = current_value
            
        # Calculate drawdown
        if self.peak_capital > 0:
            self.drawdown_percentage = (self.peak_capital - self.current_capital) / self.peak_capital * 100.0
        else:
            self.drawdown_percentage = 0.0
            
        # Check recovery mode
        max_drawdown = self.user_prefs.max_drawdown_threshold
        if self.drawdown_percentage > max_drawdown and not self.recovery_mode:
            self.logger.warning(
                f"Entering recovery mode: Drawdown {self.drawdown_percentage:.2f}% exceeds "
                f"threshold {max_drawdown:.2f}%"
            )
            self.recovery_mode = True
        elif self.recovery_mode and self.drawdown_percentage < max_drawdown * RECOVERY_FACTOR:
            self.logger.info(
                f"Exiting recovery mode: Drawdown {self.drawdown_percentage:.2f}% below "
                f"recovery threshold {max_drawdown * RECOVERY_FACTOR:.2f}%"
            )
            self.recovery_mode = False
            
        self.logger.debug(
            f"Account value updated: Current: {current_value:.2f}, Peak: {self.peak_capital:.2f}, "
            f"Drawdown: {self.drawdown_percentage:.2f}%, Recovery mode: {self.recovery_mode}"
        )

    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Record a completed trade for performance tracking.
        
        Args:
            trade_data: Dictionary with trade details
        """
        # Validate required fields
        required_fields = ['symbol', 'pnl', 'entry_time', 'exit_time', 'entry_price', 
                         'exit_price', 'size', 'side', 'pnl_percentage']
        
        for field in required_fields:
            if field not in trade_data:
                raise CapitalManagementError(f"Missing required trade field: {field}")
                
        # Record in history
        self.trades_history.append(trade_data)
        self.recent_trades.append(trade_data)
        
        # Update running counters
        pnl = trade_data['pnl']
        self.running_pnl += pnl
        
        # Update win/loss stats
        if pnl > 0:
            self.n_winning_trades += 1
            self.total_win_amount += pnl
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.n_losing_trades += 1
            self.total_loss_amount += abs(pnl)
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
        # Update asset-specific stats
        symbol = trade_data['symbol']
        self._update_asset_stats(symbol, trade_data)
        
        # Update account value if provided
        if 'new_balance' in trade_data:
            self.update_account_value(trade_data['new_balance'])
            
        self.logger.info(
            f"Recorded trade: {symbol} {trade_data['side']} {trade_data['size']:.4f} "
            f"PnL: {pnl:.2f} ({trade_data['pnl_percentage']:.2f}%)"
        )

    def _update_asset_stats(self, symbol: str, trade_data: Dict[str, Any]) -> None:
        """
        Update performance statistics for a specific asset.
        
        Args:
            symbol: The asset symbol
            trade_data: Trade data dictionary
        """
        # Add trade to asset history
        self.asset_performance[symbol]['trades'].append(trade_data)
        self.asset_performance[symbol]['n_trades'] += 1
        
        # Recalculate stats
        trades = self.asset_performance[symbol]['trades']
        
        # Win rate
        wins = sum(1 for t in trades if t['pnl'] > 0)
        n_trades = len(trades)
        win_rate = wins / n_trades if n_trades > 0 else 0
        
        # Average return
        avg_return = sum(t['pnl_percentage'] for t in trades) / n_trades if n_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = sum(abs(t['pnl']) for t in trades if t['pnl'] < 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit == 0 else float('inf'))
        
        # Average win/loss
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(abs(t['pnl']) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio with daily returns
        daily_returns = self._calculate_daily_returns(trades)
        sharpe = calculate_sharpe_ratio(daily_returns) if daily_returns else 0
        
        # Update asset stats
        self.asset_performance[symbol].update({
            'win_rate': win_rate,
            'avg_return': avg_return,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'sharpe': sharpe
        })

    def _calculate_daily_returns(self, trades: List[Dict]) -> List[float]:
        """
        Calculate daily returns from trade data.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            List of daily return values
        """
        if not trades:
            return []
            
        # Group trades by day
        trade_days = {}
        
        for trade in trades:
            # Convert timestamp to date string
            day = time.strftime('%Y-%m-%d', time.localtime(trade['exit_time']))
            
            if day not in trade_days:
                trade_days[day] = []
                
            trade_days[day].append(trade)
            
        # Calculate daily returns
        daily_returns = []
        
        for day, day_trades in trade_days.items():
            day_pnl = sum(trade['pnl'] for trade in day_trades)
            day_return = day_pnl / self.initial_capital
            daily_returns.append(day_return)
            
        return daily_returns

    def calculate_position_size(self, symbol: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the optimal position size for a trade.
        
        Args:
            symbol: The asset symbol
            signal_data: Dictionary with signal information including confidence,
                         risk/reward, and other relevant metrics
                         
        Returns:
            Dictionary with position sizing information
        """
        self.logger.debug(f"Calculating position size for {symbol}")
        
        # Extract key metrics from signal
        confidence = signal_data.get('confidence', 0.5)
        risk_reward = signal_data.get('risk_reward', 1.0)
        stop_loss_percent = signal_data.get('stop_loss_percent', None)
        market_volatility = signal_data.get('market_volatility', 0.5)  # 0-1 scale
        
        # Get asset-specific performance metrics
        asset_metrics = self.asset_performance.get(symbol, {
            'win_rate': 0.5,  # Default if no history
            'profit_factor': 1.0,
            'win_loss_ratio': 1.0
        })
        
        # Determine if this is a new asset or has very little history
        is_new_asset = asset_metrics.get('n_trades', 0) < 10
        
        # Base size calculation - Kelly Criterion with user modifier
        if stop_loss_percent is None or stop_loss_percent <= 0:
            # Default to a reasonable stop loss if not provided
            stop_loss_percent = 1.0
            
        # Calculate adjusted win rate
        historical_win_rate = asset_metrics.get('win_rate', 0.5)
        
        # Blend historical and signal confidence for experienced assets
        # For new assets rely more on signal confidence
        if is_new_asset:
            adjusted_win_rate = 0.2 * historical_win_rate + 0.8 * confidence
        else:
            adjusted_win_rate = 0.6 * historical_win_rate + 0.4 * confidence
            
        # Calculate odds based on risk/reward
        odds = risk_reward
        
        # Basic Kelly formula: f = (p*(b+1)-1)/b where:
        # f = fraction of bankroll to bet
        # p = probability of win
        # b = net odds received on the bet (payout-1)
        
        kelly_percentage = (adjusted_win_rate * odds - (1 - adjusted_win_rate)) / odds
        
        # Apply the Kelly modifier and ensure non-negative
        kelly_percentage = max(0, kelly_percentage * self.capital_settings.kelly_criterion_modifier)
        
        # Adjust for recovery mode
        if self.recovery_mode:
            recovery_factor = self._calculate_recovery_factor()
            kelly_percentage *= recovery_factor
            
        # Adjust for risk per trade preference
        risk_percentage = self.user_prefs.risk_per_trade
        
        # Calculate position size based on risk percentage and stop loss
        # Normalize to percentage of account
        position_percentage = min(kelly_percentage, risk_percentage / stop_loss_percent)
        
        # Apply market volatility adjustment
        volatility_factor = 1.0 - (market_volatility * 0.5)  # Reduce size in high volatility
        position_percentage *= volatility_factor
        
        # Apply consecutive wins/losses adjustment
        streak_factor = self._calculate_streak_factor()
        position_percentage *= streak_factor
        
        # Apply drawdown adjustment
        drawdown_factor = 1.0 - (self.drawdown_percentage / 100.0)  # Reduce size in drawdown
        position_percentage *= drawdown_factor
        
        # Apply portfolio diversity constraint
        portfolio_factor = self._calculate_portfolio_factor(symbol)
        position_percentage *= portfolio_factor
        
        # Ensure within limits
        position_percentage = max(
            self.capital_settings.min_position_size / 100.0,
            min(position_percentage, self.capital_settings.max_position_size_percentage / 100.0)
        )
        
        # Calculate nominal size
        position_size = self.current_capital * position_percentage
        
        # Calculate leverage
        leverage = self._calculate_optimal_leverage(symbol, signal_data)
        
        # Apply leverage to position size
        leveraged_size = position_size * leverage
        
        # Calculate expected risk
        expected_risk = position_size * stop_loss_percent / 100.0
        
        result = {
            'symbol': symbol,
            'position_size': position_size,
            'position_percentage': position_percentage * 100.0,
            'leveraged_size': leveraged_size,
            'leverage': leverage,
            'expected_risk_amount': expected_risk,
            'expected_risk_percentage': position_percentage * stop_loss_percent,
            'adjusted_win_rate': adjusted_win_rate,
            'kelly_percentage': kelly_percentage * 100.0,
            'factors_applied': {
                'volatility_factor': volatility_factor,
                'streak_factor': streak_factor,
                'drawdown_factor': drawdown_factor,
                'portfolio_factor': portfolio_factor,
                'recovery_mode': self.recovery_mode
            }
        }
        
        self.logger.info(
            f"Position size for {symbol}: {position_size:.4f} ({position_percentage * 100:.2f}%), "
            f"Leverage: {leverage:.2f}x, Expected risk: {expected_risk:.4f}"
        )
        
        return result

    def _calculate_recovery_factor(self) -> float:
        """
        Calculate a factor to adjust position sizing during recovery mode.
        
        Returns:
            A multiplier to apply to position sizes (0.0-1.0)
        """
        # Base factor depends on drawdown severity and recovery aggressiveness
        recovery_aggressiveness = self.user_prefs.recovery_aggressiveness
        
        # More conservative for larger drawdowns
        if self.drawdown_percentage > 30:
            base_factor = 0.3  # Very conservative
        elif self.drawdown_percentage > 20:
            base_factor = 0.4
        else:
            base_factor = 0.5
            
        # Adjust based on user preference
        recovery_factor = base_factor * (0.5 + recovery_aggressiveness)
        
        # Cap at reasonable range
        return max(0.1, min(0.7, recovery_factor))

    def _calculate_streak_factor(self) -> float:
        """
        Calculate a factor to adjust position sizing based on winning/losing streaks.
        
        Returns:
            A multiplier to apply to position sizes (0.7-1.2 typically)
        """
        if self.consecutive_wins >= 5:
            # Increase size on winning streaks, but cap the increase
            return min(1.2, 1.0 + 0.04 * self.consecutive_wins)
        elif self.consecutive_losses >= 3:
            # Decrease size on losing streaks
            return max(0.7, 1.0 - 0.1 * self.consecutive_losses)
        else:
            return 1.0

    def _calculate_portfolio_factor(self, symbol: str) -> float:
        """
        Calculate a factor to ensure portfolio diversity.
        
        Args:
            symbol: The asset symbol
            
        Returns:
            A multiplier to apply to position sizes (0.0-1.0)
        """
        # Check current exposure to this asset
        current_exposure = self.current_exposure.get(symbol, 0)
        current_exposure_pct = current_exposure / self.current_capital if self.current_capital > 0 else 0
        
        # Check exposure to correlated assets
        # In a full system, we'd have a correlation matrix
        # Here we use a simple approach based on asset type/class
        asset_class = self._get_asset_class(symbol)
        correlated_exposure = sum(
            self.current_exposure.get(sym, 0) 
            for sym in self.current_exposure 
            if self._get_asset_class(sym) == asset_class
        )
        correlated_exposure_pct = correlated_exposure / self.current_capital if self.current_capital > 0 else 0
        
        # Factor based on current exposure
        if current_exposure_pct >= self.capital_settings.max_position_size_percentage / 100.0:
            direct_factor = 0  # Already at max for this asset
        else:
            # Scale down as we approach max
            direct_factor = 1.0 - (current_exposure_pct / (self.capital_settings.max_position_size_percentage / 100.0))
            
        # Factor based on correlated exposure
        max_correlated = self.capital_settings.max_correlated_exposure / 100.0
        if correlated_exposure_pct >= max_correlated:
            correlation_factor = 0  # Already at max for this asset class
        else:
            # Scale down as we approach max
            correlation_factor = 1.0 - (correlated_exposure_pct / max_correlated)
            
        # Take the minimum of the two factors
        return min(direct_factor, correlation_factor)

    def _calculate_optimal_leverage(self, symbol: str, signal_data: Dict[str, Any]) -> float:
        """
        Calculate the optimal leverage for a position.
        
        Args:
            symbol: The asset symbol
            signal_data: Signal data dictionary
            
        Returns:
            Optimal leverage multiplier (1.0 or greater)
        """
        # Start with user preference
        base_leverage = self.user_prefs.leverage_preference
        
        # Extract key metrics from signal
        confidence = signal_data.get('confidence', 0.5)
        risk_reward = signal_data.get('risk_reward', 1.0)
        market_volatility = signal_data.get('market_volatility', 0.5)  # 0-1 scale
        
        # Get asset-specific performance metrics
        asset_metrics = self.asset_performance.get(symbol, {
            'win_rate': 0.5,  # Default if no history
            'profit_factor': 1.0,
            'sharpe': 0.0
        })
        
        # Adjust based on signal quality
        if confidence > 0.7 and risk_reward > 2.0:
            confidence_factor = 1.2  # High quality signal
        elif confidence > 0.6 and risk_reward > 1.5:
            confidence_factor = 1.1  # Good quality signal
        elif confidence < 0.4 or risk_reward < 1.0:
            confidence_factor = 0.8  # Poor quality signal
        else:
            confidence_factor = 1.0  # Average signal
            
        # Adjust based on historical performance
        performance_factor = 1.0
        
        if asset_metrics.get('n_trades', 0) >= 20:  # Enough history
            win_rate = asset_metrics.get('win_rate', 0.5)
            profit_factor = asset_metrics.get('profit_factor', 1.0)
            sharpe = asset_metrics.get('sharpe', 0.0)
            
            if win_rate > 0.6 and profit_factor > 1.5 and sharpe > 1.0:
                performance_factor = 1.25  # Excellent performance
            elif win_rate > 0.55 and profit_factor > 1.3 and sharpe > 0.8:
                performance_factor = 1.1  # Good performance
            elif win_rate < 0.45 or profit_factor < 0.9 or sharpe < 0:
                performance_factor = 0.7  # Poor performance
                
        # Adjust based on market volatility
        volatility_factor = 1.0 - (market_volatility * 0.5)  # Reduce leverage in high volatility
        
        # Adjust for recovery mode
        recovery_factor = 0.5 if self.recovery_mode else 1.0
        
        # Calculate final leverage
        leverage = base_leverage * confidence_factor * performance_factor * volatility_factor * recovery_factor
        
        # Ensure within exchange limits for this asset
        max_leverage = MAX_LEVERAGE_SETTINGS.get(symbol, {}).get(
            'default', MAX_LEVERAGE_SETTINGS.get('default', 10.0)
        )
        
        leverage = max(1.0, min(leverage, max_leverage))
        
        return leverage

    def _get_asset_class(self, symbol: str) -> str:
        """
        Get the asset class for a symbol.
        
        Args:
            symbol: The asset symbol
            
        Returns:
            The asset class as a string
        """
        # In a full system, this would look up in a proper database
        # Here we use a simple approach based on symbol structure
        
        # Crypto examples
        if symbol.endswith('USDT') or symbol.endswith('BTC') or symbol.endswith('ETH'):
            return 'crypto'
            
        # Forex examples
        if any(symbol.startswith(curr) for curr in ['EUR', 'GBP', 'JPY', 'USD', 'AUD']):
            if any(symbol.endswith(curr) for curr in ['EUR', 'GBP', 'JPY', 'USD', 'AUD']):
                return 'forex'
                
        # Stock examples
        if len(symbol) <= 5 and symbol.isalpha():
            return 'stock'
            
        # Default
        return 'unknown'

    def update_current_exposure(self, positions: Dict[str, Dict[str, Any]]) -> None:
        """
        Update the current exposure across all assets.
        
        Args:
            positions: Dictionary mapping symbols to position information
        """
        new_exposure = {}
        
        for symbol, position in positions.items():
            position_value = position.get('position_value', 0)
            new_exposure[symbol] = position_value
            
        self.current_exposure = new_exposure
        
        # Log exposure summary
        total_exposure = sum(new_exposure.values())
        exposure_percentage = total_exposure / self.current_capital * 100 if self.current_capital > 0 else 0
        
        self.logger.info(
            f"Updated exposure: {total_exposure:.2f} ({exposure_percentage:.2f}% of capital), "
            f"Positions: {len(new_exposure)}"
        )
        
        # Detailed debug logging
        for symbol, value in new_exposure.items():
            self.logger.debug(
                f"  {symbol}: {value:.2f} ({value/self.current_capital*100:.2f}% of capital)"
            )

    def get_portfolio_allocation(self) -> Dict[str, float]:
        """
        Get the optimal portfolio allocation across assets.
        
        Returns:
            Dictionary mapping symbols to allocation percentages
        """
        # For a sophisticated implementation, this would use modern portfolio theory
        # Here we use a simpler approach based on asset performance
        
        # Get performance metrics for assets with history
        assets_with_history = {}
        
        for symbol, metrics in self.asset_performance.items():
            if metrics.get('n_trades', 0) >= 10:  # Enough history
                assets_with_history[symbol] = metrics
                
        # If no assets have history, use defaults
        if not assets_with_history:
            self.logger.warning("No assets with trading history, using default allocation")
            return {'default': 100.0}
            
        # Calculate scores based on key metrics
        scores = {}
        
        for symbol, metrics in assets_with_history.items():
            win_rate = metrics.get('win_rate', 0.5)
            profit_factor = metrics.get('profit_factor', 1.0)
            sharpe = metrics.get('sharpe', 0.0)
            
            # Combined score 0-100
            score = (
                win_rate * 30 +                     # 30% weight to win rate
                min(3, profit_factor) / 3 * 40 +    # 40% weight to profit factor (capped at 3)
                max(0, min(2, sharpe)) / 2 * 30     # 30% weight to sharpe (capped at 2)
            )
            
            scores[symbol] = score
            
        # Normalize scores to allocation percentages
        total_score = sum(scores.values())
        
        if total_score <= 0:
            self.logger.warning("All asset scores <= 0, using equal allocation")
            allocation = {symbol: 100.0 / len(assets_with_history) for symbol in assets_with_history}
        else:
            allocation = {symbol: score / total_score * 100.0 for symbol, score in scores.items()}
            
        # Ensure minimum allocations and cap maximum
        for symbol in allocation:
            allocation[symbol] = max(5.0, min(allocation[symbol], 50.0))
            
        # Renormalize to 100%
        total_allocation = sum(allocation.values())
        allocation = {symbol: alloc / total_allocation * 100.0 for symbol, alloc in allocation.items()}
        
        self.portfolio_allocation = allocation
        return allocation

    def get_available_capital(self) -> float:
        """
        Get the available capital for new positions.
        
        Returns:
            Available capital amount
        """
        # Calculate reserved capital
        reserve = self.current_capital * (self.capital_settings.reserve_percentage / 100.0)
        
        # Calculate current exposure
        total_exposure = sum(self.current_exposure.values())
        
        # Available = Current - Reserved - Exposure
        available = self.current_capital - reserve - total_exposure
        
        # Ensure non-negative
        return max(0, available)

    def get_trading_state(self) -> TradingState:
        """
        Get the current trading state and performance metrics.
        
        Returns:
            TradingState data class
        """
        # Calculate win rate
        total_trades = self.n_winning_trades + self.n_losing_trades
        win_rate = self.n_winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit factor
        profit_factor = self.total_win_amount / self.total_loss_amount if self.total_loss_amount > 0 else float('inf')
        
        # Calculate average win/loss
        avg_win = self.total_win_amount / self.n_winning_trades if self.n_winning_trades > 0 else 0.0
        avg_loss = self.total_loss_amount / self.n_losing_trades if self.n_losing_trades > 0 else 0.0
        
        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Get recent trades
        last_10_trades = list(self.recent_trades)[-10:] if self.recent_trades else []
        last_10_returns = [t.get('pnl_percentage', 0) for t in last_10_trades]
        
        # Calculate trade frequency (trades per day)
        if self.trades_history and len(self.trades_history) >= 2:
            first_trade_time = self.trades_history[0]['entry_time']
            last_trade_time = self.trades_history[-1]['exit_time']
            trading_days = (last_trade_time - first_trade_time) / (60 * 60 * 24)  # Convert seconds to days
            
            if trading_days > 0:
                trade_frequency = len(self.trades_history) / trading_days
            else:
                trade_frequency = len(self.trades_history)  # All in one day
        else:
            trade_frequency = 0.0
            
        # Calculate Sharpe and Sortino ratios
        daily_returns = self._calculate_daily_returns(self.trades_history)
        sharpe_ratio = calculate_sharpe_ratio(daily_returns) if daily_returns else 0.0
        sortino_ratio = calculate_sortino_ratio(daily_returns) if daily_returns else 0.0
        
        # Create and return the state
        return TradingState(
            current_equity=self.current_capital,
            starting_equity=self.initial_capital,
            peak_equity=self.peak_capital,
            drawdown_percentage=self.drawdown_percentage,
            win_rate=win_rate,
            profit_factor=profit_factor,
            recent_wins=sum(1 for t in last_10_trades if t.get('pnl', 0) > 0),
            recent_losses=sum(1 for t in last_10_trades if t.get('pnl', 0) <= 0),
            avg_win_size=avg_win,
            avg_loss_size=avg_loss,
            win_loss_ratio=win_loss_ratio,
            consecutive_wins=self.consecutive_wins,
            consecutive_losses=self.consecutive_losses,
            last_10_trades=last_10_returns,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            trade_frequency=trade_frequency,
            recovery_mode=self.recovery_mode,
            current_exposure=self.current_exposure,
            timestamp=time.time()
        )

    def adjust_for_correlation(self, proposed_trades: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Adjust position sizes for correlated assets.
        
        Args:
            proposed_trades: Dictionary mapping symbols to proposed trade information
            
        Returns:
            Adjusted trade information
        """
        if len(proposed_trades) <= 1:
            return proposed_trades  # No correlation to consider
            
        # Group by asset class
        asset_class_trades = defaultdict(list)
        
        for symbol, trade_info in proposed_trades.items():
            asset_class = self._get_asset_class(symbol)
            asset_class_trades[asset_class].append((symbol, trade_info))
            
        # Adjust trades within each asset class
        adjusted_trades = {}
        
        for asset_class, trades in asset_class_trades.items():
            if len(trades) <= 1:
                # No correlation adjustment needed for single asset in class
                for symbol, trade_info in trades:
                    adjusted_trades[symbol] = trade_info
                continue
                
            # Calculate total proposed exposure for this asset class
            total_proposed = sum(t[1].get('position_size', 0) for t in trades)
            
            # Get max allowed correlated exposure
            max_correlated_pct = self.capital_settings.max_correlated_exposure / 100.0
            max_correlated_amount = self.current_capital * max_correlated_pct
            
            # Check if adjustment needed
            if total_proposed <= max_correlated_amount:
                # No adjustment needed
                for symbol, trade_info in trades:
                    adjusted_trades[symbol] = trade_info
            else:
                # Proportionally reduce all trades in this asset class
                reduction_factor = max_correlated_amount / total_proposed
                
                for symbol, trade_info in trades:
                    adjusted_trade = trade_info.copy()
                    adjusted_trade['position_size'] = trade_info.get('position_size', 0) * reduction_factor
                    adjusted_trade['leveraged_size'] = trade_info.get('leveraged_size', 0) * reduction_factor
                    adjusted_trade['expected_risk_amount'] = trade_info.get('expected_risk_amount', 0) * reduction_factor
                    
                    # Add adjustment note
                    if 'factors_applied' not in adjusted_trade:
                        adjusted_trade['factors_applied'] = {}
                    adjusted_trade['factors_applied']['correlation_adjustment'] = reduction_factor
                    
                    adjusted_trades[symbol] = adjusted_trade
                    
                self.logger.info(
                    f"Applied correlation adjustment to {asset_class} trades: "
                    f"Reduction factor {reduction_factor:.2f}"
                )
                
        return adjusted_trades

    def adaptive_position_sizing(self, symbol: str, signal_data: Dict[str, Any], 
                                market_condition: str) -> Dict[str, Any]:
        """
        Calculate position size with adaptive sizing based on market conditions.
        
        Args:
            symbol: The asset symbol
            signal_data: Signal data dictionary
            market_condition: Current market condition (trending, ranging, volatile, etc.)
            
        Returns:
            Position sizing information
        """
        # Get base position sizing
        base_position = self.calculate_position_size(symbol, signal_data)
        
        # Apply condition-specific adjustments
        condition_factors = {
            'trending': 1.2,     # Increase size in trending markets
            'ranging': 0.9,      # Reduce size in ranging markets
            'volatile': 0.7,     # Substantially reduce size in volatile markets
            'low_volatility': 1.1,  # Slightly increase size in low volatility
            'breakout': 1.15,    # Slightly increase for breakouts
            'reversal': 0.85     # Reduce for reversals (higher risk)
        }
        
        # Get adjustment factor with default of 1.0 if condition not recognized
        condition_factor = condition_factors.get(market_condition, 1.0)
        
        # Apply adjustment
        adjusted_position = base_position.copy()
        adjusted_position['position_size'] *= condition_factor
        adjusted_position['leveraged_size'] *= condition_factor
        adjusted_position['expected_risk_amount'] *= condition_factor
        
        # Record adjustment
        if 'factors_applied' not in adjusted_position:
            adjusted_position['factors_applied'] = {}
        adjusted_position['factors_applied']['market_condition'] = market_condition
        adjusted_position['factors_applied']['condition_factor'] = condition_factor
        
        self.logger.info(
            f"Applied adaptive sizing for {market_condition} condition: "
            f"Factor {condition_factor:.2f}, Size {adjusted_position['position_size']:.4f}"
        )
        
        return adjusted_position

    def calculate_asset_allocation(self, total_capital: float) -> Dict[str, Dict[str, Any]]:
        """
        Calculate optimal asset allocation for portfolio.
        
        Args:
            total_capital: Total capital to allocate
            
        Returns:
            Dictionary mapping symbols to allocation information
        """
        # Get optimal portfolio allocation percentages
        allocation_percentages = self.get_portfolio_allocation()
        
        # Calculate allocation amounts
        allocations = {}
        
        for symbol, percentage in allocation_percentages.items():
            allocation_amount = total_capital * (percentage / 100.0)
            
            # Get asset-specific metrics
            asset_metrics = self.asset_performance.get(symbol, {})
            
            allocations[symbol] = {
                'percentage': percentage,
                'amount': allocation_amount,
                'performance_metrics': {
                    'win_rate': asset_metrics.get('win_rate', 0.5),
                    'profit_factor': asset_metrics.get('profit_factor', 1.0),
                    'sharpe': asset_metrics.get('sharpe', 0.0)
                }
            }
            
        self.logger.info(f"Calculated allocation for {len(allocations)} assets")
        
        return allocations

    def analyze_drawdown(self) -> Dict[str, Any]:
        """
        Analyze drawdown periods and recovery.
        
        Returns:
            Dictionary with drawdown analysis
        """
        if not self.trades_history:
            return {
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'recovery_time': 0.0,
                'in_drawdown': False
            }
            
        # Calculate equity curve
        equity_curve = self._calculate_equity_curve()
        
        # Calculate drawdown statistics
        max_dd, current_dd, dd_start_time, dd_duration = calculate_drawdown(equity_curve)
        
        # Determine if currently in drawdown
        in_drawdown = self.current_capital < self.peak_capital
        
        # Estimate recovery time based on recent performance
        recovery_time = self._estimate_recovery_time(current_dd)
        
        analysis = {
            'current_drawdown': self.drawdown_percentage,
            'max_drawdown': max_dd * 100.0,
            'drawdown_start_time': dd_start_time,
            'drawdown_duration_days': dd_duration / (60 * 60 * 24) if dd_duration else 0,
            'in_drawdown': in_drawdown,
            'estimated_recovery_time_days': recovery_time,
            'recovery_mode': self.recovery_mode
        }
        
        self.logger.debug(f"Drawdown analysis: {analysis}")
        
        return analysis

    def _calculate_equity_curve(self) -> Dict[float, float]:
        """
        Calculate historical equity curve from trades.
        
        Returns:
            Dictionary mapping timestamps to equity values
        """
        equity_curve = {0: self.initial_capital}  # Start with initial capital
        
        # Create ordered list of trades
        sorted_trades = sorted(self.trades_history, key=lambda t: t['exit_time'])
        
        current_equity = self.initial_capital
        
        for trade in sorted_trades:
            current_equity += trade['pnl']
            equity_curve[trade['exit_time']] = current_equity
            
        # Add current state
        equity_curve[time.time()] = self.current_capital
        
        return equity_curve

    def _estimate_recovery_time(self, current_drawdown: float) -> float:
        """
        Estimate recovery time from current drawdown.
        
        Args:
            current_drawdown: Current drawdown as a decimal (0.0-1.0)
            
        Returns:
            Estimated recovery time in days
        """
        if current_drawdown <= 0:
            return 0.0
            
        # Need to calculate average daily return
        if not self.trades_history:
            return float('inf')  # Can't estimate without trades
            
        # Get daily returns
        daily_returns = self._calculate_daily_returns(self.trades_history)
        
        if not daily_returns:
            return float('inf')
            
        avg_daily_return = sum(daily_returns) / len(daily_returns)
        
        if avg_daily_return <= 0:
            return float('inf')  # Never recover with negative returns
            
        # Calculate required return to recover
        # If down 20%, need 25% to get back to original
        required_return = current_drawdown / (1 - current_drawdown)
        
        # Estimate days
        estimated_days = required_return / avg_daily_return
        
        return estimated_days

    def risk_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis.
        
        Returns:
            Dictionary with risk analysis
        """
        trading_state = self.get_trading_state()
        drawdown_analysis = self.analyze_drawdown()
        
        # Calculate risk of ruin
        risk_of_ruin = self._calculate_risk_of_ruin(trading_state)
        
        # Analyze per-asset risk
        asset_risk = {}
        
        for symbol, metrics in self.asset_performance.items():
            if metrics.get('n_trades', 0) >= 10:
                win_rate = metrics.get('win_rate', 0.5)
                profit_factor = metrics.get('profit_factor', 1.0)
                
                # Assess risk level
                if win_rate > 0.6 and profit_factor > 1.5:
                    risk_level = 'low'
                elif win_rate > 0.5 and profit_factor > 1.0:
                    risk_level = 'moderate'
                else:
                    risk_level = 'high'
                    
                # Calculate max recommended exposure
                max_exposure = 0.0
                
                if risk_level == 'low':
                    max_exposure = 0.15  # 15% of capital
                elif risk_level == 'moderate':
                    max_exposure = 0.1   # 10% of capital
                else:
                    max_exposure = 0.05  # 5% of capital
                    
                asset_risk[symbol] = {
                    'risk_level': risk_level,
                    'max_recommended_exposure': max_exposure * 100.0,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor
                }
        
        # Portfolio concentration risk
        exposure_values = list(self.current_exposure.values())
        if exposure_values:
            max_exposure = max(exposure_values)
            total_exposure = sum(exposure_values)
            concentration_ratio = max_exposure / total_exposure if total_exposure > 0 else 0
        else:
            concentration_ratio = 0
            
        # Overall risk assessment
        if trading_state.drawdown_percentage > 25 or risk_of_ruin > 0.1 or concentration_ratio > 0.5:
            overall_risk = 'high'
        elif trading_state.drawdown_percentage > 15 or risk_of_ruin > 0.05 or concentration_ratio > 0.3:
            overall_risk = 'moderate'
        else:
            overall_risk = 'low'
            
        risk_analysis = {
            'overall_risk': overall_risk,
            'risk_of_ruin': risk_of_ruin,
            'portfolio_concentration': concentration_ratio,
            'drawdown': drawdown_analysis,
            'total_exposure_percentage': sum(exposure_values) / self.current_capital * 100 if self.current_capital > 0 else 0,
            'asset_risk': asset_risk,
            'recovery_mode': self.recovery_mode
        }
        
        self.logger.info(f"Risk analysis: Overall risk {overall_risk}, Risk of ruin {risk_of_ruin:.4f}")
        
        return risk_analysis

    def _calculate_risk_of_ruin(self, state: TradingState) -> float:
        """
        Calculate the risk of ruin (probability of losing all capital).
        
        Args:
            state: Current trading state
            
        Returns:
            Risk of ruin probability (0.0-1.0)
        """
        # Simplified risk of ruin calculation
        if state.win_rate <= 0.5:
            # With win rate <= 50%, risk of ruin is theoretically 1.0
            # But we'll use a more realistic approach
            base_risk = 0.5
        else:
            # Formula: (1-edge/per_trade_risk)^capital_units
            # where edge = 2*win_rate - 1
            edge = 2 * state.win_rate - 1
            per_trade_risk = self.user_prefs.risk_per_trade / 100.0
            
            if edge >= per_trade_risk:
                # Very low risk of ruin
                base_risk = 0.01
            else:
                capital_units = 100  # Simplification: assuming 100 units of capital
                base_risk = (1 - edge / per_trade_risk) ** capital_units
                base_risk = min(1.0, max(0.0, base_risk))
                
        # Adjust for drawdown - higher drawdown increases risk
        drawdown_factor = 1.0 + (state.drawdown_percentage / 100.0) * 2
        
        # Adjust for profit factor - higher profit factor reduces risk
        profit_factor_adj = 1.0 / max(1.0, state.profit_factor)
        
        # Final risk calculation
        risk = base_risk * drawdown_factor * profit_factor_adj
        
        # Cap at reasonable range
        return min(0.99, max(0.01, risk))

    async def run_management_loop(self, interval: float = 60.0) -> None:
        """
        Run a continuous capital management loop.
        
        Args:
            interval: Update interval in seconds
        """
        self.logger.info(f"Starting capital management loop with {interval}s interval")
        
        try:
            while True:
                # Update trading state
                state = self.get_trading_state()
                
                # Perform risk analysis
                risk = self.risk_analysis()
                
                # Log status
                self.logger.info(
                    f"Capital: {self.current_capital:.2f}, Drawdown: {state.drawdown_percentage:.2f}%, "
                    f"Win rate: {state.win_rate:.2f}, Recovery mode: {self.recovery_mode}"
                )
                
                # Dynamic allocation update
                if self.user_prefs.capital_allocation_strategy == "dynamic":
                    self.get_portfolio_allocation()
                    
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            self.logger.info("Capital management loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in capital management loop: {e}")
            raise CapitalManagementError(f"Management loop failed: {str(e)}")
