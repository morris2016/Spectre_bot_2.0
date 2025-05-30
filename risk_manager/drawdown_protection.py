

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Risk Manager - Drawdown Protection Module

This module implements sophisticated drawdown protection mechanisms to preserve
capital during periods of account drawdown. It dynamically adjusts trading
parameters, position sizes, and risk levels based on drawdown metrics to
ensure account recovery and prevent catastrophic losses.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Set, Type


class BaseDrawdownProtector:
    """Base class for drawdown protection strategies."""

    registry: Dict[str, Type["BaseDrawdownProtector"]] = {}

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        key = name or cls.__name__
        BaseDrawdownProtector.registry[key] = cls

    async def apply_protection(self, *args, **kwargs) -> None:
        raise NotImplementedError

import time
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from enum import Enum, auto
import json

from common.constants import (
    DRAWDOWN_PROTECTION_LEVELS, MAX_ALLOWED_DRAWDOWN,
    RECOVERY_MODE_THRESHOLDS, TIMEFRAMES, DEFAULT_RISK_PERCENTAGE
)
from common.logger import get_logger
from common.utils import calculate_zscore, detect_outliers, float_round
from common.metrics import MetricsCollector
from common.exceptions import DrawdownLimitExceededException, RiskManagementException

from data_storage.market_data import MarketDataRepository
from data_storage.models.strategy_data import StrategyPerformance

logger = get_logger("risk_manager.drawdown_protection")


class DrawdownState(Enum):
    """Enum for the possible drawdown states of the account."""
    NORMAL = auto()            # No significant drawdown
    MILD = auto()              # Mild drawdown, slightly reduced risk
    MODERATE = auto()          # Moderate drawdown, significantly reduced risk
    SEVERE = auto()            # Severe drawdown, heavily restricted trading
    CRITICAL = auto()          # Critical drawdown, trading paused
    RECOVERY = auto()          # In recovery mode after drawdown


class RecoveryMode(Enum):
    """Enum for the possible recovery modes after drawdown."""
    CONSERVATIVE = auto()      # Very cautious recovery, only highest probability trades
    BALANCED = auto()          # Moderate recovery pace
    AGGRESSIVE = auto()        # Faster recovery, more risk acceptance
    CUSTOMIZED = auto()        # User-defined recovery parameters


@dataclass
class DrawdownConfig:
    """Configuration settings for drawdown protection."""
    max_drawdown_percent: float         # Maximum allowed drawdown percentage
    mild_threshold: float               # Threshold for mild drawdown state
    moderate_threshold: float           # Threshold for moderate drawdown state
    severe_threshold: float             # Threshold for severe drawdown state
    critical_threshold: float           # Threshold for critical drawdown state
    default_risk_percentage: float      # Default risk per trade (base risk)
    min_win_rate_threshold: float       # Minimum win rate required for recovery mode
    recovery_mode: RecoveryMode         # Selected recovery mode
    custom_recovery_params: Optional[Dict[str, Any]] = None  # For CUSTOMIZED mode
    drawdown_calculation_window: int = 30  # Days to calculate drawdown over
    recovery_reset_threshold: float = 0.05  # When drawdown is below this, reset to NORMAL


@dataclass
class DrawdownStatus:
    """Current drawdown status information."""
    timestamp: datetime
    current_drawdown: float
    peak_value: float
    current_value: float
    drawdown_state: DrawdownState
    adjusted_risk_percentage: float
    in_recovery_mode: bool
    recovery_progress: float  # 0.0 to 1.0
    recovery_mode: Optional[RecoveryMode]
    restricted_assets: List[str]
    restricted_strategies: List[str]
    message: str


class DrawdownProtection(BaseDrawdownProtector):
    """
    Implements sophisticated drawdown protection mechanisms to preserve capital
    and ensure consistent trading performance even during drawdown periods.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        market_data_repo: MarketDataRepository,
        config: Optional[DrawdownConfig] = None
    ):
        """
        Initialize the drawdown protection system.
        
        Args:
            metrics_collector: Metrics collection system for performance tracking
            market_data_repo: Repository for accessing market data
            config: Optional custom configuration
        """
        self.metrics_collector = metrics_collector
        self.market_data_repo = market_data_repo
        self.config = config or self._get_default_config()
        
        # Current state tracking
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.drawdown_state = DrawdownState.NORMAL
        self.adjusted_risk_pct = self.config.default_risk_percentage
        
        # Recovery mode tracking
        self.in_recovery_mode = False
        self.recovery_start_time = None
        self.recovery_start_equity = 0.0
        self.recovery_progress = 0.0
        self.recovery_target_equity = 0.0
        
        # Restricted assets and strategies (those performing poorly during drawdown)
        self.restricted_assets = set()
        self.restricted_strategies = set()
        
        # Strategy performance tracking
        self.strategy_performance = {}
        self.asset_performance = {}
        
        # Historical equity and drawdown tracking
        self.equity_history = []
        self.drawdown_history = []
        
        # State change tracking
        self.last_state_change = datetime.now()
        self.state_change_history = []
        
        logger.info("Drawdown Protection system initialized")
    
    def _get_default_config(self) -> DrawdownConfig:
        """Create default configuration for drawdown protection."""
        return DrawdownConfig(
            max_drawdown_percent=MAX_ALLOWED_DRAWDOWN,
            mild_threshold=DRAWDOWN_PROTECTION_LEVELS["mild"],
            moderate_threshold=DRAWDOWN_PROTECTION_LEVELS["moderate"],
            severe_threshold=DRAWDOWN_PROTECTION_LEVELS["severe"],
            critical_threshold=DRAWDOWN_PROTECTION_LEVELS["critical"],
            default_risk_percentage=DEFAULT_RISK_PERCENTAGE,
            min_win_rate_threshold=0.55,  # Require 55% win rate for recovery
            recovery_mode=RecoveryMode.BALANCED,
            drawdown_calculation_window=30  # 30 days
        )
    
    async def update_account_status(
        self, 
        current_equity: float, 
        exchange: str = "global"
    ) -> DrawdownStatus:
        """
        Update the current drawdown status based on account equity.
        
        Args:
            current_equity: Current account equity
            exchange: Exchange identifier or 'global' for overall
            
        Returns:
            DrawdownStatus: Updated drawdown status
        """
        # Update current equity
        self.current_equity = current_equity
        
        # Update peak equity if needed
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0
        
        # Track history
        history_entry = {
            "timestamp": datetime.now(),
            "equity": current_equity,
            "drawdown": self.current_drawdown
        }
        
        self.equity_history.append(history_entry)
        self.drawdown_history.append(history_entry)
        
        # Trim history to configured window
        cutoff_date = datetime.now() - timedelta(days=self.config.drawdown_calculation_window)
        self.equity_history = [entry for entry in self.equity_history if entry["timestamp"] >= cutoff_date]
        self.drawdown_history = [entry for entry in self.drawdown_history if entry["timestamp"] >= cutoff_date]
        
        # Determine new drawdown state
        old_state = self.drawdown_state
        self.drawdown_state = self._determine_drawdown_state()
        
        # Track state changes
        if self.drawdown_state != old_state:
            state_change = {
                "timestamp": datetime.now(),
                "from_state": old_state.name,
                "to_state": self.drawdown_state.name,
                "drawdown": self.current_drawdown,
                "equity": current_equity
            }
            self.state_change_history.append(state_change)
            self.last_state_change = datetime.now()
            
            logger.info(f"Drawdown state changed: {old_state.name} -> {self.drawdown_state.name} " +
                      f"(Drawdown: {self.current_drawdown:.2%}, Equity: {current_equity:.2f})")
            
            # Record metric for state change
            self.metrics_collector.record_event(
                "drawdown_state_change",
                {
                    "from_state": old_state.name,
                    "to_state": self.drawdown_state.name,
                    "drawdown": float_round(self.current_drawdown, 4),
                    "equity": float_round(current_equity, 2),
                    "exchange": exchange
                }
            )
        
        # Update risk percentage based on drawdown state
        self.adjusted_risk_pct = self._calculate_adjusted_risk()
        
        # Check if we should enter or exit recovery mode
        if self.drawdown_state == DrawdownState.NORMAL and self.in_recovery_mode:
            if self.current_drawdown <= self.config.recovery_reset_threshold:
                # Exit recovery mode if drawdown is below the reset threshold
                self._exit_recovery_mode()
        elif self.drawdown_state != DrawdownState.NORMAL and not self.in_recovery_mode:
            if self.drawdown_state == DrawdownState.CRITICAL:
                # Pause all trading activity and restrict assets/strategies
                self.restricted_assets.update(self.asset_performance.keys())
                self.restricted_strategies.update(self.strategy_performance.keys())
                logger.critical(
                    f"Critical drawdown reached ({self.current_drawdown:.2%}). Trading paused."
                )
                self.metrics_collector.record_event(
                    "critical_drawdown",
                    {
                        "drawdown": float_round(self.current_drawdown, 4),
                        "equity": float_round(self.current_equity, 2),
                    },
                )
            elif self.current_drawdown >= self.config.mild_threshold:
                # Enter recovery mode
                self._enter_recovery_mode()
        
        # Update recovery progress if in recovery mode
        if self.in_recovery_mode:
            self._update_recovery_progress()
        
        # Record metrics
        self._record_metrics(exchange)
        
        # Create and return current status
        status = DrawdownStatus(
            timestamp=datetime.now(),
            current_drawdown=self.current_drawdown,
            peak_value=self.peak_equity,
            current_value=self.current_equity,
            drawdown_state=self.drawdown_state,
            adjusted_risk_percentage=self.adjusted_risk_pct,
            in_recovery_mode=self.in_recovery_mode,
            recovery_progress=self.recovery_progress,
            recovery_mode=self.config.recovery_mode if self.in_recovery_mode else None,
            restricted_assets=list(self.restricted_assets),
            restricted_strategies=list(self.restricted_strategies),
            message=self._get_status_message()
        )
        
        return status
    
    def _determine_drawdown_state(self) -> DrawdownState:
        """Determine the current drawdown state based on drawdown percentage."""
        if self.current_drawdown >= self.config.critical_threshold:
            return DrawdownState.CRITICAL
        elif self.current_drawdown >= self.config.severe_threshold:
            return DrawdownState.SEVERE
        elif self.current_drawdown >= self.config.moderate_threshold:
            return DrawdownState.MODERATE
        elif self.current_drawdown >= self.config.mild_threshold:
            return DrawdownState.MILD
        elif self.in_recovery_mode and self.recovery_progress < 1.0:
            return DrawdownState.RECOVERY
        else:
            return DrawdownState.NORMAL
    
    def _calculate_adjusted_risk(self) -> float:
        """Calculate the adjusted risk percentage based on drawdown state."""
        base_risk = self.config.default_risk_percentage
        
        if self.drawdown_state == DrawdownState.CRITICAL:
            # No trading in critical state
            return 0.0
        elif self.drawdown_state == DrawdownState.SEVERE:
            # Severe reduction in risk
            return base_risk * 0.25
        elif self.drawdown_state == DrawdownState.MODERATE:
            # Moderate reduction in risk
            return base_risk * 0.5
        elif self.drawdown_state == DrawdownState.MILD:
            # Mild reduction in risk
            return base_risk * 0.75
        elif self.drawdown_state == DrawdownState.RECOVERY:
            # Risk during recovery depends on recovery mode and progress
            return self._calculate_recovery_risk()
        else:  # NORMAL
            return base_risk
    
    def _enter_recovery_mode(self):
        """Enter the recovery mode after drawdown."""
        if self.in_recovery_mode:
            return
        
        self.in_recovery_mode = True
        self.recovery_start_time = datetime.now()
        self.recovery_start_equity = self.current_equity
        self.recovery_target_equity = self.peak_equity
        self.recovery_progress = 0.0
        
        logger.info(f"Entering recovery mode: Start equity={self.recovery_start_equity:.2f}, " +
                  f"Target equity={self.recovery_target_equity:.2f}, Mode={self.config.recovery_mode.name}")
        
        # Record metric for entering recovery mode
        self.metrics_collector.record_event(
            "recovery_mode_entered",
            {
                "start_equity": float_round(self.recovery_start_equity, 2),
                "target_equity": float_round(self.recovery_target_equity, 2),
                "mode": self.config.recovery_mode.name,
                "drawdown": float_round(self.current_drawdown, 4)
            }
        )
    
    def _exit_recovery_mode(self):
        """Exit the recovery mode after successful recovery."""
        if not self.in_recovery_mode:
            return
        
        recovery_duration = (datetime.now() - self.recovery_start_time).total_seconds() / 86400  # days
        equity_gain = self.current_equity - self.recovery_start_equity
        
        self.in_recovery_mode = False
        self.recovery_progress = 1.0
        
        logger.info(f"Exiting recovery mode: Duration={recovery_duration:.2f} days, " +
                  f"Equity gain={equity_gain:.2f}, Final progress={self.recovery_progress:.2%}")
        
        # Clear restrictions
        self.restricted_assets.clear()
        self.restricted_strategies.clear()
        
        # Record metric for recovery completion
        self.metrics_collector.record_event(
            "recovery_mode_completed",
            {
                "duration_days": float_round(recovery_duration, 2),
                "equity_gain": float_round(equity_gain, 2),
                "start_equity": float_round(self.recovery_start_equity, 2),
                "final_equity": float_round(self.current_equity, 2)
            }
        )
    
    def _update_recovery_progress(self):
        """Update the recovery progress calculation."""
        if not self.in_recovery_mode:
            return
        
        equity_range = self.recovery_target_equity - self.recovery_start_equity
        if equity_range <= 0:
            self.recovery_progress = 1.0
            return
        
        current_gain = self.current_equity - self.recovery_start_equity
        self.recovery_progress = min(1.0, max(0.0, current_gain / equity_range))
        
        # Record metric for recovery progress
        self.metrics_collector.record_gauge(
            "recovery_progress",
            float_round(self.recovery_progress, 4),
            {"mode": self.config.recovery_mode.name}
        )
    
    def _calculate_recovery_risk(self) -> float:
        """Calculate appropriate risk percentage during recovery mode."""
        base_risk = self.config.default_risk_percentage
        
        if self.config.recovery_mode == RecoveryMode.CONSERVATIVE:
            # Very cautious recovery
            max_recovery_risk = base_risk * 0.6
            min_recovery_risk = base_risk * 0.3
        elif self.config.recovery_mode == RecoveryMode.BALANCED:
            # Balanced recovery pace
            max_recovery_risk = base_risk * 0.8
            min_recovery_risk = base_risk * 0.4
        elif self.config.recovery_mode == RecoveryMode.AGGRESSIVE:
            # Faster recovery
            max_recovery_risk = base_risk * 1.0
            min_recovery_risk = base_risk * 0.5
        elif self.config.recovery_mode == RecoveryMode.CUSTOMIZED and self.config.custom_recovery_params:
            # Use custom parameters
            params = self.config.custom_recovery_params
            max_recovery_risk = params.get("max_risk", base_risk * 0.8)
            min_recovery_risk = params.get("min_risk", base_risk * 0.4)
        else:
            # Default to balanced
            max_recovery_risk = base_risk * 0.8
            min_recovery_risk = base_risk * 0.4
        
        # Scale risk based on recovery progress - more progress means more risk allowed
        risk_range = max_recovery_risk - min_recovery_risk
        risk = min_recovery_risk + (risk_range * self.recovery_progress)
        
        return risk
    
    def _record_metrics(self, exchange: str):
        """Record relevant metrics for monitoring drawdown protection."""
        self.metrics_collector.record_gauge(
            "account_drawdown_percent", 
            float_round(self.current_drawdown * 100, 2),
            {"exchange": exchange}
        )
        
        self.metrics_collector.record_gauge(
            "account_equity", 
            float_round(self.current_equity, 2),
            {"exchange": exchange}
        )
        
        self.metrics_collector.record_gauge(
            "account_peak_equity", 
            float_round(self.peak_equity, 2),
            {"exchange": exchange}
        )
        
        self.metrics_collector.record_gauge(
            "adjusted_risk_percentage", 
            float_round(self.adjusted_risk_pct, 4),
            {"exchange": exchange, "drawdown_state": self.drawdown_state.name}
        )
        
        # State as numeric value for easier graphing
        self.metrics_collector.record_gauge(
            "drawdown_state", 
            self.drawdown_state.value,
            {"exchange": exchange}
        )
        
        # Recovery mode status
        self.metrics_collector.record_gauge(
            "in_recovery_mode", 
            1 if self.in_recovery_mode else 0,
            {"exchange": exchange}
        )
    
    def _get_status_message(self) -> str:
        """Generate a human-readable status message for the current drawdown state."""
        if self.drawdown_state == DrawdownState.NORMAL:
            return "Normal trading: No significant drawdown detected."
        elif self.drawdown_state == DrawdownState.MILD:
            return f"Mild drawdown ({self.current_drawdown:.2%}): Slightly reduced risk settings."
        elif self.drawdown_state == DrawdownState.MODERATE:
            return f"Moderate drawdown ({self.current_drawdown:.2%}): Significantly reduced risk settings."
        elif self.drawdown_state == DrawdownState.SEVERE:
            return f"Severe drawdown ({self.current_drawdown:.2%}): Heavily restricted trading."
        elif self.drawdown_state == DrawdownState.CRITICAL:
            return f"Critical drawdown ({self.current_drawdown:.2%}): Trading paused for capital preservation."
        elif self.drawdown_state == DrawdownState.RECOVERY:
            return (f"Recovery mode ({self.config.recovery_mode.name}): " +
                   f"Progress {self.recovery_progress:.2%}, adjusted risk {self.adjusted_risk_pct:.2%}")
    
    async def update_strategy_performance(
        self, 
        strategy_id: str, 
        performance_data: StrategyPerformance
    ):
        """
        Update performance tracking for a specific strategy.
        
        Args:
            strategy_id: Identifier for the strategy
            performance_data: Performance metrics for the strategy
        """
        self.strategy_performance[strategy_id] = performance_data
        
        # Update restricted strategies list based on performance
        if self.in_recovery_mode or self.drawdown_state != DrawdownState.NORMAL:
            # During drawdown or recovery, restrict poor performing strategies
            if (performance_data.win_rate < self.config.min_win_rate_threshold or
                performance_data.sharpe_ratio < 0.5 or
                performance_data.profit_factor < 1.0):
                
                if strategy_id not in self.restricted_strategies:
                    self.restricted_strategies.add(strategy_id)
                    logger.info(f"Restricting strategy {strategy_id} due to poor performance during drawdown")
            else:
                # Good performing strategy can be unrestricted
                if strategy_id in self.restricted_strategies:
                    self.restricted_strategies.remove(strategy_id)
                    logger.info(f"Unrestricting strategy {strategy_id} due to improved performance")
    
    async def update_asset_performance(
        self, 
        asset: str, 
        win_rate: float, 
        profit_factor: float,
        expectancy: float
    ):
        """
        Update performance tracking for a specific asset.
        
        Args:
            asset: Asset symbol
            win_rate: Win rate for this asset
            profit_factor: Profit factor for this asset
            expectancy: Average expectancy per trade
        """
        self.asset_performance[asset] = {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "last_updated": datetime.now()
        }
        
        # Update restricted assets list based on performance
        if self.in_recovery_mode or self.drawdown_state != DrawdownState.NORMAL:
            # During drawdown or recovery, restrict poor performing assets
            if (win_rate < self.config.min_win_rate_threshold or
                profit_factor < 1.0 or
                expectancy <= 0):
                
                if asset not in self.restricted_assets:
                    self.restricted_assets.add(asset)
                    logger.info(f"Restricting asset {asset} due to poor performance during drawdown")
            else:
                # Good performing asset can be unrestricted
                if asset in self.restricted_assets:
                    self.restricted_assets.remove(asset)
                    logger.info(f"Unrestricting asset {asset} due to improved performance")
    
    def calculate_position_size(
        self, 
        account_balance: float,
        asset: str,
        strategy_id: str,
        base_risk_pct: Optional[float] = None,
        trade_expectancy: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Calculate appropriate position size based on current drawdown state.
        
        Args:
            account_balance: Current account balance
            asset: Asset symbol for the trade
            strategy_id: Strategy identifier
            base_risk_pct: Base risk percentage (overrides default)
            trade_expectancy: Expected value of this specific trade
            
        Returns:
            Tuple containing:
            - float: Risk amount in account currency
            - str: Explanation of the calculation
        """
        # Start with adjusted risk based on drawdown state
        risk_pct = self.adjusted_risk_pct
        explanation = [f"Base adjusted risk: {risk_pct:.2%} (Drawdown state: {self.drawdown_state.name})"]
        
        # Check if trading is allowed at all
        if self.drawdown_state == DrawdownState.CRITICAL:
            return 0.0, "Trading paused: Critical drawdown state"
        
        # Check if asset is restricted
        if asset in self.restricted_assets:
            return 0.0, f"Trading paused for asset {asset}: Poor performance during drawdown"
        
        # Check if strategy is restricted
        if strategy_id in self.restricted_strategies:
            return 0.0, f"Trading paused for strategy {strategy_id}: Poor performance during drawdown"
        
        # Override base risk if provided
        if base_risk_pct is not None:
            # Still cap it by our adjusted risk
            risk_pct = min(base_risk_pct, self.adjusted_risk_pct)
            explanation.append(f"Custom base risk: {base_risk_pct:.2%}, capped at {risk_pct:.2%}")
        
        # Apply expectancy modifier if provided
        modifier = 1.0
        if trade_expectancy is not None:
            if trade_expectancy <= 0:
                return 0.0, f"Trade rejected: Negative expectancy ({trade_expectancy:.2f})"
            
            # Scale risk based on expectancy (higher expectancy = more risk allowed)
            if trade_expectancy >= 2.0:
                modifier = 1.2  # Exceptional expectancy
            elif trade_expectancy >= 1.5:
                modifier = 1.1  # Very good expectancy
            elif trade_expectancy >= 1.0:
                modifier = 1.0  # Good expectancy
            elif trade_expectancy >= 0.5:
                modifier = 0.8  # Moderate expectancy
            else:
                modifier = 0.6  # Poor expectancy
            
            explanation.append(f"Expectancy modifier: {modifier:.2f} (Expectancy: {trade_expectancy:.2f})")
        
        # Apply strategy performance modifier
        if strategy_id in self.strategy_performance:
            perf = self.strategy_performance[strategy_id]
            
            # High-performing strategies can get more allocation during recovery
            if self.in_recovery_mode and perf.win_rate >= 0.6 and perf.profit_factor >= 1.5:
                strategy_modifier = 1.2
                explanation.append(f"High-performing strategy bonus: {strategy_modifier:.2f}")
                modifier *= strategy_modifier
        
        # Apply asset performance modifier
        if asset in self.asset_performance:
            asset_perf = self.asset_performance[asset]
            
            # High-performing assets can get more allocation during recovery
            if self.in_recovery_mode and asset_perf["win_rate"] >= 0.6 and asset_perf["profit_factor"] >= 1.5:
                asset_modifier = 1.2
                explanation.append(f"High-performing asset bonus: {asset_modifier:.2f}")
                modifier *= asset_modifier
        
        # Apply final modifier
        final_risk_pct = risk_pct * modifier
        explanation.append(f"Final risk percentage: {final_risk_pct:.2%}")
        
        # Calculate risk amount
        risk_amount = account_balance * final_risk_pct
        explanation.append(f"Risk amount: {risk_amount:.2f} (Balance: {account_balance:.2f})")
        
        return risk_amount, " | ".join(explanation)
    
    def set_recovery_mode(self, mode: RecoveryMode, custom_params: Optional[Dict[str, Any]] = None):
        """
        Set the recovery mode for drawdown recovery.
        
        Args:
            mode: Recovery mode to use
            custom_params: Custom parameters for CUSTOMIZED mode
        """
        self.config.recovery_mode = mode
        
        if mode == RecoveryMode.CUSTOMIZED and custom_params:
            self.config.custom_recovery_params = custom_params
        
        logger.info(f"Recovery mode set to {mode.name}")
        
        # If already in recovery, update messaging
        if self.in_recovery_mode:
            logger.info(f"Recovery mode updated during active recovery: {mode.name}, Progress: {self.recovery_progress:.2%}")
    
    def reset_peak_equity(self, new_peak: Optional[float] = None):
        """
        Reset the peak equity tracking, optionally to a specific value.
        This is typically used after account deposits or withdrawals.
        
        Args:
            new_peak: Optional new peak equity value, defaults to current equity
        """
        if new_peak is not None:
            self.peak_equity = new_peak
        else:
            self.peak_equity = self.current_equity
        
        # Recalculate drawdown
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0
        
        logger.info(f"Peak equity reset to {self.peak_equity:.2f}, current drawdown: {self.current_drawdown:.2%}")
        
        # Update drawdown state
        old_state = self.drawdown_state
        self.drawdown_state = self._determine_drawdown_state()
        
        if self.drawdown_state != old_state:
            logger.info(f"Drawdown state changed after peak reset: {old_state.name} -> {self.drawdown_state.name}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get detailed status information about the current drawdown state.
        
        Returns:
            Dict containing detailed drawdown status information
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "current_drawdown": float_round(self.current_drawdown, 4),
            "current_drawdown_pct": f"{self.current_drawdown:.2%}",
            "peak_equity": float_round(self.peak_equity, 2),
            "current_equity": float_round(self.current_equity, 2),
            "drawdown_state": self.drawdown_state.name,
            "adjusted_risk_pct": float_round(self.adjusted_risk_pct, 4),
            "adjusted_risk_pct_str": f"{self.adjusted_risk_pct:.2%}",
            "in_recovery_mode": self.in_recovery_mode,
            "recovery_mode": self.config.recovery_mode.name if self.in_recovery_mode else None,
            "recovery_progress": float_round(self.recovery_progress, 4) if self.in_recovery_mode else 0,
            "recovery_progress_pct": f"{self.recovery_progress:.2%}" if self.in_recovery_mode else "0%",
            "recovery_start_time": self.recovery_start_time.isoformat() if self.recovery_start_time else None,
            "recovery_start_equity": float_round(self.recovery_start_equity, 2) if self.in_recovery_mode else 0,
            "recovery_target_equity": float_round(self.recovery_target_equity, 2) if self.in_recovery_mode else 0,
            "restricted_assets_count": len(self.restricted_assets),
            "restricted_strategies_count": len(self.restricted_strategies),
            "restricted_assets": list(self.restricted_assets),
            "restricted_strategies": list(self.restricted_strategies),
            "last_state_change": self.last_state_change.isoformat(),
            "time_in_current_state": (datetime.now() - self.last_state_change).total_seconds() / 3600,  # hours
            "state_change_history": [
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "from_state": entry["from_state"],
                    "to_state": entry["to_state"],
                    "drawdown": float_round(entry["drawdown"], 4),
                    "equity": float_round(entry["equity"], 2)
                }
                for entry in self.state_change_history[-10:]  # Last 10 state changes
            ],
            "message": self._get_status_message()
        }
    
    def export_configuration(self) -> Dict[str, Any]:
        """
        Export the current configuration for persistence.
        
        Returns:
            Dict containing configuration settings
        """
        config_data = {
            "max_drawdown_percent": self.config.max_drawdown_percent,
            "mild_threshold": self.config.mild_threshold,
            "moderate_threshold": self.config.moderate_threshold,
            "severe_threshold": self.config.severe_threshold,
            "critical_threshold": self.config.critical_threshold,
            "default_risk_percentage": self.config.default_risk_percentage,
            "min_win_rate_threshold": self.config.min_win_rate_threshold,
            "recovery_mode": self.config.recovery_mode.name,
            "drawdown_calculation_window": self.config.drawdown_calculation_window,
            "recovery_reset_threshold": self.config.recovery_reset_threshold
        }
        
        if self.config.recovery_mode == RecoveryMode.CUSTOMIZED and self.config.custom_recovery_params:
            config_data["custom_recovery_params"] = self.config.custom_recovery_params
        
        return config_data
    
    def import_configuration(self, config_data: Dict[str, Any]):
        """
        Import configuration settings.
        
        Args:
            config_data: Configuration dictionary to import
        """
        if "max_drawdown_percent" in config_data:
            self.config.max_drawdown_percent = config_data["max_drawdown_percent"]
        
        if "mild_threshold" in config_data:
            self.config.mild_threshold = config_data["mild_threshold"]
        
        if "moderate_threshold" in config_data:
            self.config.moderate_threshold = config_data["moderate_threshold"]
        
        if "severe_threshold" in config_data:
            self.config.severe_threshold = config_data["severe_threshold"]
        
        if "critical_threshold" in config_data:
            self.config.critical_threshold = config_data["critical_threshold"]
        
        if "default_risk_percentage" in config_data:
            self.config.default_risk_percentage = config_data["default_risk_percentage"]
        
        if "min_win_rate_threshold" in config_data:
            self.config.min_win_rate_threshold = config_data["min_win_rate_threshold"]
        
        if "recovery_mode" in config_data:
            mode_str = config_data["recovery_mode"]
            try:
                self.config.recovery_mode = RecoveryMode[mode_str]
            except KeyError:
                logger.warning(f"Unknown recovery mode '{mode_str}', defaulting to BALANCED")
                self.config.recovery_mode = RecoveryMode.BALANCED
        
        if "drawdown_calculation_window" in config_data:
            self.config.drawdown_calculation_window = config_data["drawdown_calculation_window"]
        
        if "recovery_reset_threshold" in config_data:
            self.config.recovery_reset_threshold = config_data["recovery_reset_threshold"]
        
        if "custom_recovery_params" in config_data:
            self.config.custom_recovery_params = config_data["custom_recovery_params"]
        
        logger.info("Drawdown protection configuration imported")
    
    def generate_drawdown_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate a comprehensive report about drawdown history and recovery.
        
        Args:
            days: Number of days to include in the report
            
        Returns:
            Dict containing report data
        """
        # Determine the date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter history to the date range
        filtered_equity = [entry for entry in self.equity_history if entry["timestamp"] >= start_date]
        filtered_drawdown = [entry for entry in self.drawdown_history if entry["timestamp"] >= start_date]
        filtered_state_changes = [entry for entry in self.state_change_history if entry["timestamp"] >= start_date]
        
        # Calculate key metrics
        max_drawdown = max([entry["drawdown"] for entry in filtered_drawdown], default=0)
        min_equity = min([entry["equity"] for entry in filtered_equity], default=0)
        max_equity = max([entry["equity"] for entry in filtered_equity], default=0)
        
        # Create time series for charting
        equity_series = [
            {"timestamp": entry["timestamp"].isoformat(), "value": entry["equity"]}
            for entry in filtered_equity
        ]
        
        drawdown_series = [
            {"timestamp": entry["timestamp"].isoformat(), "value": entry["drawdown"]}
            for entry in filtered_drawdown
        ]
        
        # Calculate recovery statistics
        recovery_periods = []
        current_recovery = None
        
        for entry in filtered_state_changes:
            if entry["to_state"] == DrawdownState.RECOVERY.name:
                # Start of a recovery period
                current_recovery = {
                    "start_time": entry["timestamp"],
                    "start_equity": entry["equity"],
                    "start_drawdown": entry["drawdown"]
                }
            elif current_recovery and entry["from_state"] == DrawdownState.RECOVERY.name:
                # End of a recovery period
                current_recovery["end_time"] = entry["timestamp"]
                current_recovery["end_equity"] = entry["equity"]
                current_recovery["duration_days"] = (
                    current_recovery["end_time"] - current_recovery["start_time"]
                ).total_seconds() / 86400
                current_recovery["equity_change"] = current_recovery["end_equity"] - current_recovery["start_equity"]
                current_recovery["equity_change_pct"] = (
                    current_recovery["equity_change"] / current_recovery["start_equity"]
                    if current_recovery["start_equity"] > 0 else 0
                )
                
                recovery_periods.append(current_recovery)
                current_recovery = None
        
        # Include current recovery period if active
        if self.in_recovery_mode:
            current_recovery = {
                "start_time": self.recovery_start_time,
                "start_equity": self.recovery_start_equity,
                "current_equity": self.current_equity,
                "target_equity": self.recovery_target_equity,
                "duration_days": (datetime.now() - self.recovery_start_time).total_seconds() / 86400,
                "equity_change": self.current_equity - self.recovery_start_equity,
                "equity_change_pct": (
                    (self.current_equity - self.recovery_start_equity) / self.recovery_start_equity
                    if self.recovery_start_equity > 0 else 0
                ),
                "progress": self.recovery_progress,
                "recovery_mode": self.config.recovery_mode.name,
                "is_active": True
            }
            recovery_periods.append(current_recovery)
        
        # Compile the report
        report = {
            "report_date": datetime.now().isoformat(),
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "days_covered": days,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": max_drawdown,
            "min_equity": min_equity,
            "max_equity": max_equity,
            "current_equity": self.current_equity,
            "current_state": self.drawdown_state.name,
            "in_recovery": self.in_recovery_mode,
            "recovery_progress": self.recovery_progress if self.in_recovery_mode else None,
            "recovery_periods": recovery_periods,
            "state_changes": [
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "from_state": entry["from_state"],
                    "to_state": entry["to_state"],
                    "drawdown": entry["drawdown"],
                    "equity": entry["equity"]
                }
                for entry in filtered_state_changes
            ],
            "equity_series": equity_series,
            "drawdown_series": drawdown_series,
            "restricted_assets": list(self.restricted_assets),
            "restricted_strategies": list(self.restricted_strategies),
            "config": self.export_configuration()
        }
        
        return report


class ProgressiveDrawdownProtector(BaseDrawdownProtector, name="ProgressiveDrawdownProtector"):
    """
    Progressive drawdown protection that adjusts risk parameters based on drawdown levels.
    
    This protector implements a tiered approach to drawdown protection, progressively
    reducing risk as drawdown increases, and implementing recovery strategies
    when drawdown exceeds certain thresholds.
    """
    
    def __init__(self,
                 max_drawdown: float = 0.25,
                 recovery_threshold: float = 0.15,
                 risk_reduction_factor: float = 0.5,
                 min_risk_percentage: float = 0.25,
                 cooldown_days: int = 5,
                 **kwargs):
        """
        Initialize the progressive drawdown protector.
        
        Args:
            max_drawdown: Maximum allowed drawdown before halting trading (default: 25%)
            recovery_threshold: Drawdown threshold to activate recovery mode (default: 15%)
            risk_reduction_factor: Factor to reduce risk by in recovery mode (default: 0.5)
            min_risk_percentage: Minimum risk percentage allowed (default: 0.25%)
            cooldown_days: Days to maintain reduced risk after recovery (default: 5)
        """
        self.max_drawdown = max_drawdown
        self.recovery_threshold = recovery_threshold
        self.risk_reduction_factor = risk_reduction_factor
        self.min_risk_percentage = min_risk_percentage
        self.cooldown_days = cooldown_days
        
        self.current_drawdown = 0.0
        self.in_recovery_mode = False
        self.recovery_start_date = None
        self.risk_adjustment = 1.0
        
        self.logger = get_logger("risk_manager.drawdown_protection")
        self.metrics = MetricsCollector("drawdown_protection")
        self.logger.info(f"Progressive drawdown protector initialized with max drawdown {max_drawdown:.1%}")
    
    async def apply_protection(self,
                              current_drawdown: float,
                              account_value: float,
                              base_risk_percentage: float = DEFAULT_RISK_PERCENTAGE,
                              strategy_performance: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply drawdown protection based on current drawdown level.
        
        Args:
            current_drawdown: Current drawdown as a decimal (e.g., 0.1 for 10%)
            account_value: Current account value
            base_risk_percentage: Base risk percentage to adjust
            strategy_performance: Optional performance metrics for strategies
            
        Returns:
            Dict containing adjusted risk parameters
        """
        self.current_drawdown = current_drawdown
        self.metrics.set("current_drawdown", current_drawdown * 100)
        
        # Check if drawdown exceeds maximum allowed
        if current_drawdown >= self.max_drawdown:
            self.logger.warning(f"Maximum drawdown exceeded: {current_drawdown:.2%} >= {self.max_drawdown:.2%}")
            self.metrics.increment("max_drawdown_exceeded")
            return {
                "halt_trading": True,
                "reason": f"Maximum drawdown exceeded: {current_drawdown:.2%}",
                "risk_percentage": 0.0,
                "position_size_multiplier": 0.0
            }
        
        # Check if we should enter recovery mode
        now = datetime.now()
        if current_drawdown >= self.recovery_threshold and not self.in_recovery_mode:
            self.in_recovery_mode = True
            self.recovery_start_date = now
            self.logger.info(f"Entering drawdown recovery mode at {current_drawdown:.2%} drawdown")
            self.metrics.increment("recovery_mode_activated")
        
        # Calculate risk adjustment based on drawdown level
        if self.in_recovery_mode:
            # Progressive risk reduction based on drawdown severity
            severity_factor = min(current_drawdown / self.max_drawdown, 1.0)
            self.risk_adjustment = max(1.0 - (severity_factor * self.risk_reduction_factor),
                                      self.min_risk_percentage)
            
            # Check if we should exit recovery mode
            if (current_drawdown < self.recovery_threshold / 2 and
                self.recovery_start_date and
                (now - self.recovery_start_date).days >= self.cooldown_days):
                self.in_recovery_mode = False
                self.recovery_start_date = None
                self.logger.info(f"Exiting drawdown recovery mode, drawdown reduced to {current_drawdown:.2%}")
                self.metrics.increment("recovery_mode_deactivated")
                self.risk_adjustment = 1.0
        else:
            # Normal operation - minor risk adjustment based on drawdown
            self.risk_adjustment = 1.0 - (current_drawdown / 2)
        
        # Calculate adjusted risk percentage
        adjusted_risk = base_risk_percentage * self.risk_adjustment
        
        return {
            "halt_trading": False,
            "risk_percentage": adjusted_risk,
            "position_size_multiplier": self.risk_adjustment,
            "in_recovery_mode": self.in_recovery_mode,
            "drawdown_level": current_drawdown
        }


def get_drawdown_protector(name: str, *args, **kwargs) -> BaseDrawdownProtector:
    """Instantiate a registered drawdown protector by name."""
    cls = BaseDrawdownProtector.registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown drawdown protector: {name}")
    return cls(*args, **kwargs)

__all__ = ["BaseDrawdownProtector", "ProgressiveDrawdownProtector", "get_drawdown_protector"]
