#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Brain Council - Performance Tracker

This module implements a comprehensive performance tracking system for strategy brains
and councils. It monitors strategy performance, calculates key metrics, tracks improvement
over time, and maintains historical records for strategy evaluation and weighting adjustments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import time
from datetime import datetime, timedelta
import asyncio
import json
from collections import defaultdict, deque
import math
from enum import Enum
import copy

from common.logger import get_logger
from common.constants import (
    TIMEFRAMES, MARKET_REGIMES, ASSETS, PLATFORMS,
    STRATEGY_CATEGORIES, PERFORMANCE_METRICS
)
from common.metrics import (
    performance_metrics, sharpe_ratio, sortino_ratio, calmar_ratio, win_rate, 
    profit_factor, drawdown, expectancy
)
from common.exceptions import PerformanceTrackerError, InvalidStrategyError
from common.db_client import get_db_client
from common.utils import time_weighted_average

logger = get_logger("brain_council.performance_tracker")


class SignalOutcome(Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    UNKNOWN = "unknown"


class PerformanceTracker:
    """
    Comprehensive performance tracking system for strategy brains and councils.
    """
    
    def __init__(
        self,
        council_name: str,
        config: dict = None,
        db_connector=None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """Initialize and schedule performance tracking setup.

        Args:
            council_name: Name of the brain council this tracker belongs to
            config: Configuration for the performance tracker
            db_connector: Optional database connector for performance data
        """
        self.council_name = council_name
        self.config = config or {}
        self._initialize_config()
        
        # Event loop for asynchronous initialization
        try:
            self.loop = loop or asyncio.get_running_loop()
        except RuntimeError:
            self.loop = loop or asyncio.get_event_loop()

        # Database client initialized asynchronously
        self.db = None
        
        # Performance cache for fast lookups
        self.strategy_cache = {}
        self.council_cache = {}
        
        # Real-time tracking
        self.active_signals = {}  # Currently active trading signals
        self.recent_outcomes = defaultdict(lambda: deque(maxlen=self.config['max_recent_outcomes']))
        
        # Performance windows for different timeframes
        self.timeframe_windows = {
            'hourly': deque(maxlen=24),  # Last 24 hours
            'daily': deque(maxlen=30),   # Last 30 days
            'weekly': deque(maxlen=12),  # Last 12 weeks
            'monthly': deque(maxlen=12)  # Last 12 months
        }
        
        # Asset and platform-specific tracking
        self.asset_performance = defaultdict(lambda: defaultdict(dict))
        self.platform_performance = defaultdict(lambda: defaultdict(dict))
        
        # Regime performance
        self.regime_performance = defaultdict(lambda: defaultdict(dict))

        # Schedule database initialization
        self.initialization_task = self.loop.create_task(
            self.initialize(db_connector)
        )

    async def initialize(self, db_connector=None) -> None:
        """Obtain a database client and create required tables.

        This coroutine is scheduled automatically during object creation, but it
        can be awaited if explicit synchronization is required.
        """
        self.db = db_connector or await get_db_client()
        await self._initialize_database()
        logger.info(
            f"Initialized PerformanceTracker for council: {self.council_name}"
        )
    
    def _initialize_config(self):
        """Initialize configuration with defaults if not provided."""
        self.config.setdefault('max_recent_outcomes', 100)
        self.config.setdefault('min_trades_for_metrics', 10)
        self.config.setdefault('breakeven_threshold', 0.0001)  # 0.01%
        self.config.setdefault('performance_update_interval', 3600)  # seconds
        self.config.setdefault('auto_cleanup_interval', 86400 * 7)  # 7 days
        self.config.setdefault('max_signal_age', 86400 * 30)  # 30 days
        self.config.setdefault('store_complete_history', True)
        self.config.setdefault('metrics_to_track', {
            'win_rate': True,
            'profit_factor': True,
            'sharpe_ratio': True,
            'sortino_ratio': True,
            'calmar_ratio': True,
            'expectancy': True,
            'max_drawdown': True,
            'avg_win_size': True,
            'avg_loss_size': True,
            'avg_trade_size': True,
            'total_profit': True,
            'trade_count': True
        })
        
    async def _initialize_database(self) -> None:
        """Create database tables if they don't exist."""
        if not self.db:
            logger.warning("No database connector available")
            return

        try:
            # Create tables for performance tracking
            await self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_signals (
                    signal_id TEXT PRIMARY KEY,
                    strategy_id TEXT,
                    council_id TEXT,
                    asset TEXT,
                    platform TEXT,
                    timeframe TEXT,
                    market_regime TEXT,
                    entry_time REAL,
                    entry_price REAL,
                    position_size REAL,
                    direction TEXT,
                    exit_time REAL,
                    exit_price REAL,
                    profit_loss REAL,
                    profit_loss_pct REAL,
                    outcome TEXT,
                    metadata TEXT
                )
                """
            )

            await self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy_id TEXT,
                    council_id TEXT,
                    timestamp REAL,
                    timeframe TEXT,
                    asset TEXT,
                    platform TEXT,
                    market_regime TEXT,
                    metrics TEXT,
                    PRIMARY KEY (strategy_id, timestamp, timeframe)
                )
                """
            )

            await self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS council_performance (
                    council_id TEXT,
                    timestamp REAL,
                    timeframe TEXT,
                    asset TEXT,
                    platform TEXT,
                    market_regime TEXT,
                    metrics TEXT,
                    PRIMARY KEY (council_id, timestamp, timeframe)
                )
                """
            )

            # Indices for performance query optimization
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategy_signals_strategy_id "
                "ON strategy_signals(strategy_id)"
            )
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategy_signals_council_id "
                "ON strategy_signals(council_id)"
            )
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategy_signals_asset "
                "ON strategy_signals(asset)"
            )
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategy_signals_platform "
                "ON strategy_signals(platform)"
            )
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategy_signals_entry_time "
                "ON strategy_signals(entry_time)"
            )
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy_id "
                "ON strategy_performance(strategy_id)"
            )
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_council_performance_council_id "
                "ON council_performance(council_id)"
            )

            # Commit if supported
            if hasattr(self.db, "commit"):
                await self.db.commit()
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    async def register_signal(self, 
                           signal_id: str,
                           strategy_id: str,
                           asset: str,
                           platform: str,
                           timeframe: str,
                           market_regime: str,
                           entry_time: float,
                           entry_price: float,
                           position_size: float,
                           direction: str,
                           metadata: Dict[str, Any] = None) -> bool:
        """
        Register a new trading signal for performance tracking.
        
        Args:
            signal_id: Unique identifier for the signal
            strategy_id: Identifier of the strategy brain that generated the signal
            asset: Asset being traded
            platform: Trading platform
            timeframe: Timeframe of the signal
            market_regime: Current market regime
            entry_time: Timestamp of entry
            entry_price: Price at entry
            position_size: Size of the position
            direction: 'long' or 'short'
            metadata: Additional signal metadata
            
        Returns:
            True if successfully registered, False otherwise
        """
        if signal_id in self.active_signals:
            logger.warning(f"Signal {signal_id} already registered")
            return False
            
        # Create signal record
        signal = {
            'signal_id': signal_id,
            'strategy_id': strategy_id,
            'council_id': self.council_name,
            'asset': asset,
            'platform': platform,
            'timeframe': timeframe,
            'market_regime': market_regime,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'position_size': position_size,
            'direction': direction,
            'exit_time': None,
            'exit_price': None,
            'profit_loss': None,
            'profit_loss_pct': None,
            'outcome': SignalOutcome.UNKNOWN.value,
            'metadata': json.dumps(metadata or {})
        }
        
        # Store in active signals
        self.active_signals[signal_id] = signal
        
        # If configured to store complete history, add to database
        if self.config['store_complete_history'] and self.db:
            try:
                await self._store_signal(signal)
                logger.debug(f"Registered signal {signal_id} for strategy {strategy_id}")
                return True
            except Exception as e:
                logger.error(f"Error storing signal {signal_id}: {str(e)}")
                return False
        else:
            return True
            
    async def update_signal(self,
                          signal_id: str,
                          exit_time: float,
                          exit_price: float,
                          market_regime: str = None,
                          metadata_updates: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update a signal with exit information and calculate performance metrics.
        
        Args:
            signal_id: Unique identifier for the signal
            exit_time: Timestamp of exit
            exit_price: Price at exit
            market_regime: Updated market regime if changed
            metadata_updates: Updates to signal metadata
            
        Returns:
            Dictionary with signal outcome and performance metrics
        """
        if signal_id not in self.active_signals:
            logger.warning(f"Signal {signal_id} not found in active signals")
            return {}
            
        signal = self.active_signals[signal_id]
        
        # Update exit information
        signal['exit_time'] = exit_time
        signal['exit_price'] = exit_price
        
        # Update market regime if provided
        if market_regime:
            signal['market_regime'] = market_regime
            
        # Update metadata if provided
        if metadata_updates:
            existing_metadata = json.loads(signal['metadata'])
            existing_metadata.update(metadata_updates)
            signal['metadata'] = json.dumps(existing_metadata)
            
        # Calculate profit/loss
        direction = signal['direction']
        entry_price = signal['entry_price']
        position_size = signal['position_size']
        
        if direction == 'long':
            price_change = exit_price - entry_price
        else:  # short
            price_change = entry_price - exit_price
            
        # Calculate profit/loss in absolute terms
        profit_loss = price_change * position_size
        
        # Calculate percentage profit/loss
        if entry_price > 0:
            profit_loss_pct = price_change / entry_price
        else:
            profit_loss_pct = 0.0
            
        # Determine outcome
        breakeven_threshold = self.config['breakeven_threshold']
        if abs(profit_loss_pct) <= breakeven_threshold:
            outcome = SignalOutcome.BREAKEVEN.value
        elif profit_loss_pct > 0:
            outcome = SignalOutcome.WIN.value
        else:
            outcome = SignalOutcome.LOSS.value
            
        # Update signal with performance metrics
        signal['profit_loss'] = profit_loss
        signal['profit_loss_pct'] = profit_loss_pct
        signal['outcome'] = outcome
        
        # Store in recent outcomes
        strategy_id = signal['strategy_id']
        self.recent_outcomes[strategy_id].append(signal)
        
        # Update timeframe windows
        self._update_timeframe_windows(signal)
        
        # Update asset and platform performance
        self._update_asset_platform_performance(signal)
        
        # Update regime performance
        self._update_regime_performance(signal)
        
        # Store updated signal in database
        if self.db:
            try:
                await self._update_stored_signal(signal)
            except Exception as e:
                logger.error(f"Error updating signal {signal_id}: {str(e)}")
                
        # Remove from active signals
        del self.active_signals[signal_id]
        
        # Trigger performance metrics update if needed
        now = time.time()
        last_update = self.strategy_cache.get(strategy_id, {}).get('last_update', 0)
        
        if now - last_update > self.config['performance_update_interval']:
            await self.update_strategy_metrics(strategy_id)
            
        return {
            'signal_id': signal_id,
            'strategy_id': strategy_id,
            'outcome': outcome,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct
        }
        
    async def cancel_signal(self, signal_id: str, reason: str = None) -> bool:
        """
        Cancel a signal without execution.
        
        Args:
            signal_id: Unique identifier for the signal
            reason: Reason for cancellation
            
        Returns:
            True if successfully cancelled, False otherwise
        """
        if signal_id not in self.active_signals:
            logger.warning(f"Signal {signal_id} not found in active signals")
            return False
            
        # Remove signal without counting it in performance metrics
        signal = self.active_signals[signal_id]
        
        # If configured to store complete history, update database
        if self.config['store_complete_history'] and self.db:
            try:
                # Mark as cancelled in metadata
                metadata = json.loads(signal['metadata'])
                metadata['cancelled'] = True
                metadata['cancellation_reason'] = reason
                signal['metadata'] = json.dumps(metadata)
                
                # Update in database
                await self._update_stored_signal(signal)
            except Exception as e:
                logger.error(f"Error cancelling signal {signal_id}: {str(e)}")
                
        # Remove from active signals
        del self.active_signals[signal_id]
        return True
        
    async def update_strategy_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dictionary of updated performance metrics
        """
        # Get recent signals for the strategy
        signals = list(self.recent_outcomes[strategy_id])
        
        # Check if we have enough data
        if len(signals) < self.config['min_trades_for_metrics']:
            logger.debug(f"Not enough trades for strategy {strategy_id} to calculate metrics")
            return {}
            
        # Calculate performance metrics
        metrics = await self._calculate_metrics(signals)
        
        # Store metrics in cache
        self.strategy_cache[strategy_id] = {
            'metrics': metrics,
            'last_update': time.time(),
            'trade_count': len(signals)
        }
        
        # Store in database if available
        if self.db:
            try:
                await self._store_strategy_metrics(strategy_id, metrics)
            except Exception as e:
                logger.error(f"Error storing metrics for strategy {strategy_id}: {str(e)}")
                
        return metrics
        
    async def update_council_metrics(self) -> Dict[str, Any]:
        """
        Update performance metrics for the entire council.
        
        Returns:
            Dictionary of updated performance metrics
        """
        # Get all signals for the council
        try:
            if self.db:
                signals = await self._get_council_signals()
            else:
                # Fallback to cached signals if no DB
                signals = []
                for strategy_outcomes in self.recent_outcomes.values():
                    signals.extend(list(strategy_outcomes))
        except Exception as e:
            logger.error(f"Error retrieving council signals: {str(e)}")
            return {}
            
        # Check if we have enough data
        if len(signals) < self.config['min_trades_for_metrics']:
            logger.debug(f"Not enough trades for council {self.council_name} to calculate metrics")
            return {}
            
        # Calculate performance metrics
        metrics = await self._calculate_metrics(signals)
        
        # Store metrics in cache
        self.council_cache = {
            'metrics': metrics,
            'last_update': time.time(),
            'trade_count': len(signals)
        }
        
        # Store in database if available
        if self.db:
            try:
                await self._store_council_metrics(metrics)
            except Exception as e:
                logger.error(f"Error storing metrics for council {self.council_name}: {str(e)}")
                
        return metrics
        
    async def get_strategy_performance(self, 
                                     strategy_id: str,
                                     timeframe: str = 'daily',
                                     lookback_periods: int = 10,
                                     asset: str = None,
                                     platform: str = None,
                                     market_regime: str = None) -> Dict[str, Any]:
        """
        Get performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            timeframe: Time period for metrics (hourly, daily, weekly, monthly)
            lookback_periods: Number of periods to look back
            asset: Filter by specific asset
            platform: Filter by specific platform
            market_regime: Filter by specific market regime
            
        Returns:
            Dictionary of performance metrics
        """
        # Check if we have fresh cached metrics
        cached = self.strategy_cache.get(strategy_id, {})
        now = time.time()
        cache_age = now - cached.get('last_update', 0)
        
        if cache_age < self.config['performance_update_interval'] and not (asset or platform or market_regime):
            # Return cached metrics if they're fresh and no filters applied
            return cached.get('metrics', {})
            
        # Otherwise, get from database
        if self.db:
            try:
                metrics = await self._get_strategy_metrics(
                    strategy_id=strategy_id,
                    timeframe=timeframe,
                    lookback_periods=lookback_periods,
                    asset=asset,
                    platform=platform,
                    market_regime=market_regime
                )
                return metrics
            except Exception as e:
                logger.error(f"Error retrieving metrics for strategy {strategy_id}: {str(e)}")
                
        # Fallback to cached metrics
        return cached.get('metrics', {})
        
    async def get_council_performance(self,
                                    timeframe: str = 'daily',
                                    lookback_periods: int = 10,
                                    asset: str = None,
                                    platform: str = None,
                                    market_regime: str = None) -> Dict[str, Any]:
        """
        Get performance metrics for the entire council.
        
        Args:
            timeframe: Time period for metrics (hourly, daily, weekly, monthly)
            lookback_periods: Number of periods to look back
            asset: Filter by specific asset
            platform: Filter by specific platform
            market_regime: Filter by specific market regime
            
        Returns:
            Dictionary of performance metrics
        """
        # Check if we have fresh cached metrics
        now = time.time()
        cache_age = now - self.council_cache.get('last_update', 0)
        
        if cache_age < self.config['performance_update_interval'] and not (asset or platform or market_regime):
            # Return cached metrics if they're fresh and no filters applied
            return self.council_cache.get('metrics', {})
            
        # Otherwise, get from database
        if self.db:
            try:
                metrics = await self._get_council_metrics(
                    timeframe=timeframe,
                    lookback_periods=lookback_periods,
                    asset=asset,
                    platform=platform,
                    market_regime=market_regime
                )
                return metrics
            except Exception as e:
                logger.error(f"Error retrieving metrics for council {self.council_name}: {str(e)}")
                
        # Fallback to cached metrics
        return self.council_cache.get('metrics', {})
        
    async def get_strategy_performance_timeline(self,
                                             strategy_id: str,
                                             timeframe: str = 'daily',
                                             start_time: float = None,
                                             end_time: float = None,
                                             metric: str = 'win_rate') -> List[Tuple[float, float]]:
        """
        Get a timeline of performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            timeframe: Time period for metrics (hourly, daily, weekly, monthly)
            start_time: Start timestamp (defaults to 30 days ago)
            end_time: End timestamp (defaults to now)
            metric: Specific metric to retrieve
            
        Returns:
            List of (timestamp, metric_value) tuples
        """
        if not self.db:
            logger.warning("No database available for performance timeline")
            return []
            
        # Set default time range if not provided
        if not end_time:
            end_time = time.time()
        if not start_time:
            start_time = end_time - (30 * 86400)  # 30 days
            
        try:
            timeline = await self._get_strategy_metric_timeline(
                strategy_id=strategy_id,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                metric=metric
            )
            return timeline
        except Exception as e:
            logger.error(f"Error retrieving timeline for strategy {strategy_id}: {str(e)}")
            return []
            
    async def get_asset_performance(self, 
                                  asset: str,
                                  timeframe: str = 'daily',
                                  lookback_periods: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all strategies on a specific asset.
        
        Args:
            asset: Asset identifier
            timeframe: Time period for metrics
            lookback_periods: Number of periods to look back
            
        Returns:
            Dictionary of strategy_id -> metrics
        """
        if not self.db:
            logger.warning("No database available for asset performance")
            return {}
            
        try:
            results = {}
            # Get all strategies that have traded this asset
            strategies = await self._get_strategies_for_asset(asset)
            
            for strategy_id in strategies:
                metrics = await self._get_strategy_metrics(
                    strategy_id=strategy_id,
                    timeframe=timeframe,
                    lookback_periods=lookback_periods,
                    asset=asset
                )
                if metrics:
                    results[strategy_id] = metrics
                    
            return results
        except Exception as e:
            logger.error(f"Error retrieving performance for asset {asset}: {str(e)}")
            return {}
            
    async def get_improvement_trends(self, 
                                   strategy_id: str = None, 
                                   metric: str = 'win_rate',
                                   lookback_periods: int = 5) -> Dict[str, float]:
        """
        Calculate improvement trends for strategies.
        
        Args:
            strategy_id: Strategy identifier (if None, calculate for all strategies)
            metric: Metric to track improvement
            lookback_periods: Number of periods to analyze
            
        Returns:
            Dictionary of strategy_id -> trend value (-1 to 1)
        """
        if not self.db:
            logger.warning("No database available for improvement trends")
            return {}
            
        try:
            trends = {}
            
            # Get strategies to analyze
            if strategy_id:
                strategies = [strategy_id]
            else:
                # Get all strategies in the council
                strategies = await self._get_all_strategies()
                
            # Calculate trend for each strategy
            for s_id in strategies:
                timeline = await self._get_strategy_metric_timeline(
                    strategy_id=s_id,
                    timeframe='daily',
                    start_time=time.time() - (lookback_periods * 86400),
                    end_time=time.time(),
                    metric=metric
                )
                
                if len(timeline) >= 3:  # Need enough data points
                    # Extract values
                    _, values = zip(*timeline)
                    
                    # Calculate trend (simple linear regression slope normalized to [-1, 1])
                    trend = self._calculate_trend(values)
                    trends[s_id] = trend
                    
            return trends
        except Exception as e:
            logger.error(f"Error calculating improvement trends: {str(e)}")
            return {}
            
    def get_active_signals_count(self, strategy_id: str = None) -> int:
        """
        Get the number of active signals.
        
        Args:
            strategy_id: Strategy identifier (if None, count all active signals)
            
        Returns:
            Number of active signals
        """
        if not strategy_id:
            return len(self.active_signals)
            
        # Count signals for specific strategy
        count = 0
        for signal in self.active_signals.values():
            if signal['strategy_id'] == strategy_id:
                count += 1
                
        return count
    
    async def cleanup_old_signals(self, max_age: float = None) -> int:
        """
        Clean up old signals from the database.
        
        Args:
            max_age: Maximum age of signals to keep (in seconds)
            
        Returns:
            Number of signals cleaned up
        """
        if not self.db or not self.config['store_complete_history']:
            return 0
            
        if max_age is None:
            max_age = self.config['max_signal_age']
            
        cutoff_time = time.time() - max_age
        
        try:
            # Delete old signals
            query = """
                DELETE FROM strategy_signals
                WHERE entry_time < ? AND council_id = ?
            """
            result = await self.db.execute(query, (cutoff_time, self.council_name))
            count = result.rowcount
            
            await self.db.commit()
            logger.info(f"Cleaned up {count} old signals")
            return count
        except Exception as e:
            logger.error(f"Error cleaning up old signals: {str(e)}")
            return 0
    
    def _update_timeframe_windows(self, signal: Dict[str, Any]):
        """
        Update timeframe windows with a completed signal.
        
        Args:
            signal: Completed signal data
        """
        # Get the signal's timestamp
        timestamp = signal['exit_time']
        
        # Determine which windows to update
        now = time.time()
        
        # Update hourly window
        if now - timestamp <= 3600 * 24:  # Last 24 hours
            self.timeframe_windows['hourly'].append(signal)
            
        # Update daily window
        if now - timestamp <= 86400 * 30:  # Last 30 days
            self.timeframe_windows['daily'].append(signal)
            
        # Update weekly window
        if now - timestamp <= 86400 * 7 * 12:  # Last 12 weeks
            self.timeframe_windows['weekly'].append(signal)
            
        # Update monthly window
        if now - timestamp <= 86400 * 30 * 12:  # Last 12 months
            self.timeframe_windows['monthly'].append(signal)
    
    def _update_asset_platform_performance(self, signal: Dict[str, Any]):
        """
        Update asset and platform specific performance tracking.
        
        Args:
            signal: Completed signal data
        """
        strategy_id = signal['strategy_id']
        asset = signal['asset']
        platform = signal['platform']
        outcome = signal['outcome']
        profit_loss = signal['profit_loss']
        
        # Update asset performance
        asset_perf = self.asset_performance[strategy_id][asset]
        
        asset_perf.setdefault('trade_count', 0)
        asset_perf.setdefault('win_count', 0)
        asset_perf.setdefault('loss_count', 0)
        asset_perf.setdefault('total_profit', 0.0)
        
        asset_perf['trade_count'] += 1
        asset_perf['total_profit'] += profit_loss
        
        if outcome == SignalOutcome.WIN.value:
            asset_perf['win_count'] += 1
        elif outcome == SignalOutcome.LOSS.value:
            asset_perf['loss_count'] += 1
            
        # Update platform performance
        platform_perf = self.platform_performance[strategy_id][platform]
        
        platform_perf.setdefault('trade_count', 0)
        platform_perf.setdefault('win_count', 0)
        platform_perf.setdefault('loss_count', 0)
        platform_perf.setdefault('total_profit', 0.0)
        
        platform_perf['trade_count'] += 1
        platform_perf['total_profit'] += profit_loss
        
        if outcome == SignalOutcome.WIN.value:
            platform_perf['win_count'] += 1
        elif outcome == SignalOutcome.LOSS.value:
            platform_perf['loss_count'] += 1
    
    def _update_regime_performance(self, signal: Dict[str, Any]):
        """
        Update market regime specific performance tracking.
        
        Args:
            signal: Completed signal data
        """
        strategy_id = signal['strategy_id']
        regime = signal['market_regime']
        outcome = signal['outcome']
        profit_loss = signal['profit_loss']
        
        # Update regime performance
        regime_perf = self.regime_performance[strategy_id][regime]
        
        regime_perf.setdefault('trade_count', 0)
        regime_perf.setdefault('win_count', 0)
        regime_perf.setdefault('loss_count', 0)
        regime_perf.setdefault('total_profit', 0.0)
        
        regime_perf['trade_count'] += 1
        regime_perf['total_profit'] += profit_loss
        
        if outcome == SignalOutcome.WIN.value:
            regime_perf['win_count'] += 1
        elif outcome == SignalOutcome.LOSS.value:
            regime_perf['loss_count'] += 1
    
    async def _calculate_metrics(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics from a list of signals.
        
        Args:
            signals: List of completed trading signals
            
        Returns:
            Dictionary of performance metrics
        """
        if not signals:
            return {}
            
        # Extract key data
        outcomes = [s['outcome'] for s in signals]
        profit_loss = [s['profit_loss'] for s in signals]
        profit_loss_pct = [s['profit_loss_pct'] for s in signals]
        entry_times = [s['entry_time'] for s in signals]
        exit_times = [s['exit_time'] for s in signals]
        
        # Filter win/loss outcomes (exclude breakeven)
        win_indices = [i for i, o in enumerate(outcomes) if o == SignalOutcome.WIN.value]
        loss_indices = [i for i, o in enumerate(outcomes) if o == SignalOutcome.LOSS.value]
        
        # Calculate metrics
        metrics = {}
        
        # Trade counts
        metrics['trade_count'] = len(signals)
        metrics['win_count'] = len(win_indices)
        metrics['loss_count'] = len(loss_indices)
        metrics['breakeven_count'] = metrics['trade_count'] - metrics['win_count'] - metrics['loss_count']
        
        # Win rate
        if metrics['trade_count'] > 0:
            metrics['win_rate'] = metrics['win_count'] / metrics['trade_count']
        else:
            metrics['win_rate'] = 0.0
            
        # Profit/loss
        metrics['total_profit'] = sum(profit_loss)
        metrics['total_profit_pct'] = sum(profit_loss_pct)
        
        # Average trade
        if metrics['trade_count'] > 0:
            metrics['avg_trade_size'] = metrics['total_profit'] / metrics['trade_count']
            metrics['avg_trade_pct'] = metrics['total_profit_pct'] / metrics['trade_count']
        else:
            metrics['avg_trade_size'] = 0.0
            metrics['avg_trade_pct'] = 0.0
            
        # Average win
        if metrics['win_count'] > 0:
            win_profits = [profit_loss[i] for i in win_indices]
            win_profits_pct = [profit_loss_pct[i] for i in win_indices]
            metrics['avg_win_size'] = sum(win_profits) / metrics['win_count']
            metrics['avg_win_pct'] = sum(win_profits_pct) / metrics['win_count']
        else:
            metrics['avg_win_size'] = 0.0
            metrics['avg_win_pct'] = 0.0
            
        # Average loss
        if metrics['loss_count'] > 0:
            loss_profits = [profit_loss[i] for i in loss_indices]
            loss_profits_pct = [profit_loss_pct[i] for i in loss_indices]
            metrics['avg_loss_size'] = sum(loss_profits) / metrics['loss_count']
            metrics['avg_loss_pct'] = sum(loss_profits_pct) / metrics['loss_count']
        else:
            metrics['avg_loss_size'] = 0.0
            metrics['avg_loss_pct'] = 0.0
            
        # Profit factor
        if metrics['loss_count'] > 0 and sum(profit_loss[i] for i in loss_indices) != 0:
            gross_profit = sum(profit_loss[i] for i in win_indices)
            gross_loss = abs(sum(profit_loss[i] for i in loss_indices))
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            metrics['profit_factor'] = float('inf') if metrics['win_count'] > 0 else 0.0
            
        # Expectancy
        if metrics['trade_count'] > 0:
            metrics['expectancy'] = (
                metrics['win_rate'] * metrics['avg_win_pct'] + 
                (1 - metrics['win_rate']) * metrics['avg_loss_pct']
            )
        else:
            metrics['expectancy'] = 0.0
            
        # Maximum drawdown
        if len(profit_loss) > 1:
            # Calculate cumulative equity curve
            equity = np.cumsum(profit_loss)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(equity)
            
            # Calculate drawdown
            drawdown = running_max - equity
            
            # Get maximum drawdown
            metrics['max_drawdown'] = float(np.max(drawdown))
            metrics['max_drawdown_pct'] = float(metrics['max_drawdown'] / np.max(running_max)) if np.max(running_max) > 0 else 0.0
        else:
            metrics['max_drawdown'] = 0.0
            metrics['max_drawdown_pct'] = 0.0
            
        # Risk-adjusted metrics
        if len(profit_loss_pct) > 1:
            # Calculate daily returns for Sharpe/Sortino
            daily_returns = []
            
            # Group trades by day
            day_trades = defaultdict(list)
            for i, (entry, exit_t, pnl) in enumerate(zip(entry_times, exit_times, profit_loss_pct)):
                day = int(exit_t / 86400)  # Convert to day number
                day_trades[day].append(pnl)
                
            # Calculate daily returns
            for day, trades in day_trades.items():
                daily_return = sum(trades)  # Simple sum for now
                daily_returns.append(daily_return)
                
            # Calculate Sharpe ratio (annualized)
            if len(daily_returns) > 0:
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                
                # Sharpe ratio (annualized)
                if std_return > 0:
                    metrics['sharpe_ratio'] = (mean_return / std_return) * np.sqrt(252)
                else:
                    metrics['sharpe_ratio'] = 0.0 if mean_return <= 0 else float('inf')
                    
                # Sortino ratio (annualized)
                neg_returns = [r for r in daily_returns if r < 0]
                if neg_returns:
                    downside_std = np.std(neg_returns)
                    if downside_std > 0:
                        metrics['sortino_ratio'] = (mean_return / downside_std) * np.sqrt(252)
                    else:
                        metrics['sortino_ratio'] = 0.0 if mean_return <= 0 else float('inf')
                else:
                    metrics['sortino_ratio'] = float('inf') if mean_return > 0 else 0.0
                    
                # Calmar ratio (annualized return / max drawdown)
                if metrics['max_drawdown_pct'] > 0:
                    annual_return = mean_return * 252
                    metrics['calmar_ratio'] = annual_return / metrics['max_drawdown_pct']
                else:
                    metrics['calmar_ratio'] = float('inf') if mean_return > 0 else 0.0
            else:
                metrics['sharpe_ratio'] = 0.0
                metrics['sortino_ratio'] = 0.0
                metrics['calmar_ratio'] = 0.0
                
        # Add trade frequency metrics
        if len(entry_times) > 1:
            # Average time between trades
            entry_times.sort()
            time_diffs = [entry_times[i+1] - entry_times[i] for i in range(len(entry_times)-1)]
            metrics['avg_time_between_trades'] = sum(time_diffs) / len(time_diffs)
            
            # Trades per day
            time_span = max(entry_times) - min(entry_times)
            days = time_span / 86400  # Convert to days
            metrics['trades_per_day'] = len(entry_times) / max(1, days)
        else:
            metrics['avg_time_between_trades'] = 0.0
            metrics['trades_per_day'] = 0.0
            
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend value from a list of data points.
        
        Args:
            values: List of metric values
            
        Returns:
            Trend value from -1 to 1
        """
        if len(values) < 2:
            return 0.0
            
        # X values (indices)
        x = np.arange(len(values))
        
        # Fit linear regression
        slope, _, _, _, _ = stats.linregress(x, values)
        
        # Normalize slope to [-1, 1] range
        # This is a simple heuristic - adjust based on typical slope magnitudes
        norm_slope = np.tanh(slope * 10)  # tanh squashes to [-1, 1]
        
        return float(norm_slope)
    
    async def _store_signal(self, signal: Dict[str, Any]):
        """
        Store a signal in the database.
        
        Args:
            signal: Signal data to store
        """
        if not self.db:
            return
            
        query = """
            INSERT INTO strategy_signals (
                signal_id, strategy_id, council_id, asset, platform, timeframe,
                market_regime, entry_time, entry_price, position_size, direction,
                exit_time, exit_price, profit_loss, profit_loss_pct, outcome, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            signal['signal_id'],
            signal['strategy_id'],
            signal['council_id'],
            signal['asset'],
            signal['platform'],
            signal['timeframe'],
            signal['market_regime'],
            signal['entry_time'],
            signal['entry_price'],
            signal['position_size'],
            signal['direction'],
            signal['exit_time'],
            signal['exit_price'],
            signal['profit_loss'],
            signal['profit_loss_pct'],
            signal['outcome'],
            signal['metadata']
        )
        
        await self.db.execute(query, values)
        await self.db.commit()
    
    async def _update_stored_signal(self, signal: Dict[str, Any]):
        """
        Update a signal in the database.
        
        Args:
            signal: Updated signal data
        """
        if not self.db:
            return
            
        query = """
            UPDATE strategy_signals
            SET exit_time = ?, exit_price = ?, profit_loss = ?,
                profit_loss_pct = ?, outcome = ?, metadata = ?,
                market_regime = ?
            WHERE signal_id = ?
        """
        
        values = (
            signal['exit_time'],
            signal['exit_price'],
            signal['profit_loss'],
            signal['profit_loss_pct'],
            signal['outcome'],
            signal['metadata'],
            signal['market_regime'],
            signal['signal_id']
        )
        
        await self.db.execute(query, values)
        await self.db.commit()
    
    async def _store_strategy_metrics(self, strategy_id: str, metrics: Dict[str, Any],
                                   timeframe: str = 'daily',
                                   asset: str = None,
                                   platform: str = None,
                                   market_regime: str = None):
        """
        Store strategy metrics in the database.
        
        Args:
            strategy_id: Strategy identifier
            metrics: Performance metrics
            timeframe: Time period for metrics
            asset: Asset identifier (or None for all)
            platform: Platform identifier (or None for all)
            market_regime: Market regime (or None for all)
        """
        if not self.db:
            return
            
        query = """
            INSERT INTO strategy_performance (
                strategy_id, council_id, timestamp, timeframe, asset,
                platform, market_regime, metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            strategy_id,
            self.council_name,
            time.time(),
            timeframe,
            asset or 'all',
            platform or 'all',
            market_regime or 'all',
            json.dumps(metrics)
        )
        
        await self.db.execute(query, values)
        await self.db.commit()
    
    async def _store_council_metrics(self, metrics: Dict[str, Any],
                                  timeframe: str = 'daily',
                                  asset: str = None,
                                  platform: str = None,
                                  market_regime: str = None):
        """
        Store council metrics in the database.
        
        Args:
            metrics: Performance metrics
            timeframe: Time period for metrics
            asset: Asset identifier (or None for all)
            platform: Platform identifier (or None for all)
            market_regime: Market regime (or None for all)
        """
        if not self.db:
            return
            
        query = """
            INSERT INTO council_performance (
                council_id, timestamp, timeframe, asset,
                platform, market_regime, metrics
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            self.council_name,
            time.time(),
            timeframe,
            asset or 'all',
            platform or 'all',
            market_regime or 'all',
            json.dumps(metrics)
        )
        
        await self.db.execute(query, values)
        await self.db.commit()
    
    async def _get_council_signals(self, 
                                lookback_days: int = 30,
                                asset: str = None,
                                platform: str = None,
                                market_regime: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve signals for the council from the database.
        
        Args:
            lookback_days: Number of days to look back
            asset: Filter by asset
            platform: Filter by platform
            market_regime: Filter by market regime
            
        Returns:
            List of signals
        """
        if not self.db:
            return []
            
        cutoff_time = time.time() - (lookback_days * 86400)
        
        # Build query
        query = """
            SELECT * FROM strategy_signals
            WHERE council_id = ? AND entry_time > ?
        """
        params = [self.council_name, cutoff_time]
        
        # Add filters
        if asset:
            query += " AND asset = ?"
            params.append(asset)
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        if market_regime:
            query += " AND market_regime = ?"
            params.append(market_regime)
            
        # Add order
        query += " ORDER BY entry_time DESC"
        
        # Execute query
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        
        # Convert to dictionaries
        signals = []
        columns = [column[0] for column in cursor.description]
        
        for row in rows:
            signal = dict(zip(columns, row))
            signals.append(signal)
            
        return signals
    
    async def _get_strategy_metrics(self, 
                                 strategy_id: str,
                                 timeframe: str = 'daily',
                                 lookback_periods: int = 10,
                                 asset: str = None,
                                 platform: str = None,
                                 market_regime: str = None) -> Dict[str, Any]:
        """
        Retrieve strategy metrics from the database.
        
        Args:
            strategy_id: Strategy identifier
            timeframe: Time period for metrics
            lookback_periods: Number of periods to look back
            asset: Filter by asset
            platform: Filter by platform
            market_regime: Filter by market regime
            
        Returns:
            Aggregated performance metrics
        """
        if not self.db:
            return {}
            
        # Build query
        query = """
            SELECT metrics FROM strategy_performance
            WHERE strategy_id = ? AND council_id = ? AND timeframe = ?
        """
        params = [strategy_id, self.council_name, timeframe]
        
        # Add filters
        if asset:
            query += " AND asset = ?"
            params.append(asset)
        else:
            query += " AND asset = 'all'"
            
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        else:
            query += " AND platform = 'all'"
            
        if market_regime:
            query += " AND market_regime = ?"
            params.append(market_regime)
        else:
            query += " AND market_regime = 'all'"
            
        # Add order and limit
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(lookback_periods)
        
        # Execute query
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        
        if not rows:
            return {}
            
        # Aggregate metrics from multiple periods
        all_metrics = []
        for row in rows:
            metrics_json = row[0]
            metrics = json.loads(metrics_json)
            all_metrics.append(metrics)
            
        # If only one period, return it directly
        if len(all_metrics) == 1:
            return all_metrics[0]
            
        # Otherwise, calculate weighted average of metrics
        return self._aggregate_metrics(all_metrics)
    
    async def _get_council_metrics(self,
                                timeframe: str = 'daily',
                                lookback_periods: int = 10,
                                asset: str = None,
                                platform: str = None,
                                market_regime: str = None) -> Dict[str, Any]:
        """
        Retrieve council metrics from the database.
        
        Args:
            timeframe: Time period for metrics
            lookback_periods: Number of periods to look back
            asset: Filter by asset
            platform: Filter by platform
            market_regime: Filter by market regime
            
        Returns:
            Aggregated performance metrics
        """
        if not self.db:
            return {}
            
        # Build query
        query = """
            SELECT metrics FROM council_performance
            WHERE council_id = ? AND timeframe = ?
        """
        params = [self.council_name, timeframe]
        
        # Add filters
        if asset:
            query += " AND asset = ?"
            params.append(asset)
        else:
            query += " AND asset = 'all'"
            
        if platform:
            query += " AND platform = ?"
            params.append(platform)
        else:
            query += " AND platform = 'all'"
            
        if market_regime:
            query += " AND market_regime = ?"
            params.append(market_regime)
        else:
            query += " AND market_regime = 'all'"
            
        # Add order and limit
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(lookback_periods)
        
        # Execute query
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        
        if not rows:
            return {}
            
        # Aggregate metrics from multiple periods
        all_metrics = []
        for row in rows:
            metrics_json = row[0]
            metrics = json.loads(metrics_json)
            all_metrics.append(metrics)
            
        # If only one period, return it directly
        if len(all_metrics) == 1:
            return all_metrics[0]
            
        # Otherwise, calculate weighted average of metrics
        return self._aggregate_metrics(all_metrics)
    
    async def _get_strategy_metric_timeline(self,
                                         strategy_id: str,
                                         timeframe: str = 'daily',
                                         start_time: float = None,
                                         end_time: float = None,
                                         metric: str = 'win_rate') -> List[Tuple[float, float]]:
        """
        Retrieve a timeline of a specific metric for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            timeframe: Time period for metrics
            start_time: Start timestamp
            end_time: End timestamp
            metric: Specific metric to retrieve
            
        Returns:
            List of (timestamp, metric_value) tuples
        """
        if not self.db:
            return []
            
        # Build query
        query = """
            SELECT timestamp, metrics FROM strategy_performance
            WHERE strategy_id = ? AND council_id = ? AND timeframe = ?
            AND asset = 'all' AND platform = 'all' AND market_regime = 'all'
        """
        params = [strategy_id, self.council_name, timeframe]
        
        # Add time range
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
            
        # Add order
        query += " ORDER BY timestamp ASC"
        
        # Execute query
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        
        # Extract metric values
        timeline = []
        for timestamp, metrics_json in rows:
            metrics = json.loads(metrics_json)
            if metric in metrics:
                timeline.append((timestamp, metrics[metric]))
                
        return timeline
    
    async def _get_all_strategies(self) -> List[str]:
        """
        Get all strategy IDs from the database.
        
        Returns:
            List of strategy IDs
        """
        if not self.db:
            return []
            
        query = """
            SELECT DISTINCT strategy_id FROM strategy_signals
            WHERE council_id = ?
        """
        
        cursor = await self.db.execute(query, (self.council_name,))
        rows = await cursor.fetchall()
        
        return [row[0] for row in rows]
    
    async def _get_strategies_for_asset(self, asset: str) -> List[str]:
        """
        Get all strategy IDs that have traded a specific asset.
        
        Args:
            asset: Asset identifier
            
        Returns:
            List of strategy IDs
        """
        if not self.db:
            return []
            
        query = """
            SELECT DISTINCT strategy_id FROM strategy_signals
            WHERE council_id = ? AND asset = ?
        """
        
        cursor = await self.db.execute(query, (self.council_name, asset))
        rows = await cursor.fetchall()
        
        return [row[0] for row in rows]
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple periods of metrics into a single value.
        
        Args:
            metrics_list: List of metrics dictionaries
            
        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}
            
        if len(metrics_list) == 1:
            return metrics_list[0]
            
        # For some metrics, we can use recency-weighted average
        weighted_metrics = ['win_rate', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                           'profit_factor', 'expectancy', 'avg_win_size', 'avg_loss_size',
                           'avg_trade_size', 'avg_win_pct', 'avg_loss_pct', 'avg_trade_pct']
        
        # For cumulative metrics, we can sum
        sum_metrics = ['total_profit', 'total_profit_pct', 'trade_count', 'win_count',
                     'loss_count', 'breakeven_count']
        
        # For max metrics, we take the max
        max_metrics = ['max_drawdown', 'max_drawdown_pct']
        
        result = {}
        
        # Calculate weights based on recency
        # More recent periods get higher weight
        weights = [0.6 ** i for i in range(len(metrics_list))]
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted metrics
        for metric in weighted_metrics:
            values = [m.get(metric, 0) for m in metrics_list]
            result[metric] = sum(v * w for v, w in zip(values, normalized_weights))
            
        # Sum cumulative metrics
        for metric in sum_metrics:
            result[metric] = sum(m.get(metric, 0) for m in metrics_list)
            
        # Max metrics
        for metric in max_metrics:
            result[metric] = max((m.get(metric, 0) for m in metrics_list), default=0)
            
        # Recalculate derived metrics if necessary
        if 'win_count' in result and 'trade_count' in result and result['trade_count'] > 0:
            result['win_rate'] = result['win_count'] / result['trade_count']
            
        return result

