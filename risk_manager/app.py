
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Risk Manager Service Application

This module implements the main Risk Manager service that provides comprehensive 
risk management capabilities to the QuantumSpectre Elite Trading System. It orchestrates
various risk components including position sizing, stop-loss management, take-profit strategies,
exposure monitoring, circuit breakers, and drawdown protection.

The Risk Manager is a critical component for targeting high win rates by ensuring capital
preservation while maximizing returns. It adapts risk parameters based on market conditions,
strategy performance, and account growth.
"""

import os
import sys
import time
import asyncio
import argparse
import traceback
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import json
import logging
from concurrent.futures import ThreadPoolExecutor

# Internal imports
from config import Config
from common.logger import get_logger
from common.utils import AsyncService, Signal, SignalBus
from common.constants import (
    RISK_MANAGER_UPDATE_INTERVAL,
    SERVICE_NAMES,
    LOG_LEVELS,
)
from common.exceptions import (
    RiskManagerError, PositionSizingError, StopLossError, 
    TakeProfitError, ExposureError, CircuitBreakerError, DrawdownError
)
from common.metrics import MetricsCollector
from common.db_client import DatabaseClient, get_db_client
from common.redis_client import RedisClient

# Risk management component imports
import risk_manager
from risk_manager.position_sizing import BasePositionSizer, get_position_sizer
from risk_manager.stop_loss import BaseStopLossStrategy, get_stop_loss_strategy
from risk_manager.take_profit import BaseTakeProfitStrategy, get_take_profit_strategy
from risk_manager.exposure import BaseExposureManager, get_exposure_manager
from risk_manager.circuit_breaker import BaseCircuitBreaker, get_circuit_breaker
from risk_manager.drawdown_protection import BaseDrawdownProtector, get_drawdown_protector


class RiskManagerService(AsyncService):
    """
    The main Risk Manager service that orchestrates all risk management components
    and provides a unified interface for risk assessment and management.
    
    This service is responsible for:
    1. Position sizing calculations
    2. Stop-loss level determination
    3. Take-profit level optimization
    4. Exposure management across multiple assets and positions
    5. Circuit breaker triggering during extreme market conditions
    6. Drawdown protection and recovery strategies
    
    The service adapts its risk parameters based on:
    - Current market volatility and regime
    - Account size and growth phase
    - Strategy performance history
    - Trading signal confidence
    - Current drawdown status
    """
    
    def __init__(self, config: Config, signal_bus: SignalBus):
        """
        Initialize the Risk Manager service with configuration and signal bus.
        
        Args:
            config: System configuration
            signal_bus: Signal bus for inter-service communication
        """
        super().__init__(SERVICE_NAMES["risk_manager"], config, signal_bus)
        self.logger = get_logger(f"{SERVICE_NAMES['risk_manager']}.service")
        self.logger.info("Initializing Risk Manager Service")
        
        # Initialize database and Redis clients
        self.db_client = None
        self._db_params = config.get('database', {}) if isinstance(config, dict) else {}
        self.redis_client = RedisClient(config)
        
        # Initialize metrics collector
        self.metrics = MetricsCollector("risk_manager", config)
        
        # Initialize component configuration
        self._load_configuration()
        
        # Initialize risk component instances
        self._initialize_components()
        
        # Service state
        self.active = False
        self.last_update_time = time.time()
        self.update_interval = config.get('risk_manager.update_interval', RISK_MANAGER_UPDATE_INTERVAL)
        
        # Risk state tracking
        self.current_exposure = {}
        self.global_risk_level = "normal"  # normal, elevated, high, extreme
        self.circuit_breaker_active = False
        self.current_drawdown = 0.0
        self.account_peak_value = 0.0
        self.risk_budget_remaining = 1.0  # Normalized risk budget (1.0 = 100%)
        
        # Performance tracking for adaptive risk
        self.win_streak = 0
        self.loss_streak = 0
        self.recent_trades = []
        self.recent_win_rate = 0.0
        
        # Asset-specific risk profiles
        self.asset_risk_profiles = {}
        
        # Register signal handlers
        self._register_signal_handlers()

        self.logger.info("Risk Manager Service initialized successfully")

    async def initialize(self, db_connector: Optional[DatabaseClient] = None) -> None:
        """Obtain a database client and create tables."""
        if db_connector is not None:
            self.db_client = db_connector
        if self.db_client is None:
            try:
                self.db_client = await get_db_client(**self._db_params)
                if getattr(self.db_client, "pool", None) is None:
                    await self.db_client.initialize()
                    await self.db_client.create_tables()
            except Exception as e:
                self.logger.warning(f"Failed to initialize database client: {e}. Using in-memory storage.")
                self.db_client = None
    
    def _load_configuration(self) -> None:
        """Load risk management configuration from config files."""
        self.logger.info("Loading risk management configuration")
        
        # Load component configurations
        self.position_sizing_config = self.config.get('risk_manager.position_sizing', {})
        self.stop_loss_config = self.config.get('risk_manager.stop_loss', {})
        self.take_profit_config = self.config.get('risk_manager.take_profit', {})
        self.exposure_config = self.config.get('risk_manager.exposure', {})
        self.circuit_breaker_config = self.config.get('risk_manager.circuit_breaker', {})
        self.drawdown_config = self.config.get('risk_manager.drawdown_protection', {})
        
        # Load risk profiles for different assets/markets
        self.risk_profiles = self.config.get('risk_manager.risk_profiles', {})
        
        # Load platform-specific risk settings
        self.binance_risk_config = self.config.get('risk_manager.platforms.binance', {})
        self.deriv_risk_config = self.config.get('risk_manager.platforms.deriv', {})
        
        # Load global risk parameters
        self.max_risk_per_trade = self.config.get('risk_manager.max_risk_per_trade', 0.02)  # 2% default
        self.max_portfolio_risk = self.config.get('risk_manager.max_portfolio_risk', 0.06)  # 6% default
        self.max_correlated_risk = self.config.get('risk_manager.max_correlated_risk', 0.08)  # 8% default
        self.max_drawdown_threshold = self.config.get('risk_manager.max_drawdown_threshold', 0.15)  # 15% default
        
        # Adaptive risk parameters
        self.enable_adaptive_risk = self.config.get('risk_manager.enable_adaptive_risk', True)
        self.adaptive_risk_factors = self.config.get('risk_manager.adaptive_risk_factors', {
            'win_streak_boost': 0.2,  # Increase risk by up to 20% during win streaks
            'loss_streak_reduction': 0.5,  # Reduce risk by 50% during loss streaks
            'volatility_scaling': True,  # Scale risk based on volatility
            'confidence_scaling': True,  # Scale risk based on signal confidence
        })
        
        self.logger.info("Risk management configuration loaded successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all risk management component instances."""
        self.logger.info("Initializing risk management components")
        
        # Initialize position sizers for different strategies/assets
        self.position_sizers = {}
        for strategy_type, sizer_config in self.position_sizing_config.items():
            sizer_class = sizer_config.get('class', 'KellyPositionSizer')
            self.position_sizers[strategy_type] = get_position_sizer(
                sizer_class, 
                **sizer_config.get('params', {})
            )
        
        # Initialize stop-loss strategies
        self.stop_loss_strategies = {}
        for strategy_type, sl_config in self.stop_loss_config.items():
            sl_class = sl_config.get('class', 'ATRStopLoss')
            self.stop_loss_strategies[strategy_type] = get_stop_loss_strategy(
                sl_class,
                **sl_config.get('params', {})
            )
        
        # Initialize take-profit strategies
        self.take_profit_strategies = {}
        for strategy_type, tp_config in self.take_profit_config.items():
            tp_class = tp_config.get('class', 'RiskRewardTakeProfit')
            self.take_profit_strategies[strategy_type] = get_take_profit_strategy(
                tp_class,
                **tp_config.get('params', {})
            )
        
        # Initialize exposure manager
        exposure_class = self.exposure_config.get('class', 'ExposureManager')
        self.exposure_manager = get_exposure_manager(
            exposure_class,
            **self.exposure_config.get('params', {})
        )
        
        # Initialize circuit breaker
        cb_class = self.circuit_breaker_config.get('class', 'VolatilityCircuitBreaker')
        self.circuit_breaker = get_circuit_breaker(
            cb_class,
            **self.circuit_breaker_config.get('params', {})
        )
        
        # Initialize drawdown protector
        dd_class = self.drawdown_config.get('class', 'ProgressiveDrawdownProtector')
        self.drawdown_protector = get_drawdown_protector(
            dd_class,
            **self.drawdown_config.get('params', {})
        )
        
        self.logger.info("Risk management components initialized successfully")
    
    def _register_signal_handlers(self) -> None:
        """Register handlers for relevant signals from the signal bus."""
        self.logger.info("Registering signal handlers")
        
        # Register handlers for account updates
        self.signal_bus.register(
            Signal.ACCOUNT_BALANCE_UPDATED,
            self._handle_account_update
        )
        
        # Register handlers for market data updates
        self.signal_bus.register(
            Signal.MARKET_DATA_UPDATED,
            self._handle_market_data_update
        )
        
        # Register handlers for trade events
        self.signal_bus.register(
            Signal.TRADE_EXECUTED,
            self._handle_trade_executed
        )
        self.signal_bus.register(
            Signal.TRADE_CLOSED,
            self._handle_trade_closed
        )
        
        # Register handlers for position management
        self.signal_bus.register(
            Signal.POSITION_SIZE_REQUESTED,
            self._handle_position_size_request
        )
        self.signal_bus.register(
            Signal.STOP_LOSS_REQUESTED,
            self._handle_stop_loss_request
        )
        self.signal_bus.register(
            Signal.TAKE_PROFIT_REQUESTED,
            self._handle_take_profit_request
        )
        
        # Register handlers for risk assessment
        self.signal_bus.register(
            Signal.RISK_ASSESSMENT_REQUESTED,
            self._handle_risk_assessment_request
        )
        
        # Register handlers for system events
        self.signal_bus.register(
            Signal.MARKET_REGIME_CHANGED,
            self._handle_market_regime_change
        )
        self.signal_bus.register(
            Signal.VOLATILITY_SPIKE_DETECTED,
            self._handle_volatility_spike
        )
        
        self.logger.info("Signal handlers registered successfully")
    
    async def start(self) -> None:
        """Start the Risk Manager service."""
        self.logger.info("Starting Risk Manager Service")

        try:
            await self.initialize()
            # Initialize the service state
            self.active = True
            
            # Load initial account data
            await self._load_initial_account_data()
            
            # Start the main service loop
            asyncio.create_task(self._service_loop())
            
            # Signal that service is started
            self.signal_bus.emit(
                Signal.SERVICE_STARTED,
                {
                    'service': SERVICE_NAMES["risk_manager"],
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            self.logger.info("Risk Manager Service started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start Risk Manager Service: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def stop(self) -> None:
        """Stop the Risk Manager service."""
        self.logger.info("Stopping Risk Manager Service")
        
        try:
            # Update service state
            self.active = False
            
            # Persist any remaining state if needed
            await self._persist_risk_state()
            
            # Signal that service is stopped
            self.signal_bus.emit(
                Signal.SERVICE_STOPPED,
                {
                    'service': SERVICE_NAMES["risk_manager"],
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            self.logger.info("Risk Manager Service stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping Risk Manager Service: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    async def _service_loop(self) -> None:
        """Main service loop for periodic risk assessment and updates."""
        self.logger.info("Starting Risk Manager service loop")
        
        # Make sure self.active is set to True
        self.active = True
        
        # Keep running until explicitly stopped
        while True:
            # Check if service should still be active
            if not self.active:
                self.logger.info("Risk Manager service loop stopping (active=False)")
                break
                
            try:
                # Update time tracking
                current_time = time.time()
                elapsed = current_time - self.last_update_time
                
                if elapsed >= self.update_interval:
                    # Perform periodic risk assessment
                    if hasattr(self, '_perform_periodic_risk_assessment'):
                        await self._perform_periodic_risk_assessment()
                    else:
                        self.logger.debug("Skipping periodic risk assessment - method not available")
                    
                    # Update stored state
                    await self._persist_risk_state()
                    
                    # Update metrics
                    if hasattr(self, '_update_metrics'):
                        self._update_metrics()
                    
                    # Update last run time
                    self.last_update_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error in Risk Manager service loop: {e}")
                self.logger.error(traceback.format_exc())
                
                # Report error but continue running
                if hasattr(self, 'metrics'):
                    self.metrics.increment('risk_manager_service_loop_errors')
            
            # Sleep for a short interval to prevent CPU hogging
            # This is outside the try-except block to ensure the loop continues
            try:
                await asyncio.sleep(5.0)  # Longer sleep to reduce CPU usage
            except asyncio.CancelledError:
                self.logger.info("Risk Manager service loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error during sleep in service loop: {e}")
                # Use a shorter sleep if there was an error
                await asyncio.sleep(1.0)
                
        self.logger.info("Risk Manager service loop exited")
    
    async def _load_initial_account_data(self) -> None:
        """Load initial account data to establish risk baselines."""
        self.logger.info("Loading initial account data")
        
        try:
            if self.db_client is None:
                self.logger.warning("Database client is None, using default values for account data")
                self.account_peak_value = 10000.0  # Default value
                self.current_drawdown = 0.0
                self.current_exposure = {}
                self.win_streak = 0
                self.loss_streak = 0
                self.recent_win_rate = 0.5  # Default 50% win rate
                return
                
            # Fetch account balance history
            account_history = await self.db_client.fetch_account_history(
                days=30,  # Look back 30 days for establishing baseline
                platforms=['binance', 'deriv']
            )
            
            if account_history:
                # Find peak account value for drawdown calculation
                peak_value = max([entry['balance'] for entry in account_history])
                self.account_peak_value = peak_value
                
                # Set current account value
                latest_entry = max(account_history, key=lambda x: x['timestamp'])
                current_value = latest_entry['balance']
                
                # Calculate current drawdown
                if peak_value > 0:
                    self.current_drawdown = 1.0 - (current_value / peak_value)
                    self.logger.info(f"Current drawdown: {self.current_drawdown:.2%}")
            
            # Fetch open positions to calculate current exposure
            open_positions = await self.db_client.fetch_open_positions(
                platforms=['binance', 'deriv']
            )
            
            # Calculate current exposure
            self.current_exposure = self.exposure_manager.calculate_current_exposure(open_positions)
            
            # Fetch recent trade history for win/loss streaks
            recent_trades = await self.db_client.fetch_recent_trades(
                limit=50,
                platforms=['binance', 'deriv']
            )
            
            # Update trading performance statistics
            self._update_performance_stats(recent_trades)
            
            # Load asset-specific risk profiles
            for asset, profile in self.risk_profiles.items():
                self.asset_risk_profiles[asset] = profile
            
            self.logger.info("Initial account data loaded successfully")
        
        except Exception as e:
            self.logger.error(f"Error loading initial account data: {e}")
            self.logger.error(traceback.format_exc())
            
            # Set defaults for safe operation even if data loading fails
            self.account_peak_value = 0.0
            self.current_drawdown = 0.0
            self.current_exposure = {}
            self.risk_budget_remaining = 0.8  # Conservative default if data loading fails
    
    async def _perform_periodic_risk_assessment(self) -> None:
        """Perform periodic risk assessment for global risk management."""
        self.logger.info("Performing periodic risk assessment")
        
        try:
            # Check global market conditions
            market_data = await self._fetch_latest_market_data()
            
            # Check if circuit breaker should be activated
            circuit_breaker_triggered = False
            if hasattr(self, 'circuit_breaker') and hasattr(self.circuit_breaker, 'check_conditions'):
                try:
                    circuit_breaker_triggered = self.circuit_breaker.check_conditions(
                        market_data,
                        self.current_exposure
                    )
                except Exception as e:
                    self.logger.error(f"Error checking circuit breaker conditions: {e}")
            
            if circuit_breaker_triggered and not self.circuit_breaker_active:
                self.circuit_breaker_active = True
                self.logger.warning("CIRCUIT BREAKER ACTIVATED - Suspending new trades")
                
                # Emit circuit breaker signal if signal_bus is available
                if hasattr(self, 'signal_bus') and hasattr(self.signal_bus, 'emit'):
                    reason = "Unknown"
                    duration = "Unknown"
                    
                    if hasattr(self.circuit_breaker, 'get_trigger_reason'):
                        reason = self.circuit_breaker.get_trigger_reason()
                    
                    if hasattr(self.circuit_breaker, 'get_cooldown_period'):
                        duration = self.circuit_breaker.get_cooldown_period()
                    
                    self.signal_bus.emit(
                        Signal.CIRCUIT_BREAKER_ACTIVATED,
                        {
                            'timestamp': datetime.now().isoformat(),
                            'reason': reason,
                            'expected_duration': duration
                        }
                    )
            elif self.circuit_breaker_active:
                # Check if circuit breaker can be deactivated
                can_deactivate = False
                if hasattr(self.circuit_breaker, 'can_deactivate'):
                    try:
                        can_deactivate = self.circuit_breaker.can_deactivate(market_data)
                    except Exception as e:
                        self.logger.error(f"Error checking if circuit breaker can be deactivated: {e}")
                
                if can_deactivate:
                    self.circuit_breaker_active = False
                    self.logger.info("CIRCUIT BREAKER DEACTIVATED - Resuming normal operations")
                    
                    # Emit circuit breaker deactivated signal
                    if hasattr(self, 'signal_bus') and hasattr(self.signal_bus, 'emit'):
                        self.signal_bus.emit(
                            Signal.CIRCUIT_BREAKER_DEACTIVATED,
                            {
                                'timestamp': datetime.now().isoformat()
                            }
                        )
            
            # Update global risk level based on market conditions and account state
            if hasattr(self, '_update_global_risk_level'):
                try:
                    self._update_global_risk_level(market_data)
                except Exception as e:
                    self.logger.error(f"Error updating global risk level: {e}")
            
            # Update risk budget based on drawdown state
            if hasattr(self, '_update_risk_budget'):
                try:
                    await self._update_risk_budget()
                except Exception as e:
                    self.logger.error(f"Error updating risk budget: {e}")
            
            # Adjust position sizes for open positions if necessary
            if hasattr(self, '_adjust_open_positions'):
                try:
                    await self._adjust_open_positions()
                except Exception as e:
                    self.logger.error(f"Error adjusting open positions: {e}")
            
            self.logger.info(f"Risk assessment completed - Global risk level: {self.global_risk_level}")
            
        except Exception as e:
            self.logger.error(f"Error performing risk assessment: {e}")
            self.logger.error(traceback.format_exc())
            self.metrics.increment('risk_assessment_errors')
    
    async def _fetch_latest_market_data(self) -> Dict[str, Any]:
        """Fetch latest market data for risk assessment."""
        try:
            # Get list of assets we're interested in
            assets = list(self.current_exposure.keys()) if hasattr(self, 'current_exposure') else []
            
            # Add any assets from risk profiles that aren't already included
            if hasattr(self, 'asset_risk_profiles'):
                for asset in self.asset_risk_profiles:
                    if asset not in assets:
                        assets.append(asset)
            
            # Fetch market data from Redis (faster) or database
            market_data = {}
            
            # If no assets to check, return empty data
            if not assets:
                self.logger.debug("No assets to fetch market data for")
                return market_data
                
            for asset in assets:
                asset_data = None
                
                # Try to get from Redis first if available
                if hasattr(self, 'redis_client') and self.redis_client is not None:
                    try:
                        asset_data = await self.redis_client.get_market_data(asset)
                    except Exception as redis_err:
                        self.logger.debug(f"Error getting market data from Redis for {asset}: {redis_err}")
                
                # Fall back to database if Redis failed or returned no data
                if not asset_data and hasattr(self, 'db_client') and self.db_client is not None:
                    try:
                        asset_data = await self.db_client.fetch_latest_market_data(asset)
                    except Exception as db_err:
                        self.logger.debug(f"Error getting market data from database for {asset}: {db_err}")
                
                # If we got data from either source, add it to the result
                if asset_data:
                    market_data[asset] = asset_data
            
            # Get global market indicators if Redis is available
            if hasattr(self, 'redis_client') and self.redis_client is not None:
                try:
                    global_indicators = await self.redis_client.get_global_market_indicators()
                    if global_indicators:
                        market_data['global'] = global_indicators
                except Exception as e:
                    self.logger.debug(f"Error getting global market indicators: {e}")
            
            return market_data
        
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            self.logger.error(traceback.format_exc())
            return {}
    
    def _update_global_risk_level(self, market_data: Dict[str, Any]) -> None:
        """Update the global risk level based on market conditions and account state."""
        # Start with a normal risk level
        risk_level = "normal"
        
        # Check if we're in a drawdown
        if self.current_drawdown > 0.1:
            risk_level = "elevated"
        
        if self.current_drawdown > 0.2:
            risk_level = "high"
        
        # Check volatility conditions in market data
        if 'global' in market_data:
            global_data = market_data['global']
            
            # Check for elevated volatility
            if global_data.get('volatility_percentile', 0) > 80:
                risk_level = max(risk_level, "elevated")
            
            # Check for high volatility
            if global_data.get('volatility_percentile', 0) > 90:
                risk_level = max(risk_level, "high")
            
            # Check for extreme market conditions
            if global_data.get('volatility_percentile', 0) > 95:
                risk_level = "extreme"
        
        # Check current exposure relative to maximum allowed
        total_exposure = sum(self.current_exposure.values())
        if total_exposure > self.max_portfolio_risk * 0.8:
            risk_level = max(risk_level, "elevated")
        
        if total_exposure > self.max_portfolio_risk:
            risk_level = max(risk_level, "high")
        
        # Check performance streak implications
        if self.loss_streak >= 3:
            risk_level = max(risk_level, "elevated")
        
        if self.loss_streak >= 5:
            risk_level = max(risk_level, "high")
        
        # Update global risk level if it has changed
        if risk_level != self.global_risk_level:
            self.logger.info(f"Global risk level changed: {self.global_risk_level} -> {risk_level}")
            self.global_risk_level = risk_level
            
            # Emit risk level changed signal
            self.signal_bus.emit(
                Signal.RISK_LEVEL_CHANGED,
                {
                    'timestamp': datetime.now().isoformat(),
                    'previous_level': self.global_risk_level,
                    'new_level': risk_level,
                    'drawdown': self.current_drawdown,
                    'exposure': total_exposure
                }
            )
    
    async def _update_risk_budget(self) -> None:
        """Update the risk budget based on drawdown and performance."""
        # Base risk budget on drawdown protection rules
        self.risk_budget_remaining = self.drawdown_protector.calculate_risk_budget(
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown_threshold,
            account_peak=self.account_peak_value
        )
        
        # Adjust risk budget based on performance if adaptive risk is enabled
        if self.enable_adaptive_risk:
            # Boost risk budget during win streaks
            if self.win_streak >= 3:
                win_boost = min(self.win_streak * 0.05, self.adaptive_risk_factors['win_streak_boost'])
                self.risk_budget_remaining = min(1.0, self.risk_budget_remaining * (1 + win_boost))
            
            # Reduce risk budget during loss streaks
            if self.loss_streak >= 2:
                loss_reduction = min(self.loss_streak * 0.1, self.adaptive_risk_factors['loss_streak_reduction'])
                self.risk_budget_remaining *= (1 - loss_reduction)
        
        # Ensure risk budget stays within valid range
        self.risk_budget_remaining = max(0.1, min(1.0, self.risk_budget_remaining))
        
        self.logger.info(f"Updated risk budget: {self.risk_budget_remaining:.2%}")
    
    async def _adjust_open_positions(self) -> None:
        """Adjust open positions based on current risk assessment."""
        # Only adjust positions if risk level is elevated or higher
        if self.global_risk_level in ["elevated", "high", "extreme"]:
            try:
                # Fetch currently open positions
                open_positions = await self.db_client.fetch_open_positions(
                    platforms=['binance', 'deriv']
                )
                
                # Evaluate position adjustments
                for position in open_positions:
                    # Get current market data for this asset
                    asset = position['asset']
                    market_data = await self._fetch_latest_market_data_for_asset(asset)
                    
                    if not market_data:
                        continue
                    
                    # Determine if stop loss needs adjustment
                    new_stop_loss = None
                    strategy_type = position.get('strategy_type', 'default')
                    
                    if strategy_type in self.stop_loss_strategies:
                        stop_loss_strategy = self.stop_loss_strategies[strategy_type]
                        
                        # Check if stop loss needs tightening
                        if self.global_risk_level == "extreme":
                            # In extreme conditions, move to breakeven or better if possible
                            new_stop_loss = stop_loss_strategy.calculate_breakeven_stop(
                                position, market_data
                            )
                        elif self.global_risk_level == "high":
                            # In high risk, tighten stops substantially
                            new_stop_loss = stop_loss_strategy.calculate_tightened_stop(
                                position, market_data, tightening_factor=0.7
                            )
                        elif self.global_risk_level == "elevated":
                            # In elevated risk, tighten stops moderately
                            new_stop_loss = stop_loss_strategy.calculate_tightened_stop(
                                position, market_data, tightening_factor=0.3
                            )
                        
                        # Only update if we have a new stop and it's more conservative than current
                        if new_stop_loss is not None:
                            current_stop = position.get('stop_loss_price')
                            
                            if current_stop is None or (
                                position['direction'] == 'long' and new_stop_loss > current_stop or
                                position['direction'] == 'short' and new_stop_loss < current_stop
                            ):
                                # Emit signal to update stop loss
                                self.signal_bus.emit(
                                    Signal.ADJUST_STOP_LOSS,
                                    {
                                        'position_id': position['id'],
                                        'platform': position['platform'],
                                        'asset': position['asset'],
                                        'new_stop_loss': new_stop_loss,
                                        'reason': f"Risk level: {self.global_risk_level}"
                                    }
                                )
                                
                                self.logger.info(
                                    f"Adjusting stop loss for {position['platform']} {position['asset']} "
                                    f"from {current_stop} to {new_stop_loss} due to {self.global_risk_level} risk"
                                )
                
                # If risk level is extreme, consider reducing position sizes
                if self.global_risk_level == "extreme" and not self.circuit_breaker_active:
                    # Activate circuit breaker in extreme conditions
                    self.circuit_breaker_active = True
                    cooldown_period = self.circuit_breaker.get_cooldown_period()
                    
                    self.logger.warning(
                        f"CIRCUIT BREAKER ACTIVATED due to extreme risk level. "
                        f"Cooldown period: {cooldown_period} minutes"
                    )
                    
                    # Emit circuit breaker signal
                    self.signal_bus.emit(
                        Signal.CIRCUIT_BREAKER_ACTIVATED,
                        {
                            'timestamp': datetime.now().isoformat(),
                            'reason': f"Extreme risk level detected",
                            'expected_duration': cooldown_period
                        }
                    )
            
            except Exception as e:
                self.logger.error(f"Error adjusting open positions: {e}")
                self.logger.error(traceback.format_exc())
                self.metrics.increment('position_adjustment_errors')
    
    async def _fetch_latest_market_data_for_asset(self, asset: str) -> Dict[str, Any]:
        """Fetch latest market data for a specific asset."""
        try:
            # Try to get from Redis first (for speed)
            asset_data = await self.redis_client.get_market_data(asset)
            
            if not asset_data:
                # Fall back to database
                asset_data = await self.db_client.fetch_latest_market_data(asset)
            
            return asset_data
        
        except Exception as e:
            self.logger.error(f"Error fetching market data for {asset}: {e}")
            return None
    
    def _update_performance_stats(self, recent_trades: List[Dict[str, Any]]) -> None:
        """Update performance statistics based on recent trades."""
        if not recent_trades:
            return
        
        # Sort trades by close time
        sorted_trades = sorted(recent_trades, key=lambda x: x.get('close_time', datetime.min))
        
        # Store recent trades
        self.recent_trades = sorted_trades[-50:]  # Keep last 50 trades
        
        # Calculate win/loss streaks
        self.win_streak = 0
        self.loss_streak = 0
        
        # Start from most recent and count backward
        for trade in reversed(sorted_trades):
            profit = trade.get('realized_pnl', 0)
            
            if profit > 0:
                # Winning trade
                if self.loss_streak > 0:
                    break  # End of current streak
                self.win_streak += 1
            elif profit < 0:
                # Losing trade
                if self.win_streak > 0:
                    break  # End of current streak
                self.loss_streak += 1
            else:
                # Breakeven trade
                break
        
        # Calculate recent win rate
        if self.recent_trades:
            wins = sum(1 for trade in self.recent_trades if trade.get('realized_pnl', 0) > 0)
            self.recent_win_rate = wins / len(self.recent_trades)
        
        self.logger.info(
            f"Performance stats updated: Win streak={self.win_streak}, "
            f"Loss streak={self.loss_streak}, Recent win rate={self.recent_win_rate:.2%}"
        )
    
    async def _persist_risk_state(self) -> None:
        """Persist current risk state to database for recovery."""
        try:
            risk_state = {
                'timestamp': datetime.now().isoformat(),
                'global_risk_level': self.global_risk_level,
                'circuit_breaker_active': self.circuit_breaker_active,
                'current_drawdown': self.current_drawdown,
                'account_peak_value': self.account_peak_value,
                'risk_budget_remaining': self.risk_budget_remaining,
                'win_streak': self.win_streak,
                'loss_streak': self.loss_streak,
                'recent_win_rate': self.recent_win_rate,
                'current_exposure': self.current_exposure
            }
            
            # Store in database
            if self.db_client is not None:
                await self.db_client.store_risk_state(risk_state)
            else:
                self.logger.debug("Database client is None, skipping database persistence")
            
            # Also cache in Redis for quick access
            if hasattr(self.redis_client, 'set_risk_state'):
                await self.redis_client.set_risk_state(risk_state)
            
        except Exception as e:
            self.logger.error(f"Error persisting risk state: {e}")
            self.logger.error(traceback.format_exc())
            self.metrics.increment('risk_state_persistence_errors')
    
    def _update_metrics(self) -> None:
        """Update service metrics."""
        try:
            # Record risk metrics
            self.metrics.gauge('global_risk_level', {
                'normal': 1,
                'elevated': 2,
                'high': 3,
                'extreme': 4
            }.get(self.global_risk_level, 0))
            
            self.metrics.gauge('circuit_breaker_active', int(self.circuit_breaker_active))
            self.metrics.gauge('current_drawdown', self.current_drawdown)
            self.metrics.gauge('risk_budget_remaining', self.risk_budget_remaining)
            self.metrics.gauge('win_streak', self.win_streak)
            self.metrics.gauge('loss_streak', self.loss_streak)
            self.metrics.gauge('recent_win_rate', self.recent_win_rate)
            
            # Record exposure metrics
            total_exposure = sum(self.current_exposure.values())
            self.metrics.gauge('total_portfolio_exposure', total_exposure)
            
            for asset, exposure in self.current_exposure.items():
                self.metrics.gauge(f'asset_exposure', exposure, {'asset': asset})
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    # Signal handlers
    
    async def _handle_account_update(self, data: Dict[str, Any]) -> None:
        """Handle account balance update signals."""
        try:
            platform = data.get('platform')
            new_balance = data.get('balance')
            timestamp = data.get('timestamp')
            
            self.logger.info(f"Account update received: {platform} balance = {new_balance}")
            
            # Update account peak value if needed
            if new_balance > self.account_peak_value:
                self.account_peak_value = new_balance
                self.logger.info(f"New account peak value: {self.account_peak_value}")
            
            # Recalculate drawdown
            if self.account_peak_value > 0:
                self.current_drawdown = 1.0 - (new_balance / self.account_peak_value)
                self.logger.info(f"Updated drawdown: {self.current_drawdown:.2%}")
                
                # Check if drawdown protection should be activated
                if self.current_drawdown > self.max_drawdown_threshold:
                    protection_actions = self.drawdown_protector.get_protection_actions(
                        self.current_drawdown,
                        self.max_drawdown_threshold
                    )
                    
                    if protection_actions:
                        self.logger.warning(
                            f"DRAWDOWN PROTECTION ACTIVATED - Current drawdown: {self.current_drawdown:.2%}"
                        )
                        
                        # Emit drawdown protection signal
                        self.signal_bus.emit(
                            Signal.DRAWDOWN_PROTECTION_ACTIVATED,
                            {
                                'timestamp': datetime.now().isoformat(),
                                'current_drawdown': self.current_drawdown,
                                'threshold': self.max_drawdown_threshold,
                                'actions': protection_actions
                            }
                        )
            
            # Store updated account state
            await self._persist_risk_state()
            
        except Exception as e:
            self.logger.error(f"Error handling account update: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _handle_market_data_update(self, data: Dict[str, Any]) -> None:
        """Handle market data update signals."""
        # This is a high-frequency signal, so keep processing minimal
        try:
            # Only process if we need to check circuit breaker or volatility
            if self.circuit_breaker_active or self.global_risk_level in ["high", "extreme"]:
                asset = data.get('asset')
                
                # Check if this is an asset we're monitoring
                if asset in self.current_exposure or asset in self.asset_risk_profiles:
                    # Check circuit breaker conditions
                    if self.circuit_breaker_active:
                        if self.circuit_breaker.can_deactivate({asset: data}):
                            self.circuit_breaker_active = False
                            self.logger.info("CIRCUIT BREAKER DEACTIVATED - Market conditions improved")
                            
                            # Emit circuit breaker deactivated signal
                            self.signal_bus.emit(
                                Signal.CIRCUIT_BREAKER_DEACTIVATED,
                                {
                                    'timestamp': datetime.now().isoformat(),
                                    'asset': asset
                                }
                            )
                
        except Exception as e:
            # Log but don't spam with full traceback for high-frequency events
            self.logger.error(f"Error handling market data update: {e}")
    
    async def _handle_trade_executed(self, data: Dict[str, Any]) -> None:
        """Handle trade executed signals."""
        try:
            platform = data.get('platform')
            asset = data.get('asset')
            position_id = data.get('position_id')
            direction = data.get('direction')
            size = data.get('size')
            entry_price = data.get('entry_price')
            
            self.logger.info(
                f"Trade executed: {platform} {asset} {direction} {size} @ {entry_price}"
            )
            
            # Update exposure
            if asset in self.current_exposure:
                # Update existing exposure
                if direction == 'long':
                    self.current_exposure[asset] += size * entry_price
                else:
                    self.current_exposure[asset] -= size * entry_price
            else:
                # Add new exposure
                if direction == 'long':
                    self.current_exposure[asset] = size * entry_price
                else:
                    self.current_exposure[asset] = -size * entry_price
            
            # Log updated exposure
            self.logger.info(f"Updated exposure for {asset}: {self.current_exposure[asset]}")
            
            # Update stored state
            await self._persist_risk_state()
            
        except Exception as e:
            self.logger.error(f"Error handling trade executed: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _handle_trade_closed(self, data: Dict[str, Any]) -> None:
        """Handle trade closed signals."""
        try:
            platform = data.get('platform')
            asset = data.get('asset')
            position_id = data.get('position_id')
            realized_pnl = data.get('realized_pnl')
            close_reason = data.get('close_reason')
            
            self.logger.info(
                f"Trade closed: {platform} {asset} - P&L: {realized_pnl} - Reason: {close_reason}"
            )
            
            # Update exposure
            if asset in self.current_exposure:
                self.current_exposure[asset] = 0
                self.logger.info(f"Cleared exposure for {asset}")
            
            # Update performance tracking
            # Add to recent trades list
            trade_data = {
                'platform': platform,
                'asset': asset,
                'realized_pnl': realized_pnl,
                'close_time': datetime.now(),
                'close_reason': close_reason
            }
            
            self.recent_trades.append(trade_data)
            if len(self.recent_trades) > 50:
                self.recent_trades.pop(0)  # Keep only the most recent 50 trades
            
            # Update streak counting
            if realized_pnl > 0:
                # Winning trade
                self.win_streak += 1
                self.loss_streak = 0
                self.logger.info(f"Win streak increased to {self.win_streak}")
            elif realized_pnl < 0:
                # Losing trade
                self.loss_streak += 1
                self.win_streak = 0
                self.logger.info(f"Loss streak increased to {self.loss_streak}")
            
            # Update win rate
            wins = sum(1 for trade in self.recent_trades if trade.get('realized_pnl', 0) > 0)
            self.recent_win_rate = wins / len(self.recent_trades)
            self.logger.info(f"Updated win rate: {self.recent_win_rate:.2%}")
            
            # Update stored state
            await self._persist_risk_state()
            
        except Exception as e:
            self.logger.error(f"Error handling trade closed: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _handle_position_size_request(self, data: Dict[str, Any]) -> None:
        """Handle position size request signals."""
        try:
            request_id = data.get('request_id')
            platform = data.get('platform')
            asset = data.get('asset')
            direction = data.get('direction')
            strategy_type = data.get('strategy_type', 'default')
            signal_strength = data.get('signal_strength', 0.5)
            
            self.logger.info(
                f"Position size request: {platform} {asset} {direction} - "
                f"Strategy: {strategy_type}, Signal strength: {signal_strength:.2f}"
            )
            
            # Get account balance
            account_balance = await self._get_account_balance(platform)
            
            # Determine risk per trade based on global state
            risk_per_trade = self.max_risk_per_trade * self.risk_budget_remaining
            
            # Adjust for global risk level
            if self.global_risk_level == "elevated":
                risk_per_trade *= 0.7
            elif self.global_risk_level == "high":
                risk_per_trade *= 0.5
            elif self.global_risk_level == "extreme":
                risk_per_trade *= 0.25
            
            # Adjust for circuit breaker
            if self.circuit_breaker_active:
                self.logger.warning(f"Position size request while circuit breaker active - denying trade")
                
                # Emit response with zero size
                self.signal_bus.emit(
                    Signal.POSITION_SIZE_RESPONSE,
                    {
                        'request_id': request_id,
                        'position_size': 0,
                        'reason': "Circuit breaker active",
                        'risk_used': 0
                    }
                )
                return
            
            # Get market data for the asset
            market_data = await self._fetch_latest_market_data_for_asset(asset)
            
            if not market_data:
                self.logger.error(f"Cannot determine position size - no market data for {asset}")
                
                # Emit response with zero size
                self.signal_bus.emit(
                    Signal.POSITION_SIZE_RESPONSE,
                    {
                        'request_id': request_id,
                        'position_size': 0,
                        'reason': "No market data available",
                        'risk_used': 0
                    }
                )
                return
            
            # Check correlation-aware exposure
            exposure_check_result = self.exposure_manager.check_additional_exposure(
                asset=asset,
                direction=direction,
                current_exposure=self.current_exposure,
                max_portfolio_risk=self.max_portfolio_risk,
                max_correlated_risk=self.max_correlated_risk
            )
            
            if not exposure_check_result['allowed']:
                self.logger.warning(
                    f"Position size request for {asset} denied due to exposure limits: "
                    f"{exposure_check_result['reason']}"
                )
                
                # Emit response with zero size
                self.signal_bus.emit(
                    Signal.POSITION_SIZE_RESPONSE,
                    {
                        'request_id': request_id,
                        'position_size': 0,
                        'reason': exposure_check_result['reason'],
                        'risk_used': 0
                    }
                )
                return
            
            # Get position sizer for this strategy type
            position_sizer = self.position_sizers.get(
                strategy_type, 
                self.position_sizers.get('default')
            )
            
            if not position_sizer:
                self.logger.error(f"No position sizer available for strategy type: {strategy_type}")
                
                # Use a conservative default (fixed percentage)
                from risk_manager.position_sizing import FixedPercentagePositionSizer
                position_sizer = FixedPercentagePositionSizer(percentage=0.01)
            
            # Calculate position size
            position_size_result = position_sizer.calculate_position_size(
                account_balance=account_balance,
                risk_per_trade=risk_per_trade,
                asset=asset,
                direction=direction,
                market_data=market_data,
                signal_strength=signal_strength
            )
            
            # Log the position size calculation
            self.logger.info(
                f"Position size calculation for {asset}: {position_size_result['position_size']} "
                f"(Risk: {position_size_result['risk_amount']}, "
                f"Stop distance: {position_size_result.get('stop_distance', 'N/A')})"
            )
            
            # Emit position size response
            self.signal_bus.emit(
                Signal.POSITION_SIZE_RESPONSE,
                {
                    'request_id': request_id,
                    'position_size': position_size_result['position_size'],
                    'risk_amount': position_size_result['risk_amount'],
                    'risk_percentage': position_size_result['risk_percentage'],
                    'stop_distance': position_size_result.get('stop_distance'),
                    'potential_profit': position_size_result.get('potential_profit')
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling position size request: {e}")
            self.logger.error(traceback.format_exc())
            
            # Emit error response
            self.signal_bus.emit(
                Signal.POSITION_SIZE_RESPONSE,
                {
                    'request_id': data.get('request_id'),
                    'position_size': 0,
                    'reason': f"Error: {str(e)}",
                    'risk_used': 0
                }
            )
    
    async def _handle_stop_loss_request(self, data: Dict[str, Any]) -> None:
        """Handle stop loss calculation request signals."""
        try:
            request_id = data.get('request_id')
            platform = data.get('platform')
            asset = data.get('asset')
            direction = data.get('direction')
            entry_price = data.get('entry_price')
            strategy_type = data.get('strategy_type', 'default')
            
            self.logger.info(
                f"Stop loss request: {platform} {asset} {direction} @ {entry_price} - "
                f"Strategy: {strategy_type}"
            )
            
            # Get market data for the asset
            market_data = await self._fetch_latest_market_data_for_asset(asset)
            
            if not market_data:
                self.logger.error(f"Cannot determine stop loss - no market data for {asset}")
                
                # Emit error response
                self.signal_bus.emit(
                    Signal.STOP_LOSS_RESPONSE,
                    {
                        'request_id': request_id,
                        'stop_loss_price': None,
                        'reason': "No market data available"
                    }
                )
                return
            
            # Get stop loss strategy for this strategy type
            stop_loss_strategy = self.stop_loss_strategies.get(
                strategy_type, 
                self.stop_loss_strategies.get('default')
            )
            
            if not stop_loss_strategy:
                self.logger.error(f"No stop loss strategy available for strategy type: {strategy_type}")
                
                # Use a conservative default (percentage-based)
                from risk_manager.stop_loss import PercentageStopLoss
                stop_loss_strategy = PercentageStopLoss(percentage=0.02)
            
            # Create position data structure for stop loss calculation
            position_data = {
                'platform': platform,
                'asset': asset,
                'direction': direction,
                'entry_price': entry_price
            }
            
            # Calculate stop loss
            stop_loss_result = stop_loss_strategy.calculate_stop_loss(
                position_data, market_data
            )
            
            # Log stop loss calculation
            self.logger.info(
                f"Stop loss calculation for {asset}: {stop_loss_result['stop_loss_price']} "
                f"(Method: {stop_loss_result.get('method', 'unknown')})"
            )
            
            # Emit stop loss response
            self.signal_bus.emit(
                Signal.STOP_LOSS_RESPONSE,
                {
                    'request_id': request_id,
                    'stop_loss_price': stop_loss_result['stop_loss_price'],
                    'method': stop_loss_result.get('method', 'unknown'),
                    'confidence': stop_loss_result.get('confidence', 1.0)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling stop loss request: {e}")
            self.logger.error(traceback.format_exc())
            
            # Emit error response
            self.signal_bus.emit(
                Signal.STOP_LOSS_RESPONSE,
                {
                    'request_id': data.get('request_id'),
                    'stop_loss_price': None,
                    'reason': f"Error: {str(e)}"
                }
            )
    
    async def _handle_take_profit_request(self, data: Dict[str, Any]) -> None:
        """Handle take profit calculation request signals."""
        try:
            request_id = data.get('request_id')
            platform = data.get('platform')
            asset = data.get('asset')
            direction = data.get('direction')
            entry_price = data.get('entry_price')
            stop_loss_price = data.get('stop_loss_price')
            strategy_type = data.get('strategy_type', 'default')
            
            self.logger.info(
                f"Take profit request: {platform} {asset} {direction} @ {entry_price} - "
                f"Stop loss: {stop_loss_price}, Strategy: {strategy_type}"
            )
            
            # Get market data for the asset
            market_data = await self._fetch_latest_market_data_for_asset(asset)
            
            if not market_data:
                self.logger.error(f"Cannot determine take profit - no market data for {asset}")
                
                # Emit error response
                self.signal_bus.emit(
                    Signal.TAKE_PROFIT_RESPONSE,
                    {
                        'request_id': request_id,
                        'take_profit_price': None,
                        'reason': "No market data available"
                    }
                )
                return
            
            # Get take profit strategy for this strategy type
            take_profit_strategy = self.take_profit_strategies.get(
                strategy_type, 
                self.take_profit_strategies.get('default')
            )
            
            if not take_profit_strategy:
                self.logger.error(f"No take profit strategy available for strategy type: {strategy_type}")
                
                # Use a conservative default (fixed R:R)
                from risk_manager.take_profit import RiskRewardTakeProfit
                take_profit_strategy = RiskRewardTakeProfit(risk_reward_ratio=2.0)
            
            # Create position data structure for take profit calculation
            position_data = {
                'platform': platform,
                'asset': asset,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price
            }
            
            # Calculate take profit
            take_profit_result = take_profit_strategy.calculate_take_profit(
                position_data, market_data
            )
            
            # Log take profit calculation
            self.logger.info(
                f"Take profit calculation for {asset}: {take_profit_result['take_profit_price']} "
                f"(Method: {take_profit_result.get('method', 'unknown')}, "
                f"R:R: {take_profit_result.get('risk_reward_ratio', 'N/A')})"
            )
            
            # Emit take profit response
            self.signal_bus.emit(
                Signal.TAKE_PROFIT_RESPONSE,
                {
                    'request_id': request_id,
                    'take_profit_price': take_profit_result['take_profit_price'],
                    'method': take_profit_result.get('method', 'unknown'),
                    'risk_reward_ratio': take_profit_result.get('risk_reward_ratio'),
                    'confidence': take_profit_result.get('confidence', 1.0)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling take profit request: {e}")
            self.logger.error(traceback.format_exc())
            
            # Emit error response
            self.signal_bus.emit(
                Signal.TAKE_PROFIT_RESPONSE,
                {
                    'request_id': data.get('request_id'),
                    'take_profit_price': None,
                    'reason': f"Error: {str(e)}"
                }
            )
    
    async def _handle_risk_assessment_request(self, data: Dict[str, Any]) -> None:
        """Handle risk assessment request signals."""
        try:
            request_id = data.get('request_id')
            platform = data.get('platform')
            asset = data.get('asset')
            direction = data.get('direction', None)
            strategy_type = data.get('strategy_type', 'default')
            
            self.logger.info(
                f"Risk assessment request: {platform} {asset} - "
                f"Direction: {direction}, Strategy: {strategy_type}"
            )
            
            # Prepare risk assessment response
            risk_assessment = {
                'request_id': request_id,
                'timestamp': datetime.now().isoformat(),
                'global_risk_level': self.global_risk_level,
                'circuit_breaker_active': self.circuit_breaker_active,
                'current_drawdown': self.current_drawdown,
                'risk_budget_remaining': self.risk_budget_remaining,
                'asset_exposure': self.current_exposure.get(asset, 0),
                'total_exposure': sum(self.current_exposure.values()),
                'win_streak': self.win_streak,
                'loss_streak': self.loss_streak,
                'recent_win_rate': self.recent_win_rate,
                'trading_allowed': not self.circuit_breaker_active
            }
            
            # Add asset-specific risk assessment if direction provided
            if direction:
                # Get market data for the asset
                market_data = await self._fetch_latest_market_data_for_asset(asset)
                
                if market_data:
                    # Check correlation-aware exposure
                    exposure_check = self.exposure_manager.check_additional_exposure(
                        asset=asset,
                        direction=direction,
                        current_exposure=self.current_exposure,
                        max_portfolio_risk=self.max_portfolio_risk,
                        max_correlated_risk=self.max_correlated_risk
                    )
                    
                    risk_assessment['exposure_check'] = {
                        'allowed': exposure_check['allowed'],
                        'reason': exposure_check.get('reason', ''),
                        'risk_contribution': exposure_check.get('risk_contribution', 0)
                    }
                    
                    # Add volatility assessment
                    volatility = market_data.get('volatility', {})
                    risk_assessment['volatility_assessment'] = {
                        'current_volatility': volatility.get('current', 0),
                        'historical_percentile': volatility.get('percentile', 0),
                        'trend': volatility.get('trend', 'stable')
                    }
                    
                    # Add market regime information if available
                    regime = market_data.get('regime', {})
                    if regime:
                        risk_assessment['market_regime'] = {
                            'current_regime': regime.get('current', 'unknown'),
                            'regime_strength': regime.get('strength', 0),
                            'regime_duration': regime.get('duration', 0)
                        }
            
            # Emit risk assessment response
            self.signal_bus.emit(
                Signal.RISK_ASSESSMENT_RESPONSE,
                risk_assessment
            )
            
        except Exception as e:
            self.logger.error(f"Error handling risk assessment request: {e}")
            self.logger.error(traceback.format_exc())
            
            # Emit error response
            self.signal_bus.emit(
                Signal.RISK_ASSESSMENT_RESPONSE,
                {
                    'request_id': data.get('request_id'),
                    'error': str(e),
                    'trading_allowed': False
                }
            )
    
    async def _handle_market_regime_change(self, data: Dict[str, Any]) -> None:
        """Handle market regime change signals."""
        try:
            asset = data.get('asset')
            previous_regime = data.get('previous_regime')
            new_regime = data.get('new_regime')
            regime_strength = data.get('regime_strength', 0)
            
            self.logger.info(
                f"Market regime change for {asset}: {previous_regime} -> {new_regime} "
                f"(Strength: {regime_strength})"
            )
            
            # Update global risk level if this is a significant asset
            if asset in self.current_exposure and abs(self.current_exposure[asset]) > 0:
                if new_regime in ['bear', 'high_volatility', 'crash']:
                    # Increase risk level for negative regimes
                    if self.global_risk_level == "normal":
                        self.global_risk_level = "elevated"
                    elif self.global_risk_level == "elevated" and regime_strength > 0.7:
                        self.global_risk_level = "high"
                
                elif new_regime in ['bull', 'low_volatility', 'accumulation'] and regime_strength > 0.8:
                    # Potentially decrease risk level for positive regimes
                    # but only if very confident and not in drawdown
                    if self.global_risk_level == "elevated" and self.current_drawdown < 0.05:
                        self.global_risk_level = "normal"
            
            # Update stored state
            await self._persist_risk_state()
            
        except Exception as e:
            self.logger.error(f"Error handling market regime change: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _handle_volatility_spike(self, data: Dict[str, Any]) -> None:
        """Handle volatility spike signals."""
        try:
            asset = data.get('asset')
            spike_magnitude = data.get('magnitude')
            spike_direction = data.get('direction')
            
            self.logger.warning(
                f"Volatility spike detected for {asset}: {spike_magnitude:.2f}x "
                f"(Direction: {spike_direction})"
            )
            
            # If this is a significant spike (> 3x normal volatility)
            if spike_magnitude > 3.0:
                # Check if circuit breaker should be activated
                if asset in self.current_exposure and abs(self.current_exposure[asset]) > 0:
                    # Only activate if we have exposure to this asset
                    self.circuit_breaker_active = True
                    cooldown_period = self.circuit_breaker.get_cooldown_period()
                    
                    self.logger.warning(
                        f"CIRCUIT BREAKER ACTIVATED due to volatility spike in {asset}. "
                        f"Cooldown period: {cooldown_period} minutes"
                    )
                    
                    # Emit circuit breaker signal
                    self.signal_bus.emit(
                        Signal.CIRCUIT_BREAKER_ACTIVATED,
                        {
                            'timestamp': datetime.now().isoformat(),
                            'reason': f"Volatility spike in {asset} ({spike_magnitude:.2f}x)",
                            'expected_duration': cooldown_period
                        }
                    )
                
                # Increase global risk level
                if self.global_risk_level == "normal":
                    self.global_risk_level = "elevated"
                elif self.global_risk_level == "elevated":
                    self.global_risk_level = "high"
            
            # Update stored state
            await self._persist_risk_state()
            
        except Exception as e:
            self.logger.error(f"Error handling volatility spike: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _get_account_balance(self, platform: str) -> float:
        """Get current account balance for the specified platform."""
        try:
            # Try to get from Redis first (faster)
            balance = await self.redis_client.get_account_balance(platform)
            
            if balance is not None:
                return balance
            
            # Fall back to database
            account_data = await self.db_client.fetch_latest_account_data(platform)
            
            if account_data and 'balance' in account_data:
                return account_data['balance']
            
            # If we still don't have a balance, use a very conservative placeholder
            # This should almost never happen in practice
            self.logger.error(f"Could not determine account balance for {platform}")
            return 100.0  # Conservative placeholder
            
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            self.logger.error(traceback.format_exc())
            return 100.0  # Conservative placeholder

async def main():
    """Main entry point for running the Risk Manager service standalone."""
    parser = argparse.ArgumentParser(description='QuantumSpectre Risk Manager Service')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO', choices=LOG_LEVELS.keys(),
                       help='Logging level')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    logger = get_logger("risk_manager.main")
    logger.info("Starting Risk Manager Service")
    
    try:
        # Load configuration
        from config import load_config
        config = load_config(args.config)
        
        # Create signal bus
        from common.utils import SignalBus
        signal_bus = SignalBus()
        
        # Create and start service
        service = RiskManagerService(config, signal_bus)
        await service.start()
        
        # Keep service running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        if 'service' in locals():
            await service.stop()
    except Exception as e:
        logger.error(f"Error in Risk Manager service: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
