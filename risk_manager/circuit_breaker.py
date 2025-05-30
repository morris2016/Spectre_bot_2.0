#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Risk Manager - Circuit Breaker Module

This module implements advanced circuit breaker functionality to protect
against extreme market conditions and system anomalies. The circuit breakers
can temporarily pause trading activities when predefined risk thresholds are
exceeded, protecting capital during periods of high volatility or when
unusual market behaviors are detected.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Type

from common.constants import TIMEFRAMES, CIRCUIT_BREAKER_THRESHOLDS
from common.logger import get_logger
from common.utils import calculate_zscore, detect_outliers, exponential_backoff
from common.async_utils import create_task_with_retry
from common.metrics import MetricsCollector
from common.exceptions import CircuitBreakerTrippedException, SystemCriticalError

from data_storage.market_data import MarketDataRepository
from feature_service.features.volatility import VolatilityCalculator

logger = get_logger("risk_manager.circuit_breaker")


# Define this function at the top level to avoid circular imports
def get_circuit_breaker(name: str, *args, **kwargs):
    """Instantiate a registered circuit breaker by name."""
    from risk_manager.circuit_breaker import BaseCircuitBreaker
    cls = BaseCircuitBreaker.registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown circuit breaker: {name}")
    return cls(*args, **kwargs)


class BaseCircuitBreaker:
    """Base class for circuit breaker implementations."""

    registry: Dict[str, Type["BaseCircuitBreaker"]] = {}

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        key = name or cls.__name__
        BaseCircuitBreaker.registry[key] = cls

    async def check(self, *args, **kwargs) -> bool:
        raise NotImplementedError


class CircuitBreakerState(Enum):
    """
    Enum representing the possible states of a circuit breaker.
    
    States:
        CLOSED: Normal operation, circuit breaker is not triggered
        OPEN: Circuit breaker is triggered, operations are blocked
        HALF_OPEN: Transitional state during recovery
    """
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()
    """Enum for the possible states of a circuit breaker."""
    NORMAL = auto()         # Circuit breaker is inactive, trading permitted
    WARNING = auto()        # Approaching threshold, caution advised
    TRIPPED = auto()        # Circuit breaker is active, trading paused
    COOLING = auto()        # Cool-down period after being tripped
    RECOVERY = auto()       # Gradual recovery phase, limited trading


class CircuitBreakerType(Enum):
    """Enum for the types of circuit breakers in the system."""
    VOLATILITY = auto()     # Based on market volatility metrics
    DRAWDOWN = auto()       # Based on account drawdown
    LOSS_STREAK = auto()    # Based on consecutive losses
    LATENCY = auto()        # Based on execution latency
    PRICE_GAP = auto()      # Based on large price gaps
    LIQUIDITY = auto()      # Based on available liquidity
    SYSTEM = auto()         # Based on system performance metrics
    ANOMALY = auto()        # Based on anomaly detection
    CUSTOM = auto()         # Custom defined circuit breaker


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    type: CircuitBreakerType
    warning_threshold: float
    trip_threshold: float
    cool_down_period: int  # seconds
    recovery_period: int   # seconds
    recovery_steps: int    # number of steps for gradual recovery
    scope: str             # global, asset, asset_class, exchange, etc.
    scope_id: Optional[str] = None  # specific ID within scope if applicable
    check_interval: int = 5  # seconds between checks
    description: str = ""
    custom_condition: Optional[callable] = None  # for CUSTOM type only
    metadata: Dict[str, Any] = None


@dataclass
class CircuitBreakerStatus:
    """Current status of a circuit breaker."""
    id: str
    config: CircuitBreakerConfig
    state: CircuitBreakerState
    current_value: float
    last_checked: datetime
    last_tripped: Optional[datetime] = None
    trip_count: int = 0
    warning_count: int = 0
    cool_down_end: Optional[datetime] = None
    recovery_end: Optional[datetime] = None
    recovery_level: int = 0  # 0 = full restriction, recovery_steps = no restriction
    last_state_change: datetime = None
    additional_info: Dict[str, Any] = None


class CircuitBreaker(BaseCircuitBreaker):
    """
    Advanced circuit breaker implementation for protecting trading operations
    during extreme market conditions or system anomalies.
    """
    
    def __init__(self, market_data_repo: MarketDataRepository, metrics_collector: MetricsCollector):
        """
        Initialize the circuit breaker system.
        
        Args:
            market_data_repo: Repository for accessing market data
            metrics_collector: System for collecting metrics
        """
        self.market_data_repo = market_data_repo
        self.metrics_collector = metrics_collector
        self.volatility_calculator = VolatilityCalculator()
        self.circuit_breakers: Dict[str, CircuitBreakerStatus] = {}
        self.active_checks: Set[str] = set()
        self.cb_lock = asyncio.Lock()
        
        # Load default circuit breakers from configuration
        self._load_default_circuit_breakers()
        
        # Start background task for circuit breaker monitoring
        self.monitoring_task = None
        self.is_running = False
        logger.info("Circuit Breaker system initialized")
    
    async def start(self):
        """Start the circuit breaker monitoring system."""
        if self.is_running:
            return
            
        self.is_running = True
        self.monitoring_task = create_task_with_retry(self._monitoring_loop())
        logger.info("Circuit Breaker monitoring started")
    
    async def stop(self):
        """Stop the circuit breaker monitoring system."""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Circuit Breaker monitoring stopped")
    
    def _load_default_circuit_breakers(self):
        """Load the default set of circuit breakers from configuration."""
        # Global volatility circuit breaker
        self.register_circuit_breaker(CircuitBreakerConfig(
            type=CircuitBreakerType.VOLATILITY,
            warning_threshold=CIRCUIT_BREAKER_THRESHOLDS["volatility"]["warning"],
            trip_threshold=CIRCUIT_BREAKER_THRESHOLDS["volatility"]["trip"],
            cool_down_period=300,  # 5 minutes
            recovery_period=600,   # 10 minutes
            recovery_steps=5,
            scope="global",
            description="Global market volatility circuit breaker"
        ))
        
        # Global drawdown circuit breaker
        self.register_circuit_breaker(CircuitBreakerConfig(
            type=CircuitBreakerType.DRAWDOWN,
            warning_threshold=CIRCUIT_BREAKER_THRESHOLDS["drawdown"]["warning"],
            trip_threshold=CIRCUIT_BREAKER_THRESHOLDS["drawdown"]["trip"],
            cool_down_period=600,  # 10 minutes
            recovery_period=1800,  # 30 minutes
            recovery_steps=6,
            scope="global",
            description="Global account drawdown circuit breaker"
        ))
        
        # Loss streak circuit breaker
        self.register_circuit_breaker(CircuitBreakerConfig(
            type=CircuitBreakerType.LOSS_STREAK,
            warning_threshold=CIRCUIT_BREAKER_THRESHOLDS["loss_streak"]["warning"],
            trip_threshold=CIRCUIT_BREAKER_THRESHOLDS["loss_streak"]["trip"],
            cool_down_period=900,  # 15 minutes
            recovery_period=1800,  # 30 minutes
            recovery_steps=4,
            scope="global",
            description="Consecutive loss streak circuit breaker"
        ))
        
        # System performance circuit breaker
        self.register_circuit_breaker(CircuitBreakerConfig(
            type=CircuitBreakerType.SYSTEM,
            warning_threshold=CIRCUIT_BREAKER_THRESHOLDS["system"]["warning"],
            trip_threshold=CIRCUIT_BREAKER_THRESHOLDS["system"]["trip"],
            cool_down_period=120,  # 2 minutes
            recovery_period=300,   # 5 minutes
            recovery_steps=3,
            scope="global",
            description="System performance circuit breaker"
        ))
        
        # Latency circuit breaker
        self.register_circuit_breaker(CircuitBreakerConfig(
            type=CircuitBreakerType.LATENCY,
            warning_threshold=CIRCUIT_BREAKER_THRESHOLDS["latency"]["warning"],
            trip_threshold=CIRCUIT_BREAKER_THRESHOLDS["latency"]["trip"],
            cool_down_period=180,  # 3 minutes
            recovery_period=300,   # 5 minutes
            recovery_steps=3,
            scope="global",
            description="Execution latency circuit breaker"
        ))
        
        # Liquidity circuit breaker
        self.register_circuit_breaker(CircuitBreakerConfig(
            type=CircuitBreakerType.LIQUIDITY,
            warning_threshold=CIRCUIT_BREAKER_THRESHOLDS["liquidity"]["warning"],
            trip_threshold=CIRCUIT_BREAKER_THRESHOLDS["liquidity"]["trip"],
            cool_down_period=300,  # 5 minutes
            recovery_period=600,   # 10 minutes
            recovery_steps=4,
            scope="global",
            description="Market liquidity circuit breaker"
        ))
        
        # Anomaly detection circuit breaker
        self.register_circuit_breaker(CircuitBreakerConfig(
            type=CircuitBreakerType.ANOMALY,
            warning_threshold=CIRCUIT_BREAKER_THRESHOLDS["anomaly"]["warning"],
            trip_threshold=CIRCUIT_BREAKER_THRESHOLDS["anomaly"]["trip"],
            cool_down_period=600,  # 10 minutes
            recovery_period=1200,  # 20 minutes
            recovery_steps=5,
            scope="global",
            description="Market anomaly circuit breaker"
        ))
    
    def register_circuit_breaker(self, config: CircuitBreakerConfig) -> str:
        """
        Register a new circuit breaker with the system.
        
        Args:
            config: Configuration for the circuit breaker
            
        Returns:
            str: Unique ID of the registered circuit breaker
        """
        # Generate a unique ID for this circuit breaker
        cb_id = f"{config.type.name.lower()}_{config.scope}"
        if config.scope_id:
            cb_id += f"_{config.scope_id}"
        
        # Create initial status
        status = CircuitBreakerStatus(
            id=cb_id,
            config=config,
            state=CircuitBreakerState.NORMAL,
            current_value=0.0,
            last_checked=datetime.now(),
            last_state_change=datetime.now(),
            additional_info={}
        )
        
        # Add to managed circuit breakers
        self.circuit_breakers[cb_id] = status
        logger.info(f"Registered circuit breaker: {cb_id} - {config.description}")
        
        return cb_id
    
    def unregister_circuit_breaker(self, cb_id: str) -> bool:
        """
        Unregister a circuit breaker from the system.
        
        Args:
            cb_id: ID of the circuit breaker to unregister
            
        Returns:
            bool: True if successful, False if circuit breaker not found
        """
        if cb_id in self.circuit_breakers:
            del self.circuit_breakers[cb_id]
            logger.info(f"Unregistered circuit breaker: {cb_id}")
            return True
        return False
    
    async def _monitoring_loop(self):
        """Background task that continuously monitors all circuit breakers."""
        logger.info("Circuit breaker monitoring loop started")
        while self.is_running:
            try:
                # Gather all circuit breakers that need checking
                current_time = datetime.now()
                to_check = []
                
                async with self.cb_lock:
                    for cb_id, status in self.circuit_breakers.items():
                        if (cb_id not in self.active_checks and 
                            (current_time - status.last_checked).total_seconds() >= status.config.check_interval):
                            to_check.append(cb_id)
                
                # Check each circuit breaker
                for cb_id in to_check:
                    if cb_id not in self.active_checks:
                        self.active_checks.add(cb_id)
                        asyncio.create_task(self._check_circuit_breaker(cb_id))
                
                # Wait before next round of checks
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in circuit breaker monitoring loop: {str(e)}", exc_info=True)
                await asyncio.sleep(5)  # Wait longer on error
    
    async def _check_circuit_breaker(self, cb_id: str):
        """
        Check a specific circuit breaker and update its state as needed.
        
        Args:
            cb_id: ID of the circuit breaker to check
        """
        try:
            status = self.circuit_breakers.get(cb_id)
            if not status:
                logger.warning(f"Attempted to check non-existent circuit breaker: {cb_id}")
                return
            
            # Get the current value for this circuit breaker type
            current_value = await self._get_current_value(status.config)
            
            # Update the status with the current value and check time
            async with self.cb_lock:
                status.current_value = current_value
                status.last_checked = datetime.now()
                old_state = status.state
                
                # Process state transitions based on current state
                if status.state == CircuitBreakerState.NORMAL:
                    # Check if we should transition to WARNING or TRIPPED
                    if current_value >= status.config.trip_threshold:
                        status.state = CircuitBreakerState.TRIPPED
                        status.last_tripped = datetime.now()
                        status.trip_count += 1
                        status.cool_down_end = datetime.now() + timedelta(seconds=status.config.cool_down_period)
                        logger.warning(f"Circuit breaker {cb_id} TRIPPED: value={current_value} threshold={status.config.trip_threshold}")
                    elif current_value >= status.config.warning_threshold:
                        status.state = CircuitBreakerState.WARNING
                        status.warning_count += 1
                        logger.info(f"Circuit breaker {cb_id} WARNING: value={current_value} threshold={status.config.warning_threshold}")
                
                elif status.state == CircuitBreakerState.WARNING:
                    # Check if we should transition to NORMAL or TRIPPED
                    if current_value >= status.config.trip_threshold:
                        status.state = CircuitBreakerState.TRIPPED
                        status.last_tripped = datetime.now()
                        status.trip_count += 1
                        status.cool_down_end = datetime.now() + timedelta(seconds=status.config.cool_down_period)
                        logger.warning(f"Circuit breaker {cb_id} TRIPPED from WARNING: value={current_value} threshold={status.config.trip_threshold}")
                    elif current_value < status.config.warning_threshold:
                        status.state = CircuitBreakerState.NORMAL
                        logger.info(f"Circuit breaker {cb_id} returned to NORMAL from WARNING: value={current_value}")
                
                elif status.state == CircuitBreakerState.TRIPPED:
                    # Check if cool-down period has passed
                    if datetime.now() >= status.cool_down_end:
                        status.state = CircuitBreakerState.COOLING
                        status.recovery_level = 0
                        status.recovery_end = datetime.now() + timedelta(seconds=status.config.recovery_period)
                        logger.info(f"Circuit breaker {cb_id} entered COOLING phase: value={current_value}")
                
                elif status.state == CircuitBreakerState.COOLING:
                    # Check if we should transition to RECOVERY or back to TRIPPED
                    if current_value >= status.config.trip_threshold:
                        status.state = CircuitBreakerState.TRIPPED
                        status.last_tripped = datetime.now()
                        status.trip_count += 1
                        status.cool_down_end = datetime.now() + timedelta(seconds=status.config.cool_down_period)
                        logger.warning(f"Circuit breaker {cb_id} re-TRIPPED during COOLING: value={current_value}")
                    elif datetime.now() >= status.recovery_end:
                        status.state = CircuitBreakerState.RECOVERY
                        status.recovery_level = 1  # Start recovery at first level
                        recovery_step_time = status.config.recovery_period / status.config.recovery_steps
                        status.recovery_end = datetime.now() + timedelta(seconds=recovery_step_time)
                        logger.info(f"Circuit breaker {cb_id} entered RECOVERY phase: value={current_value}, level=1/{status.config.recovery_steps}")
                
                elif status.state == CircuitBreakerState.RECOVERY:
                    # Check if we should increase recovery level, return to NORMAL, or go back to TRIPPED
                    if current_value >= status.config.trip_threshold:
                        status.state = CircuitBreakerState.TRIPPED
                        status.last_tripped = datetime.now()
                        status.trip_count += 1
                        status.cool_down_end = datetime.now() + timedelta(seconds=status.config.cool_down_period)
                        logger.warning(f"Circuit breaker {cb_id} re-TRIPPED during RECOVERY: value={current_value}")
                    elif datetime.now() >= status.recovery_end:
                        status.recovery_level += 1
                        if status.recovery_level >= status.config.recovery_steps:
                            status.state = CircuitBreakerState.NORMAL
                            logger.info(f"Circuit breaker {cb_id} returned to NORMAL after recovery: value={current_value}")
                        else:
                            recovery_step_time = status.config.recovery_period / status.config.recovery_steps
                            status.recovery_end = datetime.now() + timedelta(seconds=recovery_step_time)
                            logger.info(f"Circuit breaker {cb_id} recovery level increased: value={current_value}, level={status.recovery_level}/{status.config.recovery_steps}")
                
                # If state changed, record the time
                if status.state != old_state:
                    status.last_state_change = datetime.now()
                    # Record metrics for state change
                    self.metrics_collector.record_gauge(
                        "circuit_breaker_state", 
                        status.state.value,
                        {"id": cb_id, "type": status.config.type.name, "scope": status.config.scope}
                    )
                
                # Always record the current value
                self.metrics_collector.record_gauge(
                    "circuit_breaker_value", 
                    status.current_value,
                    {"id": cb_id, "type": status.config.type.name, "scope": status.config.scope}
                )
        
        except Exception as e:
            logger.error(f"Error checking circuit breaker {cb_id}: {str(e)}", exc_info=True)
        
        finally:
            # Make sure we remove this from active checks
            self.active_checks.discard(cb_id)
    
    async def _get_current_value(self, config: CircuitBreakerConfig) -> float:
        """
        Get the current value for a specific circuit breaker type.
        
        Args:
            config: Circuit breaker configuration
            
        Returns:
            float: The current value to compare against thresholds
        """
        try:
            if config.type == CircuitBreakerType.VOLATILITY:
                # Get market volatility measure
                return await self._get_volatility_value(config)
                
            elif config.type == CircuitBreakerType.DRAWDOWN:
                # Get account drawdown measure
                return await self._get_drawdown_value(config)
                
            elif config.type == CircuitBreakerType.LOSS_STREAK:
                # Get consecutive loss streak count
                return await self._get_loss_streak_value(config)
                
            elif config.type == CircuitBreakerType.LATENCY:
                # Get execution latency measure
                return await self._get_latency_value(config)
                
            elif config.type == CircuitBreakerType.PRICE_GAP:
                # Get price gap measure
                return await self._get_price_gap_value(config)
                
            elif config.type == CircuitBreakerType.LIQUIDITY:
                # Get liquidity measure
                return await self._get_liquidity_value(config)
                
            elif config.type == CircuitBreakerType.SYSTEM:
                # Get system performance measure
                return await self._get_system_value(config)
                
            elif config.type == CircuitBreakerType.ANOMALY:
                # Get anomaly measure
                return await self._get_anomaly_value(config)
                
            elif config.type == CircuitBreakerType.CUSTOM:
                # Use custom condition function
                if config.custom_condition:
                    return await config.custom_condition()
                else:
                    logger.error(f"Custom circuit breaker has no condition function")
                    return 0.0
            
            logger.warning(f"Unknown circuit breaker type: {config.type}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting value for circuit breaker {config.type}: {str(e)}", exc_info=True)
            return 0.0
    
    async def _get_volatility_value(self, config: CircuitBreakerConfig) -> float:
        """Get the current volatility value for circuit breaker evaluation."""
        try:
            # For global scope, check multiple assets and take the average
            if config.scope == "global":
                assets = await self.market_data_repo.get_active_assets()
                if not assets:
                    return 0.0
                
                total_volatility = 0.0
                asset_count = 0
                
                for asset in assets[:10]:  # Limit to 10 most active assets for performance
                    # Get recent price data
                    ohlc_data = await self.market_data_repo.get_ohlc(
                        asset.symbol, 
                        TIMEFRAMES.M5, 
                        limit=60  # Last 5 hours of 5-minute data
                    )
                    
                    if len(ohlc_data) >= 20:  # Need enough data
                        # Calculate normalized volatility (as percentage of price)
                        volatility = self.volatility_calculator.calculate_realized_volatility(
                            ohlc_data, period=20, annualize=False
                        )
                        total_volatility += volatility
                        asset_count += 1
                
                if asset_count == 0:
                    return 0.0
                
                # Return average volatility across assets
                return total_volatility / asset_count
            
            # For asset-specific scope
            elif config.scope == "asset" and config.scope_id:
                ohlc_data = await self.market_data_repo.get_ohlc(
                    config.scope_id, 
                    TIMEFRAMES.M5, 
                    limit=60
                )
                
                if len(ohlc_data) >= 20:
                    return self.volatility_calculator.calculate_realized_volatility(
                        ohlc_data, period=20, annualize=False
                    )
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility value: {str(e)}", exc_info=True)
            return 0.0
    
    async def _get_drawdown_value(self, config: CircuitBreakerConfig) -> float:
        """Get the current drawdown value for circuit breaker evaluation."""
        try:
            # For global scope, get the overall account drawdown
            if config.scope == "global":
                metrics = self.metrics_collector.get_metrics("account_drawdown_percent")
                if metrics and len(metrics) > 0:
                    # Get the maximum drawdown across all accounts
                    return max([m.value for m in metrics])
                return 0.0
            
            # For exchange-specific scope
            elif config.scope == "exchange" and config.scope_id:
                metrics = self.metrics_collector.get_metrics(
                    "account_drawdown_percent", 
                    {"exchange": config.scope_id}
                )
                if metrics and len(metrics) > 0:
                    return metrics[0].value
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating drawdown value: {str(e)}", exc_info=True)
            return 0.0
    
    async def _get_loss_streak_value(self, config: CircuitBreakerConfig) -> float:
        """Get the current loss streak value for circuit breaker evaluation."""
        try:
            # For global scope, get the maximum consecutive loss count
            if config.scope == "global":
                metrics = self.metrics_collector.get_metrics("consecutive_losses")
                if metrics and len(metrics) > 0:
                    return max([m.value for m in metrics])
                return 0.0
            
            # For strategy-specific scope
            elif config.scope == "strategy" and config.scope_id:
                metrics = self.metrics_collector.get_metrics(
                    "consecutive_losses", 
                    {"strategy": config.scope_id}
                )
                if metrics and len(metrics) > 0:
                    return metrics[0].value
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating loss streak value: {str(e)}", exc_info=True)
            return 0.0
    
    async def _get_latency_value(self, config: CircuitBreakerConfig) -> float:
        """Get the current execution latency value for circuit breaker evaluation."""
        try:
            # For global scope, get the average execution latency
            if config.scope == "global":
                metrics = self.metrics_collector.get_metrics("execution_latency_ms")
                if metrics and len(metrics) > 0:
                    return np.mean([m.value for m in metrics])
                return 0.0
            
            # For exchange-specific scope
            elif config.scope == "exchange" and config.scope_id:
                metrics = self.metrics_collector.get_metrics(
                    "execution_latency_ms", 
                    {"exchange": config.scope_id}
                )
                if metrics and len(metrics) > 0:
                    return np.mean([m.value for m in metrics])
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating latency value: {str(e)}", exc_info=True)
            return 0.0
    
    async def _get_price_gap_value(self, config: CircuitBreakerConfig) -> float:
        """Get the current price gap value for circuit breaker evaluation."""
        try:
            # For asset-specific scope
            if config.scope == "asset" and config.scope_id:
                # Get recent price data
                ohlc_data = await self.market_data_repo.get_ohlc(
                    config.scope_id, 
                    TIMEFRAMES.M1, 
                    limit=60  # Last hour of 1-minute data
                )
                
                if len(ohlc_data) < 2:
                    return 0.0
                
                # Calculate maximum gap between consecutive candles
                max_gap = 0.0
                for i in range(1, len(ohlc_data)):
                    prev_close = ohlc_data[i-1].close
                    curr_open = ohlc_data[i].open
                    gap = abs(curr_open - prev_close) / prev_close * 100.0  # Gap as percentage
                    max_gap = max(max_gap, gap)
                
                return max_gap
            
            # For global scope, check multiple assets and take the maximum
            elif config.scope == "global":
                assets = await self.market_data_repo.get_active_assets()
                if not assets:
                    return 0.0
                
                max_global_gap = 0.0
                
                for asset in assets[:10]:  # Limit to 10 most active assets for performance
                    # Get recent price data
                    ohlc_data = await self.market_data_repo.get_ohlc(
                        asset.symbol, 
                        TIMEFRAMES.M1, 
                        limit=60
                    )
                    
                    if len(ohlc_data) < 2:
                        continue
                    
                    # Calculate maximum gap between consecutive candles
                    max_gap = 0.0
                    for i in range(1, len(ohlc_data)):
                        prev_close = ohlc_data[i-1].close
                        curr_open = ohlc_data[i].open
                        gap = abs(curr_open - prev_close) / prev_close * 100.0
                        max_gap = max(max_gap, gap)
                    
                    max_global_gap = max(max_global_gap, max_gap)
                
                return max_global_gap
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating price gap value: {str(e)}", exc_info=True)
            return 0.0
    
    async def _get_liquidity_value(self, config: CircuitBreakerConfig) -> float:
        """Get the current liquidity value for circuit breaker evaluation."""
        try:
            # For asset-specific scope
            if config.scope == "asset" and config.scope_id:
                # Get order book data
                order_book = await self.market_data_repo.get_order_book(config.scope_id)
                
                if not order_book or not order_book.bids or not order_book.asks:
                    return 0.0
                
                # Calculate liquidity score (higher score means lower liquidity, which is worse)
                # We use the spread and depth as measures of liquidity
                
                # Get best bid and ask
                best_bid = order_book.bids[0].price if order_book.bids else 0
                best_ask = order_book.asks[0].price if order_book.asks else 0
                
                if best_bid == 0 or best_ask == 0:
                    return 0.0
                
                # Calculate spread as percentage
                spread_pct = (best_ask - best_bid) / ((best_ask + best_bid) / 2) * 100
                
                # Calculate depth (sum of volume within 1% of best prices)
                mid_price = (best_bid + best_ask) / 2
                depth_threshold = mid_price * 0.01  # 1% of price
                
                bid_depth = sum(b.amount for b in order_book.bids 
                               if b.price >= best_bid - depth_threshold)
                ask_depth = sum(a.amount for a in order_book.asks 
                               if a.price <= best_ask + depth_threshold)
                
                total_depth = bid_depth + ask_depth
                
                # Convert depth to a score (lower depth = higher score)
                # Normalize based on typical values for this asset
                depth_metrics = self.metrics_collector.get_metrics(
                    "order_book_depth", 
                    {"asset": config.scope_id}
                )
                
                if depth_metrics and len(depth_metrics) > 0:
                    avg_depth = np.mean([m.value for m in depth_metrics])
                    if avg_depth > 0:
                        depth_score = 100 * (1 - min(1.0, total_depth / avg_depth))
                    else:
                        depth_score = 50  # Default if no history
                else:
                    depth_score = 50  # Default if no metrics
                
                # Combine spread and depth scores (higher score is worse liquidity)
                liquidity_score = (spread_pct * 0.7) + (depth_score * 0.3)
                
                return liquidity_score
            
            # For global scope, average the scores of active assets
            elif config.scope == "global":
                assets = await self.market_data_repo.get_active_assets()
                if not assets:
                    return 0.0
                
                total_liquidity_score = 0.0
                asset_count = 0
                
                for asset in assets[:5]:  # Limit to 5 most active assets for performance
                    config_copy = CircuitBreakerConfig(
                        type=config.type,
                        warning_threshold=config.warning_threshold,
                        trip_threshold=config.trip_threshold,
                        cool_down_period=config.cool_down_period,
                        recovery_period=config.recovery_period,
                        recovery_steps=config.recovery_steps,
                        scope="asset",
                        scope_id=asset.symbol,
                        check_interval=config.check_interval
                    )
                    
                    liquidity_score = await self._get_liquidity_value(config_copy)
                    
                    if liquidity_score > 0:
                        total_liquidity_score += liquidity_score
                        asset_count += 1
                
                if asset_count == 0:
                    return 0.0
                
                return total_liquidity_score / asset_count
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating liquidity value: {str(e)}", exc_info=True)
            return 0.0
    
    async def _get_system_value(self, config: CircuitBreakerConfig) -> float:
        """Get the current system performance value for circuit breaker evaluation."""
        try:
            # Collect various system performance metrics
            metrics = {}
            
            # CPU usage (0-100%)
            cpu_metrics = self.metrics_collector.get_metrics("system_cpu_usage_percent")
            if cpu_metrics and len(cpu_metrics) > 0:
                metrics["cpu"] = np.mean([m.value for m in cpu_metrics])
            else:
                metrics["cpu"] = 0.0
            
            # Memory usage (0-100%)
            mem_metrics = self.metrics_collector.get_metrics("system_memory_usage_percent")
            if mem_metrics and len(mem_metrics) > 0:
                metrics["memory"] = np.mean([m.value for m in mem_metrics])
            else:
                metrics["memory"] = 0.0
            
            # API rate limit usage (0-100%)
            api_metrics = self.metrics_collector.get_metrics("api_rate_limit_usage_percent")
            if api_metrics and len(api_metrics) > 0:
                metrics["api"] = np.mean([m.value for m in api_metrics])
            else:
                metrics["api"] = 0.0
            
            # Error rate (errors per minute)
            error_metrics = self.metrics_collector.get_metrics("system_error_rate")
            if error_metrics and len(error_metrics) > 0:
                metrics["error"] = np.mean([m.value for m in error_metrics])
            else:
                metrics["error"] = 0.0
            
            # Calculate weighted system load score (higher is worse)
            system_score = (
                (metrics["cpu"] * 0.3) + 
                (metrics["memory"] * 0.2) + 
                (metrics["api"] * 0.3) + 
                (min(100.0, metrics["error"] * 10) * 0.2)  # Scale error rate
            )
            
            return system_score
            
        except Exception as e:
            logger.error(f"Error calculating system value: {str(e)}", exc_info=True)
            return 0.0
    
    async def _get_anomaly_value(self, config: CircuitBreakerConfig) -> float:
        """Get the current anomaly detection value for circuit breaker evaluation."""
        try:
            # For asset-specific scope
            if config.scope == "asset" and config.scope_id:
                # Get recent price and volume data
                ohlc_data = await self.market_data_repo.get_ohlc(
                    config.scope_id, 
                    TIMEFRAMES.M1, 
                    limit=120  # Last 2 hours of 1-minute data
                )
                
                if len(ohlc_data) < 30:  # Need enough data for anomaly detection
                    return 0.0
                
                # Extract price and volume series
                prices = np.array([candle.close for candle in ohlc_data])
                volumes = np.array([candle.volume for candle in ohlc_data])
                
                # Detect price anomalies using Z-score
                price_zscores = calculate_zscore(prices, window=30)
                price_anomaly_score = np.max(np.abs(price_zscores[-10:]))  # Max z-score in recent 10 periods
                
                # Detect volume anomalies
                volume_zscores = calculate_zscore(volumes, window=30)
                volume_anomaly_score = np.max(np.abs(volume_zscores[-10:]))
                
                # Calculate price velocity (rate of change)
                price_velocity = np.diff(prices) / prices[:-1] * 100  # Percentage change
                velocity_zscores = calculate_zscore(price_velocity, window=29)  # One less due to diff
                velocity_anomaly_score = np.max(np.abs(velocity_zscores[-10:]))
                
                # Combine anomaly scores (higher score means more anomalous)
                anomaly_score = max(
                    price_anomaly_score,
                    volume_anomaly_score,
                    velocity_anomaly_score
                )
                
                return anomaly_score
            
            # For global scope, check multiple assets and take the maximum
            elif config.scope == "global":
                assets = await self.market_data_repo.get_active_assets()
                if not assets:
                    return 0.0
                
                max_anomaly_score = 0.0
                
                for asset in assets[:10]:  # Limit to 10 most active assets for performance
                    config_copy = CircuitBreakerConfig(
                        type=config.type,
                        warning_threshold=config.warning_threshold,
                        trip_threshold=config.trip_threshold,
                        cool_down_period=config.cool_down_period,
                        recovery_period=config.recovery_period,
                        recovery_steps=config.recovery_steps,
                        scope="asset",
                        scope_id=asset.symbol,
                        check_interval=config.check_interval
                    )
                    
                    anomaly_score = await self._get_anomaly_value(config_copy)
                    max_anomaly_score = max(max_anomaly_score, anomaly_score)
                
                return max_anomaly_score
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating anomaly value: {str(e)}", exc_info=True)
            return 0.0
    
    def check_trading_allowed(self, asset: str, exchange: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Check if trading is allowed for a specific asset and exchange based on circuit breaker status.
        
        Args:
            asset: Asset symbol to check
            exchange: Exchange name to check
            
        Returns:
            Tuple containing:
            - bool: True if trading is allowed, False if not
            - Optional[str]: Reason for disallowing trading if applicable
            - Optional[Dict]: Additional information about restrictions
        """
        # Check global circuit breakers first
        for cb_id, status in self.circuit_breakers.items():
            if status.config.scope == "global":
                if status.state == CircuitBreakerState.TRIPPED:
                    return False, f"Global circuit breaker {cb_id} is tripped", {
                        "circuit_breaker": cb_id,
                        "type": status.config.type.name,
                        "value": status.current_value,
                        "threshold": status.config.trip_threshold,
                        "cool_down_end": status.cool_down_end.isoformat() if status.cool_down_end else None
                    }
                
                if status.state == CircuitBreakerState.COOLING:
                    return False, f"Global circuit breaker {cb_id} is cooling down", {
                        "circuit_breaker": cb_id,
                        "type": status.config.type.name,
                        "value": status.current_value,
                        "cool_down_end": status.cool_down_end.isoformat() if status.cool_down_end else None
                    }
                
                if status.state == CircuitBreakerState.RECOVERY:
                    # Allow limited trading during recovery based on recovery level
                    recovery_ratio = status.recovery_level / status.config.recovery_steps
                    if np.random.random() > recovery_ratio:  # Probabilistic trading based on recovery level
                        return False, f"Global circuit breaker {cb_id} is in recovery", {
                            "circuit_breaker": cb_id,
                            "type": status.config.type.name,
                            "value": status.current_value,
                            "recovery_level": f"{status.recovery_level}/{status.config.recovery_steps}",
                            "recovery_end": status.recovery_end.isoformat() if status.recovery_end else None
                        }
        
        # Check exchange-specific circuit breakers
        for cb_id, status in self.circuit_breakers.items():
            if status.config.scope == "exchange" and status.config.scope_id == exchange:
                if status.state == CircuitBreakerState.TRIPPED:
                    return False, f"Exchange circuit breaker {cb_id} is tripped", {
                        "circuit_breaker": cb_id,
                        "type": status.config.type.name,
                        "value": status.current_value,
                        "threshold": status.config.trip_threshold,
                        "cool_down_end": status.cool_down_end.isoformat() if status.cool_down_end else None
                    }
                
                if status.state == CircuitBreakerState.COOLING:
                    return False, f"Exchange circuit breaker {cb_id} is cooling down", {
                        "circuit_breaker": cb_id,
                        "type": status.config.type.name,
                        "value": status.current_value,
                        "cool_down_end": status.cool_down_end.isoformat() if status.cool_down_end else None
                    }
                
                if status.state == CircuitBreakerState.RECOVERY:
                    recovery_ratio = status.recovery_level / status.config.recovery_steps
                    if np.random.random() > recovery_ratio:
                        return False, f"Exchange circuit breaker {cb_id} is in recovery", {
                            "circuit_breaker": cb_id,
                            "type": status.config.type.name,
                            "recovery_level": f"{status.recovery_level}/{status.config.recovery_steps}",
                            "recovery_end": status.recovery_end.isoformat() if status.recovery_end else None
                        }
        
        # Check asset-specific circuit breakers
        for cb_id, status in self.circuit_breakers.items():
            if status.config.scope == "asset" and status.config.scope_id == asset:
                if status.state == CircuitBreakerState.TRIPPED:
                    return False, f"Asset circuit breaker {cb_id} is tripped", {
                        "circuit_breaker": cb_id,
                        "type": status.config.type.name,
                        "value": status.current_value,
                        "threshold": status.config.trip_threshold,
                        "cool_down_end": status.cool_down_end.isoformat() if status.cool_down_end else None
                    }
                
                if status.state == CircuitBreakerState.COOLING:
                    return False, f"Asset circuit breaker {cb_id} is cooling down", {
                        "circuit_breaker": cb_id,
                        "type": status.config.type.name,
                        "value": status.current_value,
                        "cool_down_end": status.cool_down_end.isoformat() if status.cool_down_end else None
                    }
                
                if status.state == CircuitBreakerState.RECOVERY:
                    recovery_ratio = status.recovery_level / status.config.recovery_steps
                    if np.random.random() > recovery_ratio:
                        return False, f"Asset circuit breaker {cb_id} is in recovery", {
                            "circuit_breaker": cb_id,
                            "type": status.config.type.name,
                            "recovery_level": f"{status.recovery_level}/{status.config.recovery_steps}",
                            "recovery_end": status.recovery_end.isoformat() if status.recovery_end else None
                        }
        
        # If we reached here, trading is allowed
        return True, None, None
    
    def get_circuit_breaker_status(self, cb_id: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """
        Get the current status of one or all circuit breakers.
        
        Args:
            cb_id: Optional ID of a specific circuit breaker
            
        Returns:
            Dictionary or list of dictionaries with circuit breaker status
        """
        if cb_id:
            if cb_id in self.circuit_breakers:
                status = self.circuit_breakers[cb_id]
                return {
                    "id": status.id,
                    "type": status.config.type.name,
                    "scope": status.config.scope,
                    "scope_id": status.config.scope_id,
                    "state": status.state.name,
                    "current_value": status.current_value,
                    "warning_threshold": status.config.warning_threshold,
                    "trip_threshold": status.config.trip_threshold,
                    "last_checked": status.last_checked.isoformat(),
                    "last_tripped": status.last_tripped.isoformat() if status.last_tripped else None,
                    "trip_count": status.trip_count,
                    "warning_count": status.warning_count,
                    "cool_down_end": status.cool_down_end.isoformat() if status.cool_down_end else None,
                    "recovery_end": status.recovery_end.isoformat() if status.recovery_end else None,
                    "recovery_level": status.recovery_level,
                    "recovery_steps": status.config.recovery_steps,
                    "last_state_change": status.last_state_change.isoformat() if status.last_state_change else None,
                    "description": status.config.description
                }
            else:
                return {"error": f"Circuit breaker {cb_id} not found"}
        
        # Return all circuit breakers
        result = []
        for cb_id, status in self.circuit_breakers.items():
            result.append({
                "id": status.id,
                "type": status.config.type.name,
                "scope": status.config.scope,
                "scope_id": status.config.scope_id,
                "state": status.state.name,
                "current_value": status.current_value,
                "warning_threshold": status.config.warning_threshold,
                "trip_threshold": status.config.trip_threshold,
                "last_state_change": status.last_state_change.isoformat() if status.last_state_change else None,
                "trip_count": status.trip_count
            })
        
        return result
    
    def reset_circuit_breaker(self, cb_id: str) -> bool:
        """
        Reset a circuit breaker to normal state. This is an emergency override.
        
        Args:
            cb_id: ID of the circuit breaker to reset
            
        Returns:
            bool: True if successful, False if circuit breaker not found
        """
        if cb_id in self.circuit_breakers:
            status = self.circuit_breakers[cb_id]
            status.state = CircuitBreakerState.NORMAL
            status.last_state_change = datetime.now()
            status.cool_down_end = None
            status.recovery_end = None
            status.recovery_level = 0
            logger.warning(f"Circuit breaker {cb_id} manually reset to NORMAL state")
            
            # Record the manual reset in metrics
            self.metrics_collector.record_count(
                "circuit_breaker_manual_reset", 
                1,
                {"id": cb_id, "type": status.config.type.name, "scope": status.config.scope}
            )
            
            return True
        
        logger.warning(f"Attempted to reset non-existent circuit breaker: {cb_id}")
        return False
    
    def register_custom_circuit_breaker(
        self, 
        name: str, 
        warning_threshold: float,
        trip_threshold: float,
        scope: str,
        scope_id: Optional[str],
        condition_func: callable,
        description: str,
        cool_down_period: int = 300,
        recovery_period: int = 600,
        recovery_steps: int = 5,
        check_interval: int = 30
    ) -> str:
        """
        Register a custom circuit breaker with a user-defined condition function.
        
        Args:
            name: Name for the custom circuit breaker
            warning_threshold: Threshold for warning state
            trip_threshold: Threshold for tripped state
            scope: Scope of application (global, asset, exchange, etc.)
            scope_id: Specific ID within scope if applicable
            condition_func: Async function that returns the current value
            description: Description of the circuit breaker
            cool_down_period: Cooling period in seconds after tripping
            recovery_period: Recovery period in seconds after cooling
            recovery_steps: Number of steps for gradual recovery
            check_interval: Interval in seconds between checks
            
        Returns:
            str: ID of the registered circuit breaker
        """
        config = CircuitBreakerConfig(
            type=CircuitBreakerType.CUSTOM,
            warning_threshold=warning_threshold,
            trip_threshold=trip_threshold,
            cool_down_period=cool_down_period,
            recovery_period=recovery_period,
            recovery_steps=recovery_steps,
            scope=scope,
            scope_id=scope_id,
            check_interval=check_interval,
            description=description,
            custom_condition=condition_func,
            metadata={"name": name}
        )
        
        cb_id = self.register_circuit_breaker(config)
        logger.info(f"Registered custom circuit breaker: {cb_id} - {name}")

        return cb_id
    
    
    class VolatilityCircuitBreaker(BaseCircuitBreaker, name="VolatilityCircuitBreaker"):
        """
        Circuit breaker that triggers when market volatility exceeds a threshold.
        
        This circuit breaker monitors market volatility and trips when it exceeds
        a predefined threshold, protecting against extreme market conditions.
        """
        
        def __init__(self,
                     volatility_threshold: float = 3.0,
                     lookback_periods: int = 20,
                     cooldown_minutes: int = 30,
                     asset_specific_thresholds: Dict[str, float] = None,
                     **kwargs):
            """
            Initialize the volatility circuit breaker.
            
            Args:
                volatility_threshold: Z-score threshold for volatility (default: 3.0)
                lookback_periods: Number of periods to look back for volatility calculation
                cooldown_minutes: Minutes to wait before resetting after triggering
                asset_specific_thresholds: Optional dict of asset-specific thresholds
            """
            self.volatility_threshold = volatility_threshold
            self.lookback_periods = lookback_periods
            self.cooldown_minutes = cooldown_minutes
            self.asset_specific_thresholds = asset_specific_thresholds or {}
            
            self.last_triggered = {}
            self.state = CircuitBreakerState.CLOSED
            self.metrics = MetricsCollector("circuit_breaker.volatility")
            self.logger = get_logger("risk_manager.circuit_breaker.volatility")
            self.logger.info(f"Volatility circuit breaker initialized with threshold {volatility_threshold}")
        
        async def check(self, asset: str, current_volatility: float = None,
                       market_data: Dict[str, Any] = None) -> bool:
            """
            Check if the circuit breaker should be triggered.
            
            Args:
                asset: Asset to check
                current_volatility: Current volatility value (optional)
                market_data: Market data for volatility calculation (optional)
                
            Returns:
                True if circuit breaker is triggered, False otherwise
            """
            # Check if we're in cooldown period
            now = datetime.now()
            if asset in self.last_triggered:
                cooldown_end = self.last_triggered[asset] + timedelta(minutes=self.cooldown_minutes)
                if now < cooldown_end:
                    self.logger.debug(f"Volatility circuit breaker for {asset} in cooldown until {cooldown_end}")
                    return False
            
            # Get asset-specific threshold or use default
            threshold = self.asset_specific_thresholds.get(asset, self.volatility_threshold)
            
            # Use provided volatility or calculate it
            volatility = current_volatility
            if volatility is None and market_data is not None:
                # Simple volatility calculation if market data is provided
                if 'close' in market_data and len(market_data['close']) >= self.lookback_periods:
                    prices = market_data['close'][-self.lookback_periods:]
                    returns = np.diff(prices) / prices[:-1]
                    volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            if volatility is None:
                self.logger.warning(f"Cannot check volatility circuit breaker for {asset}: no volatility data")
                return False
            
            # Check if volatility exceeds threshold
            self.metrics.set(f"volatility.{asset}", volatility)
            
            if volatility > threshold:
                self.logger.warning(f"Volatility circuit breaker triggered for {asset}: {volatility:.4f} > {threshold:.4f}")
                self.last_triggered[asset] = now
                self.state = CircuitBreakerState.OPEN
                self.metrics.increment(f"triggers.{asset}")
                return True
            
            return False
    
    
    # Export the necessary symbols
    __all__ = ["BaseCircuitBreaker", "VolatilityCircuitBreaker", "get_circuit_breaker"]
