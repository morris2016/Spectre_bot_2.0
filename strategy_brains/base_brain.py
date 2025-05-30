#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Base Strategy Brain

This module defines the base class for all strategy brains in the system.
"""

import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union

from .historical_memory import HistoricalMemoryMixin


from common.logger import get_logger
from common.metrics import MetricsCollector
from common.exceptions import StrategyError, SignalGenerationError
from common.models import AssetType
from common.constants import SignalStrength


class TradeDirection(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    WEAK_BUY = "weak_buy"
    NEUTRAL = "neutral"
    WEAK_SELL = "weak_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class SignalEvent:
    """Lightweight signal representation used in tests."""
    timestamp: float
    asset: str
    direction: TradeDirection
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Base configuration for strategy brains."""
    risk_per_trade: float = 0.01
    max_position_size: float = 0.05


@dataclass
class BrainConfig:
    """Base configuration for strategy brains."""
    pass


    # Memory windows for performance tracking
    short_memory: int = 50
    long_memory: int = 500


    # Arbitrary strategy parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Generic risk management settings
    risk_per_trade: float = 0.01
    max_position_size: float = 0.05
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 3.0



class StrategyBrain(HistoricalMemoryMixin, ABC):
    """
    Base class for all trading strategy brains.
    
    Each strategy brain implements a specific trading strategy approach,
    generates signals, and adapts to changing market conditions.
    """
    
    def __init__(
        self,
        config: Optional[Union[BrainConfig, Dict[str, Any]]] = None,
        *args,
        name: str = None,
        redis_client=None,
        db_client=None,
        loop=None,
        **kwargs,
    ):
        """
        Initialize a strategy brain.
        
        Args:
            config: Brain-specific configuration
            name: Brain name
            redis_client: Redis client for communication
            db_client: Database client
            loop: Asyncio event loop
            *args: Additional positional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)
        """
        self.config = config or BrainConfig()
        if isinstance(self.config, dict):
            self.config = BrainConfig(**self.config)
        self._validate_config(self.config)
        self.name = name or self.__class__.__name__
        self.redis_client = redis_client
        self.db_client = db_client
        try:
            self.loop = loop or asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        HistoricalMemoryMixin.__init__(
            self,
            short_window=getattr(self.config, "short_memory", 50),
            long_window=getattr(self.config, "long_memory", 500),
        )
        
        self.logger = get_logger(f"Brain.{self.name}")
        self.metrics = MetricsCollector(f"brain.{self.name}")
        
        self.initialized = False
        self.running = False
        self.features = {}
        self.signals_generated = 0
        self.successful_signals = 0
        self.failed_signals = 0
        
        # Performance tracking
        self.performance_metrics = {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expected_value": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "overall_score": 0.5  # Default neutral score
        }
        
        # Strategy parameters
        self.parameters = getattr(self.config, "parameters", {})

    def _validate_config(self, config: BrainConfig) -> None:
        """Validate core configuration options."""
        risk = getattr(config, "risk_per_trade", 0.01)
        if risk <= 0 or risk > 1:
            raise ValueError("risk_per_trade must be between 0 and 1")
        config.risk_per_trade = risk

        max_pos = getattr(config, "max_position_size", 1.0)

        if max_pos <= 0:
            raise ValueError("max_position_size must be positive")
        config.max_position_size = max_pos
        
    async def initialize(self):
        """Initialize the strategy brain."""
        self.logger.info(f"Initializing strategy brain: {self.name}")
        
        try:
            # Load historical performance if available
            await self._load_performance_history()
            
            # Load any saved state
            await self._load_state()
            
            # Initialize strategy-specific resources
            await self._initialize_strategy()
            
            self.initialized = True
            self.logger.info(f"Strategy brain {self.name} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategy brain {self.name}: {str(e)}")
            raise StrategyError(f"Failed to initialize brain {self.name}: {str(e)}")
            
    async def stop(self):
        """Stop the strategy brain and release resources."""
        if not self.initialized:
            return
            
        self.logger.info(f"Stopping strategy brain: {self.name}")
        self.running = False
        
        try:
            # Save current state
            await self._save_state()
            
            # Release strategy-specific resources
            await self._release_resources()
            
            self.logger.info(f"Strategy brain {self.name} stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy brain {self.name}: {str(e)}")
            
    async def health_check(self) -> bool:
        """
        Perform a health check.
        
        Returns:
            bool: True if brain is healthy, False otherwise
        """
        return self.initialized and not self._check_for_errors()
        
    @abstractmethod
    async def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on market conditions and strategy rules.
        
        Returns:
            List of signal dictionaries
        """
        pass
        
    @abstractmethod
    async def on_regime_change(self, new_regime: str):
        """
        Handle market regime changes.
        
        Args:
            new_regime: New market regime
        """
        pass
        
    async def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get brain performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        await self._update_performance_metrics()
        return self.performance_metrics
        
    async def update_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        """
        Update strategy parameters.
        
        Args:
            new_parameters: New parameter values
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate parameters
            valid = await self._validate_parameters(new_parameters)
            if not valid:
                self.logger.warning(f"Invalid parameters for brain {self.name}")
                return False
                
            # Update parameters
            self.parameters.update(new_parameters)
            self.logger.info(f"Updated parameters for brain {self.name}: {new_parameters}")
            
            # Apply the new parameters
            await self._apply_parameters()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating parameters for brain {self.name}: {str(e)}")
            return False
            
    async def _initialize_strategy(self):
        """Initialize strategy-specific resources."""
        # Implement in subclasses
        pass
        
    async def _release_resources(self):
        """Release strategy-specific resources."""
        # Implement in subclasses
        pass
        
    async def _load_state(self):
        """Load strategy state from persistent storage."""
        if not self.db_client:
            return
            
        try:
            # Query from database
            query = """
            SELECT state_data FROM strategy_brain_states 
            WHERE brain_name = $1 
            ORDER BY timestamp DESC LIMIT 1
            """
            
            result = await self.db_client.fetch_one(query, self.name)
            if result and 'state_data' in result:
                state_data = result['state_data']
                await self._restore_state(state_data)
                self.logger.info(f"Loaded state for brain {self.name}")
                
        except Exception as e:
            self.logger.error(f"Error loading state for brain {self.name}: {str(e)}")
            
    async def _save_state(self):
        """Save strategy state to persistent storage."""
        if not self.db_client:
            return
            
        try:
            # Get current state
            state_data = await self._get_current_state()
            if not state_data:
                return
                
            # Save to database
            query = """
            INSERT INTO strategy_brain_states (brain_name, timestamp, state_data)
            VALUES ($1, $2, $3)
            """
            
            await self.db_client.execute(
                query, self.name, time.time(), state_data
            )
            
            self.logger.info(f"Saved state for brain {self.name}")
            
        except Exception as e:
            self.logger.error(f"Error saving state for brain {self.name}: {str(e)}")
            
    async def _get_current_state(self) -> Dict[str, Any]:
        """
        Get current strategy state for persistence.
        
        Returns:
            State data dictionary
        """
        # Default implementation, override in subclasses
        return {
            "name": self.name,
            "parameters": self.parameters,
            "signals_generated": self.signals_generated,
            "successful_signals": self.successful_signals,
            "failed_signals": self.failed_signals,
            "performance_metrics": self.performance_metrics
        }
        
    async def _restore_state(self, state_data: Dict[str, Any]):
        """
        Restore strategy state from persistence.
        
        Args:
            state_data: State data dictionary
        """
        # Default implementation, override in subclasses
        if "parameters" in state_data:
            self.parameters = state_data["parameters"]
            
        if "signals_generated" in state_data:
            self.signals_generated = state_data["signals_generated"]
            
        if "successful_signals" in state_data:
            self.successful_signals = state_data["successful_signals"]
            
        if "failed_signals" in state_data:
            self.failed_signals = state_data["failed_signals"]
            
        if "performance_metrics" in state_data:
            self.performance_metrics = state_data["performance_metrics"]
            
    async def _validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate strategy parameters.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Default implementation, override in subclasses
        return True
        
    async def _apply_parameters(self):
        """Apply strategy parameters."""
        # Implement in subclasses
        pass
        
    async def _load_performance_history(self):
        """Load historical performance from database."""
        if not self.db_client:
            return
            
        try:
            # Query from database
            query = """
            SELECT win_rate, profit_factor, expected_value, sharpe_ratio, max_drawdown, overall_score
            FROM strategy_performance
            WHERE brain_name = $1
            ORDER BY timestamp DESC LIMIT 1
            """
            
            result = await self.db_client.fetch_one(query, self.name)
            if result:
                self.performance_metrics = {
                    "win_rate": result["win_rate"],
                    "profit_factor": result["profit_factor"],
                    "expected_value": result["expected_value"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "max_drawdown": result["max_drawdown"],
                    "overall_score": result["overall_score"]
                }
                
                self.logger.info(f"Loaded performance history for brain {self.name}")
                
        except Exception as e:
            self.logger.error(f"Error loading performance history for brain {self.name}: {str(e)}")
            
    async def _update_performance_metrics(self):
        """Update performance metrics based on recent results."""
        if not self.db_client:
            return
            
        try:
            # Query recent signals
            query = """
            SELECT 
                SUM(CASE WHEN result = 'success' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'failure' THEN 1 ELSE 0 END) as losses,
                SUM(profit) as total_profit,
                SUM(CASE WHEN profit < 0 THEN profit ELSE 0 END) as total_loss,
                AVG(profit) as avg_profit,
                STDDEV(profit) as stddev_profit,
                MIN(drawdown) as max_drawdown
            FROM signal_results
            WHERE brain_name = $1 AND timestamp > $2
            """
            
            # Get signals from the last 30 days
            cutoff_time = time.time() - (30 * 24 * 60 * 60)
            result = await self.db_client.fetch_one(query, self.name, cutoff_time)
            
            if result and result["wins"] is not None and (result["wins"] + result["losses"]) > 0:
                wins = result["wins"]
                losses = result["losses"]
                total = wins + losses
                
                # Calculate metrics
                win_rate = wins / total if total > 0 else 0
                profit_factor = abs(result["total_profit"] / result["total_loss"]) if result["total_loss"] and result["total_loss"] < 0 else 0
                expected_value = result["avg_profit"] if result["avg_profit"] else 0
                sharpe_ratio = result["avg_profit"] / result["stddev_profit"] if result["stddev_profit"] and result["stddev_profit"] > 0 else 0
                max_drawdown = abs(result["max_drawdown"]) if result["max_drawdown"] else 0
                
                # Overall score - weighted average of normalized metrics
                overall_score = (
                    0.3 * win_rate +
                    0.2 * min(profit_factor / 3, 1) +  # Cap at 1.0
                    0.2 * min(max(0, expected_value / 0.01), 1) +  # Normalize to 0-1
                    0.2 * min(max(0, sharpe_ratio / 2), 1) +  # Normalize to 0-1
                    0.1 * (1 - min(max_drawdown / 0.5, 1))  # Invert and normalize
                )
                
                # Update metrics
                self.performance_metrics = {
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "expected_value": expected_value,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "overall_score": overall_score
                }
                
                # Log significant performance changes
                self.logger.debug(f"Updated performance metrics for brain {self.name}")
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics for brain {self.name}: {str(e)}")
            
    def _check_for_errors(self) -> bool:
        """
        Check for error conditions.
        
        Returns:
            bool: True if errors detected, False otherwise
        """
        # Default implementation, override in subclasses
        return False


# Backwards compatibility
BaseBrain = StrategyBrain

__all__ = [
    "BrainConfig",
    "TradeDirection",
    "SignalStrength",
    "AssetType",
    "StrategyBrain",
    "BaseBrain",
]
