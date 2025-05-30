#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Adaptive Brain Module

This module implements the Adaptive Brain strategy, which continuously evolves
and adjusts its trading approaches based on market feedback and performance.

The Adaptive Brain strategy is designed to:
1. Learn from both success and failure
2. Dynamically adjust parameters based on performance
3. Switch between strategies based on effectiveness
4. Combine multiple approaches using adaptive weighting
5. Evolve new strategy variations through genetic algorithms
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from common.async_utils import gather_with_concurrency
from common.constants import DEFAULT_ADAPTIVE_CONFIG, DEFAULT_RISK_PARAMETERS, MARKET_REGIMES, SIGNAL_TYPES, STRATEGY_TYPES, TIMEFRAMES
from common.exceptions import AdaptationError, StrategyExecutionError
from common.logger import get_logger
from common.metrics import MetricsCollector
from common.utils import calculate_drawdown, calculate_sharpe_ratio, get_utc_now
from data_storage.market_data import MarketDataRepository
from feature_service.features.technical import TechnicalFeatureExtractor
from feature_service.multi_timeframe import MultiTimeframeAnalyzer
from intelligence.adaptive_learning.bayesian import BayesianOptimizer
from intelligence.adaptive_learning.genetic import GeneticOptimizer
from intelligence.adaptive_learning.reinforcement import ReinforcementLearner
from ml_models.prediction import PredictionService

from .base_brain import BaseBrain


class StrategyComponent(Enum):
    """
    Enum representing different components of a trading strategy.
    """

    ENTRY_CONDITION = auto()
    EXIT_CONDITION = auto()
    STOP_LOSS = auto()
    TAKE_PROFIT = auto()
    POSITION_SIZING = auto()
    TIMEFRAME = auto()
    INDICATOR = auto()
    FILTER = auto()


class StrategyVariant:
    """
    Represents a variant of a trading strategy with specific parameters.
    """

    def __init__(self, strategy_type: str, parameters: Dict[str, Any], creation_date=None):
        """
        Initialize a strategy variant.

        Args:
            strategy_type: Type of strategy this variant represents
            parameters: Parameters defining this strategy variant
            creation_date: When this variant was created
        """
        self.strategy_type = strategy_type
        self.parameters = parameters
        self.creation_date = creation_date or get_utc_now()

        # Performance metrics
        self.trades_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        self.last_updated = self.creation_date

        # Computed metrics
        self.win_rate = 0.0
        self.average_pnl = 0.0
        self.sharpe_ratio = 0.0
        self.fitness_score = 0.0

        # Generate a unique ID for this variant
        self.variant_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique ID for this strategy variant."""
        # Create a string representation of key parameters
        params_str = json.dumps(self.parameters, sort_keys=True)
        strategy_signature = f"{self.strategy_type}:{params_str}:{self.creation_date.isoformat()}"

        # Generate a hash
        variant_hash = hashlib.md5(strategy_signature.encode()).hexdigest()

        return f"{self.strategy_type[:3].upper()}-{variant_hash[:8]}"

    def update_performance(self, win: bool, pnl_percent: float) -> None:
        """
        Update performance metrics with a new trade result.

        Args:
            win: Whether the trade was successful
            pnl_percent: Percentage profit/loss from the trade
        """
        self.trades_count += 1
        if win:
            self.win_count += 1
        else:
            self.loss_count += 1

        self.total_pnl += pnl_percent
        self.last_updated = get_utc_now()

        # Update computed metrics
        self._update_computed_metrics()

    def _update_computed_metrics(self) -> None:
        """Update computed performance metrics."""
        if self.trades_count > 0:
            self.win_rate = self.win_count / self.trades_count
            self.average_pnl = self.total_pnl / self.trades_count

            # Simple fitness score: combination of win rate and average PnL
            # Production version would use a more sophisticated calculation
            self.fitness_score = (self.win_rate * 0.7) + ((self.average_pnl / 2) * 0.3)

            # Normalize fitness score to 0-1 range
            self.fitness_score = max(0.0, min(1.0, self.fitness_score))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "variant_id": self.variant_id,
            "strategy_type": self.strategy_type,
            "parameters": self.parameters,
            "creation_date": self.creation_date.isoformat(),
            "trades_count": self.trades_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "total_pnl": self.total_pnl,
            "last_updated": self.last_updated.isoformat(),
            "win_rate": self.win_rate,
            "average_pnl": self.average_pnl,
            "fitness_score": self.fitness_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyVariant":
        """Create from dictionary representation."""
        variant = cls(strategy_type=data["strategy_type"], parameters=data["parameters"], creation_date=datetime.fromisoformat(data["creation_date"]))

        variant.variant_id = data["variant_id"]
        variant.trades_count = data["trades_count"]
        variant.win_count = data["win_count"]
        variant.loss_count = data["loss_count"]
        variant.total_pnl = data["total_pnl"]
        variant.last_updated = datetime.fromisoformat(data["last_updated"])
        variant.win_rate = data["win_rate"]
        variant.average_pnl = data["average_pnl"]
        variant.fitness_score = data["fitness_score"]

        return variant


class AdaptiveBrain(BaseBrain):
    """
    Specialized brain that continuously adapts and evolves trading strategies
    based on performance feedback.
    """

    def __init__(
        self,
        asset_id: str,
        exchange_id: str,
        config: Dict[str, Any],
        metrics_collector: MetricsCollector,
        reinforcement_learner: ReinforcementLearner = None,
        genetic_optimizer: GeneticOptimizer = None,
        bayesian_optimizer: BayesianOptimizer = None,
        prediction_service: PredictionService = None,
        market_data_repository: MarketDataRepository = None,
        timeframes: List[str] = None,
    ):
        """
        Initialize the AdaptiveBrain.

        Args:
            asset_id: Identifier for the asset
            exchange_id: Identifier for the exchange
            config: Configuration parameters
            metrics_collector: Metrics collection service
            reinforcement_learner: Service for RL-based adaptation
            genetic_optimizer: Service for genetic optimization
            bayesian_optimizer: Service for Bayesian parameter optimization
            prediction_service: Service for ML predictions
            market_data_repository: Repository for market data
            timeframes: List of timeframes to analyze
        """
        super().__init__(asset_id=asset_id, exchange_id=exchange_id, config=config, metrics_collector=metrics_collector, brain_type="adaptive")

        self.logger = get_logger(f"AdaptiveBrain-{exchange_id}-{asset_id}")

        # Initialize services
        self.reinforcement_learner = reinforcement_learner
        self.genetic_optimizer = genetic_optimizer
        self.bayesian_optimizer = bayesian_optimizer
        self.prediction_service = prediction_service
        self.market_data_repository = market_data_repository

        # Set default timeframes if not provided
        self.timeframes = timeframes or TIMEFRAMES

        # Initialize feature extractors
        self.technical_feature_extractor = TechnicalFeatureExtractor()
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(self.timeframes)

        # Initialize adaptation parameters
        self._initialize_adaptation_parameters()

        # Initialize strategy variants
        self._initialize_strategy_variants()

        # Active strategies
        self.active_variants = set()
        self.primary_variant_id = None
        self.secondary_variant_id = None

        # Strategy selection state
        self.strategy_weights = {}
        self.last_strategy_evaluation = get_utc_now()
        self.last_evolution = get_utc_now()

        # Performance tracking
        self.recent_trades = []
        self.cumulative_pnl = 0.0
        self.overall_win_rate = 0.0

        self.logger.info(f"AdaptiveBrain initialized for {exchange_id}-{asset_id}")

    def _initialize_adaptation_parameters(self):
        """Initialize parameters that control adaptation behavior."""
        default_params = DEFAULT_ADAPTIVE_CONFIG
        config_params = self.config.get("adaptive_params", {})

        # Merge defaults with configuration
        self.adaptive_params = {**default_params, **config_params}

        # Ensure essential parameters are present
        required_params = [
            "evaluation_interval_hours",
            "evolution_interval_hours",
            "min_trades_for_evaluation",
            "exploration_rate",
            "exploitation_rate",
            "learning_rate",
            "max_active_variants",
            "tournament_size",
            "mutation_rate",
            "crossover_rate",
            "elitism_count",
        ]

        for param in required_params:
            if param not in self.adaptive_params:
                default_value = default_params.get(param, 0.5)
                self.adaptive_params[param] = default_value
                self.logger.warning(f"Missing adaptive parameter {param}, using default: {default_value}")

    def _initialize_strategy_variants(self):
        """Initialize strategy variants population."""
        # Dictionary to hold all strategy variants
        self.strategy_variants = {}

        # Create initial variants for different strategy types
        initial_strategies = [
            STRATEGY_TYPES.TREND_FOLLOWING,
            STRATEGY_TYPES.MEAN_REVERSION,
            STRATEGY_TYPES.BREAKOUT,
            STRATEGY_TYPES.MOMENTUM,
            STRATEGY_TYPES.VOLATILITY,
            STRATEGY_TYPES.SUPPORT_RESISTANCE,
        ]

        # Create variants for each strategy type
        for strategy_type in initial_strategies:
            # Create multiple variants with different parameters
            for i in range(3):  # 3 variants per strategy type
                variant = self._create_strategy_variant(strategy_type)
                self.strategy_variants[variant.variant_id] = variant

                # Make some variants active initially
                if i == 0:  # Only activate the first variant of each type
                    self.active_variants.add(variant.variant_id)

        # Set initial primary and secondary strategies
        if self.active_variants:
            active_list = list(self.active_variants)
            if len(active_list) >= 1:
                self.primary_variant_id = active_list[0]
            if len(active_list) >= 2:
                self.secondary_variant_id = active_list[1]

        # Initialize strategy weights (equal weighting initially)
        self._update_strategy_weights()

        self.logger.info(f"Initialized {len(self.strategy_variants)} strategy variants, " f"{len(self.active_variants)} active variants")

    def _create_strategy_variant(self, strategy_type: str) -> StrategyVariant:
        """
        Create a new strategy variant with randomized parameters.

        Args:
            strategy_type: Type of strategy to create

        Returns:
            A new StrategyVariant instance
        """
        # Default parameters
        default_params = {
            "lookback_periods": random.randint(10, 50),
            "signal_threshold": round(random.uniform(0.5, 0.9), 2),
            "stop_loss_pct": round(random.uniform(0.5, 3.0), 2),
            "take_profit_pct": round(random.uniform(1.0, 5.0), 2),
            "risk_factor": round(random.uniform(0.5, 1.5), 2),
            "timeframe": random.choice(self.timeframes),
        }

        # Strategy-specific parameters
        if strategy_type == STRATEGY_TYPES.TREND_FOLLOWING:
            params = {
                **default_params,
                "fast_period": random.randint(5, 20),
                "slow_period": random.randint(20, 100),
                "trend_strength_threshold": round(random.uniform(15, 40), 1),
                "reversal_tolerance": round(random.uniform(0.1, 0.5), 2),
            }

        elif strategy_type == STRATEGY_TYPES.MEAN_REVERSION:
            params = {
                **default_params,
                "bollinger_period": random.randint(10, 30),
                "bollinger_std": round(random.uniform(1.5, 3.0), 1),
                "rsi_period": random.randint(7, 21),
                "oversold_threshold": random.randint(20, 35),
                "overbought_threshold": random.randint(65, 80),
            }

        elif strategy_type == STRATEGY_TYPES.BREAKOUT:
            params = {
                **default_params,
                "breakout_periods": random.randint(10, 50),
                "confirmation_periods": random.randint(1, 5),
                "volume_multiplier": round(random.uniform(1.2, 3.0), 1),
                "volatility_filter": bool(random.randint(0, 1)),
            }

        elif strategy_type == STRATEGY_TYPES.MOMENTUM:
            params = {
                **default_params,
                "roc_period": random.randint(5, 25),
                "macd_fast": random.randint(8, 15),
                "macd_slow": random.randint(20, 35),
                "macd_signal": random.randint(7, 12),
                "momentum_threshold": round(random.uniform(0.1, 1.0), 2),
            }

        elif strategy_type == STRATEGY_TYPES.VOLATILITY:
            params = {
                **default_params,
                "atr_period": random.randint(7, 21),
                "volatility_entry_multiplier": round(random.uniform(0.5, 2.0), 1),
                "volatility_exit_multiplier": round(random.uniform(1.0, 3.0), 1),
                "bollinger_squeeze_threshold": round(random.uniform(0.1, 0.5), 2),
            }

        elif strategy_type == STRATEGY_TYPES.SUPPORT_RESISTANCE:
            params = {
                **default_params,
                "pivot_lookback": random.randint(5, 30),
                "pivot_strength": random.randint(2, 5),
                "level_tolerance": round(random.uniform(0.1, 0.5), 2),
                "bounce_likelihood_threshold": round(random.uniform(0.6, 0.9), 2),
            }

        else:
            # Default parameters for unknown strategy types
            params = default_params

        # Create and return the strategy variant
        return StrategyVariant(strategy_type, params)

    def _update_strategy_weights(self):
        """Update the weighting for each active strategy based on performance."""
        # Get active variants
        active_variants = [self.strategy_variants[v_id] for v_id in self.active_variants if v_id in self.strategy_variants]

        if not active_variants:
            self.logger.warning("No active strategy variants found for weight update")
            return

        # Calculate weights based on fitness scores
        total_fitness = sum(v.fitness_score for v in active_variants)

        # If no meaningful fitness data, use equal weights
        if total_fitness <= 0:
            equal_weight = 1.0 / len(active_variants)
            self.strategy_weights = {v.variant_id: equal_weight for v in active_variants}
            self.logger.info("Using equal weights for strategies due to insufficient performance data")
        else:
            # Weight by fitness score
            self.strategy_weights = {v.variant_id: v.fitness_score / total_fitness for v in active_variants}

            # Log top strategies
            top_strategies = sorted([(v.variant_id, self.strategy_weights[v.variant_id]) for v in active_variants], key=lambda x: x[1], reverse=True)

            self.logger.info(f"Updated strategy weights. Top strategies: {top_strategies[:3]}")

        # Update primary and secondary variants
        self._select_primary_secondary_variants()

    def _select_primary_secondary_variants(self):
        """Select the primary and secondary variants based on weights."""
        if not self.strategy_weights:
            return

        # Sort variants by weight
        sorted_variants = sorted(self.strategy_weights.items(), key=lambda x: x[1], reverse=True)

        # Update primary and secondary variants
        if len(sorted_variants) >= 1:
            self.primary_variant_id = sorted_variants[0][0]

            # Log primary variant details
            if self.primary_variant_id in self.strategy_variants:
                primary = self.strategy_variants[self.primary_variant_id]
                self.logger.info(
                    f"Primary variant: {primary.variant_id} ({primary.strategy_type}) "
                    f"with weight {self.strategy_weights[primary.variant_id]:.2f}, "
                    f"win rate: {primary.win_rate:.2f}, avg PnL: {primary.average_pnl:.2f}%"
                )

        if len(sorted_variants) >= 2:
            self.secondary_variant_id = sorted_variants[1][0]

            # Log secondary variant details
            if self.secondary_variant_id in self.strategy_variants:
                secondary = self.strategy_variants[self.secondary_variant_id]
                self.logger.info(
                    f"Secondary variant: {secondary.variant_id} ({secondary.strategy_type}) "
                    f"with weight {self.strategy_weights[secondary.variant_id]:.2f}, "
                    f"win rate: {secondary.win_rate:.2f}, avg PnL: {secondary.average_pnl:.2f}%"
                )

    async def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate signals using adaptive strategy selection.

        Args:
            market_data: DataFrame containing market data

        Returns:
            Dictionary with analysis results
        """
        try:
            self.logger.info(f"Analyzing {self.exchange_id}-{self.asset_id} with AdaptiveBrain")

            start_time = time.time()

            # Check if it's time to evaluate strategies
            current_time = get_utc_now()
            if (current_time - self.last_strategy_evaluation).total_seconds() >= self.adaptive_params["evaluation_interval_hours"] * 3600:
                self.logger.info("Evaluating strategy performance")
                await self._evaluate_strategies()
                self.last_strategy_evaluation = current_time

            # Check if it's time to evolve strategies
            if (current_time - self.last_evolution).total_seconds() >= self.adaptive_params["evolution_interval_hours"] * 3600:
                self.logger.info("Evolving strategies")
                await self._evolve_strategies()
                self.last_evolution = current_time

            # Apply active strategies to generate signals
            primary_signals = {}
            secondary_signals = {}

            # Generate signals from primary variant
            if self.primary_variant_id and self.primary_variant_id in self.strategy_variants:
                primary_variant = self.strategy_variants[self.primary_variant_id]
                primary_signals = await self._apply_strategy_variant(primary_variant, market_data)

            # Generate signals from secondary variant
            if self.secondary_variant_id and self.secondary_variant_id in self.strategy_variants:
                secondary_variant = self.strategy_variants[self.secondary_variant_id]
                secondary_signals = await self._apply_strategy_variant(secondary_variant, market_data)

            # Combine signals according to weights
            if primary_signals and secondary_signals:
                primary_weight = self.strategy_weights.get(self.primary_variant_id, 0.7)
                secondary_weight = self.strategy_weights.get(self.secondary_variant_id, 0.3)

                combined_signals = self._combine_signals(primary_signals, secondary_signals, primary_weight, secondary_weight)
            elif primary_signals:
                combined_signals = primary_signals
            elif secondary_signals:
                combined_signals = secondary_signals
            else:
                # Fallback to neutral signal if no strategies available
                combined_signals = {
                    "signal": SIGNAL_TYPES.NEUTRAL,
                    "confidence": 0.5,
                    "entry_price": None,
                    "stop_loss": None,
                    "take_profit": None,
                    "strategy_type": "none",
                    "indicators": {},
                }

            # Add metadata about which strategies were used
            strategy_info = {
                "primary_variant": self.primary_variant_id,
                "primary_type": (
                    self.strategy_variants[self.primary_variant_id].strategy_type if self.primary_variant_id in self.strategy_variants else None
                ),
                "primary_weight": self.strategy_weights.get(self.primary_variant_id, 0),
                "secondary_variant": self.secondary_variant_id,
                "secondary_type": (
                    self.strategy_variants[self.secondary_variant_id].strategy_type if self.secondary_variant_id in self.strategy_variants else None
                ),
                "secondary_weight": self.strategy_weights.get(self.secondary_variant_id, 0),
                "active_variants_count": len(self.active_variants),
                "total_variants_count": len(self.strategy_variants),
            }

            # Add information about strategy adaptation
            adaptation_info = {
                "last_strategy_evaluation": self.last_strategy_evaluation.isoformat(),
                "last_evolution": self.last_evolution.isoformat(),
                "overall_win_rate": self.overall_win_rate,
                "cumulative_pnl": self.cumulative_pnl,
                "exploration_rate": self.adaptive_params["exploration_rate"],
            }

            # Prepare result
            analysis_result = {"signals": combined_signals, "strategy_info": strategy_info, "adaptation_info": adaptation_info}

            # Track metrics
            execution_time = time.time() - start_time
            self.metrics_collector.record_timing(
                f"adaptive_brain.analysis_time.{self.exchange_id}.{self.asset_id}", execution_time * 1000  # Convert to ms
            )

            return analysis_result

        except Exception as e:
            self.logger.error(f"Error in AdaptiveBrain analysis: {str(e)}", exc_info=True)
            raise StrategyExecutionError(f"AdaptiveBrain analysis failed: {str(e)}")

    async def _apply_strategy_variant(self, variant: StrategyVariant, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply a specific strategy variant to generate trading signals.

        Args:
            variant: Strategy variant to apply
            market_data: Market data for analysis

        Returns:
            Dictionary with signal information
        """

        try:
            strategy_type = variant.strategy_type
            params = variant.parameters

            # Different strategy implementations
            if strategy_type == STRATEGY_TYPES.TREND_FOLLOWING:
                return await self._apply_trend_following(market_data, params)
            elif strategy_type == STRATEGY_TYPES.MEAN_REVERSION:
                return await self._apply_mean_reversion(market_data, params)
            elif strategy_type == STRATEGY_TYPES.BREAKOUT:
                return await self._apply_breakout(market_data, params)
            elif strategy_type == STRATEGY_TYPES.MOMENTUM:
                return await self._apply_momentum(market_data, params)
            elif strategy_type == STRATEGY_TYPES.VOLATILITY:
                return await self._apply_volatility(market_data, params)
            elif strategy_type == STRATEGY_TYPES.SUPPORT_RESISTANCE:
                return await self._apply_support_resistance(market_data, params)
            else:
                return await self._apply_neutral(market_data, params)

        except Exception as e:
            self.logger.error(
                f"Error applying strategy variant {strategy_type}: {str(e)}",
                exc_info=True,
            )
            return {
                "signal": SIGNAL_TYPES.NEUTRAL,
                "confidence": 0.3,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "strategy_type": strategy_type,
                "error": str(e),
            }

    async def _apply_trend_following(self, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder trend following strategy implementation."""
        return {
            "signal": SIGNAL_TYPES.NEUTRAL,
            "confidence": 0.5,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "strategy_type": STRATEGY_TYPES.TREND_FOLLOWING,
            "indicators": {},
        }

    async def _apply_mean_reversion(self, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder mean reversion strategy implementation."""
        return {
            "signal": SIGNAL_TYPES.NEUTRAL,
            "confidence": 0.5,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "strategy_type": STRATEGY_TYPES.MEAN_REVERSION,
            "indicators": {},
        }

    async def _apply_breakout(self, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder breakout strategy implementation."""
        return {
            "signal": SIGNAL_TYPES.NEUTRAL,
            "confidence": 0.5,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "strategy_type": STRATEGY_TYPES.BREAKOUT,
            "indicators": {},
        }

    async def _apply_momentum(self, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder momentum strategy implementation."""
        return {
            "signal": SIGNAL_TYPES.NEUTRAL,
            "confidence": 0.5,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "strategy_type": STRATEGY_TYPES.MOMENTUM,
            "indicators": {},
        }

    async def _apply_volatility(self, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder volatility strategy implementation."""
        return {
            "signal": SIGNAL_TYPES.NEUTRAL,
            "confidence": 0.5,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "strategy_type": STRATEGY_TYPES.VOLATILITY,
            "indicators": {},
        }

    async def _apply_support_resistance(self, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder support/resistance strategy implementation."""
        return {
            "signal": SIGNAL_TYPES.NEUTRAL,
            "confidence": 0.5,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "strategy_type": STRATEGY_TYPES.SUPPORT_RESISTANCE,
            "indicators": {},
        }

    async def _apply_neutral(self, market_data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Return a neutral signal when strategy is unknown."""
        return {
            "signal": SIGNAL_TYPES.NEUTRAL,
            "confidence": 0.5,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "strategy_type": STRATEGY_TYPES.NEUTRAL,
            "indicators": {},
        }
