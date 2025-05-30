#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Regime Brain Module

This module implements the Regime Brain strategy, which adapts trading approaches
based on the current market regime (trending, ranging, volatile, etc.).

The Regime Brain strategy is designed to:
1. Accurately identify the current market regime
2. Apply optimal strategies for the detected regime
3. Anticipate regime transitions for proactive strategy adjustment
4. Maintain multiple regime-specific sub-strategies
5. Adapt parameters based on regime characteristics
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from common.async_utils import gather_with_concurrency
from common.constants import DEFAULT_REGIME_CONFIG, DEFAULT_RISK_PARAMETERS, MARKET_REGIMES, SIGNAL_TYPES, STRATEGY_TYPES, TIMEFRAMES
from common.exceptions import RegimeDetectionError, StrategyExecutionError
from common.logger import get_logger
from common.metrics import MetricsCollector
from common.utils import calculate_drawdown, calculate_sharpe_ratio, get_utc_now
from data_storage.market_data import MarketDataRepository
from feature_service.features.market_structure import MarketStructureAnalyzer
from feature_service.features.pattern import PatternRecognizer
from feature_service.features.volatility import VolatilityFeatureExtractor
from feature_service.multi_timeframe import MultiTimeframeAnalyzer
from intelligence.loophole_detection.market_inefficiency import MarketInefficiencyDetector
from intelligence.pattern_recognition.support_resistance import SupportResistanceDetector
from ml_models.models.classification import ClassificationModel
from ml_models.prediction import PredictionService

from .base_brain import BaseBrain


class MarketRegime(Enum):
    """
    Enum representing different market regimes.
    """

    STRONG_UPTREND = auto()
    WEAK_UPTREND = auto()
    RANGING = auto()
    WEAK_DOWNTREND = auto()
    STRONG_DOWNTREND = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()
    TRANSITIONING = auto()
    ACCUMULATION = auto()
    DISTRIBUTION = auto()
    UNKNOWN = auto()


class RegimeBrain(BaseBrain):
    """
    Specialized brain for market regime-based trading strategies.

    This brain identifies the current market regime and applies the optimal
    strategy for that regime, with smooth transitions between regimes.
    """

    def __init__(
        self,
        asset_id: str,
        exchange_id: str,
        config: Dict[str, Any],
        metrics_collector: MetricsCollector,
        prediction_service: PredictionService = None,
        market_data_repository: MarketDataRepository = None,
        timeframes: List[str] = None,
    ):
        """
        Initialize the RegimeBrain.

        Args:
            asset_id: Identifier for the asset
            exchange_id: Identifier for the exchange
            config: Configuration parameters
            metrics_collector: Metrics collection service
            prediction_service: Service for ML predictions
            market_data_repository: Repository for market data
            timeframes: List of timeframes to analyze
        """
        super().__init__(asset_id=asset_id, exchange_id=exchange_id, config=config, metrics_collector=metrics_collector, brain_type="regime")

        self.logger = get_logger(f"RegimeBrain-{exchange_id}-{asset_id}")

        # Initialize services
        self.prediction_service = prediction_service
        self.market_data_repository = market_data_repository

        # Set default timeframes if not provided
        self.timeframes = timeframes or TIMEFRAMES

        # Initialize regime-related components
        self._initialize_regime_components()

        # Initialize strategy components
        self._initialize_strategy_components()

        # Initialize adaptive parameters
        self._initialize_adaptive_parameters()

        # Last detected regime and confidence
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.regime_history = []
        self.regime_transitions = []

        # Performance metrics by regime
        self.regime_performance = {regime: {"count": 0, "success": 0, "pnl": 0.0} for regime in MarketRegime}

        # Tracking for regime transitions
        self.transition_detected = False
        self.transition_target_regime = None
        self.transition_probability = 0.0

        self.logger.info(f"RegimeBrain initialized for {exchange_id}-{asset_id}")

    def _initialize_regime_components(self):
        """Initialize components for regime detection and analysis."""
        # Analyzers for different aspects of regime detection
        self.volatility_analyzer = VolatilityFeatureExtractor()
        self.market_structure_analyzer = MarketStructureAnalyzer()
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(self.timeframes)
        self.pattern_recognizer = PatternRecognizer()
        self.support_resistance_detector = SupportResistanceDetector()

        # ML model for regime classification
        self.regime_classifier = ClassificationModel(
            model_name="regime_classifier", model_path=self.config.get("regime_classifier_path", "models/regime_classifier")
        )

        # Regime detection parameters
        self.regime_detection_params = self.config.get("regime_detection", DEFAULT_REGIME_CONFIG)

        # Feature importance for regime detection
        self.regime_feature_importance = {
            "price_action": 0.25,
            "volatility": 0.20,
            "volume_profile": 0.15,
            "momentum": 0.15,
            "support_resistance": 0.15,
            "correlation": 0.10,
        }

    def _initialize_strategy_components(self):
        """Initialize regime-specific strategy components."""
        # Strategy selection for each regime
        self.regime_strategies = {
            MarketRegime.STRONG_UPTREND: {
                "primary": STRATEGY_TYPES.TREND_FOLLOWING,
                "secondary": STRATEGY_TYPES.BREAKOUT,
                "weight": {"primary": 0.7, "secondary": 0.3},
            },
            MarketRegime.WEAK_UPTREND: {
                "primary": STRATEGY_TYPES.MOMENTUM,
                "secondary": STRATEGY_TYPES.PULLBACK,
                "weight": {"primary": 0.6, "secondary": 0.4},
            },
            MarketRegime.RANGING: {
                "primary": STRATEGY_TYPES.MEAN_REVERSION,
                "secondary": STRATEGY_TYPES.SUPPORT_RESISTANCE,
                "weight": {"primary": 0.65, "secondary": 0.35},
            },
            MarketRegime.WEAK_DOWNTREND: {
                "primary": STRATEGY_TYPES.MOMENTUM,
                "secondary": STRATEGY_TYPES.PULLBACK,
                "weight": {"primary": 0.6, "secondary": 0.4},
            },
            MarketRegime.STRONG_DOWNTREND: {
                "primary": STRATEGY_TYPES.TREND_FOLLOWING,
                "secondary": STRATEGY_TYPES.BREAKOUT,
                "weight": {"primary": 0.7, "secondary": 0.3},
            },
            MarketRegime.HIGH_VOLATILITY: {
                "primary": STRATEGY_TYPES.VOLATILITY,
                "secondary": STRATEGY_TYPES.OPTION_BASED,
                "weight": {"primary": 0.8, "secondary": 0.2},
            },
            MarketRegime.LOW_VOLATILITY: {
                "primary": STRATEGY_TYPES.STATISTICAL_ARBITRAGE,
                "secondary": STRATEGY_TYPES.CARRY_TRADE,
                "weight": {"primary": 0.75, "secondary": 0.25},
            },
            MarketRegime.TRANSITIONING: {
                "primary": STRATEGY_TYPES.ADAPTIVE,
                "secondary": STRATEGY_TYPES.MOMENTUM,
                "weight": {"primary": 0.8, "secondary": 0.2},
            },
            MarketRegime.ACCUMULATION: {
                "primary": STRATEGY_TYPES.ACCUMULATION,
                "secondary": STRATEGY_TYPES.SUPPORT_RESISTANCE,
                "weight": {"primary": 0.7, "secondary": 0.3},
            },
            MarketRegime.DISTRIBUTION: {
                "primary": STRATEGY_TYPES.DISTRIBUTION,
                "secondary": STRATEGY_TYPES.RESISTANCE_BREAKOUT,
                "weight": {"primary": 0.7, "secondary": 0.3},
            },
            MarketRegime.UNKNOWN: {
                "primary": STRATEGY_TYPES.NEUTRAL,
                "secondary": STRATEGY_TYPES.PROTECTIVE,
                "weight": {"primary": 0.5, "secondary": 0.5},
            },
        }

        # Strategy parameters for each regime
        self.regime_parameters = {
            MarketRegime.STRONG_UPTREND: {
                "risk_factor": 1.2,
                "take_profit_multiplier": 2.0,
                "stop_loss_multiplier": 1.0,
                "entry_aggressiveness": 0.8,
                "exit_aggressiveness": 0.6,
            },
            MarketRegime.WEAK_UPTREND: {
                "risk_factor": 0.9,
                "take_profit_multiplier": 1.5,
                "stop_loss_multiplier": 1.0,
                "entry_aggressiveness": 0.6,
                "exit_aggressiveness": 0.7,
            },
            MarketRegime.RANGING: {
                "risk_factor": 0.8,
                "take_profit_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "entry_aggressiveness": 0.7,
                "exit_aggressiveness": 0.8,
            },
            MarketRegime.WEAK_DOWNTREND: {
                "risk_factor": 0.9,
                "take_profit_multiplier": 1.5,
                "stop_loss_multiplier": 1.0,
                "entry_aggressiveness": 0.6,
                "exit_aggressiveness": 0.7,
            },
            MarketRegime.STRONG_DOWNTREND: {
                "risk_factor": 1.2,
                "take_profit_multiplier": 2.0,
                "stop_loss_multiplier": 1.0,
                "entry_aggressiveness": 0.8,
                "exit_aggressiveness": 0.6,
            },
            MarketRegime.HIGH_VOLATILITY: {
                "risk_factor": 0.7,
                "take_profit_multiplier": 2.5,
                "stop_loss_multiplier": 1.2,
                "entry_aggressiveness": 0.5,
                "exit_aggressiveness": 0.9,
            },
            MarketRegime.LOW_VOLATILITY: {
                "risk_factor": 1.1,
                "take_profit_multiplier": 1.2,
                "stop_loss_multiplier": 0.8,
                "entry_aggressiveness": 0.7,
                "exit_aggressiveness": 0.5,
            },
            MarketRegime.TRANSITIONING: {
                "risk_factor": 0.6,
                "take_profit_multiplier": 1.0,
                "stop_loss_multiplier": 1.2,
                "entry_aggressiveness": 0.4,
                "exit_aggressiveness": 0.8,
            },
            MarketRegime.ACCUMULATION: {
                "risk_factor": 1.0,
                "take_profit_multiplier": 1.8,
                "stop_loss_multiplier": 0.9,
                "entry_aggressiveness": 0.7,
                "exit_aggressiveness": 0.6,
            },
            MarketRegime.DISTRIBUTION: {
                "risk_factor": 0.8,
                "take_profit_multiplier": 1.5,
                "stop_loss_multiplier": 1.1,
                "entry_aggressiveness": 0.6,
                "exit_aggressiveness": 0.8,
            },
            MarketRegime.UNKNOWN: {
                "risk_factor": 0.5,
                "take_profit_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "entry_aggressiveness": 0.3,
                "exit_aggressiveness": 0.9,
            },
        }

        # Initialize market inefficiency detector for regime-specific inefficiencies
        self.inefficiency_detector = MarketInefficiencyDetector()

    def _initialize_adaptive_parameters(self):
        """Initialize adaptive parameters that evolve based on regime performance."""
        # Adaptive parameters
        self.adaptive_params = {
            "regime_detection_sensitivity": 0.7,
            "transition_threshold": 0.6,
            "confidence_threshold": 0.65,
            "historical_weight": 0.3,
            "ml_weight": 0.7,
            "feature_adjustment_rate": 0.05,
            "performance_memory_length": 20,
            "strategy_migration_speed": 0.3,
        }

        # Regime detection performance tracking
        self.regime_detection_accuracy = 0.8  # Initialize with reasonable default
        self.regime_prediction_accuracy = 0.7  # Initialize with reasonable default

        # Parameters for strategy adaptation speed
        self.adaptation_rates = {
            "fast": 0.2,  # Rapid adaptation (e.g., for volatile regimes)
            "medium": 0.1,  # Moderate adaptation
            "slow": 0.05,  # Slow adaptation (e.g., for stable regimes)
        }

    async def detect_regime(self, market_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """
        Detect the current market regime using multiple detection methods.

        Args:
            market_data: DataFrame containing market data

        Returns:
            Tuple with detected regime and confidence score
        """
        try:
            # Multi-method regime detection for robustness
            regime_signals = await gather_with_concurrency(
                10,  # Max concurrent tasks
                self._detect_regime_price_action(market_data),
                self._detect_regime_volatility(market_data),
                self._detect_regime_volume_profile(market_data),
                self._detect_regime_momentum(market_data),
                self._detect_regime_support_resistance(market_data),
                self._detect_regime_ml(market_data),
            )

            # Extract regime predictions and confidence scores
            regimes = [signal[0] for signal in regime_signals]
            confidences = [signal[1] for signal in regime_signals]

            # Calculate weighted regime prediction
            regime_weights = [
                self.regime_feature_importance["price_action"],
                self.regime_feature_importance["volatility"],
                self.regime_feature_importance["volume_profile"],
                self.regime_feature_importance["momentum"],
                self.regime_feature_importance["support_resistance"],
                self.ml_weight,  # ML model gets its own special weight
            ]

            # Normalize weights
            total_weight = sum(regime_weights)
            normalized_weights = [w / total_weight for w in regime_weights]

            # Count occurrences of each regime weighted by confidence
            regime_scores = {regime: 0.0 for regime in MarketRegime}
            for regime, confidence, weight in zip(regimes, confidences, normalized_weights):
                regime_scores[regime] += confidence * weight

            # Select regime with highest score
            detected_regime = max(regime_scores, key=regime_scores.get)
            regime_confidence = regime_scores[detected_regime]

            # Check for regime transitions
            self.transition_detected = False
            if self.current_regime != MarketRegime.UNKNOWN and detected_regime != self.current_regime:
                # A different regime has been detected
                self.transition_detected = True
                self.transition_target_regime = detected_regime
                self.transition_probability = regime_confidence

                self.logger.info(
                    f"Regime transition detected: {self.current_regime} -> {detected_regime} " f"with confidence {regime_confidence:.2f}"
                )

                # Add to transition history
                self.regime_transitions.append(
                    {"timestamp": get_utc_now(), "from_regime": self.current_regime, "to_regime": detected_regime, "confidence": regime_confidence}
                )

            # Update tracking
            if regime_confidence >= self.adaptive_params["confidence_threshold"]:
                # Only update if confidence is sufficient
                self.current_regime = detected_regime
                self.regime_confidence = regime_confidence

                # Add to regime history
                self.regime_history.append({"timestamp": get_utc_now(), "regime": detected_regime, "confidence": regime_confidence})

                # Ensure history doesn't grow unbounded
                if len(self.regime_history) > 100:
                    self.regime_history = self.regime_history[-100:]

                # Log regime detection
                self.logger.info(f"Detected regime: {detected_regime} with confidence {regime_confidence:.2f}")

            # Track metrics
            self.metrics_collector.record_gauge(f"regime_brain.regime_confidence.{self.exchange_id}.{self.asset_id}", regime_confidence)
            self.metrics_collector.record_gauge(f"regime_brain.regime_type.{self.exchange_id}.{self.asset_id}", detected_regime.value)

            return detected_regime, regime_confidence

        except Exception as e:
            self.logger.error(f"Error in regime detection: {str(e)}", exc_info=True)
            raise RegimeDetectionError(f"Failed to detect market regime: {str(e)}")

    async def _detect_regime_price_action(self, market_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect regime based on price action patterns."""
        # Analyze price action patterns like trends, swing highs/lows, etc.
        try:
            # Extract key levels and patterns
            patterns = self.pattern_recognizer.identify_patterns(market_data)

            # Analyze market structure
            structure = self.market_structure_analyzer.analyze(market_data)

            # Determine trend direction and strength
            if structure["trend"]["direction"] == "up":
                if structure["trend"]["strength"] > 0.7:
                    return MarketRegime.STRONG_UPTREND, structure["trend"]["strength"]
                else:
                    return MarketRegime.WEAK_UPTREND, structure["trend"]["strength"]
            elif structure["trend"]["direction"] == "down":
                if structure["trend"]["strength"] > 0.7:
                    return MarketRegime.STRONG_DOWNTREND, structure["trend"]["strength"]
                else:
                    return MarketRegime.WEAK_DOWNTREND, structure["trend"]["strength"]
            elif structure["accumulation"]["probability"] > 0.7:
                return MarketRegime.ACCUMULATION, structure["accumulation"]["probability"]
            elif structure["distribution"]["probability"] > 0.7:
                return MarketRegime.DISTRIBUTION, structure["distribution"]["probability"]
            else:
                # Check for ranging conditions
                range_probability = structure["range"]["probability"]
                if range_probability > 0.6:
                    return MarketRegime.RANGING, range_probability

            # If no strong pattern is detected
            return MarketRegime.UNKNOWN, 0.5

        except Exception as e:
            self.logger.warning(f"Price action regime detection failed: {str(e)}")
            return MarketRegime.UNKNOWN, 0.3

    async def _detect_regime_volatility(self, market_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect regime based on volatility characteristics."""
        try:
            # Extract volatility features
            volatility_features = self.volatility_analyzer.extract_features(market_data)

            # Determine if we're in a high/low volatility regime
            if volatility_features["is_high_volatility"]:
                return MarketRegime.HIGH_VOLATILITY, volatility_features["volatility_confidence"]
            elif volatility_features["is_low_volatility"]:
                return MarketRegime.LOW_VOLATILITY, volatility_features["volatility_confidence"]

            # If volatility is transitioning
            if volatility_features["is_expanding"]:
                # Expanding volatility often precedes trends
                return MarketRegime.TRANSITIONING, volatility_features["volatility_direction_confidence"]

            # Look at trend direction in combination with volatility
            if volatility_features["trend_direction"] == "up":
                if volatility_features["volatility_relative_to_history"] > 1.5:
                    return MarketRegime.STRONG_UPTREND, volatility_features["trend_confidence"]
                else:
                    return MarketRegime.WEAK_UPTREND, volatility_features["trend_confidence"]
            elif volatility_features["trend_direction"] == "down":
                if volatility_features["volatility_relative_to_history"] > 1.5:
                    return MarketRegime.STRONG_DOWNTREND, volatility_features["trend_confidence"]
                else:
                    return MarketRegime.WEAK_DOWNTREND, volatility_features["trend_confidence"]

            # No strong volatility signal
            return MarketRegime.UNKNOWN, 0.4

        except Exception as e:
            self.logger.warning(f"Volatility regime detection failed: {str(e)}")
            return MarketRegime.UNKNOWN, 0.3

    async def _detect_regime_volume_profile(self, market_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect regime based on volume profile analysis."""
        try:
            # For comprehensive volume profile analysis, we need OHLCV data
            if "volume" not in market_data.columns:
                return MarketRegime.UNKNOWN, 0.3

            # Analyze volume distribution
            volume_data = market_data[["open", "high", "low", "close", "volume"]].copy()

            # Calculate volume-weighted average price
            volume_data["vwap"] = (volume_data["close"] * volume_data["volume"]).cumsum() / volume_data["volume"].cumsum()

            # Calculate volume momentum
            volume_data["volume_sma"] = volume_data["volume"].rolling(window=20).mean()
            volume_data["volume_momentum"] = volume_data["volume"] / volume_data["volume_sma"]

            # Identify accumulation and distribution periods
            volume_data["price_change"] = volume_data["close"].pct_change()
            volume_data["volume_price_correlation"] = (
                (volume_data["volume_momentum"] * np.sign(volume_data["price_change"])).rolling(window=10).mean()
            )

            # Get recent data
            recent_data = volume_data.dropna().iloc[-20:]

            # Detect accumulation (high volume, price not dropping significantly)
            if (
                recent_data["volume_momentum"].mean() > 1.2
                and recent_data["price_change"].mean() > -0.001
                and recent_data["close"].iloc[-1] < recent_data["vwap"].iloc[-1]
            ):
                return MarketRegime.ACCUMULATION, 0.7

            # Detect distribution (high volume, price not rising significantly)
            if (
                recent_data["volume_momentum"].mean() > 1.2
                and recent_data["price_change"].mean() < 0.001
                and recent_data["close"].iloc[-1] > recent_data["vwap"].iloc[-1]
            ):
                return MarketRegime.DISTRIBUTION, 0.7

            # Detect trending regimes based on volume and price alignment
            vol_price_corr = recent_data["volume_price_correlation"].mean()

            if vol_price_corr > 0.4:
                # Positive correlation - volume confirms price movement
                if recent_data["price_change"].mean() > 0:
                    return MarketRegime.STRONG_UPTREND, abs(vol_price_corr)
                else:
                    return MarketRegime.STRONG_DOWNTREND, abs(vol_price_corr)
            elif vol_price_corr < -0.4:
                # Negative correlation - volume and price diverging
                return MarketRegime.TRANSITIONING, abs(vol_price_corr)
            else:
                # No strong volume signal
                if recent_data["volume_momentum"].std() < 0.3:
                    # Low volume volatility suggests ranging
                    return MarketRegime.RANGING, 0.6

            # Default case
            return MarketRegime.UNKNOWN, 0.4

        except Exception as e:
            self.logger.warning(f"Volume profile regime detection failed: {str(e)}")
            return MarketRegime.UNKNOWN, 0.3

    async def _detect_regime_momentum(self, market_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect regime based on momentum indicators."""
        try:
            # Calculate momentum indicators
            close = market_data["close"]

            # Calculate RSI
            delta = close.diff()
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0

            avg_gain = up.rolling(window=14).mean()
            avg_loss = abs(down.rolling(window=14).mean())

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Calculate MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            # Get recent values
            recent_rsi = rsi.dropna().iloc[-5:].mean()
            recent_histogram = histogram.dropna().iloc[-5:].mean()
            histogram_slope = histogram.dropna().diff().iloc[-5:].mean()

            # Detect regimes based on momentum
            if recent_rsi > 70:
                if recent_histogram > 0 and histogram_slope > 0:
                    return MarketRegime.STRONG_UPTREND, 0.8
                elif recent_histogram > 0 and histogram_slope < 0:
                    return MarketRegime.WEAK_UPTREND, 0.7
                else:
                    return MarketRegime.DISTRIBUTION, 0.6
            elif recent_rsi < 30:
                if recent_histogram < 0 and histogram_slope < 0:
                    return MarketRegime.STRONG_DOWNTREND, 0.8
                elif recent_histogram < 0 and histogram_slope > 0:
                    return MarketRegime.WEAK_DOWNTREND, 0.7
                else:
                    return MarketRegime.ACCUMULATION, 0.6
            else:
                # RSI in middle range
                if abs(recent_rsi - 50) < 10:
                    return MarketRegime.RANGING, 0.6
                elif histogram_slope > 0 and recent_histogram < 0:
                    # Momentum shifting from negative to positive
                    return MarketRegime.TRANSITIONING, 0.7
                elif histogram_slope < 0 and recent_histogram > 0:
                    # Momentum shifting from positive to negative
                    return MarketRegime.TRANSITIONING, 0.7

            # No strong momentum signal
            return MarketRegime.UNKNOWN, 0.4

        except Exception as e:
            self.logger.warning(f"Momentum regime detection failed: {str(e)}")
            return MarketRegime.UNKNOWN, 0.3

    async def _detect_regime_support_resistance(self, market_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect regime based on support and resistance interactions."""
        try:
            # Detect key support and resistance levels
            levels = self.support_resistance_detector.detect_levels(market_data)

            if not levels:
                return MarketRegime.UNKNOWN, 0.3

            # Get current price
            current_price = market_data["close"].iloc[-1]

            # Find nearest levels
            supports = [level for level in levels if level < current_price]
            resistances = [level for level in levels if level > current_price]

            nearest_support = max(supports) if supports else None
            nearest_resistance = min(resistances) if resistances else None

            # Calculate price relative to nearest levels
            if nearest_support and nearest_resistance:
                range_size = nearest_resistance - nearest_support
                if range_size <= 0:
                    return MarketRegime.UNKNOWN, 0.3

                position_in_range = (current_price - nearest_support) / range_size

                # Check if price is respecting a range
                if 0.1 <= position_in_range <= 0.9:
                    # Price within established range
                    if len(levels) >= 5:  # Multiple levels suggest a ranging market
                        range_quality = min(1.0, len(levels) / 10)  # More levels = higher quality range
                        return MarketRegime.RANGING, 0.5 + (range_quality * 0.4)

                # Check for breakout/breakdown
                price_history = market_data["close"].iloc[-30:].values
                resistance_touches = sum(1 for p in price_history if 0.99 <= p / nearest_resistance <= 1.01)
                support_touches = sum(1 for p in price_history if 0.99 <= p / nearest_support <= 1.01)

                # Recent breakout above resistance
                if current_price > nearest_resistance and resistance_touches >= 3:
                    return MarketRegime.STRONG_UPTREND, 0.7

                # Recent breakdown below support
                if current_price < nearest_support and support_touches >= 3:
                    return MarketRegime.STRONG_DOWNTREND, 0.7

                # Price near bottom of range - possible accumulation
                if position_in_range < 0.2 and support_touches >= 2:
                    return MarketRegime.ACCUMULATION, 0.6

                # Price near top of range - possible distribution
                if position_in_range > 0.8 and resistance_touches >= 2:
                    return MarketRegime.DISTRIBUTION, 0.6

            # Multiple failed attempts to break resistance
            resistance_fails = sum(
                1 for i in range(-10, 0) if i < -1 and market_data["high"].iloc[i] > nearest_resistance > market_data["close"].iloc[i]
            )
            if resistance_fails >= 2:
                return MarketRegime.WEAK_UPTREND, 0.6

            # Multiple failed attempts to break support
            support_fails = sum(1 for i in range(-10, 0) if i < -1 and market_data["low"].iloc[i] < nearest_support < market_data["close"].iloc[i])
            if support_fails >= 2:
                return MarketRegime.WEAK_DOWNTREND, 0.6

            # No strong S/R signal
            return MarketRegime.UNKNOWN, 0.4

        except Exception as e:
            self.logger.warning(f"Support/Resistance regime detection failed: {str(e)}")
            return MarketRegime.UNKNOWN, 0.3

    async def _detect_regime_ml(self, market_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect regime using machine learning model."""
        try:
            # Prepare features for the ML model
            features = self._prepare_regime_features(market_data)

            # Use prediction service for regime classification
            if self.prediction_service:
                prediction_result = await self.prediction_service.predict(model_name="regime_classifier", features=features)

                # Extract predicted regime and probability
                regime_idx = prediction_result["prediction"]
                confidence = prediction_result["probability"]

                # Map index to regime enum
                regime_mapping = {
                    0: MarketRegime.STRONG_UPTREND,
                    1: MarketRegime.WEAK_UPTREND,
                    2: MarketRegime.RANGING,
                    3: MarketRegime.WEAK_DOWNTREND,
                    4: MarketRegime.STRONG_DOWNTREND,
                    5: MarketRegime.HIGH_VOLATILITY,
                    6: MarketRegime.LOW_VOLATILITY,
                    7: MarketRegime.TRANSITIONING,
                    8: MarketRegime.ACCUMULATION,
                    9: MarketRegime.DISTRIBUTION,
                }

                predicted_regime = regime_mapping.get(regime_idx, MarketRegime.UNKNOWN)

                return predicted_regime, confidence
            else:
                # If prediction service not available, use direct model
                if hasattr(self, "regime_classifier"):
                    # Use the local classifier model
                    prediction = self.regime_classifier.predict(features)
                    regime_idx = prediction[0]
                    probability = self.regime_classifier.predict_proba(features)[0][regime_idx]

                    # Map index to regime enum (same mapping as above)
                    regime_mapping = {
                        0: MarketRegime.STRONG_UPTREND,
                        1: MarketRegime.WEAK_UPTREND,
                        2: MarketRegime.RANGING,
                        3: MarketRegime.WEAK_DOWNTREND,
                        4: MarketRegime.STRONG_DOWNTREND,
                        5: MarketRegime.HIGH_VOLATILITY,
                        6: MarketRegime.LOW_VOLATILITY,
                        7: MarketRegime.TRANSITIONING,
                        8: MarketRegime.ACCUMULATION,
                        9: MarketRegime.DISTRIBUTION,
                    }

                    predicted_regime = regime_mapping.get(regime_idx, MarketRegime.UNKNOWN)

                    return predicted_regime, float(probability)

            # Default fallback
            return MarketRegime.UNKNOWN, 0.3

        except Exception as e:
            self.logger.warning(f"ML regime detection failed: {str(e)}")
            return MarketRegime.UNKNOWN, 0.3

    def _prepare_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for the regime classifier model."""
        try:
            # Extract relevant columns
            df = market_data[["open", "high", "low", "close"]].copy()
            if "volume" in market_data.columns:
                df["volume"] = market_data["volume"]

            # Calculate technical features

            # Trend indicators
            df["sma20"] = df["close"].rolling(window=20).mean()
            df["sma50"] = df["close"].rolling(window=50).mean()
            df["sma200"] = df["close"].rolling(window=200).mean()

            df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
            df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

            # Compute MACD
            df["macd"] = df["ema12"] - df["ema26"]
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]

            # Compute RSI
            delta = df["close"].diff()
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0

            roll_up = up.rolling(window=14).mean()
            roll_down = abs(down.rolling(window=14).mean())

            rs = roll_up / roll_down
            df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

            # Volatility indicators
            df["atr"] = self._calculate_atr(df, 14)
            df["bb_width"] = self._calculate_bollinger_bandwidth(df, 20, 2)

            # Price relative to moving averages
            df["price_to_sma20"] = df["close"] / df["sma20"]
            df["price_to_sma50"] = df["close"] / df["sma50"]
            df["price_to_sma200"] = df["close"] / df["sma200"]

            # Moving average slopes
            df["sma20_slope"] = df["sma20"].diff(5) / df["sma20"].shift(5)
            df["sma50_slope"] = df["sma50"].diff(10) / df["sma50"].shift(10)

            # Momentum indicators
            df["roc5"] = df["close"].pct_change(5)
            df["roc10"] = df["close"].pct_change(10)
            df["roc20"] = df["close"].pct_change(20)

            # Volume indicators (if available)
            if "volume" in df.columns:
                df["volume_sma20"] = df["volume"].rolling(window=20).mean()
                df["volume_ratio"] = df["volume"] / df["volume_sma20"]
                df["obv"] = self._calculate_obv(df)

            # Drop NaN values
            df = df.dropna()

            # Select the most recent data point
            features = df.iloc[-1].values

            # Return as numpy array reshaped for model input
            return features.reshape(1, -1)

        except Exception as e:
            self.logger.error(f"Error preparing regime features: {str(e)}")
            # Return a zero vector of appropriate size as fallback
            return np.zeros((1, 20))  # Adjust size as needed

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def _calculate_bollinger_bandwidth(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Calculate Bollinger Bandwidth."""
        sma = df["close"].rolling(window=period).mean()
        rolling_std = df["close"].rolling(window=period).std()

        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)

        bandwidth = (upper_band - lower_band) / sma

        return bandwidth

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index)
        obv.iloc[0] = 0

        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    async def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate regime-based trading signals.

        Args:
            market_data: DataFrame containing market data

        Returns:
            Dictionary with analysis results
        """
        try:
            self.logger.info(f"Analyzing {self.exchange_id}-{self.asset_id} with RegimeBrain")

            start_time = time.time()

            # Detect current market regime
            regime, confidence = await self.detect_regime(market_data)

            # Check if confidence is sufficient
            if confidence < self.adaptive_params["confidence_threshold"]:
                self.logger.info(f"Regime confidence too low: {confidence:.2f}, using previous regime or UNKNOWN")
                if self.current_regime == MarketRegime.UNKNOWN:
                    # No previous regime, use UNKNOWN with neutral strategy
                    regime = MarketRegime.UNKNOWN
                    self.logger.info(f"Using UNKNOWN regime with neutral strategy")
                else:
                    # Use the previous regime
                    regime = self.current_regime
                    self.logger.info(f"Using previous regime: {regime}")

            # Get regime-specific strategies and parameters
            strategies = self.regime_strategies[regime]
            parameters = self.regime_parameters[regime]

            # Adjust parameters based on confidence
            adjusted_parameters = self._adjust_parameters_by_confidence(parameters, confidence)

            # Select appropriate strategy for this regime
            primary_strategy = strategies["primary"]
            secondary_strategy = strategies["secondary"]
            strategy_weights = strategies["weight"]

            # Apply strategies to generate signals
            primary_signals = await self._apply_strategy(primary_strategy, market_data, adjusted_parameters)
            secondary_signals = await self._apply_strategy(secondary_strategy, market_data, adjusted_parameters)

            # Combine signals according to weights
            combined_signals = self._combine_signals(primary_signals, secondary_signals, strategy_weights["primary"], strategy_weights["secondary"])

            # Add regime-specific insights
            regime_insights = await self._generate_regime_insights(regime, market_data)

            # Prepare result
            analysis_result = {
                "regime": regime.name,
                "confidence": confidence,
                "signals": combined_signals,
                "parameters": adjusted_parameters,
                "insights": regime_insights,
                "strategies_used": {"primary": primary_strategy, "secondary": secondary_strategy, "weights": strategy_weights},
                "transition": {
                    "detected": self.transition_detected,
                    "target_regime": self.transition_target_regime.name if self.transition_detected else None,
                    "probability": self.transition_probability if self.transition_detected else 0.0,
                },
            }

            # Track metrics
            execution_time = time.time() - start_time
            self.metrics_collector.record_timing(
                f"regime_brain.analysis_time.{self.exchange_id}.{self.asset_id}", execution_time * 1000  # Convert to ms
            )

            return analysis_result

        except Exception as e:
            self.logger.error(f"Error in RegimeBrain analysis: {str(e)}", exc_info=True)
            raise StrategyExecutionError(f"RegimeBrain analysis failed: {str(e)}")

    def _adjust_parameters_by_confidence(self, parameters: Dict[str, float], confidence: float) -> Dict[str, float]:
        """Adjust strategy parameters based on regime confidence."""
        # Clone parameters
        adjusted = parameters.copy()

        # Confidence scaling factor (reduces parameter values when confidence is lower)
        scaling = 0.5 + (0.5 * confidence)  # scales from 0.5 to 1.0

        # Adjust aggressiveness parameters
        adjusted["entry_aggressiveness"] *= scaling
        adjusted["exit_aggressiveness"] *= scaling

        # Adjust risk parameters more conservatively
        risk_scaling = 0.3 + (0.7 * confidence)  # more conservative scaling from 0.3 to 1.0
        adjusted["risk_factor"] *= risk_scaling

        # If very low confidence, increase stop_loss_multiplier (tighter stops)
        if confidence < 0.5:
            adjusted["stop_loss_multiplier"] *= 1.0 + (0.5 - confidence)

        return adjusted

    async def _apply_strategy(self, strategy_type: str, market_data: pd.DataFrame, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Apply a specific strategy to generate trading signals."""
        try:
            # Different strategies based on the strategy type
            if strategy_type == STRATEGY_TYPES.TREND_FOLLOWING:
                return await self._apply_trend_following(market_data, parameters)
            elif strategy_type == STRATEGY_TYPES.MEAN_REVERSION:
                return await self._apply_mean_reversion(market_data, parameters)
            elif strategy_type == STRATEGY_TYPES.BREAKOUT:
                return await self._apply_breakout(market_data, parameters)
            elif strategy_type == STRATEGY_TYPES.MOMENTUM:
                return await self._apply_momentum(market_data, parameters)
            elif strategy_type == STRATEGY_TYPES.VOLATILITY:
                return await self._apply_volatility(market_data, parameters)
            elif strategy_type == STRATEGY_TYPES.SUPPORT_RESISTANCE:
                return await self._apply_support_resistance(market_data, parameters)
            elif strategy_type == STRATEGY_TYPES.NEUTRAL:
                return await self._apply_neutral(market_data, parameters)
            else:
                # Default to neutral strategy
                return await self._apply_neutral(market_data, parameters)

        except Exception as e:
            self.logger.error(f"Error applying strategy {strategy_type}: {str(e)}")
            # Return neutral signals as fallback
            return {
                "signal": SIGNAL_TYPES.NEUTRAL,
                "confidence": 0.3,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "strategy_type": strategy_type,
                "error": str(e),
            }

    async def _apply_trend_following(self, market_data: pd.DataFrame, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Apply trend following strategy."""
        # Implement trend following logic here
        # This is a simplified example - production code would be more sophisticated
        try:
            # Calculate moving averages
            short_ma = market_data["close"].rolling(window=20).mean()
            long_ma = market_data["close"].rolling(window=50).mean()

            # Determine trend direction
            current_short = short_ma.iloc[-1]
            current_long = long_ma.iloc[-1]

            # Calculate additional trend strength indicators
            adx_value = self._calculate_adx(market_data, 14)

            current_price = market_data["close"].iloc[-1]

            # Generate signal
            signal = SIGNAL_TYPES.NEUTRAL
            confidence = 0.5
            entry_price = None
            stop_loss = None
            take_profit = None

            if current_short > current_long and adx_value > 20:
                # Uptrend
                signal = SIGNAL_TYPES.LONG
                confidence = min(0.5 + (adx_value / 100), 0.9)
                entry_price = current_price
                stop_loss = current_price * (1 - 0.02 * parameters["stop_loss_multiplier"])
                take_profit = current_price * (1 + 0.04 * parameters["take_profit_multiplier"])
            elif current_short < current_long and adx_value > 20:
                # Downtrend
                signal = SIGNAL_TYPES.SHORT
                confidence = min(0.5 + (adx_value / 100), 0.9)
                entry_price = current_price
                stop_loss = current_price * (1 + 0.02 * parameters["stop_loss_multiplier"])
                take_profit = current_price * (1 - 0.04 * parameters["take_profit_multiplier"])

            # Adjust entry aggressiveness
            if entry_price is not None:
                entry_adjustment = parameters["entry_aggressiveness"] * 0.005
                if signal == SIGNAL_TYPES.LONG:
                    entry_price *= 1 + entry_adjustment
                elif signal == SIGNAL_TYPES.SHORT:
                    entry_price *= 1 - entry_adjustment

            return {
                "signal": signal,
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy_type": STRATEGY_TYPES.TREND_FOLLOWING,
                "indicators": {"short_ma": current_short, "long_ma": current_long, "adx": adx_value},
            }

        except Exception as e:
            self.logger.error(f"Error in trend following strategy: {str(e)}")
            return {
                "signal": SIGNAL_TYPES.NEUTRAL,
                "confidence": 0.3,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "strategy_type": STRATEGY_TYPES.TREND_FOLLOWING,
                "error": str(e),
            }

    async def _apply_mean_reversion(self, market_data: pd.DataFrame, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Apply mean reversion strategy."""
        try:
            # Calculate Bollinger Bands
            window = 20
            std_dev = 2

            rolling_mean = market_data["close"].rolling(window=window).mean()
            rolling_std = market_data["close"].rolling(window=window).std()

            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)

            # Calculate RSI
            delta = market_data["close"].diff()
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0

            roll_up = up.rolling(window=14).mean()
            roll_down = abs(down.rolling(window=14).mean())

            rs = roll_up / roll_down
            rsi = 100.0 - (100.0 / (1.0 + rs))

            # Get current values
            current_price = market_data["close"].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_mean = rolling_mean.iloc[-1]
            current_rsi = rsi.iloc[-1]

            # Generate signal
            signal = SIGNAL_TYPES.NEUTRAL
            confidence = 0.5
            entry_price = None
            stop_loss = None
            take_profit = None

            # Check for overbought/oversold conditions
            if current_price > current_upper and current_rsi > 70:
                # Overbought - potential short
                signal = SIGNAL_TYPES.SHORT
                confidence = 0.6 + (0.3 * min((current_rsi - 70) / 30, 1.0))
                entry_price = current_price
                stop_loss = current_price * (1 + 0.02 * parameters["stop_loss_multiplier"])
                take_profit = current_mean
            elif current_price < current_lower and current_rsi < 30:
                # Oversold - potential long
                signal = SIGNAL_TYPES.LONG
                confidence = 0.6 + (0.3 * min((30 - current_rsi) / 30, 1.0))
                entry_price = current_price
                stop_loss = current_price * (1 - 0.02 * parameters["stop_loss_multiplier"])
                take_profit = current_mean

            # Adjust entry aggressiveness
            if entry_price is not None:
                entry_adjustment = parameters["entry_aggressiveness"] * 0.005
                if signal == SIGNAL_TYPES.LONG:
                    entry_price *= 1 + entry_adjustment
                elif signal == SIGNAL_TYPES.SHORT:
                    entry_price *= 1 - entry_adjustment

            return {
                "signal": signal,
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy_type": STRATEGY_TYPES.MEAN_REVERSION,
                "indicators": {
                    "bollinger_upper": current_upper,
                    "bollinger_lower": current_lower,
                    "bollinger_mean": current_mean,
                    "rsi": current_rsi,
                },
            }

        except Exception as e:
            self.logger.error(f"Error in mean reversion strategy: {str(e)}")
            return {
                "signal": SIGNAL_TYPES.NEUTRAL,
                "confidence": 0.3,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "strategy_type": STRATEGY_TYPES.MEAN_REVERSION,
                "error": str(e),
            }

    async def _apply_breakout(self, market_data: pd.DataFrame, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Apply breakout strategy."""
        # Implement breakout strategy logic (placeholder)
        try:
            # Get price data
            high = market_data["high"]
            low = market_data["low"]
            close = market_data["close"]

            # Identify recent highs and lows (last 20 periods)
            recent_high = high.iloc[-20:].max()
            recent_low = low.iloc[-20:].min()

            # Current price
            current_price = close.iloc[-1]

            # Calculate price volatility (ATR)
            atr = self._calculate_atr(market_data, 14).iloc[-1]

            # Generate signal
            signal = SIGNAL_TYPES.NEUTRAL
            confidence = 0.5
            entry_price = None
            stop_loss = None
            take_profit = None

            # Check for breakouts
            # High breakout
            if current_price > recent_high:
                # Calculate strength of breakout
                breakout_strength = (current_price - recent_high) / atr

                if breakout_strength > 0.5:
                    signal = SIGNAL_TYPES.LONG
                    confidence = min(0.6 + (0.3 * breakout_strength / 2), 0.9)
                    entry_price = current_price
                    stop_loss = max(recent_high - (atr * parameters["stop_loss_multiplier"]), current_price * 0.97)
                    take_profit = current_price + (atr * 3 * parameters["take_profit_multiplier"])

            # Low breakout
            elif current_price < recent_low:
                # Calculate strength of breakout
                breakout_strength = (recent_low - current_price) / atr

                if breakout_strength > 0.5:
                    signal = SIGNAL_TYPES.SHORT
                    confidence = min(0.6 + (0.3 * breakout_strength / 2), 0.9)
                    entry_price = current_price
                    stop_loss = min(recent_low + (atr * parameters["stop_loss_multiplier"]), current_price * 1.03)
                    take_profit = current_price - (atr * 3 * parameters["take_profit_multiplier"])

            # Adjust entry aggressiveness
            if entry_price is not None:
                entry_adjustment = parameters["entry_aggressiveness"] * 0.005
                if signal == SIGNAL_TYPES.LONG:
                    entry_price *= 1 + entry_adjustment
                elif signal == SIGNAL_TYPES.SHORT:
                    entry_price *= 1 - entry_adjustment

            return {
                "signal": signal,
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy_type": STRATEGY_TYPES.BREAKOUT,
                "indicators": {"recent_high": recent_high, "recent_low": recent_low, "atr": atr},
            }

        except Exception as e:
            self.logger.error(f"Error in breakout strategy: {str(e)}")
            return {
                "signal": SIGNAL_TYPES.NEUTRAL,
                "confidence": 0.3,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "strategy_type": STRATEGY_TYPES.BREAKOUT,
                "error": str(e),
            }

    async def _apply_momentum(self, market_data: pd.DataFrame, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Apply momentum strategy."""
        try:
            # Calculate momentum indicators
            close = market_data["close"]

            # Rate of Change
            roc10 = close.pct_change(10)

            # Calculate MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            # RSI
            delta = close.diff()
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0

            roll_up = up.rolling(window=14).mean()
            roll_down = abs(down.rolling(window=14).mean())

            rs = roll_up / roll_down
            rsi = 100.0 - (100.0 / (1.0 + rs))

            # Current values
            current_price = close.iloc[-1]
            current_roc = roc10.iloc[-1]
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]
            current_rsi = rsi.iloc[-1]

            # Previous values
            prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else 0

            # Generate signal
            signal = SIGNAL_TYPES.NEUTRAL
            confidence = 0.5
            entry_price = None
            stop_loss = None
            take_profit = None

            # Momentum long conditions
            if current_roc > 0.02 and current_macd > current_signal and current_histogram > prev_histogram and current_rsi > 50:

                # Strong momentum to the upside
                signal = SIGNAL_TYPES.LONG
                momentum_strength = (current_roc + (current_histogram / current_price)) * 10
                confidence = min(0.6 + (0.3 * momentum_strength), 0.9)

                entry_price = current_price
                stop_loss = current_price * (1 - 0.02 * parameters["stop_loss_multiplier"])
                take_profit = current_price * (1 + 0.04 * parameters["take_profit_multiplier"])

            # Momentum short conditions
            elif current_roc < -0.02 and current_macd < current_signal and current_histogram < prev_histogram and current_rsi < 50:

                # Strong momentum to the downside
                signal = SIGNAL_TYPES.SHORT
                momentum_strength = (abs(current_roc) + (abs(current_histogram) / current_price)) * 10
                confidence = min(0.6 + (0.3 * momentum_strength), 0.9)

                entry_price = current_price
                stop_loss = current_price * (1 + 0.02 * parameters["stop_loss_multiplier"])
                take_profit = current_price * (1 - 0.04 * parameters["take_profit_multiplier"])

            # Adjust entry aggressiveness
            if entry_price is not None:
                entry_adjustment = parameters["entry_aggressiveness"] * 0.005
                if signal == SIGNAL_TYPES.LONG:
                    entry_price *= 1 + entry_adjustment
                elif signal == SIGNAL_TYPES.SHORT:
                    entry_price *= 1 - entry_adjustment

            return {
                "signal": signal,
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy_type": STRATEGY_TYPES.MOMENTUM,
                "indicators": {
                    "roc10": current_roc,
                    "macd": current_macd,
                    "macd_signal": current_signal,
                    "macd_histogram": current_histogram,
                    "rsi": current_rsi,
                },
            }

        except Exception as e:
            self.logger.error(f"Error in momentum strategy: {str(e)}")
            return {
                "signal": SIGNAL_TYPES.NEUTRAL,
                "confidence": 0.3,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "strategy_type": STRATEGY_TYPES.MOMENTUM,
                "error": str(e),
            }

    async def _apply_volatility(self, market_data: pd.DataFrame, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Apply volatility-based strategy."""
        try:
            # Get price data
            high = market_data["high"]
            low = market_data["low"]
            close = market_data["close"]

            # Calculate ATR
            atr = self._calculate_atr(market_data, 14)

            # Calculate Bollinger Bands
            window = 20
            rolling_mean = close.rolling(window=window).mean()
            rolling_std = close.rolling(window=window).std()

            upper_band = rolling_mean + (2 * rolling_std)
            lower_band = rolling_mean - (2 * rolling_std)

            # Calculate volatility ratio (current vs historical)
            current_atr = atr.iloc[-1]
            avg_atr = atr.iloc[-20:].mean()
            volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

            # Current values
            current_price = close.iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_mean = rolling_mean.iloc[-1]

            # Generate signal
            signal = SIGNAL_TYPES.NEUTRAL
            confidence = 0.5
            entry_price = None
            stop_loss = None
            take_profit = None

            # Volatility expansion strategy
            if volatility_ratio > 1.3:
                # High volatility environment - look for breakouts
                if current_price > current_upper:
                    # Upside breakout in high volatility
                    signal = SIGNAL_TYPES.LONG
                    confidence = min(0.6 + (0.3 * (volatility_ratio - 1)), 0.9)

                    entry_price = current_price
                    stop_loss = current_price * (1 - (current_atr / current_price) * parameters["stop_loss_multiplier"])
                    take_profit = current_price * (1 + (current_atr / current_price) * 2 * parameters["take_profit_multiplier"])

                elif current_price < current_lower:
                    # Downside breakout in high volatility
                    signal = SIGNAL_TYPES.SHORT
                    confidence = min(0.6 + (0.3 * (volatility_ratio - 1)), 0.9)

                    entry_price = current_price
                    stop_loss = current_price * (1 + (current_atr / current_price) * parameters["stop_loss_multiplier"])
                    take_profit = current_price * (1 - (current_atr / current_price) * 2 * parameters["take_profit_multiplier"])

            # Volatility contraction strategy
            elif volatility_ratio < 0.7:
                # Low and decreasing volatility - prepare for expansion
                # This is a more speculative signal
                bb_width = (current_upper - current_lower) / current_mean
                if bb_width < 0.03:  # Very narrow bands
                    # Coiled spring setup - can break either way
                    # Use other indicators to determine direction
                    if current_price > current_mean:
                        signal = SIGNAL_TYPES.LONG
                        confidence = 0.6
                    else:
                        signal = SIGNAL_TYPES.SHORT
                        confidence = 0.6

                    entry_price = current_price
                    # Tight stops due to low volatility
                    stop_factor = 1.5 * parameters["stop_loss_multiplier"]
                    take_factor = 3 * parameters["take_profit_multiplier"]

                    if signal == SIGNAL_TYPES.LONG:
                        stop_loss = current_price * (1 - (current_atr / current_price) * stop_factor)
                        take_profit = current_price * (1 + (current_atr / current_price) * take_factor)
                    else:
                        stop_loss = current_price * (1 + (current_atr / current_price) * stop_factor)
                        take_profit = current_price * (1 - (current_atr / current_price) * take_factor)

            # Adjust entry aggressiveness
            if entry_price is not None:
                entry_adjustment = parameters["entry_aggressiveness"] * 0.005
                if signal == SIGNAL_TYPES.LONG:
                    entry_price *= 1 + entry_adjustment
                elif signal == SIGNAL_TYPES.SHORT:
                    entry_price *= 1 - entry_adjustment

            return {
                "signal": signal,
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy_type": STRATEGY_TYPES.VOLATILITY,
                "indicators": {
                    "atr": current_atr,
                    "volatility_ratio": volatility_ratio,
                    "bollinger_upper": current_upper,
                    "bollinger_lower": current_lower,
                },
            }

        except Exception as e:
            self.logger.error(f"Error in volatility strategy: {str(e)}")
            return {
                "signal": SIGNAL_TYPES.NEUTRAL,
                "confidence": 0.3,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "strategy_type": STRATEGY_TYPES.VOLATILITY,
                "error": str(e),
            }

    async def _apply_support_resistance(self, market_data: pd.DataFrame, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Apply support and resistance based strategy."""
        try:
            # Detect key support and resistance levels
            levels = self.support_resistance_detector.detect_levels(market_data)

            # Get current price
            current_price = market_data["close"].iloc[-1]

            # Calculate ATR for stop loss placement
            atr = self._calculate_atr(market_data, 14).iloc[-1]

            # Find nearest levels
            supports = sorted([level for level in levels if level < current_price])
            resistances = sorted([level for level in levels if level > current_price])

            nearest_support = supports[-1] if supports else None
            nearest_resistance = resistances[0] if resistances else None

            # Generate signal
            signal = SIGNAL_TYPES.NEUTRAL
            confidence = 0.5
            entry_price = None
            stop_loss = None
            take_profit = None

            # Check if price is near support or resistance
            if nearest_support and (current_price - nearest_support) / current_price < 0.01:
                # Price is very close to support - potential long
                signal = SIGNAL_TYPES.LONG
                distance_factor = 1 - ((current_price - nearest_support) / current_price)
                confidence = min(0.6 + (0.3 * distance_factor), 0.9)

                entry_price = current_price
                # Place stop just below support
                stop_loss = nearest_support - (0.2 * atr * parameters["stop_loss_multiplier"])

                # Target next resistance or risk:reward ratio
                if resistances and len(resistances) > 0:
                    take_profit = resistances[0]
                else:
                    take_profit = current_price + (3 * (current_price - stop_loss))

            elif nearest_resistance and (nearest_resistance - current_price) / current_price < 0.01:
                # Price is very close to resistance - potential short
                signal = SIGNAL_TYPES.SHORT
                distance_factor = 1 - ((nearest_resistance - current_price) / current_price)
                confidence = min(0.6 + (0.3 * distance_factor), 0.9)

                entry_price = current_price
                # Place stop just above resistance
                stop_loss = nearest_resistance + (0.2 * atr * parameters["stop_loss_multiplier"])

                # Target next support or risk:reward ratio
                if supports and len(supports) > 0:
                    take_profit = supports[-1]
                else:
                    take_profit = current_price - (3 * (stop_loss - current_price))

            # Adjust entry aggressiveness
            if entry_price is not None:
                entry_adjustment = parameters["entry_aggressiveness"] * 0.005
                if signal == SIGNAL_TYPES.LONG:
                    entry_price *= 1 + entry_adjustment
                elif signal == SIGNAL_TYPES.SHORT:
                    entry_price *= 1 - entry_adjustment

            return {
                "signal": signal,
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "strategy_type": STRATEGY_TYPES.SUPPORT_RESISTANCE,
                "indicators": {"nearest_support": nearest_support, "nearest_resistance": nearest_resistance, "atr": atr, "levels": levels},
            }

        except Exception as e:
            self.logger.error(f"Error in support/resistance strategy: {str(e)}")
            return {
                "signal": SIGNAL_TYPES.NEUTRAL,
                "confidence": 0.3,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "strategy_type": STRATEGY_TYPES.SUPPORT_RESISTANCE,
                "error": str(e),
            }

    async def _apply_neutral(self, market_data: pd.DataFrame, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Apply neutral strategy when no clear signal exists."""
        return {
            "signal": SIGNAL_TYPES.NEUTRAL,
            "confidence": 0.5,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "strategy_type": STRATEGY_TYPES.NEUTRAL,
            "indicators": {},
        }

    def _combine_signals(
        self, primary_signals: Dict[str, Any], secondary_signals: Dict[str, Any], primary_weight: float, secondary_weight: float
    ) -> Dict[str, Any]:
        """Combine signals from primary and secondary strategies."""
        # Clone primary signals as the base
        combined = primary_signals.copy()

        # If primary and secondary have the same signal direction, boost confidence
        if primary_signals["signal"] == secondary_signals["signal"]:
            combined["confidence"] = min(primary_signals["confidence"] * primary_weight + secondary_signals["confidence"] * secondary_weight, 0.95)
            combined["signal_agreement"] = True
        else:
            # If they disagree, weight them accordingly
            primary_score = primary_signals["confidence"] * primary_weight
            secondary_score = secondary_signals["confidence"] * secondary_weight

            if primary_score >= secondary_score:
                # Keep primary signal but reduce confidence
                combined["confidence"] = primary_signals["confidence"] * (primary_weight + (secondary_weight * 0.3))
                combined["signal_agreement"] = False
            else:
                # Switch to secondary signal
                combined["signal"] = secondary_signals["signal"]
                combined["confidence"] = secondary_signals["confidence"] * (secondary_weight + (primary_weight * 0.3))
                combined["entry_price"] = secondary_signals["entry_price"]
                combined["stop_loss"] = secondary_signals["stop_loss"]
                combined["take_profit"] = secondary_signals["take_profit"]
                combined["signal_agreement"] = False

        # Add both strategies' indicators
        combined["indicators"] = {"primary": primary_signals.get("indicators", {}), "secondary": secondary_signals.get("indicators", {})}

        return combined

    async def _generate_regime_insights(self, regime: MarketRegime, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate regime-specific insights and recommendations."""
        insights = {
            "regime_description": self._get_regime_description(regime),
            "typical_duration": self._get_typical_regime_duration(regime),
            "optimal_strategies": self._get_optimal_strategies(regime),
            "common_patterns": self._get_common_patterns(regime),
            "market_inefficiencies": await self._detect_regime_inefficiencies(regime, market_data),
            "historical_performance": self._get_historical_performance(regime),
            "regime_transition_probability": await self._calculate_transition_probability(regime, market_data),
        }

        return insights

    def _get_regime_description(self, regime: MarketRegime) -> str:
        """Get a description of the current market regime."""
        descriptions = {
            MarketRegime.STRONG_UPTREND: (
                "A powerful uptrend with consistent higher highs and higher lows. "
                "Strong buying pressure with limited pullbacks. Momentum indicators are bullish with good volume confirmation."
            ),
            MarketRegime.WEAK_UPTREND: (
                "An uptrend that is losing momentum or facing resistance. Still making higher highs and higher lows, "
                "but with decreasing strength. May be transitioning to a ranging or distribution phase."
            ),
            MarketRegime.RANGING: (
                "Price moving sideways within a defined range. Multiple tests of support and resistance "
                "with neither buyers nor sellers gaining control. "
                "Often a period of consolidation before the next directional move."
            ),
            MarketRegime.WEAK_DOWNTREND: (
                "A downtrend that is losing momentum or finding support. Still making lower highs and lower lows, "
                "but with decreasing strength. May be transitioning to a ranging or accumulation phase."
            ),
            MarketRegime.STRONG_DOWNTREND: (
                "A powerful downtrend with consistent lower highs and lower lows. Strong selling pressure with limited rallies. "
                "Momentum indicators are bearish with good volume confirmation."
            ),
            MarketRegime.HIGH_VOLATILITY: (
                "Elevated price fluctuations with larger than normal candles and wide ranges. "
                "Often seen during market uncertainty, news events, or during trend reversals. Risk is heightened in both directions."
            ),
            MarketRegime.LOW_VOLATILITY: (
                "Compressed price action with smaller than normal candles and narrow ranges. "
                "Often precedes a volatility expansion and can signal a coming powerful move. Patience is required during this phase."
            ),
            MarketRegime.TRANSITIONING: (
                "Market in flux between regimes. Old patterns breaking down while new patterns have not yet established. "
                "Often marked by increased volatility and conflicting signals."
            ),
            MarketRegime.ACCUMULATION: (
                "Sideways price action near support levels with decreasing selling pressure. "
                "Early smart money beginning to accumulate positions before a potential uptrend. "
                "Often characterized by decreasing volume but absorption of selling."
            ),
            MarketRegime.DISTRIBUTION: (
                "Sideways price action near resistance levels with decreasing buying pressure. "
                "Smart money beginning to distribute (sell) positions before a potential downtrend. "
                "Often characterized by decreasing volume but absorption of buying."
            ),
        }

        return descriptions.get(regime, "Unknown market regime")

    def _get_typical_regime_duration(self, regime: MarketRegime) -> str:
        """Get typical duration for the current regime based on historical data."""
        durations = {
            MarketRegime.STRONG_UPTREND: "Typically lasts 2-6 weeks in crypto markets, longer in traditional markets.",
            MarketRegime.WEAK_UPTREND: "Usually lasts 1-3 weeks before transitioning to another regime.",
            MarketRegime.RANGING: "Can persist for 1-8 weeks, especially after strong trends.",
            MarketRegime.WEAK_DOWNTREND: "Usually lasts 1-3 weeks before transitioning to another regime.",
            MarketRegime.STRONG_DOWNTREND: "Typically lasts 2-4 weeks in crypto markets, can be longer in bear markets.",
            MarketRegime.HIGH_VOLATILITY: "Usually brief, lasting 3-10 days before reverting to the mean.",
            MarketRegime.LOW_VOLATILITY: "Can persist for 1-4 weeks, often preceding significant moves.",
            MarketRegime.TRANSITIONING: "Brief period of 3-10 days while the market establishes a new regime.",
            MarketRegime.ACCUMULATION: "Extended periods of 2-12 weeks, particularly at major bottoms.",
            MarketRegime.DISTRIBUTION: "Extended periods of 2-8 weeks, particularly at major tops.",
        }

        return durations.get(regime, "Variable duration based on market conditions")

    def _get_optimal_strategies(self, regime: MarketRegime) -> List[str]:
        """Get optimal strategies for the current regime."""
        strategies = {
            MarketRegime.STRONG_UPTREND: ["Trend Following", "Breakout Trading", "Momentum Trading"],
            MarketRegime.WEAK_UPTREND: ["Pullback Entries", "Momentum Divergence", "Swing Trading"],
            MarketRegime.RANGING: ["Mean Reversion", "Range Trading", "Support/Resistance Bounces"],
            MarketRegime.WEAK_DOWNTREND: ["Pullback Entries (Short)", "Momentum Divergence", "Swing Trading"],
            MarketRegime.STRONG_DOWNTREND: ["Trend Following (Short)", "Breakdown Trading", "Momentum Trading (Short)"],
            MarketRegime.HIGH_VOLATILITY: ["Option Strategies", "Reduced Position Sizing", "Wider Stop Losses"],
            MarketRegime.LOW_VOLATILITY: ["Breakout Anticipation", "Tight Range Trading", "Coiled Spring Setups"],
            MarketRegime.TRANSITIONING: ["Reduced Exposure", "Waiting for Confirmation", "Multiple Timeframe Analysis"],
            MarketRegime.ACCUMULATION: ["Wyckoff Method", "Swing Lows Entries", "Divergence Trading"],
            MarketRegime.DISTRIBUTION: ["Wyckoff Method (Short)", "Lower Highs Entries", "Divergence Trading (Short)"],
        }

        return strategies.get(regime, ["Adaptive Strategies", "Neutral Exposure", "Wait for Clear Signals"])

    def _get_common_patterns(self, regime: MarketRegime) -> List[str]:
        """Get common patterns observed in the current regime."""
        patterns = {
            MarketRegime.STRONG_UPTREND: ["Bull Flags", "Ascending Triangles", "Cup and Handle", "Higher Highs and Higher Lows"],
            MarketRegime.WEAK_UPTREND: ["Rising Wedge", "Weakening Momentum Divergence", "Decreasing Volume"],
            MarketRegime.RANGING: ["Rectangle Patterns", "Symmetrical Triangles", "Double Tops/Bottoms", "Head and Shoulders"],
            MarketRegime.WEAK_DOWNTREND: ["Falling Wedge", "Weakening Momentum Divergence", "Decreasing Volume"],
            MarketRegime.STRONG_DOWNTREND: ["Bear Flags", "Descending Triangles", "Lower Highs and Lower Lows"],
            MarketRegime.HIGH_VOLATILITY: ["Wide Range Candles", "Gaps", "V-Shaped Reversals", "Climactic Volume"],
            MarketRegime.LOW_VOLATILITY: ["Narrow Range Candles", "Pennants", "Bollinger Band Squeeze"],
            MarketRegime.TRANSITIONING: ["Double Tops/Bottoms", "Reversal Candlestick Patterns", "Momentum Divergence"],
            MarketRegime.ACCUMULATION: ["Wyckoff Accumulation", "Falling Volume", "Rounding Bottom", "Bullish Divergence"],
            MarketRegime.DISTRIBUTION: ["Wyckoff Distribution", "Rising Volume on Drops", "Rounding Top", "Bearish Divergence"],
        }

        return patterns.get(regime, ["Mixed Signals", "Unclear Patterns", "Waiting for Confirmation"])

    async def _detect_regime_inefficiencies(self, regime: MarketRegime, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect regime-specific market inefficiencies."""
        try:
            # Use the inefficiency detector to find regime-specific inefficiencies
            inefficiencies = await self.inefficiency_detector.detect_inefficiencies(
                market_data=market_data, asset_id=self.asset_id, exchange_id=self.exchange_id, market_regime=regime.name
            )

            # Filter and return the top 3 inefficiencies by score
            if inefficiencies:
                top_inefficiencies = sorted(inefficiencies, key=lambda x: x.get("score", 0), reverse=True)[:3]

                return top_inefficiencies

            return []

        except Exception as e:
            self.logger.warning(f"Error detecting regime inefficiencies: {str(e)}")
            return []

    def _get_historical_performance(self, regime: MarketRegime) -> Dict[str, Any]:
        """Get historical performance metrics for the current regime."""
        # In production, this would query a database of historical performance
        if regime in self.regime_performance:
            perf = self.regime_performance[regime]
            if perf["count"] > 0:
                win_rate = perf["success"] / perf["count"] if perf["count"] > 0 else 0
                avg_pnl = perf["pnl"] / perf["count"] if perf["count"] > 0 else 0

                return {
                    "trades": perf["count"],
                    "win_rate": win_rate,
                    "avg_pnl_percent": avg_pnl,
                    "regime_effectiveness": "High" if win_rate > 0.7 else ("Medium" if win_rate > 0.5 else "Low"),
                }

        # Default return if no historical data
        return {"trades": 0, "win_rate": None, "avg_pnl_percent": None, "regime_effectiveness": "Unknown - Insufficient data"}

    async def _calculate_transition_probability(self, current_regime: MarketRegime, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate probability of transitioning to another regime."""
        try:
            # Simple transition model based on regime characteristics
            # In production, this would use a more sophisticated ML model

            # Default transition probabilities
            transitions = {regime.name: 0.05 for regime in MarketRegime}
            transitions[current_regime.name] = 0.60  # High probability of staying in current regime

            # Adjust based on regime-specific transition patterns
            if current_regime == MarketRegime.STRONG_UPTREND:
                transitions[MarketRegime.WEAK_UPTREND.name] = 0.20
                transitions[MarketRegime.DISTRIBUTION.name] = 0.10

            elif current_regime == MarketRegime.WEAK_UPTREND:
                transitions[MarketRegime.RANGING.name] = 0.15
                transitions[MarketRegime.STRONG_UPTREND.name] = 0.10
                transitions[MarketRegime.DISTRIBUTION.name] = 0.10

            elif current_regime == MarketRegime.RANGING:
                transitions[MarketRegime.WEAK_UPTREND.name] = 0.10
                transitions[MarketRegime.WEAK_DOWNTREND.name] = 0.10
                transitions[MarketRegime.ACCUMULATION.name] = 0.05
                transitions[MarketRegime.DISTRIBUTION.name] = 0.05

            elif current_regime == MarketRegime.WEAK_DOWNTREND:
                transitions[MarketRegime.RANGING.name] = 0.15
                transitions[MarketRegime.STRONG_DOWNTREND.name] = 0.10
                transitions[MarketRegime.ACCUMULATION.name] = 0.10

            elif current_regime == MarketRegime.STRONG_DOWNTREND:
                transitions[MarketRegime.WEAK_DOWNTREND.name] = 0.20
                transitions[MarketRegime.ACCUMULATION.name] = 0.10

            elif current_regime == MarketRegime.HIGH_VOLATILITY:
                transitions[MarketRegime.TRANSITIONING.name] = 0.20
                transitions[MarketRegime.STRONG_UPTREND.name] = 0.05
                transitions[MarketRegime.STRONG_DOWNTREND.name] = 0.05

            elif current_regime == MarketRegime.LOW_VOLATILITY:
                transitions[MarketRegime.HIGH_VOLATILITY.name] = 0.20
                transitions[MarketRegime.TRANSITIONING.name] = 0.15

            elif current_regime == MarketRegime.TRANSITIONING:
                transitions[MarketRegime.STRONG_UPTREND.name] = 0.10
                transitions[MarketRegime.STRONG_DOWNTREND.name] = 0.10
                transitions[MarketRegime.RANGING.name] = 0.15

            elif current_regime == MarketRegime.ACCUMULATION:
                transitions[MarketRegime.WEAK_UPTREND.name] = 0.15
                transitions[MarketRegime.RANGING.name] = 0.10

            elif current_regime == MarketRegime.DISTRIBUTION:
                transitions[MarketRegime.WEAK_DOWNTREND.name] = 0.15
                transitions[MarketRegime.RANGING.name] = 0.10

            # Further adjust based on current market conditions
            # This would use a more sophisticated approach in production

            return transitions

        except Exception as e:
            self.logger.warning(f"Error calculating regime transition probabilities: {str(e)}")
            return {regime.name: 0.0 for regime in MarketRegime}

    def _calculate_adx(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)."""
        try:
            high = market_data["high"]
            low = market_data["low"]
            close = market_data["close"]

            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()

            # Calculate Directional Movement
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = abs(minus_dm)

            # Conditions for +DM and -DM
            cond_plus = (high.diff() > -low.diff()) & (high.diff() > 0)
            cond_minus = (-low.diff() > high.diff()) & (-low.diff() > 0)

            plus_dm = plus_dm.where(cond_plus, 0.0)
            minus_dm = minus_dm.where(cond_minus, 0.0)

            # Calculate +DI and -DI
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

            # Calculate Directional Index (DX)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

            # Calculate ADX
            adx = dx.rolling(window=period).mean()

            return adx.iloc[-1]

        except Exception as e:
            self.logger.error(f"Error calculating ADX: {str(e)}")
            return 0.0

    async def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """Update regime performance tracking with trade results."""
        if "regime" in trade_result and "success" in trade_result and "pnl_percent" in trade_result:
            regime = MarketRegime[trade_result["regime"]] if trade_result["regime"] in MarketRegime.__members__ else None

            if regime and regime in self.regime_performance:
                # Update regime performance metrics
                self.regime_performance[regime]["count"] += 1
                if trade_result["success"]:
                    self.regime_performance[regime]["success"] += 1
                self.regime_performance[regime]["pnl"] += trade_result["pnl_percent"]

                # Log performance update
                perf = self.regime_performance[regime]
                win_rate = perf["success"] / perf["count"] if perf["count"] > 0 else 0
                self.logger.info(
                    f"Updated {regime.name} performance: {perf['count']} trades, " f"{win_rate:.2f} win rate, {perf['pnl']:.2f}% cumulative PnL"
                )

                # Adjust regime feature importance based on performance
                if perf["count"] >= 5:
                    self._adjust_feature_importance(regime, win_rate)

    def _adjust_feature_importance(self, regime: MarketRegime, win_rate: float) -> None:
        """Adjust feature importance weights based on regime performance."""
        # Only adjust if we have significant data
        if win_rate > 0.7:
            # Successful regime detection - reinforce current weights
            self.logger.info(f"Reinforcing feature weights for successful {regime.name} detection")
            # No changes needed, current weights are working well
        elif win_rate < 0.5:
            # Poor regime detection - adjust weights
            self.logger.info(f"Adjusting feature weights for poor {regime.name} detection")

            adjustment = self.adaptive_params["feature_adjustment_rate"]

            # Rebalance weights - reduce weakest, increase strongest
            min_feature = min(self.regime_feature_importance, key=self.regime_feature_importance.get)
            max_feature = max(self.regime_feature_importance, key=self.regime_feature_importance.get)

            if self.regime_feature_importance[min_feature] >= adjustment:
                self.regime_feature_importance[min_feature] -= adjustment
                self.regime_feature_importance[max_feature] += adjustment

                self.logger.info(f"Adjusted weights: decreased {min_feature}, increased {max_feature}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert brain state to dictionary for serialization."""
        return {
            "brain_type": "regime",
            "asset_id": self.asset_id,
            "exchange_id": self.exchange_id,
            "current_regime": self.current_regime.name,
            "regime_confidence": self.regime_confidence,
            "adaptive_params": self.adaptive_params,
            "regime_feature_importance": self.regime_feature_importance,
            "regime_performance": {k.name: v for k, v in self.regime_performance.items()},
            "regime_history": self.regime_history[-10:] if self.regime_history else [],
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], metrics_collector: MetricsCollector, prediction_service=None, market_data_repository=None
    ) -> "RegimeBrain":
        """Create RegimeBrain instance from dictionary."""
        brain = cls(
            asset_id=data["asset_id"],
            exchange_id=data["exchange_id"],
            config={},  # Default config, will be overridden
            metrics_collector=metrics_collector,
            prediction_service=prediction_service,
            market_data_repository=market_data_repository,
        )

        # Restore state
        brain.current_regime = MarketRegime[data["current_regime"]]
        brain.regime_confidence = data["regime_confidence"]
        brain.adaptive_params = data["adaptive_params"]
        brain.regime_feature_importance = data["regime_feature_importance"]
        brain.regime_performance = {MarketRegime[k]: v for k, v in data["regime_performance"].items()}
        brain.regime_history = data["regime_history"]

        return brain
