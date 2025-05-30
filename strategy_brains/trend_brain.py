#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Trend Following Brain Strategy

This module implements a sophisticated trend-following strategy that adapts to market conditions,
identifies and follows trends with multiple confirmation mechanisms, and implements smart
entry/exit logic for optimized trade execution.
"""

import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from common.constants import DIRECTIONAL_BIAS_THRESHOLD, FILTER_STRENGTH_LEVELS, REGIME_TYPES, TREND_FOLLOWING_CONFIG
from common.exceptions import InsufficientDataError, SignalGenerationError, StrategyError
from common.utils import calculate_atr, is_higher_high, is_lower_low, zigzag_identification
from feature_service.features.market_structure import analyze_swing_points, identify_market_structure
from feature_service.features.technical import (
    calculate_adx,
    calculate_donchian_channel,
    calculate_macd,
    calculate_moving_average,
    calculate_supertrend,
)
from feature_service.features.volatility import calculate_keltner_channel
from strategy_brains.base_brain import BaseBrain


class TrendBrain(BaseBrain):
    """
    A sophisticated trend-following strategy brain that identifies and follows market trends
    with multiple confirmation mechanisms and adaptive parameters.

    Features:
    - Multi-timeframe trend confirmation
    - Trend strength assessment
    - Dynamic parameter adjustment based on volatility
    - Progressive position building
    - Multiple exit strategies
    - Trend reversal early detection
    - Market regime adaptation
    """

    def __init__(self, name: str = "trend_brain", **kwargs):
        """
        Initialize the TrendBrain strategy.

        Args:
            name: Unique name for this brain instance
            **kwargs: Additional configuration parameters
        """
        super().__init__(name=name, **kwargs)

        # Strategy-specific configuration with smart defaults
        self.config = {
            "long_lookback_period": kwargs.get("long_lookback_period", 200),
            "medium_lookback_period": kwargs.get("medium_lookback_period", 50),
            "short_lookback_period": kwargs.get("short_lookback_period", 20),
            "adx_period": kwargs.get("adx_period", 14),
            "adx_threshold": kwargs.get("adx_threshold", 25),
            "macd_fast": kwargs.get("macd_fast", 12),
            "macd_slow": kwargs.get("macd_slow", 26),
            "macd_signal": kwargs.get("macd_signal", 9),
            "supertrend_period": kwargs.get("supertrend_period", 10),
            "supertrend_multiplier": kwargs.get("supertrend_multiplier", 3.0),
            "donchian_period": kwargs.get("donchian_period", 20),
            "keltner_period": kwargs.get("keltner_period", 20),
            "keltner_atr_multiplier": kwargs.get("keltner_atr_multiplier", 2.0),
            "volatility_adjustment_enabled": kwargs.get("volatility_adjustment_enabled", True),
            "multi_timeframe_confirmation": kwargs.get("multi_timeframe_confirmation", True),
            "trend_strength_filter": kwargs.get("trend_strength_filter", "medium"),  # low, medium, high
            "progressive_entry": kwargs.get("progressive_entry", True),
            "early_reversal_detection": kwargs.get("early_reversal_detection", True),
            "profit_target_atr_multiplier": kwargs.get("profit_target_atr_multiplier", 3.0),
            "stop_loss_atr_multiplier": kwargs.get("stop_loss_atr_multiplier", 1.5),
            "trailing_stop_enabled": kwargs.get("trailing_stop_enabled", True),
            "trailing_stop_activation_atr_multiplier": kwargs.get("trailing_stop_activation_atr_multiplier", 1.0),
            "trailing_stop_atr_multiplier": kwargs.get("trailing_stop_atr_multiplier", 2.0),
            "time_stop_bars": kwargs.get("time_stop_bars", 15),
            "regime_adaptation": kwargs.get("regime_adaptation", True),
            "min_bars_in_trend": kwargs.get("min_bars_in_trend", 3),
            "suboptimal_exit_threshold": kwargs.get("suboptimal_exit_threshold", 0.5),  # 0.0-1.0, lower is more strict
        }

        # Internal state tracking
        self.current_trend = None  # 'up', 'down', or None
        self.trend_start_price = None
        self.trend_start_time = None
        self.trend_strength = 0.0
        self.entry_prices = []
        self.position_size_percentages = []
        self.current_volatility = None
        self.current_atr = None
        self.current_market_regime = None
        self.consecutive_signals = 0
        self.current_swing_points = []
        self.last_signal_time = None
        self.current_exit_targets = {"profit_target": None, "stop_loss": None, "trailing_stop": None, "time_stop": None}

        # Performance metrics
        self.signals_generated = 0
        self.successful_signals = 0
        self.false_signals = 0
        self.premature_exits = 0
        self.optimal_exits = 0

        self.logger = logging.getLogger(f"strategy_brains.{self.name}")
        self.logger.info(f"Initialized {self.name} with configuration: {self.config}")

    async def analyze(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        Analyze market data to determine trend direction, strength, and generate trading signals.

        Args:
            data: Dict of DataFrames with market data for different timeframes
            **kwargs: Additional parameters

        Returns:
            Dict containing analysis results and trading signals
        """
        self.logger.info(f"Running trend analysis on {len(data)} timeframes")

        try:
            # Ensure we have the primary timeframe data
            if not self.primary_timeframe in data:
                raise InsufficientDataError(f"Primary timeframe {self.primary_timeframe} data not available")

            # Get the primary timeframe data
            df = data[self.primary_timeframe].copy()
            if len(df) < self.config["long_lookback_period"]:
                raise InsufficientDataError(f"Insufficient data for analysis. Need at least {self.config['long_lookback_period']} bars.")

            # Calculate technical indicators
            self._calculate_indicators(df)

            # Detect market regime
            self._detect_market_regime(df)

            # Analyze trend in multiple timeframes if enabled
            trend_signals = {}
            trend_strengths = {}

            # Primary timeframe analysis
            primary_trend_data = self._analyze_single_timeframe(df, self.primary_timeframe)
            trend_signals[self.primary_timeframe] = primary_trend_data["trend_direction"]
            trend_strengths[self.primary_timeframe] = primary_trend_data["trend_strength"]

            # Multi-timeframe analysis if enabled
            if self.config["multi_timeframe_confirmation"]:
                for timeframe, tf_data in data.items():
                    if timeframe != self.primary_timeframe and len(tf_data) >= 100:  # Ensure sufficient data
                        tf_data_copy = tf_data.copy()
                        self._calculate_indicators(tf_data_copy)
                        tf_trend_data = self._analyze_single_timeframe(tf_data_copy, timeframe)
                        trend_signals[timeframe] = tf_trend_data["trend_direction"]
                        trend_strengths[timeframe] = tf_trend_data["trend_strength"]

            # Combine trend signals from multiple timeframes
            overall_trend = self._combine_timeframe_signals(trend_signals, trend_strengths)

            # Update current trend and metrics
            self._update_trend_state(overall_trend, df)

            # Generate trading signals
            signals = self._generate_signals(df, overall_trend)

            # Calculate exit targets if we have an active position
            if self.current_trend is not None:
                self._calculate_exit_targets(df, overall_trend)

            # Prepare analysis results
            analysis_results = {
                "trend_direction": overall_trend["trend_direction"],
                "trend_strength": overall_trend["trend_strength"],
                "market_regime": self.current_market_regime,
                "current_volatility": self.current_volatility,
                "signals": signals,
                "exit_targets": self.current_exit_targets,
                "swing_points": self.current_swing_points,
                "timeframe_agreement": overall_trend["timeframe_agreement"],
                "raw_metrics": {
                    "adx": df["adx"].iloc[-1] if "adx" in df else None,
                    "supertrend": {
                        "value": df["supertrend"].iloc[-1] if "supertrend" in df else None,
                        "direction": df["supertrend_direction"].iloc[-1] if "supertrend_direction" in df else None,
                    },
                    "macd": {
                        "macd": df["macd"].iloc[-1] if "macd" in df else None,
                        "signal": df["macd_signal"].iloc[-1] if "macd_signal" in df else None,
                        "histogram": df["macd_hist"].iloc[-1] if "macd_hist" in df else None,
                    },
                    "moving_averages": {
                        "short_ma": df["short_ma"].iloc[-1] if "short_ma" in df else None,
                        "medium_ma": df["medium_ma"].iloc[-1] if "medium_ma" in df else None,
                        "long_ma": df["long_ma"].iloc[-1] if "long_ma" in df else None,
                    },
                },
                "timestamps": {"analysis_time": datetime.now(), "trend_start_time": self.trend_start_time, "last_signal_time": self.last_signal_time},
            }

            self.logger.info(
                "Trend analysis complete. Direction: %s, Strength: %.2f",
                overall_trend["trend_direction"],
                overall_trend["trend_strength"],
            )
            return analysis_results

        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise StrategyError(f"Trend analysis failed: {str(e)}")

    def _calculate_indicators(self, df: pd.DataFrame) -> None:
        """
        Calculate all required technical indicators for trend analysis.

        Args:
            df: DataFrame with market data
        """
        # Make sure we have OHLCV data
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column {col} not found in data")

        # Calculate moving averages
        df["short_ma"] = calculate_moving_average(df["close"], period=self.config["short_lookback_period"], ma_type="ema")
        df["medium_ma"] = calculate_moving_average(df["close"], period=self.config["medium_lookback_period"], ma_type="ema")
        df["long_ma"] = calculate_moving_average(df["close"], period=self.config["long_lookback_period"], ma_type="ema")

        # Calculate MACD
        macd_result = calculate_macd(
            df["close"], fast_period=self.config["macd_fast"], slow_period=self.config["macd_slow"], signal_period=self.config["macd_signal"]
        )
        df["macd"] = macd_result["macd"]
        df["macd_signal"] = macd_result["signal"]
        df["macd_hist"] = macd_result["histogram"]

        # Calculate ADX for trend strength
        df["adx"] = calculate_adx(df["high"], df["low"], df["close"], period=self.config["adx_period"])

        # Calculate SuperTrend
        supertrend_result = calculate_supertrend(
            df["high"], df["low"], df["close"], period=self.config["supertrend_period"], multiplier=self.config["supertrend_multiplier"]
        )
        df["supertrend"] = supertrend_result["supertrend"]
        df["supertrend_direction"] = supertrend_result["direction"]

        # Calculate Donchian Channels
        donchian = calculate_donchian_channel(df["high"], df["low"], period=self.config["donchian_period"])
        df["donchian_high"] = donchian["upper"]
        df["donchian_mid"] = donchian["middle"]
        df["donchian_low"] = donchian["lower"]

        # Calculate Keltner Channels
        keltner = calculate_keltner_channel(
            df["high"], df["low"], df["close"], period=self.config["keltner_period"], atr_multiplier=self.config["keltner_atr_multiplier"]
        )
        df["keltner_high"] = keltner["upper"]
        df["keltner_mid"] = keltner["middle"]
        df["keltner_low"] = keltner["lower"]

        # Calculate volatility and ATR
        df["atr"] = calculate_atr(df["high"], df["low"], df["close"], period=14)
        self.current_atr = df["atr"].iloc[-1]
        self.current_volatility = df["atr"].iloc[-1] / df["close"].iloc[-1]

        # Identify swing points
        swing_points = analyze_swing_points(df["high"], df["low"], period=5)
        df["swing_high"] = swing_points["swing_highs"]
        df["swing_low"] = swing_points["swing_lows"]
        self.current_swing_points = [
            {"type": "high", "price": price, "index": idx} for idx, price in swing_points["swing_highs"].dropna().iloc[-5:].items()
        ] + [{"type": "low", "price": price, "index": idx} for idx, price in swing_points["swing_lows"].dropna().iloc[-5:].items()]
        self.current_swing_points.sort(key=lambda x: x["index"])

    def _detect_market_regime(self, df: pd.DataFrame) -> None:
        """
        Detect the current market regime (trending, ranging, volatile, etc.)

        Args:
            df: DataFrame with market data and indicators
        """
        # Use ADX to distinguish between trending and ranging markets
        latest_adx = df["adx"].iloc[-1]

        # Calculate recent volatility relative to historical
        recent_volatility = df["atr"].iloc[-20:].mean() / df["close"].iloc[-20:].mean()
        historical_volatility = df["atr"].iloc[-100:-20].mean() / df["close"].iloc[-100:-20].mean()
        volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1.0

        # Calculate price efficiency (directional movement / total movement)
        price_change = abs(df["close"].iloc[-1] - df["close"].iloc[-20])
        total_movement = sum(abs(df["close"].diff().iloc[-20:].fillna(0)))
        efficiency_ratio = price_change / total_movement if total_movement > 0 else 0

        # Determine regime based on combination of factors
        if latest_adx > 30:
            if volatility_ratio > 1.5:
                self.current_market_regime = REGIME_TYPES.VOLATILE_TREND
            else:
                self.current_market_regime = REGIME_TYPES.STRONG_TREND
        elif latest_adx > 20:
            self.current_market_regime = REGIME_TYPES.WEAK_TREND
        else:
            if volatility_ratio > 1.5:
                self.current_market_regime = REGIME_TYPES.VOLATILE_RANGE
            elif efficiency_ratio < 0.2:
                self.current_market_regime = REGIME_TYPES.CHOPPY
            else:
                self.current_market_regime = REGIME_TYPES.RANGING

        self.logger.debug(f"Detected market regime: {self.current_market_regime}")

    def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        Analyze trend direction and strength for a single timeframe.

        Args:
            df: DataFrame with market data and indicators
            timeframe: The timeframe being analyzed

        Returns:
            Dict with trend direction and strength
        """
        # Check moving average alignment for trend direction
        ma_trend = self._evaluate_ma_trend(df)

        # Check SuperTrend direction
        supertrend_direction = 1 if df["supertrend_direction"].iloc[-1] > 0 else -1

        # Check MACD for trend momentum
        macd_above_signal = df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]
        macd_direction = 1 if macd_above_signal else -1

        # Price in relation to Donchian Channel
        price_in_donchian = self._evaluate_price_in_donchian(df)

        # Check if price is making higher highs/lows or lower highs/lows
        price_structure = self._evaluate_price_structure(df)

        # Calculate trend strength (0-100)
        adx_contribution = min(df["adx"].iloc[-1] / 50, 1.0) * 40  # Max 40 points from ADX

        # MA alignment contribution (0-20)
        ma_alignment_score = 0
        if ma_trend == 1:  # Uptrend
            if df["short_ma"].iloc[-1] > df["medium_ma"].iloc[-1] > df["long_ma"].iloc[-1]:
                ma_alignment_score = 20
            elif df["short_ma"].iloc[-1] > df["medium_ma"].iloc[-1]:
                ma_alignment_score = 10
        elif ma_trend == -1:  # Downtrend
            if df["short_ma"].iloc[-1] < df["medium_ma"].iloc[-1] < df["long_ma"].iloc[-1]:
                ma_alignment_score = 20
            elif df["short_ma"].iloc[-1] < df["medium_ma"].iloc[-1]:
                ma_alignment_score = 10

        # MACD contribution (0-20)
        macd_contribution = abs(df["macd_hist"].iloc[-1]) / (abs(df["macd"].iloc[-1]) + 0.0001) * 20
        if np.isnan(macd_contribution):
            macd_contribution = 0

        # Price structure contribution (0-20)
        price_structure_contribution = 0
        if price_structure == 1 and ma_trend == 1:  # Uptrend with higher highs/lows
            price_structure_contribution = 20
        elif price_structure == -1 and ma_trend == -1:  # Downtrend with lower highs/lows
            price_structure_contribution = 20
        elif price_structure != 0 and price_structure == ma_trend:
            price_structure_contribution = 10

        # Calculate final trend strength
        trend_strength = adx_contribution + ma_alignment_score + macd_contribution + price_structure_contribution
        trend_strength = min(trend_strength, 100)  # Cap at 100

        # Determine overall trend direction using weighted voting
        direction_votes = {
            "ma_trend": (ma_trend, 3),  # Weight of 3
            "supertrend": (supertrend_direction, 2),  # Weight of 2
            "macd": (macd_direction, 1),  # Weight of 1
            "price_structure": (price_structure, 2),  # Weight of 2
            "donchian": (price_in_donchian, 1),  # Weight of 1
        }

        weighted_sum = sum(direction * weight for direction, weight in direction_votes.values())
        total_weight = sum(weight for _, weight in direction_votes.values())

        # Final trend direction is the sign of the weighted sum
        if weighted_sum > 0:
            trend_direction = 1  # Uptrend
        elif weighted_sum < 0:
            trend_direction = -1  # Downtrend
        else:
            trend_direction = 0  # No clear trend

        # Apply trend strength filter
        if self.config["trend_strength_filter"] == "high" and trend_strength < 70:
            trend_direction = 0
        elif self.config["trend_strength_filter"] == "medium" and trend_strength < 50:
            trend_direction = 0
        elif self.config["trend_strength_filter"] == "low" and trend_strength < 30:
            trend_direction = 0

        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "component_scores": {
                "adx": adx_contribution,
                "ma_alignment": ma_alignment_score,
                "macd": macd_contribution,
                "price_structure": price_structure_contribution,
            },
            "direction_votes": direction_votes,
        }

    def _evaluate_ma_trend(self, df: pd.DataFrame) -> int:
        """
        Evaluate trend direction based on moving average relationships.

        Returns:
            1 for uptrend, -1 for downtrend, 0 for no clear trend
        """
        # Get latest values for moving averages
        short_ma = df["short_ma"].iloc[-1]
        medium_ma = df["medium_ma"].iloc[-1]
        long_ma = df["long_ma"].iloc[-1]

        # Check for clear uptrend (short > medium > long)
        if short_ma > medium_ma and medium_ma > long_ma:
            return 1

        # Check for clear downtrend (short < medium < long)
        if short_ma < medium_ma and medium_ma < long_ma:
            return -1

        # Check for recent trend change (short > medium but medium < long)
        if short_ma > medium_ma and medium_ma < long_ma:
            # Check if medium MA is turning up
            if df["medium_ma"].iloc[-1] > df["medium_ma"].iloc[-2]:
                return 1
            # No clear direction
            return 0

        # Check for recent trend change (short < medium but medium > long)
        if short_ma < medium_ma and medium_ma > long_ma:
            # Check if medium MA is turning down
            if df["medium_ma"].iloc[-1] < df["medium_ma"].iloc[-2]:
                return -1
            # No clear direction
            return 0

        # No clear trend
        return 0

    def _evaluate_price_in_donchian(self, df: pd.DataFrame) -> int:
        """
        Evaluate where price is in relation to Donchian Channel.

        Returns:
            1 for bullish, -1 for bearish, 0 for neutral
        """
        latest_close = df["close"].iloc[-1]
        upper_band = df["donchian_high"].iloc[-1]
        lower_band = df["donchian_low"].iloc[-1]
        middle_band = df["donchian_mid"].iloc[-1]

        # Calculate position within the channel (0-1)
        channel_width = upper_band - lower_band
        if channel_width > 0:
            position = (latest_close - lower_band) / channel_width
        else:
            position = 0.5  # Default to middle if channel has no width

        # Interpret position
        if position > 0.8:  # Near top of channel
            return 1
        elif position < 0.2:  # Near bottom of channel
            return -1
        elif position > 0.5:  # Above midpoint
            return 0.5
        elif position < 0.5:  # Below midpoint
            return -0.5
        else:  # At midpoint
            return 0

    def _evaluate_price_structure(self, df: pd.DataFrame) -> int:
        """
        Evaluate if price is making higher highs/lows or lower highs/lows.

        Returns:
            1 for higher highs/lows, -1 for lower highs/lows, 0 for no clear pattern
        """
        # Use recent swing points to determine structure
        recent_swing_highs = df["swing_high"].dropna().tail(3)
        recent_swing_lows = df["swing_low"].dropna().tail(3)

        # Need at least 2 swing points of each type to determine pattern
        if len(recent_swing_highs) >= 2 and len(recent_swing_lows) >= 2:
            # Check for higher highs and higher lows (uptrend)
            higher_highs = is_higher_high(recent_swing_highs)
            higher_lows = is_higher_high(recent_swing_lows)

            # Check for lower highs and lower lows (downtrend)
            lower_highs = is_lower_low(recent_swing_highs)
            lower_lows = is_lower_low(recent_swing_lows)

            if higher_highs and higher_lows:
                return 1  # Strong uptrend structure
            elif lower_highs and lower_lows:
                return -1  # Strong downtrend structure
            elif higher_highs or higher_lows:
                return 0.5  # Weak uptrend structure
            elif lower_highs or lower_lows:
                return -0.5  # Weak downtrend structure

        # Not enough swing points or no clear pattern
        return 0

    def _combine_timeframe_signals(self, trend_signals: Dict[str, int], trend_strengths: Dict[str, float]) -> Dict[str, Any]:
        """
        Combine trend signals from multiple timeframes to get an overall trend direction.

        Args:
            trend_signals: Dict mapping timeframes to trend directions
            trend_strengths: Dict mapping timeframes to trend strengths

        Returns:
            Dict with overall trend direction and strength
        """
        if not trend_signals:
            return {"trend_direction": 0, "trend_strength": 0, "timeframe_agreement": 0}

        # Define timeframe weights (higher timeframes have more weight)
        timeframe_weights = {"1m": 1, "5m": 2, "15m": 3, "30m": 4, "1h": 5, "4h": 6, "1d": 7, "1w": 8}

        # Calculate weighted trend direction
        weighted_sum = 0
        total_weight = 0

        for timeframe, direction in trend_signals.items():
            # Get weight for this timeframe
            weight = timeframe_weights.get(timeframe, 3)  # Default weight if timeframe not in dict

            # Adjust weight by trend strength
            adjusted_weight = weight * (trend_strengths.get(timeframe, 50) / 50)

            weighted_sum += direction * adjusted_weight
            total_weight += adjusted_weight

        # Calculate overall trend direction
        if total_weight > 0:
            overall_direction = 1 if weighted_sum / total_weight > 0.2 else (-1 if weighted_sum / total_weight < -0.2 else 0)
        else:
            overall_direction = 0

        # Calculate overall trend strength (average of trend strengths)
        overall_strength = sum(trend_strengths.values()) / len(trend_strengths) if trend_strengths else 0

        # Calculate timeframe agreement (-100 to 100)
        agreement = (weighted_sum / total_weight * 100) if total_weight > 0 else 0

        return {"trend_direction": overall_direction, "trend_strength": overall_strength, "timeframe_agreement": agreement}

    def _update_trend_state(self, overall_trend: Dict[str, Any], df: pd.DataFrame) -> None:
        """
        Update the internal trend state based on the overall trend analysis.

        Args:
            overall_trend: Dict with trend direction and strength
            df: DataFrame with market data
        """
        current_direction = overall_trend["trend_direction"]
        current_price = df["close"].iloc[-1]
        current_time = pd.Timestamp.now() if df.index[-1].tzinfo else df.index[-1]

        # Initialize trend if none exists
        if self.current_trend is None and current_direction != 0:
            self.current_trend = "up" if current_direction > 0 else "down"
            self.trend_start_price = current_price
            self.trend_start_time = current_time
            self.consecutive_signals = 1
            self.logger.info(f"New {self.current_trend} trend started at {current_price}")

        # Detect trend change
        elif self.current_trend == "up" and current_direction < 0:
            self.logger.info(f"Trend changed from up to down at {current_price}")
            self.current_trend = "down"
            self.trend_start_price = current_price
            self.trend_start_time = current_time
            self.consecutive_signals = 1
            self.entry_prices = []
            self.position_size_percentages = []

        elif self.current_trend == "down" and current_direction > 0:
            self.logger.info(f"Trend changed from down to up at {current_price}")
            self.current_trend = "up"
            self.trend_start_price = current_price
            self.trend_start_time = current_time
            self.consecutive_signals = 1
            self.entry_prices = []
            self.position_size_percentages = []

        # Same trend continues
        elif self.current_trend is not None and (
            (self.current_trend == "up" and current_direction > 0) or (self.current_trend == "down" and current_direction < 0)
        ):
            self.consecutive_signals += 1

        # No clear trend
        elif current_direction == 0:
            # Only clear trend if it's been unclear for some time
            if self.consecutive_signals <= 0:
                self.current_trend = None
                self.trend_start_price = None
                self.trend_start_time = None
                self.consecutive_signals = 0
                self.entry_prices = []
                self.position_size_percentages = []
                self.logger.info("Trend cleared due to lack of clear direction")
            else:
                # Reduce confidence in current trend
                self.consecutive_signals -= 1

        # Update trend strength
        self.trend_strength = overall_trend["trend_strength"]

    def _generate_signals(self, df: pd.DataFrame, overall_trend: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on trend analysis.

        Args:
            df: DataFrame with market data
            overall_trend: Dict with trend direction and strength

        Returns:
            Dict with trading signals and metadata
        """
        signals = {"entry": None, "exit": None, "confidence": 0.0, "sizing_factor": 1.0, "metadata": {}}  # 'long', 'short', or None  # 'exit' or None

        current_price = df["close"].iloc[-1]
        current_time = pd.Timestamp.now() if df.index[-1].tzinfo else df.index[-1]

        # Check if we're in a sufficiently strong trend
        strong_trend = overall_trend["trend_strength"] >= FILTER_STRENGTH_LEVELS[self.config["trend_strength_filter"]]

        # Entry logic for new positions
        if self.current_trend is not None and strong_trend:
            # Calculate confidence based on trend strength and consecutive signals
            base_confidence = overall_trend["trend_strength"] / 100
            signal_confidence = min(self.consecutive_signals / 5, 1.0)
            regime_factor = self._get_regime_confidence_factor()

            # Apply volatility adjustment if enabled
            volatility_factor = 1.0
            if self.config["volatility_adjustment_enabled"]:
                volatility_percentile = self._calculate_volatility_percentile(df["atr"], 100)
                # Reduce confidence in high volatility environments
                if volatility_percentile > 0.8:
                    volatility_factor = 0.8
                # Increase confidence in stable environments
                elif volatility_percentile < 0.3:
                    volatility_factor = 1.2

            # Final confidence score (0-1)
            confidence = base_confidence * signal_confidence * regime_factor * volatility_factor

            # Determine position sizing factor based on confidence
            sizing_factor = confidence

            # Check for entry conditions
            if self.current_trend == "up" and self._check_uptrend_entry_conditions(df):
                signals["entry"] = "long"
                signals["confidence"] = confidence
                signals["sizing_factor"] = sizing_factor
                signals["metadata"]["trend_duration"] = (
                    (current_time - self.trend_start_time).total_seconds() / 60 if hasattr(current_time, "total_seconds") else 0
                )
                signals["metadata"]["trend_strength"] = overall_trend["trend_strength"]
                signals["metadata"]["consecutive_signals"] = self.consecutive_signals

                # Record signal
                self.signals_generated += 1
                self.last_signal_time = current_time

                # Add to entry prices if using progressive entry
                if self.config["progressive_entry"]:
                    self.entry_prices.append(current_price)
                    self.position_size_percentages.append(sizing_factor)

                self.logger.info(
                    "Generated LONG signal at %s with confidence %.2f",
                    current_price,
                    confidence,
                )

            elif self.current_trend == "down" and self._check_downtrend_entry_conditions(df):
                signals["entry"] = "short"
                signals["confidence"] = confidence
                signals["sizing_factor"] = sizing_factor
                signals["metadata"]["trend_duration"] = (
                    (current_time - self.trend_start_time).total_seconds() / 60 if hasattr(current_time, "total_seconds") else 0
                )
                signals["metadata"]["trend_strength"] = overall_trend["trend_strength"]
                signals["metadata"]["consecutive_signals"] = self.consecutive_signals

                # Record signal
                self.signals_generated += 1
                self.last_signal_time = current_time

                # Add to entry prices if using progressive entry
                if self.config["progressive_entry"]:
                    self.entry_prices.append(current_price)
                    self.position_size_percentages.append(sizing_factor)

                self.logger.info(
                    "Generated SHORT signal at %s with confidence %.2f",
                    current_price,
                    confidence,
                )

        # Exit logic
        if self.current_trend is not None and len(self.entry_prices) > 0:
            exit_signal = self._check_exit_conditions(df, overall_trend)
            if exit_signal is not None:
                signals["exit"] = "exit"
                signals["metadata"]["exit_reason"] = exit_signal["reason"]
                signals["metadata"]["exit_strength"] = exit_signal["strength"]

                self.logger.info(f"Generated EXIT signal at {current_price} due to {exit_signal['reason']}")

        return signals

    def _check_uptrend_entry_conditions(self, df: pd.DataFrame) -> bool:
        """
        Check entry conditions for long positions in an uptrend.

        Args:
            df: DataFrame with market data

        Returns:
            bool: True if entry conditions are met
        """
        # Minimum required ADX for a trending market
        if df["adx"].iloc[-1] < self.config["adx_threshold"]:
            return False

        # SuperTrend confirmation
        if df["supertrend_direction"].iloc[-1] <= 0:
            return False

        # Check for pullback to moving average (entry opportunity)
        close = df["close"].iloc[-1]

        # Identify market regime appropriate entry conditions
        if self.current_market_regime in [REGIME_TYPES.STRONG_TREND, REGIME_TYPES.VOLATILE_TREND]:
            # In strong trends, look for pullbacks to short or medium MA
            short_ma_pullback = df["close"].iloc[-2] < df["short_ma"].iloc[-2] and close > df["short_ma"].iloc[-1]

            medium_ma_pullback = close <= df["medium_ma"].iloc[-1] * 1.01 and close >= df["medium_ma"].iloc[-1] * 0.99

            # MACD momentum confirmation
            macd_crossover = df["macd"].iloc[-2] < df["macd_signal"].iloc[-2] and df["macd"].iloc[-1] > df["macd_signal"].iloc[-1]

            # Price above Keltner midline
            above_keltner_mid = close > df["keltner_mid"].iloc[-1]

            return above_keltner_mid and (short_ma_pullback or medium_ma_pullback or macd_crossover)

        elif self.current_market_regime in [REGIME_TYPES.WEAK_TREND, REGIME_TYPES.RANGING]:
            # In weak trends or ranges, be more conservative
            # Look for bounces off support or moving averages
            medium_ma_support = df["close"].iloc[-2] < df["medium_ma"].iloc[-2] and close > df["medium_ma"].iloc[-1]

            # Bounce off Keltner lower band
            keltner_bounce = df["low"].iloc[-2] <= df["keltner_low"].iloc[-2] and close > df["keltner_low"].iloc[-1]

            # MACD positive momentum
            macd_positive = df["macd_hist"].iloc[-1] > 0 and df["macd_hist"].iloc[-1] > df["macd_hist"].iloc[-2]

            return macd_positive and (medium_ma_support or keltner_bounce)

        elif self.current_market_regime in [REGIME_TYPES.VOLATILE_RANGE, REGIME_TYPES.CHOPPY]:
            # In highly volatile or choppy markets, be very selective
            # Consider only strong bounces with multiple confirmations
            strong_support_bounce = df["low"].iloc[-3:].min() <= df["keltner_low"].iloc[-3:].min() and close > df["keltner_mid"].iloc[-1]

            # Strong MACD signal
            strong_macd = df["macd_hist"].iloc[-1] > 0 and df["macd_hist"].iloc[-1] > df["macd_hist"].iloc[-2] * 1.5

            # ADX showing increasing trend strength
            increasing_adx = df["adx"].iloc[-1] > df["adx"].iloc[-2] > df["adx"].iloc[-3]

            return strong_support_bounce and strong_macd and increasing_adx

        # Default case
        return False

    def _check_downtrend_entry_conditions(self, df: pd.DataFrame) -> bool:
        """
        Check entry conditions for short positions in a downtrend.

        Args:
            df: DataFrame with market data

        Returns:
            bool: True if entry conditions are met
        """
        # Minimum required ADX for a trending market
        if df["adx"].iloc[-1] < self.config["adx_threshold"]:
            return False

        # SuperTrend confirmation
        if df["supertrend_direction"].iloc[-1] >= 0:
            return False

        # Check for rallies to moving average (entry opportunity)
        close = df["close"].iloc[-1]

        # Identify market regime appropriate entry conditions
        if self.current_market_regime in [REGIME_TYPES.STRONG_TREND, REGIME_TYPES.VOLATILE_TREND]:
            # In strong trends, look for rallies to short or medium MA
            short_ma_rally = df["close"].iloc[-2] > df["short_ma"].iloc[-2] and close < df["short_ma"].iloc[-1]

            medium_ma_rally = close >= df["medium_ma"].iloc[-1] * 0.99 and close <= df["medium_ma"].iloc[-1] * 1.01

            # MACD momentum confirmation
            macd_crossover = df["macd"].iloc[-2] > df["macd_signal"].iloc[-2] and df["macd"].iloc[-1] < df["macd_signal"].iloc[-1]

            # Price below Keltner midline
            below_keltner_mid = close < df["keltner_mid"].iloc[-1]

            return below_keltner_mid and (short_ma_rally or medium_ma_rally or macd_crossover)

        elif self.current_market_regime in [REGIME_TYPES.WEAK_TREND, REGIME_TYPES.RANGING]:
            # In weak trends or ranges, be more conservative
            # Look for bounces off resistance or moving averages
            medium_ma_resistance = df["close"].iloc[-2] > df["medium_ma"].iloc[-2] and close < df["medium_ma"].iloc[-1]

            # Bounce off Keltner upper band
            keltner_bounce = df["high"].iloc[-2] >= df["keltner_high"].iloc[-2] and close < df["keltner_high"].iloc[-1]

            # MACD negative momentum
            macd_negative = df["macd_hist"].iloc[-1] < 0 and df["macd_hist"].iloc[-1] < df["macd_hist"].iloc[-2]

            return macd_negative and (medium_ma_resistance or keltner_bounce)

        elif self.current_market_regime in [REGIME_TYPES.VOLATILE_RANGE, REGIME_TYPES.CHOPPY]:
            # In highly volatile or choppy markets, be very selective
            # Consider only strong bounces with multiple confirmations
            strong_resistance_bounce = df["high"].iloc[-3:].max() >= df["keltner_high"].iloc[-3:].max() and close < df["keltner_mid"].iloc[-1]

            # Strong MACD signal
            strong_macd = df["macd_hist"].iloc[-1] < 0 and df["macd_hist"].iloc[-1] < df["macd_hist"].iloc[-2] * 1.5

            # ADX showing increasing trend strength
            increasing_adx = df["adx"].iloc[-1] > df["adx"].iloc[-2] > df["adx"].iloc[-3]

            return strong_resistance_bounce and strong_macd and increasing_adx

        # Default case
        return False

    def _calculate_exit_targets(self, df: pd.DataFrame, overall_trend: Dict[str, Any]) -> None:
        """
        Calculate exit targets based on current position and market conditions.

        Args:
            df: DataFrame with market data
            overall_trend: Dict with trend direction and strength
        """
        if not self.entry_prices:
            return

        # Calculate average entry price
        avg_entry = sum(price * size for price, size in zip(self.entry_prices, self.position_size_percentages))
        avg_entry /= sum(self.position_size_percentages)

        current_price = df["close"].iloc[-1]
        current_atr = self.current_atr

        # Calculate profit target
        if self.current_trend == "up":
            profit_target = avg_entry + (current_atr * self.config["profit_target_atr_multiplier"])
            stop_loss = avg_entry - (current_atr * self.config["stop_loss_atr_multiplier"])
            # Trailing stop activates after price has moved in our favor by a certain amount
            activation_threshold = avg_entry + (current_atr * self.config["trailing_stop_activation_atr_multiplier"])
            if current_price >= activation_threshold and self.config["trailing_stop_enabled"]:
                trailing_stop = current_price - (current_atr * self.config["trailing_stop_atr_multiplier"])
                trailing_stop = max(trailing_stop, stop_loss)  # Ensure trailing stop is above initial stop
            else:
                trailing_stop = None
        else:  # downtrend
            profit_target = avg_entry - (current_atr * self.config["profit_target_atr_multiplier"])
            stop_loss = avg_entry + (current_atr * self.config["stop_loss_atr_multiplier"])
            # Trailing stop activates after price has moved in our favor by a certain amount
            activation_threshold = avg_entry - (current_atr * self.config["trailing_stop_activation_atr_multiplier"])
            if current_price <= activation_threshold and self.config["trailing_stop_enabled"]:
                trailing_stop = current_price + (current_atr * self.config["trailing_stop_atr_multiplier"])
                trailing_stop = min(trailing_stop, stop_loss)  # Ensure trailing stop is below initial stop
            else:
                trailing_stop = None

        # Time-based stop
        last_entry_index = df.index.get_loc(df.index[-1])
        time_stop_index = last_entry_index + self.config["time_stop_bars"]
        if time_stop_index < len(df.index):
            time_stop = df.index[time_stop_index]
        else:
            # Estimate future time based on average bar duration
            avg_bar_duration = (df.index[-1] - df.index[0]) / (len(df.index) - 1)
            time_stop = df.index[-1] + (avg_bar_duration * self.config["time_stop_bars"])

        # Update exit targets
        self.current_exit_targets = {"profit_target": profit_target, "stop_loss": stop_loss, "trailing_stop": trailing_stop, "time_stop": time_stop}

    def _check_exit_conditions(self, df: pd.DataFrame, overall_trend: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if exit conditions are met.

        Args:
            df: DataFrame with market data
            overall_trend: Dict with trend direction and strength

        Returns:
            Dict with exit reason and strength if exit conditions met, None otherwise
        """
        if not self.entry_prices:
            return None

        current_price = df["close"].iloc[-1]
        current_time = pd.Timestamp.now() if df.index[-1].tzinfo else df.index[-1]

        # 1. Trend reversal exit
        trend_changed = (self.current_trend == "up" and overall_trend["trend_direction"] < 0) or (
            self.current_trend == "down" and overall_trend["trend_direction"] > 0
        )

        if trend_changed:
            return {"reason": "trend_reversal", "strength": 0.9}  # High confidence exit

        # 2. Profit target hit
        profit_target_hit = (self.current_trend == "up" and current_price >= self.current_exit_targets["profit_target"]) or (
            self.current_trend == "down" and current_price <= self.current_exit_targets["profit_target"]
        )

        if profit_target_hit:
            return {"reason": "profit_target", "strength": 0.8}  # High confidence exit

        # 3. Stop loss hit
        stop_loss_hit = (self.current_trend == "up" and current_price <= self.current_exit_targets["stop_loss"]) or (
            self.current_trend == "down" and current_price >= self.current_exit_targets["stop_loss"]
        )

        if stop_loss_hit:
            return {"reason": "stop_loss", "strength": 1.0}  # Highest confidence exit - must exit

        # 4. Trailing stop hit
        trailing_stop_hit = False
        if self.current_exit_targets["trailing_stop"] is not None:
            trailing_stop_hit = (self.current_trend == "up" and current_price <= self.current_exit_targets["trailing_stop"]) or (
                self.current_trend == "down" and current_price >= self.current_exit_targets["trailing_stop"]
            )

        if trailing_stop_hit:
            return {"reason": "trailing_stop", "strength": 0.9}  # High confidence exit

        # 5. Time stop reached
        time_stop_reached = current_time >= self.current_exit_targets["time_stop"]

        if time_stop_reached:
            # Only exit on time stop if we're in profit or trend is weakening
            avg_entry = sum(price * size for price, size in zip(self.entry_prices, self.position_size_percentages))
            avg_entry /= sum(self.position_size_percentages)

            in_profit = (self.current_trend == "up" and current_price > avg_entry) or (self.current_trend == "down" and current_price < avg_entry)

            trend_weakening = overall_trend["trend_strength"] < 50

            if in_profit or trend_weakening:
                return {"reason": "time_stop", "strength": 0.7 if in_profit else 0.5}

        # 6. Early reversal detection
        if self.config["early_reversal_detection"]:
            # Check for potential reversal signals
            potential_reversal = False

            if self.current_trend == "up":
                # Check for bearish reversal patterns
                # Example: Lower high followed by lower low
                recent_highs = df["high"].iloc[-4:]
                recent_lows = df["low"].iloc[-4:]

                lower_high = recent_highs.iloc[-1] < recent_highs.iloc[-3]
                lower_low = recent_lows.iloc[-1] < recent_lows.iloc[-2]

                # Bearish MACD divergence
                price_higher = df["close"].iloc[-1] > df["close"].iloc[-5]
                macd_lower = df["macd"].iloc[-1] < df["macd"].iloc[-5]

                bearish_divergence = price_higher and macd_lower

                # SuperTrend turning bearish
                supertrend_bearish = df["supertrend_direction"].iloc[-1] < 0

                # Combine signals
                potential_reversal = (lower_high and lower_low) or bearish_divergence or supertrend_bearish

            elif self.current_trend == "down":
                # Check for bullish reversal patterns
                # Example: Higher low followed by higher high
                recent_highs = df["high"].iloc[-4:]
                recent_lows = df["low"].iloc[-4:]

                higher_low = recent_lows.iloc[-1] > recent_lows.iloc[-3]
                higher_high = recent_highs.iloc[-1] > recent_highs.iloc[-2]

                # Bullish MACD divergence
                price_lower = df["close"].iloc[-1] < df["close"].iloc[-5]
                macd_higher = df["macd"].iloc[-1] > df["macd"].iloc[-5]

                bullish_divergence = price_lower and macd_higher

                # SuperTrend turning bullish
                supertrend_bullish = df["supertrend_direction"].iloc[-1] > 0

                # Combine signals
                potential_reversal = (higher_low and higher_high) or bullish_divergence or supertrend_bullish

            if potential_reversal:
                # Check if we're in profit
                in_profit = (self.current_trend == "up" and current_price > avg_entry) or (self.current_trend == "down" and current_price < avg_entry)

                if in_profit:
                    return {"reason": "early_reversal", "strength": 0.6}  # Medium confidence exit

        # No exit conditions met
        return None

    def _get_regime_confidence_factor(self) -> float:
        """
        Get confidence factor based on current market regime.

        Returns:
            float: Confidence factor (0-1)
        """
        regime_factors = {
            REGIME_TYPES.STRONG_TREND: 1.0,
            REGIME_TYPES.VOLATILE_TREND: 0.8,
            REGIME_TYPES.WEAK_TREND: 0.7,
            REGIME_TYPES.RANGING: 0.6,
            REGIME_TYPES.VOLATILE_RANGE: 0.5,
            REGIME_TYPES.CHOPPY: 0.4,
        }

        return regime_factors.get(self.current_market_regime, 0.7)

    def _calculate_volatility_percentile(self, atr_series: pd.Series, lookback: int) -> float:
        """
        Calculate the percentile of current volatility compared to historical.

        Args:
            atr_series: Series of ATR values
            lookback: Number of periods to look back

        Returns:
            float: Percentile of current volatility (0-1)
        """
        if len(atr_series) < lookback:
            lookback = len(atr_series)

        # Get historical volatility data
        historical_atr = atr_series.iloc[-lookback:-1]
        current_atr = atr_series.iloc[-1]

        # Calculate percentile
        percentile = sum(historical_atr < current_atr) / len(historical_atr)

        return percentile

    async def adaptive_parameter_update(self, performance_metrics: Dict[str, Any]) -> None:
        """
        Update strategy parameters based on performance metrics.

        Args:
            performance_metrics: Dict with performance metrics
        """
        if not performance_metrics:
            return

        self.logger.info("Updating strategy parameters based on performance metrics")

        # Extract relevant metrics
        win_rate = performance_metrics.get("win_rate", 0.5)
        avg_win_loss_ratio = performance_metrics.get("avg_win_loss_ratio", 1.0)
        recent_trades = performance_metrics.get("recent_trades", [])
        current_drawdown = performance_metrics.get("current_drawdown", 0.0)
        profit_factor = performance_metrics.get("profit_factor", 1.0)

        # Adjust trend strength filter based on win rate
        if win_rate < 0.4:
            # Become more selective
            self.config["trend_strength_filter"] = "high"
        elif win_rate > 0.6:
            # Can be less selective if we're winning
            if profit_factor > 1.5:
                self.config["trend_strength_filter"] = "low"
            else:
                self.config["trend_strength_filter"] = "medium"

        # Adjust profit targets and stop losses based on win/loss ratio
        if avg_win_loss_ratio < 1.0:
            # Increase profit target relative to stop loss
            self.config["profit_target_atr_multiplier"] += 0.2
        elif avg_win_loss_ratio > 2.0:
            # Tighten stop loss and reduce profit target
            self.config["stop_loss_atr_multiplier"] -= 0.1
            self.config["profit_target_atr_multiplier"] -= 0.1

        # Adjust early reversal detection based on drawdown
        if current_drawdown > 0.1:  # 10% drawdown
            # Enable early reversal detection to reduce drawdowns
            self.config["early_reversal_detection"] = True

        # Adjust strategy parameters to the min/max allowed values
        self.config["stop_loss_atr_multiplier"] = max(0.5, min(self.config["stop_loss_atr_multiplier"], 3.0))
        self.config["profit_target_atr_multiplier"] = max(1.0, min(self.config["profit_target_atr_multiplier"], 5.0))

        self.logger.info(f"Updated parameters: {self.config}")

    async def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information and current state.

        Returns:
            Dict with strategy information
        """
        return {
            "name": self.name,
            "type": "trend_following",
            "current_trend": self.current_trend,
            "trend_strength": self.trend_strength,
            "trend_start_time": self.trend_start_time,
            "current_market_regime": self.current_market_regime,
            "current_volatility": self.current_volatility,
            "current_atr": self.current_atr,
            "config": self.config,
            "performance": {
                "signals_generated": self.signals_generated,
                "successful_signals": self.successful_signals,
                "false_signals": self.false_signals,
                "success_rate": self.successful_signals / max(1, self.signals_generated),
            },
            "exit_targets": self.current_exit_targets,
        }
