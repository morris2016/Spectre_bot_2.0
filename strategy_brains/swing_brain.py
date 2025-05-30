#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Swing Trading Brain Strategy

This module implements a sophisticated swing trading strategy that identifies and exploits
price swing points, reversals, momentum shifts, and market structure changes to execute
high-probability, short to medium-term trades across various market conditions.
"""

import asyncio
import logging
import math
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from common.constants import MARKET_STRUCTURE, REGIME_TYPES, SWING_STRENGTH_LEVELS, SWING_TRADING_CONFIG
from common.exceptions import InsufficientDataError, SignalGenerationError, StrategyError
from common.utils import calculate_atr, detect_divergence, find_swing_points, is_higher_high, is_lower_low
from feature_service.features.market_structure import analyze_swing_points, detect_wyckoff_patterns, identify_market_structure
from feature_service.features.technical import calculate_bollinger_bands, calculate_ichimoku, calculate_macd, calculate_rsi, calculate_stochastic
from feature_service.features.volatility import calculate_keltner_channel
from feature_service.features.volume import analyze_volume_profile, detect_volume_climax
from strategy_brains.base_brain import BaseBrain


class SwingBrain(BaseBrain):
    """
    A sophisticated swing trading strategy brain that identifies high-probability reversal
    and continuation points in the market for short to medium-term trades.

    Features:
    - Swing point identification and qualification
    - Market structure analysis and range/trend determination
    - Multiple oscillator convergence for confirmation
    - Support/resistance zone recognition
    - Volume profile and climax detection
    - Pattern recognition (Wyckoff, harmonics, etc.)
    - Divergence and hidden divergence detection
    - Adaptive parameters based on volatility and market regime
    """

    def __init__(self, name: str = "swing_brain", **kwargs):
        """
        Initialize the SwingBrain strategy.

        Args:
            name: Unique name for this brain instance
            **kwargs: Additional configuration parameters
        """
        super().__init__(name=name, **kwargs)

        # Strategy-specific configuration with smart defaults
        self.config = {
            "rsi_period": kwargs.get("rsi_period", 14),
            "rsi_overbought": kwargs.get("rsi_overbought", 70),
            "rsi_oversold": kwargs.get("rsi_oversold", 30),
            "stoch_k_period": kwargs.get("stoch_k_period", 14),
            "stoch_d_period": kwargs.get("stoch_d_period", 3),
            "stoch_overbought": kwargs.get("stoch_overbought", 80),
            "stoch_oversold": kwargs.get("stoch_oversold", 20),
            "bollinger_period": kwargs.get("bollinger_period", 20),
            "bollinger_std_dev": kwargs.get("bollinger_std_dev", 2.0),
            "macd_fast": kwargs.get("macd_fast", 12),
            "macd_slow": kwargs.get("macd_slow", 26),
            "macd_signal": kwargs.get("macd_signal", 9),
            "swing_lookback": kwargs.get("swing_lookback", 10),
            "swing_strength_filter": kwargs.get("swing_strength_filter", "medium"),  # low, medium, high
            "min_swing_size_atr": kwargs.get("min_swing_size_atr", 1.0),
            "volume_confirmation": kwargs.get("volume_confirmation", True),
            "divergence_detection": kwargs.get("divergence_detection", True),
            "pattern_recognition": kwargs.get("pattern_recognition", True),
            "adaptive_parameter_adjustment": kwargs.get("adaptive_parameter_adjustment", True),
            "profit_target_atr_multiplier": kwargs.get("profit_target_atr_multiplier", 2.0),
            "stop_loss_atr_multiplier": kwargs.get("stop_loss_atr_multiplier", 1.0),
            "ichimoku_confirmation": kwargs.get("ichimoku_confirmation", True),
            "require_multi_indicator_confirmation": kwargs.get("require_multi_indicator_confirmation", True),
            "min_indicators_agreement": kwargs.get("min_indicators_agreement", 3),
            "zone_recognition": kwargs.get("zone_recognition", True),
            "fib_retracement_levels": kwargs.get("fib_retracement_levels", [0.382, 0.5, 0.618, 0.786]),
            "wyckoff_detection": kwargs.get("wyckoff_detection", True),
            "harmonic_detection": kwargs.get("harmonic_detection", True),
            "counter_trend_allowed": kwargs.get("counter_trend_allowed", True),
        }

        # Internal state tracking
        self.current_market_structure = None  # 'uptrend', 'downtrend', 'range', 'accumulation', 'distribution'
        self.current_swing_high = None
        self.current_swing_low = None
        self.swing_points = []
        self.support_resistance_zones = []
        self.current_atr = None
        self.current_volatility = None
        self.divergences = []
        self.current_market_regime = None
        self.detected_patterns = []
        self.last_signal_time = None
        self.current_exit_targets = {"profit_target": None, "stop_loss": None, "trailing_stop": None, "time_stop": None}

        # Performance metrics
        self.signals_generated = 0
        self.successful_signals = 0
        self.false_signals = 0

        self.logger = logging.getLogger(f"strategy_brains.{self.name}")
        self.logger.info(f"Initialized {self.name} with configuration: {self.config}")

    async def analyze(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        Analyze market data to identify swing trading opportunities.

        Args:
            data: Dict of DataFrames with market data for different timeframes
            **kwargs: Additional parameters

        Returns:
            Dict containing analysis results and trading signals
        """
        self.logger.info(f"Running swing analysis on {len(data)} timeframes")

        try:
            # Ensure we have the primary timeframe data
            if not self.primary_timeframe in data:
                raise InsufficientDataError(f"Primary timeframe {self.primary_timeframe} data not available")

            # Get the primary timeframe data
            df = data[self.primary_timeframe].copy()
            if len(df) < 50:  # Need at least 50 bars for reliable analysis
                raise InsufficientDataError(f"Insufficient data for analysis. Need at least 50 bars.")

            # Calculate technical indicators
            self._calculate_indicators(df)

            # Detect market structure
            self._detect_market_structure(df)

            # Identify swing points
            self._identify_swing_points(df)

            # Identify support/resistance zones if enabled
            if self.config["zone_recognition"]:
                self._identify_support_resistance_zones(df)

            # Detect divergences if enabled
            if self.config["divergence_detection"]:
                self._detect_divergences(df)

            # Detect patterns if enabled
            if self.config["pattern_recognition"]:
                self._detect_patterns(df)

            # Analyze swing trading opportunities
            opportunities = self._analyze_swing_opportunities(df)

            # Generate trading signals
            signals = self._generate_signals(df, opportunities)

            # Calculate exit targets if we have an active position
            if signals["entry"] is not None:
                self._calculate_exit_targets(df, signals["entry"], opportunities)

            # Prepare analysis results
            analysis_results = {
                "market_structure": self.current_market_structure,
                "market_regime": self.current_market_regime,
                "current_volatility": self.current_volatility,
                "swing_points": self.swing_points[-5:] if self.swing_points else [],
                "support_resistance_zones": self.support_resistance_zones,
                "divergences": self.divergences,
                "detected_patterns": self.detected_patterns,
                "signals": signals,
                "exit_targets": self.current_exit_targets,
                "opportunities": opportunities,
                "raw_metrics": {
                    "rsi": df["rsi"].iloc[-1] if "rsi" in df else None,
                    "stoch_k": df["stoch_k"].iloc[-1] if "stoch_k" in df else None,
                    "stoch_d": df["stoch_d"].iloc[-1] if "stoch_d" in df else None,
                    "macd": {
                        "macd": df["macd"].iloc[-1] if "macd" in df else None,
                        "signal": df["macd_signal"].iloc[-1] if "macd_signal" in df else None,
                        "histogram": df["macd_hist"].iloc[-1] if "macd_hist" in df else None,
                    },
                    "bollinger": {
                        "upper": df["bb_upper"].iloc[-1] if "bb_upper" in df else None,
                        "middle": df["bb_middle"].iloc[-1] if "bb_middle" in df else None,
                        "lower": df["bb_lower"].iloc[-1] if "bb_lower" in df else None,
                    },
                },
                "timestamps": {"analysis_time": datetime.now(), "last_signal_time": self.last_signal_time},
            }

            self.logger.info(f"Swing analysis complete. Market structure: {self.current_market_structure}, Opportunities: {len(opportunities)}")
            return analysis_results

        except Exception as e:
            self.logger.error(f"Error in swing analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise StrategyError(f"Swing analysis failed: {str(e)}")

    def _calculate_indicators(self, df: pd.DataFrame) -> None:
        """
        Calculate all required technical indicators for swing analysis.

        Args:
            df: DataFrame with market data
        """
        # Make sure we have OHLCV data
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column {col} not found in data")

        # Calculate RSI
        df["rsi"] = calculate_rsi(df["close"], period=self.config["rsi_period"])

        # Calculate Stochastic
        stoch = calculate_stochastic(
            df["high"], df["low"], df["close"], k_period=self.config["stoch_k_period"], d_period=self.config["stoch_d_period"], smooth_k=3
        )
        df["stoch_k"] = stoch["k"]
        df["stoch_d"] = stoch["d"]

        # Calculate Bollinger Bands
        bollinger = calculate_bollinger_bands(df["close"], period=self.config["bollinger_period"], std_dev=self.config["bollinger_std_dev"])
        df["bb_upper"] = bollinger["upper"]
        df["bb_middle"] = bollinger["middle"]
        df["bb_lower"] = bollinger["lower"]

        # Calculate MACD
        macd_result = calculate_macd(
            df["close"], fast_period=self.config["macd_fast"], slow_period=self.config["macd_slow"], signal_period=self.config["macd_signal"]
        )
        df["macd"] = macd_result["macd"]
        df["macd_signal"] = macd_result["signal"]
        df["macd_hist"] = macd_result["histogram"]

        # Calculate Keltner Channels
        keltner = calculate_keltner_channel(df["high"], df["low"], df["close"], period=20, atr_multiplier=2.0)
        df["keltner_upper"] = keltner["upper"]
        df["keltner_middle"] = keltner["middle"]
        df["keltner_lower"] = keltner["lower"]

        # Calculate Ichimoku if enabled
        if self.config["ichimoku_confirmation"]:
            ichimoku = calculate_ichimoku(df["high"], df["low"], df["close"])
            df["tenkan_sen"] = ichimoku["tenkan_sen"]
            df["kijun_sen"] = ichimoku["kijun_sen"]
            df["senkou_span_a"] = ichimoku["senkou_span_a"]
            df["senkou_span_b"] = ichimoku["senkou_span_b"]
            df["chikou_span"] = ichimoku["chikou_span"]

        # Calculate volatility and ATR
        df["atr"] = calculate_atr(df["high"], df["low"], df["close"], period=14)
        self.current_atr = df["atr"].iloc[-1]
        self.current_volatility = df["atr"].iloc[-1] / df["close"].iloc[-1]

        # Volume analysis
        if self.config["volume_confirmation"]:
            volume_profile = analyze_volume_profile(df["close"], df["volume"], periods=20)
            df["volume_sma"] = volume_profile["volume_sma"]
            df["relative_volume"] = volume_profile["relative_volume"]
            df["volume_climax"] = detect_volume_climax(df["volume"], df["close"], lookback=20)

            # Calculate OBV (On-Balance Volume)
            df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    def _detect_market_structure(self, df: pd.DataFrame) -> None:
        """
        Detect the current market structure (uptrend, downtrend, range, etc.)

        Args:
            df: DataFrame with market data and indicators
        """
        # Use market structure detection from feature service
        structure_result = identify_market_structure(df["high"], df["low"], df["close"], lookback=50)

        self.current_market_structure = structure_result["structure_type"]

        # Detect market regime (trending, ranging, volatile)
        # Calculate price efficiency (directional movement / total movement)
        price_change = abs(df["close"].iloc[-1] - df["close"].iloc[-20])
        total_movement = sum(abs(df["close"].diff().iloc[-20:].fillna(0)))
        efficiency_ratio = price_change / total_movement if total_movement > 0 else 0

        # Calculate recent volatility relative to historical
        recent_volatility = df["atr"].iloc[-10:].mean() / df["close"].iloc[-10:].mean()
        historical_volatility = df["atr"].iloc[-50:-10].mean() / df["close"].iloc[-50:-10].mean()
        volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1.0

        # Determine regime based on combination of factors
        if self.current_market_structure in [MARKET_STRUCTURE.UPTREND, MARKET_STRUCTURE.DOWNTREND]:
            if efficiency_ratio > 0.7:
                self.current_market_regime = REGIME_TYPES.STRONG_TREND
            elif volatility_ratio > 1.5:
                self.current_market_regime = REGIME_TYPES.VOLATILE_TREND
            else:
                self.current_market_regime = REGIME_TYPES.WEAK_TREND
        else:  # RANGE, ACCUMULATION, DISTRIBUTION
            if volatility_ratio > 1.5:
                self.current_market_regime = REGIME_TYPES.VOLATILE_RANGE
            elif efficiency_ratio < 0.2:
                self.current_market_regime = REGIME_TYPES.CHOPPY
            else:
                self.current_market_regime = REGIME_TYPES.RANGING

        self.logger.debug(f"Detected market structure: {self.current_market_structure}, regime: {self.current_market_regime}")

    def _identify_swing_points(self, df: pd.DataFrame) -> None:
        """
        Identify significant swing points in the price data.

        Args:
            df: DataFrame with market data
        """
        # Use swing point analysis from feature service
        swing_points = analyze_swing_points(df["high"], df["low"], period=self.config["swing_lookback"])

        df["swing_high"] = swing_points["swing_highs"]
        df["swing_low"] = swing_points["swing_lows"]

        # Extract recent swing points and store them
        recent_highs = swing_points["swing_highs"].dropna().iloc[-5:]
        recent_lows = swing_points["swing_lows"].dropna().iloc[-5:]

        # Convert to list of dicts with price, index, and type
        swing_high_points = [{"type": "high", "price": price, "index": idx, "confirmed": True} for idx, price in recent_highs.items()]

        swing_low_points = [{"type": "low", "price": price, "index": idx, "confirmed": True} for idx, price in recent_lows.items()]

        # Combine swing points and sort by index
        self.swing_points = swing_high_points + swing_low_points
        self.swing_points.sort(key=lambda x: x["index"])

        # Update current swing high/low
        if len(swing_high_points) > 0:
            self.current_swing_high = swing_high_points[-1]

        if len(swing_low_points) > 0:
            self.current_swing_low = swing_low_points[-1]

    def _identify_support_resistance_zones(self, df: pd.DataFrame) -> None:
        """
        Identify key support and resistance zones.

        Args:
            df: DataFrame with market data
        """
        # Reset current zones
        self.support_resistance_zones = []

        # Use swing points to identify zones
        if len(self.swing_points) < 2:
            return

        # Group nearby swing highs and swing lows
        price_grouping_threshold = self.current_atr * 0.5  # Proximity threshold

        # Process swing highs for resistance zones
        swing_highs = [point for point in self.swing_points if point["type"] == "high"]
        high_zones = self._group_price_points(swing_highs, price_grouping_threshold)

        # Process swing lows for support zones
        swing_lows = [point for point in self.swing_points if point["type"] == "low"]
        low_zones = self._group_price_points(swing_lows, price_grouping_threshold)

        # Create resistance zones
        for zone in high_zones:
            if len(zone) >= 2:  # At least 2 touches to consider it a zone
                zone_price = sum(point["price"] for point in zone) / len(zone)
                zone_strength = len(zone) * 10  # Base strength on number of touches

                # Adjust strength based on recency (more recent = stronger)
                recency_factor = 1.0
                most_recent_index = max(point["index"] for point in zone)
                distance_from_current = len(df) - 1 - most_recent_index
                if distance_from_current < 10:
                    recency_factor = 1.5
                elif distance_from_current < 30:
                    recency_factor = 1.2

                zone_strength *= recency_factor

                self.support_resistance_zones.append(
                    {
                        "type": "resistance",
                        "price": zone_price,
                        "upper": zone_price + (price_grouping_threshold / 2),
                        "lower": zone_price - (price_grouping_threshold / 2),
                        "strength": zone_strength,
                        "touches": len(zone),
                    }
                )

        # Create support zones
        for zone in low_zones:
            if len(zone) >= 2:  # At least 2 touches to consider it a zone
                zone_price = sum(point["price"] for point in zone) / len(zone)
                zone_strength = len(zone) * 10  # Base strength on number of touches

                # Adjust strength based on recency (more recent = stronger)
                recency_factor = 1.0
                most_recent_index = max(point["index"] for point in zone)
                distance_from_current = len(df) - 1 - most_recent_index
                if distance_from_current < 10:
                    recency_factor = 1.5
                elif distance_from_current < 30:
                    recency_factor = 1.2

                zone_strength *= recency_factor

                self.support_resistance_zones.append(
                    {
                        "type": "support",
                        "price": zone_price,
                        "upper": zone_price + (price_grouping_threshold / 2),
                        "lower": zone_price - (price_grouping_threshold / 2),
                        "strength": zone_strength,
                        "touches": len(zone),
                    }
                )

        # Sort zones by price
        self.support_resistance_zones.sort(key=lambda x: x["price"])

        # Add Fibonacci retracement levels if we have a clear recent trend
        if len(self.swing_points) >= 2:
            # Find the most recent significant swing high and swing low
            recent_swing_points = sorted(self.swing_points, key=lambda x: x["index"], reverse=True)
            recent_high = next((point for point in recent_swing_points if point["type"] == "high"), None)
            recent_low = next((point for point in recent_swing_points if point["type"] == "low"), None)

            if recent_high and recent_low:
                # Determine trend direction from the order of the points
                if recent_high["index"] > recent_low["index"]:
                    # Uptrend - retrace from low to high
                    price_range = recent_high["price"] - recent_low["price"]
                    for level in self.config["fib_retracement_levels"]:
                        retracement_price = recent_high["price"] - (price_range * level)
                        self.support_resistance_zones.append(
                            {
                                "type": "fibonacci_support",
                                "price": retracement_price,
                                "upper": retracement_price + (self.current_atr * 0.2),
                                "lower": retracement_price - (self.current_atr * 0.2),
                                "strength": 50 * (1 if level == 0.618 else 0.8),  # 0.618 is the strongest level
                                "level": level,
                            }
                        )
                else:
                    # Downtrend - retrace from high to low
                    price_range = recent_high["price"] - recent_low["price"]
                    for level in self.config["fib_retracement_levels"]:
                        retracement_price = recent_low["price"] + (price_range * level)
                        self.support_resistance_zones.append(
                            {
                                "type": "fibonacci_resistance",
                                "price": retracement_price,
                                "upper": retracement_price + (self.current_atr * 0.2),
                                "lower": retracement_price - (self.current_atr * 0.2),
                                "strength": 50 * (1 if level == 0.618 else 0.8),  # 0.618 is the strongest level
                                "level": level,
                            }
                        )

        # Add Bollinger Bands and Keltner channels as dynamic zones
        bb_upper = df["bb_upper"].iloc[-1]
        bb_lower = df["bb_lower"].iloc[-1]
        keltner_upper = df["keltner_upper"].iloc[-1]
        keltner_lower = df["keltner_lower"].iloc[-1]

        self.support_resistance_zones.append(
            {
                "type": "dynamic_resistance",
                "price": bb_upper,
                "upper": bb_upper + (self.current_atr * 0.2),
                "lower": bb_upper - (self.current_atr * 0.2),
                "strength": 40,
                "source": "bollinger_upper",
            }
        )

        self.support_resistance_zones.append(
            {
                "type": "dynamic_support",
                "price": bb_lower,
                "upper": bb_lower + (self.current_atr * 0.2),
                "lower": bb_lower - (self.current_atr * 0.2),
                "strength": 40,
                "source": "bollinger_lower",
            }
        )

        self.support_resistance_zones.append(
            {
                "type": "dynamic_resistance",
                "price": keltner_upper,
                "upper": keltner_upper + (self.current_atr * 0.2),
                "lower": keltner_upper - (self.current_atr * 0.2),
                "strength": 35,
                "source": "keltner_upper",
            }
        )

        self.support_resistance_zones.append(
            {
                "type": "dynamic_support",
                "price": keltner_lower,
                "upper": keltner_lower + (self.current_atr * 0.2),
                "lower": keltner_lower - (self.current_atr * 0.2),
                "strength": 35,
                "source": "keltner_lower",
            }
        )

        # Add Ichimoku Cloud boundaries if enabled
        if self.config["ichimoku_confirmation"] and "senkou_span_a" in df and "senkou_span_b" in df:
            span_a = df["senkou_span_a"].iloc[-1]
            span_b = df["senkou_span_b"].iloc[-1]

            # Determine which is the upper and lower boundary
            if span_a > span_b:
                cloud_upper = span_a
                cloud_lower = span_b
                cloud_strength = 45  # Bullish cloud is stronger
            else:
                cloud_upper = span_b
                cloud_lower = span_a
                cloud_strength = 40  # Bearish cloud is stronger

            self.support_resistance_zones.append(
                {
                    "type": "dynamic_resistance",
                    "price": cloud_upper,
                    "upper": cloud_upper + (self.current_atr * 0.2),
                    "lower": cloud_upper - (self.current_atr * 0.2),
                    "strength": cloud_strength,
                    "source": "ichimoku_upper",
                }
            )

            self.support_resistance_zones.append(
                {
                    "type": "dynamic_support",
                    "price": cloud_lower,
                    "upper": cloud_lower + (self.current_atr * 0.2),
                    "lower": cloud_lower - (self.current_atr * 0.2),
                    "strength": cloud_strength - 5,  # Lower boundary slightly weaker
                    "source": "ichimoku_lower",
                }
            )

    def _group_price_points(self, points: List[Dict[str, Any]], threshold: float) -> List[List[Dict[str, Any]]]:
        """
        Group price points that are within a threshold distance of each other.

        Args:
            points: List of price points (dicts with 'price' and other metadata)
            threshold: Price distance threshold for grouping

        Returns:
            List of lists, where each inner list contains grouped price points
        """
        if not points:
            return []

        # Sort points by price
        sorted_points = sorted(points, key=lambda x: x["price"])

        # Initialize groups
        groups = [[sorted_points[0]]]

        # Group points
        for point in sorted_points[1:]:
            # Check if point is close enough to the last group
            last_group = groups[-1]
            avg_group_price = sum(p["price"] for p in last_group) / len(last_group)

            if abs(point["price"] - avg_group_price) <= threshold:
                # Add to existing group
                last_group.append(point)
            else:
                # Start new group
                groups.append([point])

        return groups

    def _detect_divergences(self, df: pd.DataFrame) -> None:
        """
        Detect regular and hidden divergences in oscillators.

        Args:
            df: DataFrame with market data and indicators
        """
        # Reset current divergences
        self.divergences = []

        # Find price swing points in recent data (last 50 bars)
        recent_df = df.iloc[-50:].copy() if len(df) > 50 else df.copy()

        # Regular divergence with RSI
        if "rsi" in recent_df:
            # Bullish divergence: Lower price low but higher RSI low
            price_lows = recent_df["low"].rolling(5, center=True).min()
            rsi_values_at_lows = recent_df["rsi"][price_lows == recent_df["low"]]

            if len(rsi_values_at_lows) >= 2:
                last_two_price_lows = price_lows[price_lows == recent_df["low"]].iloc[-2:]
                last_two_rsi_at_lows = rsi_values_at_lows.iloc[-2:]

                if last_two_price_lows.iloc[1] < last_two_price_lows.iloc[0] and last_two_rsi_at_lows.iloc[1] > last_two_rsi_at_lows.iloc[0]:
                    self.divergences.append(
                        {
                            "type": "bullish",
                            "indicator": "rsi",
                            "divergence_type": "regular",
                            "first_point": {
                                "price": last_two_price_lows.iloc[0],
                                "indicator": last_two_rsi_at_lows.iloc[0],
                                "index": last_two_price_lows.index[0],
                            },
                            "second_point": {
                                "price": last_two_price_lows.iloc[1],
                                "indicator": last_two_rsi_at_lows.iloc[1],
                                "index": last_two_price_lows.index[1],
                            },
                        }
                    )

            # Bearish divergence: Higher price high but lower RSI high
            price_highs = recent_df["high"].rolling(5, center=True).max()
            rsi_values_at_highs = recent_df["rsi"][price_highs == recent_df["high"]]

            if len(rsi_values_at_highs) >= 2:
                last_two_price_highs = price_highs[price_highs == recent_df["high"]].iloc[-2:]
                last_two_rsi_at_highs = rsi_values_at_highs.iloc[-2:]

                if last_two_price_highs.iloc[1] > last_two_price_highs.iloc[0] and last_two_rsi_at_highs.iloc[1] < last_two_rsi_at_highs.iloc[0]:
                    self.divergences.append(
                        {
                            "type": "bearish",
                            "indicator": "rsi",
                            "divergence_type": "regular",
                            "first_point": {
                                "price": last_two_price_highs.iloc[0],
                                "indicator": last_two_rsi_at_highs.iloc[0],
                                "index": last_two_price_highs.index[0],
                            },
                            "second_point": {
                                "price": last_two_price_highs.iloc[1],
                                "indicator": last_two_rsi_at_highs.iloc[1],
                                "index": last_two_price_highs.index[1],
                            },
                        }
                    )

        # MACD divergences
        if "macd" in recent_df:
            # Use the MACD histogram for divergence detection
            macd_hist_values = recent_df["macd_hist"]

            # Find local maxima and minima in MACD histogram
            macd_hist_maxima = macd_hist_values[(macd_hist_values.shift(1) < macd_hist_values) & (macd_hist_values.shift(-1) < macd_hist_values)]
            macd_hist_minima = macd_hist_values[(macd_hist_values.shift(1) > macd_hist_values) & (macd_hist_values.shift(-1) > macd_hist_values)]

            # Bullish divergence: Lower price low but higher MACD histogram low
            if len(macd_hist_minima) >= 2:
                last_two_macd_minima = macd_hist_minima.iloc[-2:]
                macd_min_indices = last_two_macd_minima.index

                # Find corresponding price lows around these points
                price_lows_near_macd = []
                for idx in macd_min_indices:
                    # Look for price low within 3 bars of the MACD low
                    window_start = max(0, recent_df.index.get_loc(idx) - 3)
                    window_end = min(len(recent_df), recent_df.index.get_loc(idx) + 4)
                    window = recent_df.iloc[window_start:window_end]
                    min_idx = window["low"].idxmin()
                    price_lows_near_macd.append((min_idx, recent_df.loc[min_idx, "low"]))

                if len(price_lows_near_macd) == 2:
                    if price_lows_near_macd[1][1] < price_lows_near_macd[0][1] and last_two_macd_minima.iloc[1] > last_two_macd_minima.iloc[0]:
                        self.divergences.append(
                            {
                                "type": "bullish",
                                "indicator": "macd_hist",
                                "divergence_type": "regular",
                                "first_point": {
                                    "price": price_lows_near_macd[0][1],
                                    "indicator": last_two_macd_minima.iloc[0],
                                    "index": price_lows_near_macd[0][0],
                                },
                                "second_point": {
                                    "price": price_lows_near_macd[1][1],
                                    "indicator": last_two_macd_minima.iloc[1],
                                    "index": price_lows_near_macd[1][0],
                                },
                            }
                        )

            # Bearish divergence: Higher price high but lower MACD histogram high
            if len(macd_hist_maxima) >= 2:
                last_two_macd_maxima = macd_hist_maxima.iloc[-2:]
                macd_max_indices = last_two_macd_maxima.index

                # Find corresponding price highs around these points
                price_highs_near_macd = []
                for idx in macd_max_indices:
                    # Look for price high within 3 bars of the MACD high
                    window_start = max(0, recent_df.index.get_loc(idx) - 3)
                    window_end = min(len(recent_df), recent_df.index.get_loc(idx) + 4)
                    window = recent_df.iloc[window_start:window_end]
                    max_idx = window["high"].idxmax()
                    price_highs_near_macd.append((max_idx, recent_df.loc[max_idx, "high"]))

                if len(price_highs_near_macd) == 2:
                    if price_highs_near_macd[1][1] > price_highs_near_macd[0][1] and last_two_macd_maxima.iloc[1] < last_two_macd_maxima.iloc[0]:
                        self.divergences.append(
                            {
                                "type": "bearish",
                                "indicator": "macd_hist",
                                "divergence_type": "regular",
                                "first_point": {
                                    "price": price_highs_near_macd[0][1],
                                    "indicator": last_two_macd_maxima.iloc[0],
                                    "index": price_highs_near_macd[0][0],
                                },
                                "second_point": {
                                    "price": price_highs_near_macd[1][1],
                                    "indicator": last_two_macd_maxima.iloc[1],
                                    "index": price_highs_near_macd[1][0],
                                },
                            }
                        )

    def _detect_patterns(self, df: pd.DataFrame) -> None:
        """
        Detect chart patterns in the price data.

        Args:
            df: DataFrame with market data
        """
        # Reset current patterns
        self.detected_patterns = []

        # Detect Wyckoff patterns if enabled
        if self.config["wyckoff_detection"]:
            wyckoff_patterns = detect_wyckoff_patterns(df["high"], df["low"], df["close"], df["volume"])

            if wyckoff_patterns:
                for pattern in wyckoff_patterns:
                    self.detected_patterns.append(
                        {
                            "type": "wyckoff",
                            "pattern": pattern["pattern"],
                            "phase": pattern["phase"],
                            "strength": pattern["strength"],
                            "start_idx": pattern["start_idx"],
                            "end_idx": pattern["end_idx"],
                        }
                    )

        # Detect harmonic patterns if enabled
        if self.config["harmonic_detection"]:
            # Simplified harmonic pattern detection
            # In a real implementation, this would be more sophisticated

            # Check for potential harmonic patterns using Fibonacci ratios
            if len(self.swing_points) >= 4:
                recent_points = sorted(self.swing_points, key=lambda x: x["index"])[-4:]

                # Need alternating highs and lows
                if len(set(point["type"] for point in recent_points)) == 2:
                    # Check if points form potential harmonic pattern
                    # Calculate XA, AB, BC, CD legs
                    leg_sizes = []
                    for i in range(len(recent_points) - 1):
                        leg_sizes.append(abs(recent_points[i + 1]["price"] - recent_points[i]["price"]))

                    # Calculate ratios between legs
                    ab_xa = leg_sizes[1] / leg_sizes[0] if leg_sizes[0] != 0 else 0
                    cd_ab = leg_sizes[2] / leg_sizes[1] if leg_sizes[1] != 0 else 0

                    # Check for potential Bat pattern
                    if (0.38 <= ab_xa <= 0.5) and (1.27 <= cd_ab <= 2.0):
                        self.detected_patterns.append(
                            {
                                "type": "harmonic",
                                "pattern": "bat",
                                "strength": 70,
                                "points": recent_points,
                                "completion_area": {
                                    "price": recent_points[-1]["price"],
                                    "upper": recent_points[-1]["price"] + (self.current_atr * 0.5),
                                    "lower": recent_points[-1]["price"] - (self.current_atr * 0.5),
                                },
                            }
                        )

                    # Check for potential Butterfly pattern
                    elif (0.786 <= ab_xa <= 0.886) and (1.618 <= cd_ab <= 2.618):
                        self.detected_patterns.append(
                            {
                                "type": "harmonic",
                                "pattern": "butterfly",
                                "strength": 75,
                                "points": recent_points,
                                "completion_area": {
                                    "price": recent_points[-1]["price"],
                                    "upper": recent_points[-1]["price"] + (self.current_atr * 0.5),
                                    "lower": recent_points[-1]["price"] - (self.current_atr * 0.5),
                                },
                            }
                        )

                    # Check for potential Gartley pattern
                    elif (0.618 <= ab_xa <= 0.618) and (1.272 <= cd_ab <= 1.618):
                        self.detected_patterns.append(
                            {
                                "type": "harmonic",
                                "pattern": "gartley",
                                "strength": 65,
                                "points": recent_points,
                                "completion_area": {
                                    "price": recent_points[-1]["price"],
                                    "upper": recent_points[-1]["price"] + (self.current_atr * 0.5),
                                    "lower": recent_points[-1]["price"] - (self.current_atr * 0.5),
                                },
                            }
                        )

    def _analyze_swing_opportunities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze the market for swing trading opportunities.

        Args:
            df: DataFrame with market data

        Returns:
            List of potential swing trading opportunities
        """
        opportunities = []
        current_price = df["close"].iloc[-1]

        # Check for oversold conditions in an uptrend or support zone
        if (self.current_market_structure == MARKET_STRUCTURE.UPTREND or self.current_market_structure == MARKET_STRUCTURE.ACCUMULATION) and (
            (df["rsi"].iloc[-1] < self.config["rsi_oversold"])
            or (df["stoch_k"].iloc[-1] < self.config["stoch_oversold"] and df["stoch_d"].iloc[-1] < self.config["stoch_oversold"])
        ):

            # Check if price is near a support zone
            near_support = False
            support_zone = None
            support_distance = float("inf")

            for zone in self.support_resistance_zones:
                if zone["type"] in ["support", "dynamic_support", "fibonacci_support"]:
                    distance = abs(current_price - zone["price"])
                    if distance < self.current_atr * 1.0 and distance < support_distance:
                        near_support = True
                        support_zone = zone
                        support_distance = distance

            # Calculate opportunity score
            if near_support:
                # Calculate indicator agreement
                indicators_bullish = []

                # RSI oversold
                if df["rsi"].iloc[-1] < self.config["rsi_oversold"]:
                    indicators_bullish.append(("rsi_oversold", 1))

                # Stochastic oversold
                if df["stoch_k"].iloc[-1] < self.config["stoch_oversold"] and df["stoch_d"].iloc[-1] < self.config["stoch_oversold"]:
                    indicators_bullish.append(("stoch_oversold", 1))

                # MACD turning up
                if df["macd_hist"].iloc[-1] > df["macd_hist"].iloc[-2] and df["macd_hist"].iloc[-2] > df["macd_hist"].iloc[-3]:
                    indicators_bullish.append(("macd_turning_up", 1))

                # Bollinger band bounce
                if current_price <= df["bb_lower"].iloc[-1]:
                    indicators_bullish.append(("bollinger_bounce", 1))

                # Bullish divergence
                bullish_divergences = [div for div in self.divergences if div["type"] == "bullish"]
                if bullish_divergences:
                    indicators_bullish.append(("bullish_divergence", 1.5))

                # Volume confirmation if enabled
                if self.config["volume_confirmation"] and "relative_volume" in df:
                    if df["relative_volume"].iloc[-1] > 1.5 and df["close"].iloc[-1] > df["open"].iloc[-1]:
                        indicators_bullish.append(("volume_confirmation", 1))

                # Ichimoku confirmation if enabled
                if self.config["ichimoku_confirmation"] and "tenkan_sen" in df and "kijun_sen" in df:
                    if current_price > df["senkou_span_a"].iloc[-1] and current_price > df["senkou_span_b"].iloc[-1]:
                        indicators_bullish.append(("above_ichimoku_cloud", 1))

                # Calculate score based on indicators and support zone strength
                indicator_score = sum(weight for _, weight in indicators_bullish)
                zone_strength = support_zone["strength"] / 100 if support_zone else 0.5

                # Combine scores
                opportunity_score = (indicator_score * 0.6) + (zone_strength * 0.4)
                opportunity_score = min(opportunity_score, 1.0) * 100  # Scale to 0-100

                # Check if we have enough indicator agreement
                if len(indicators_bullish) >= self.config["min_indicators_agreement"] or not self.config["require_multi_indicator_confirmation"]:
                    opportunities.append(
                        {
                            "type": "long",
                            "price": current_price,
                            "score": opportunity_score,
                            "reason": "oversold_at_support",
                            "indicators": indicators_bullish,
                            "support_zone": support_zone,
                            "stop_loss": (
                                support_zone["lower"] - (self.current_atr * 0.5)
                                if support_zone
                                else current_price - (self.current_atr * self.config["stop_loss_atr_multiplier"])
                            ),
                            "profit_target": current_price + (self.current_atr * self.config["profit_target_atr_multiplier"]),
                        }
                    )

        # Check for overbought conditions in a downtrend or resistance zone
        if (self.current_market_structure == MARKET_STRUCTURE.DOWNTREND or self.current_market_structure == MARKET_STRUCTURE.DISTRIBUTION) and (
            (df["rsi"].iloc[-1] > self.config["rsi_overbought"])
            or (df["stoch_k"].iloc[-1] > self.config["stoch_overbought"] and df["stoch_d"].iloc[-1] > self.config["stoch_overbought"])
        ):

            # Check if price is near a resistance zone
            near_resistance = False
            resistance_zone = None
            resistance_distance = float("inf")

            for zone in self.support_resistance_zones:
                if zone["type"] in ["resistance", "dynamic_resistance", "fibonacci_resistance"]:
                    distance = abs(current_price - zone["price"])
                    if distance < self.current_atr * 1.0 and distance < resistance_distance:
                        near_resistance = True
                        resistance_zone = zone
                        resistance_distance = distance

            # Calculate opportunity score
            if near_resistance:
                # Calculate indicator agreement
                indicators_bearish = []

                # RSI overbought
                if df["rsi"].iloc[-1] > self.config["rsi_overbought"]:
                    indicators_bearish.append(("rsi_overbought", 1))

                # Stochastic overbought
                if df["stoch_k"].iloc[-1] > self.config["stoch_overbought"] and df["stoch_d"].iloc[-1] > self.config["stoch_overbought"]:
                    indicators_bearish.append(("stoch_overbought", 1))

                # MACD turning down
                if df["macd_hist"].iloc[-1] < df["macd_hist"].iloc[-2] and df["macd_hist"].iloc[-2] < df["macd_hist"].iloc[-3]:
                    indicators_bearish.append(("macd_turning_down", 1))

                # Bollinger band bounce
                if current_price >= df["bb_upper"].iloc[-1]:
                    indicators_bearish.append(("bollinger_bounce", 1))

                # Bearish divergence
                bearish_divergences = [div for div in self.divergences if div["type"] == "bearish"]
                if bearish_divergences:
                    indicators_bearish.append(("bearish_divergence", 1.5))

                # Volume confirmation if enabled
                if self.config["volume_confirmation"] and "relative_volume" in df:
                    if df["relative_volume"].iloc[-1] > 1.5 and df["close"].iloc[-1] < df["open"].iloc[-1]:
                        indicators_bearish.append(("volume_confirmation", 1))

                # Ichimoku confirmation if enabled
                if self.config["ichimoku_confirmation"] and "tenkan_sen" in df and "kijun_sen" in df:
                    if current_price < df["senkou_span_a"].iloc[-1] and current_price < df["senkou_span_b"].iloc[-1]:
                        indicators_bearish.append(("below_ichimoku_cloud", 1))

                # Calculate score based on indicators and resistance zone strength
                indicator_score = sum(weight for _, weight in indicators_bearish)
                zone_strength = resistance_zone["strength"] / 100 if resistance_zone else 0.5

                # Combine scores
                opportunity_score = (indicator_score * 0.6) + (zone_strength * 0.4)
                opportunity_score = min(opportunity_score, 1.0) * 100  # Scale to 0-100

                # Check if we have enough indicator agreement
                if len(indicators_bearish) >= self.config["min_indicators_agreement"] or not self.config["require_multi_indicator_confirmation"]:
                    opportunities.append(
                        {
                            "type": "short",
                            "price": current_price,
                            "score": opportunity_score,
                            "reason": "overbought_at_resistance",
                            "indicators": indicators_bearish,
                            "resistance_zone": resistance_zone,
                            "stop_loss": (
                                resistance_zone["upper"] + (self.current_atr * 0.5)
                                if resistance_zone
                                else current_price + (self.current_atr * self.config["stop_loss_atr_multiplier"])
                            ),
                            "profit_target": current_price - (self.current_atr * self.config["profit_target_atr_multiplier"]),
                        }
                    )

        # Check for harmonic pattern completions
        for pattern in self.detected_patterns:
            if pattern["type"] == "harmonic":
                # Check if current price is in the completion area
                completion_area = pattern.get("completion_area")
                if completion_area:
                    if completion_area["lower"] <= current_price <= completion_area["upper"]:
                        # Determine trade direction based on pattern
                        if pattern["pattern"] in ["bat", "butterfly", "gartley"]:
                            # These patterns typically indicate reversals
                            last_point = pattern["points"][-1]
                            if last_point["type"] == "high":  # Bearish reversal
                                opportunities.append(
                                    {
                                        "type": "short",
                                        "price": current_price,
                                        "score": pattern["strength"],
                                        "reason": f'harmonic_{pattern["pattern"]}_completion',
                                        "pattern": pattern,
                                        "stop_loss": completion_area["upper"] + (self.current_atr * 0.5),
                                        "profit_target": current_price - (current_price - pattern["points"][-2]["price"]) * 0.618,
                                    }
                                )
                            else:  # Bullish reversal
                                opportunities.append(
                                    {
                                        "type": "long",
                                        "price": current_price,
                                        "score": pattern["strength"],
                                        "reason": f'harmonic_{pattern["pattern"]}_completion',
                                        "pattern": pattern,
                                        "stop_loss": completion_area["lower"] - (self.current_atr * 0.5),
                                        "profit_target": current_price + (pattern["points"][-2]["price"] - current_price) * 0.618,
                                    }
                                )

        # Check for Wyckoff pattern completions
        for pattern in self.detected_patterns:
            if pattern["type"] == "wyckoff":
                # Check if the pattern is recently completed (within last 5 bars)
                if pattern["end_idx"] >= len(df) - 5:
                    if pattern["phase"] == "accumulation_end":
                        opportunities.append(
                            {
                                "type": "long",
                                "price": current_price,
                                "score": pattern["strength"],
                                "reason": "wyckoff_accumulation_completion",
                                "pattern": pattern,
                                "stop_loss": df["low"].iloc[pattern["end_idx"]] - (self.current_atr * 0.5),
                                "profit_target": current_price + (self.current_atr * self.config["profit_target_atr_multiplier"] * 1.5),
                            }
                        )
                    elif pattern["phase"] == "distribution_end":
                        opportunities.append(
                            {
                                "type": "short",
                                "price": current_price,
                                "score": pattern["strength"],
                                "reason": "wyckoff_distribution_completion",
                                "pattern": pattern,
                                "stop_loss": df["high"].iloc[pattern["end_idx"]] + (self.current_atr * 0.5),
                                "profit_target": current_price - (self.current_atr * self.config["profit_target_atr_multiplier"] * 1.5),
                            }
                        )

        # Sort opportunities by score (highest first)
        opportunities.sort(key=lambda x: x["score"], reverse=True)

        return opportunities

    def _generate_signals(self, df: pd.DataFrame, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate trading signals based on swing analysis.

        Args:
            df: DataFrame with market data
            opportunities: List of swing trading opportunities

        Returns:
            Dict with trading signals and metadata
        """
        signals = {"entry": None, "exit": None, "confidence": 0.0, "sizing_factor": 1.0, "metadata": {}}  # 'long', 'short', or None  # 'exit' or None

        # Check if we have any valid opportunities
        if not opportunities:
            return signals

        # Get the best opportunity
        best_opportunity = opportunities[0]

        # Check minimum score requirement based on swing strength filter
        min_score = SWING_STRENGTH_LEVELS[self.config["swing_strength_filter"]]

        if best_opportunity["score"] >= min_score:
            # Check if the opportunity aligns with market structure
            opportunity_type = best_opportunity["type"]

            # Only consider counter-trend trades if explicitly allowed
            if not self.config["counter_trend_allowed"]:
                if (
                    opportunity_type == "long"
                    and self.current_market_structure == MARKET_STRUCTURE.DOWNTREND
                    or opportunity_type == "short"
                    and self.current_market_structure == MARKET_STRUCTURE.UPTREND
                ):
                    return signals

            # Calculate confidence based on opportunity score and market structure alignment
            base_confidence = best_opportunity["score"] / 100

            # Adjust confidence based on market structure alignment
            structure_alignment = 1.0
            if opportunity_type == "long":
                if self.current_market_structure == MARKET_STRUCTURE.UPTREND:
                    structure_alignment = 1.2
                elif self.current_market_structure == MARKET_STRUCTURE.DOWNTREND:
                    structure_alignment = 0.7
            else:  # short
                if self.current_market_structure == MARKET_STRUCTURE.DOWNTREND:
                    structure_alignment = 1.2
                elif self.current_market_structure == MARKET_STRUCTURE.UPTREND:
                    structure_alignment = 0.7

            # Adjust for market regime
            regime_factor = self._get_regime_confidence_factor()

            # Final confidence score (0-1)
            confidence = base_confidence * structure_alignment * regime_factor
            confidence = min(confidence, 1.0)

            # Determine position sizing factor based on confidence
            sizing_factor = confidence

            # Generate entry signal
            signals["entry"] = opportunity_type
            signals["confidence"] = confidence
            signals["sizing_factor"] = sizing_factor
            signals["metadata"]["reason"] = best_opportunity["reason"]
            signals["metadata"]["score"] = best_opportunity["score"]
            signals["metadata"]["market_structure"] = self.current_market_structure
            signals["metadata"]["stop_loss"] = best_opportunity["stop_loss"]
            signals["metadata"]["profit_target"] = best_opportunity["profit_target"]

            # Record signal
            self.signals_generated += 1
            self.last_signal_time = datetime.now()

            self.logger.info(f"Generated {opportunity_type.upper()} signal with confidence {confidence:.2f} due to {best_opportunity['reason']}")

        return signals

    def _calculate_exit_targets(self, df: pd.DataFrame, entry_type: str, opportunities: List[Dict[str, Any]]) -> None:
        """
        Calculate exit targets based on entry signal and market conditions.

        Args:
            df: DataFrame with market data
            entry_type: Type of entry signal ('long' or 'short')
            opportunities: List of trading opportunities
        """
        if not opportunities:
            return

        best_opportunity = opportunities[0]
        current_price = df["close"].iloc[-1]

        # Set stop loss and profit target from the opportunity
        stop_loss = best_opportunity["stop_loss"]
        profit_target = best_opportunity["profit_target"]

        # Calculate reasonable time stop based on trading style (swing trading)
        # For swing trading, typically set for 5-20 bars depending on timeframe
        time_stop_bars = 10
        if self.primary_timeframe in ["1h", "4h"]:
            time_stop_bars = 15
        elif self.primary_timeframe in ["1d"]:
            time_stop_bars = 20

        last_entry_index = df.index.get_loc(df.index[-1])
        time_stop_index = last_entry_index + time_stop_bars
        if time_stop_index < len(df.index):
            time_stop = df.index[time_stop_index]
        else:
            # Estimate future time based on average bar duration
            avg_bar_duration = (df.index[-1] - df.index[0]) / (len(df.index) - 1)
            time_stop = df.index[-1] + (avg_bar_duration * time_stop_bars)

        # Calculate trailing stop if in profit
        trailing_stop = None
        if entry_type == "long":
            # Trailing stop for long positions
            trailing_activation = current_price + (self.current_atr * 1.0)  # Activate trailing stop after 1 ATR move
            trailing_stop = {"activation_price": trailing_activation, "stop_price": current_price * 0.98}  # Initial trailing stop at 98% of entry
        else:  # short
            # Trailing stop for short positions
            trailing_activation = current_price - (self.current_atr * 1.0)  # Activate trailing stop after 1 ATR move
            trailing_stop = {"activation_price": trailing_activation, "stop_price": current_price * 1.02}  # Initial trailing stop at 102% of entry

        # Update exit targets
        self.current_exit_targets = {"profit_target": profit_target, "stop_loss": stop_loss, "trailing_stop": trailing_stop, "time_stop": time_stop}

    def _get_regime_confidence_factor(self) -> float:
        """
        Get confidence factor based on current market regime.

        Returns:
            float: Confidence factor (0-1)
        """
        regime_factors = {
            REGIME_TYPES.STRONG_TREND: 1.0,
            REGIME_TYPES.VOLATILE_TREND: 0.8,
            REGIME_TYPES.WEAK_TREND: 0.9,  # Swing trades often work well in weak trends
            REGIME_TYPES.RANGING: 1.0,  # Swing trades excel in ranging markets
            REGIME_TYPES.VOLATILE_RANGE: 0.7,
            REGIME_TYPES.CHOPPY: 0.6,
        }

        return regime_factors.get(self.current_market_regime, 0.8)
