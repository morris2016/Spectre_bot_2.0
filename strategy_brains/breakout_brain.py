#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Breakout Strategy Brain Implementation

This module implements a sophisticated breakout detection and trading strategy
that identifies and capitalizes on price breakouts from established ranges,
support/resistance levels, and chart patterns.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from common.constants import OrderSide, OrderType, SignalType, TimeInForce
from intelligence.app import SignalConfidence
from common.exceptions import InvalidDataError, StrategyError
from common.utils import calculate_risk_reward, normalize_price_series

from feature_service.features.market_structure import detect_consolidation, identify_swing_points
from feature_service.features.pattern import detect_rectangle_pattern, detect_triangle_pattern
from feature_service.features.technical import calculate_atr
from feature_service.features.volatility import volatility_expansion_indicator
from intelligence.pattern_recognition.chart_patterns import ChartPattern
from intelligence.pattern_recognition.support_resistance import identify_support_resistance_zones
from strategy_brains.base_brain import SignalEvent, StrategyBrain, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class BreakoutConfig(StrategyConfig):
    """Configuration for the Breakout Brain strategy."""

    # Basic configuration
    lookback_periods: int = 100
    confirmation_periods: int = 3
    min_consolidation_periods: int = 5
    max_consolidation_periods: int = 50

    # Breakout confirmation parameters
    volume_confirmation_threshold: float = 1.5
    price_confirmation_percentage: float = 0.5

    # False breakout protection
    false_breakout_atr_multiplier: float = 0.5
    max_retest_periods: int = 3

    # Volatility-based parameters
    volatility_expansion_threshold: float = 1.8
    pre_breakout_volatility_contraction: float = 0.7

    # Pattern-specific parameters
    pattern_recognition_enabled: bool = True
    pattern_importance_weight: float = 1.2

    # Support/Resistance specific parameters
    sr_importance_weight: float = 1.5
    sr_zone_width_percentage: float = 0.2

    # Signal generation
    min_signal_confidence: SignalConfidence = SignalConfidence.MEDIUM
    risk_reward_min: float = 1.5
    adaptive_parameters: bool = True

    # Multiple timeframe confirmation
    higher_timeframe_confirmation: bool = True

    # Asset-specific overrides
    asset_specific_config: Dict[str, Dict[str, Any]] = None


class BreakoutBrain(StrategyBrain):
    """
    Advanced implementation of a breakout trading strategy that detects and trades
    breakouts from various chart patterns, support/resistance levels, and price consolidations.

    Features:
    - Multi-timeframe breakout detection and confirmation
    - Volume-confirmed breakout validation
    - False breakout protection mechanisms
    - Pattern-based breakout detection (rectangles, triangles, etc.)
    - Support/resistance breakout identification
    - Adaptive parameter optimization based on asset behavior
    - Historical breakout success rate tracking
    """

    def __init__(self, config: Union[Dict[str, Any], BreakoutConfig] = None):
        """Initialize the Breakout Brain strategy with configuration."""
        super().__init__(config_class=BreakoutConfig, config=config)

        # Breakout detection metrics
        self.recent_breakouts = {}  # Store recent breakouts for tracking
        self.breakout_success_metrics = {}  # Track success rates by asset/pattern
        self.failed_breakout_patterns = {}  # Store characteristics of failed breakouts

        # Pattern detection components
        self.chart_pattern_detector = ChartPattern()

        # Adaptive parameter state
        self.asset_volatility_metrics = {}  # Track asset volatility for parameter adjustment
        self.asset_breakout_characteristics = {}  # Store typical breakout behavior by asset

        logger.info("Breakout Brain initialized with configuration: %s", self.config)

    async def analyze(self, market_data: pd.DataFrame, context: Dict[str, Any] = None) -> List[SignalEvent]:
        """
        Analyze market data to detect and validate breakout opportunities.

        Args:
            market_data: DataFrame containing OHLCV data
            context: Additional context data including higher timeframes, asset info, etc.

        Returns:
            List of signal events if breakout opportunities are detected
        """
        signals = []

        try:
            # Validate input data
            self._validate_data(market_data)

            # Apply asset-specific configuration if available
            self._apply_asset_specific_config(context.get("asset", ""))

            # Calculate key metrics
            atr = calculate_atr(market_data, period=14)
            latest_atr = atr.iloc[-1]

            # Step 1: Detect consolidation regions
            consolidation_regions = self._detect_consolidation_regions(market_data)

            # Step 2: Identify support and resistance zones
            sr_zones = self._identify_support_resistance(market_data)

            # Step 3: Detect chart patterns
            patterns = self._detect_chart_patterns(market_data) if self.config.pattern_recognition_enabled else []

            # Step 4: Check for breakout conditions
            breakout_signals = self._detect_breakouts(market_data, consolidation_regions, sr_zones, patterns, latest_atr)

            # Step 5: Apply higher timeframe confirmation if enabled
            if self.config.higher_timeframe_confirmation and context and "higher_timeframe_data" in context:
                breakout_signals = self._apply_higher_timeframe_confirmation(breakout_signals, context["higher_timeframe_data"])

            # Step 6: Calculate entry, stop loss, take profit for valid signals
            for signal in breakout_signals:
                complete_signal = self._generate_complete_signal(signal, market_data, latest_atr)
                if complete_signal:
                    signals.append(complete_signal)

            # Step 7: Update historical metrics for adaptive parameters
            if self.config.adaptive_parameters:
                self._update_asset_metrics(market_data, context.get("asset", ""))

            logger.debug(f"Breakout analysis completed. Found {len(signals)} signals.")

        except Exception as e:
            logger.error(f"Error in Breakout Brain analysis: {str(e)}", exc_info=True)

        return signals

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate the input data for required columns and sufficient history."""
        required_columns = ["open", "high", "low", "close", "volume"]

        if data is None or len(data) < self.config.lookback_periods:
            raise InvalidDataError(f"Insufficient data: need at least {self.config.lookback_periods} periods")

        for col in required_columns:
            if col not in data.columns:
                raise InvalidDataError(f"Missing required column: {col}")

    def _apply_asset_specific_config(self, asset: str) -> None:
        """Apply asset-specific configuration overrides if available."""
        if not self.config.asset_specific_config or asset not in self.config.asset_specific_config:
            return

        asset_config = self.config.asset_specific_config[asset]
        for key, value in asset_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Applied asset-specific config for {asset}: {key}={value}")

    def _detect_consolidation_regions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect price consolidation regions in the data.

        Returns:
            List of dictionaries with consolidation region information
        """
        consolidation_regions = []

        # Use feature service to detect consolidation
        is_consolidating = detect_consolidation(
            data,
            min_periods=self.config.min_consolidation_periods,
            max_periods=self.config.max_consolidation_periods,
            threshold=0.15,  # Adjustable threshold for consolidation detection
        )

        # Find consecutive consolidation regions
        if is_consolidating.any():
            # Get indices where consolidation starts and ends
            consolidation_changes = is_consolidating.diff().fillna(0).astype(int)
            consolidation_starts = data.index[consolidation_changes == 1].tolist()
            consolidation_ends = data.index[consolidation_changes == -1].tolist()

            # Handle edge cases if we're still in consolidation
            if len(consolidation_starts) > len(consolidation_ends):
                consolidation_ends.append(data.index[-1])

            # Process each consolidation region
            for start, end in zip(consolidation_starts, consolidation_ends):
                region_data = data.loc[start:end]
                if len(region_data) >= self.config.min_consolidation_periods:
                    region_high = region_data["high"].max()
                    region_low = region_data["low"].min()
                    region_width = region_high - region_low
                    region_volatility = region_data["close"].pct_change().std()

                    consolidation_regions.append(
                        {
                            "start_idx": start,
                            "end_idx": end,
                            "high": region_high,
                            "low": region_low,
                            "width": region_width,
                            "avg_volume": region_data["volume"].mean(),
                            "volatility": region_volatility,
                            "duration": len(region_data),
                        }
                    )

        return consolidation_regions

    def _identify_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identify key support and resistance zones.

        Returns:
            Dictionary with 'support' and 'resistance' lists of price levels
        """
        # Use the support/resistance detector from intelligence module
        sr_zones = identify_support_resistance_zones(data, zone_width_pct=self.config.sr_zone_width_percentage, min_touches=2, swing_strength=0.6)

        # Also identify swing points for additional confirmation
        swings = identify_swing_points(data, window=5)

        # Enhance SR zones with swing point information
        for point_type, points in swings.items():
            if point_type == "swing_highs":
                for point in points:
                    price = data.loc[point, "high"]
                    # Check if close to existing resistance, otherwise add
                    if not any(abs(price - r) / price < self.config.sr_zone_width_percentage for r in sr_zones["resistance"]):
                        sr_zones["resistance"].append(price)
            elif point_type == "swing_lows":
                for point in points:
                    price = data.loc[point, "low"]
                    # Check if close to existing support, otherwise add
                    if not any(abs(price - s) / price < self.config.sr_zone_width_percentage for s in sr_zones["support"]):
                        sr_zones["support"].append(price)

        # Sort levels for easier processing
        sr_zones["support"] = sorted(sr_zones["support"])
        sr_zones["resistance"] = sorted(sr_zones["resistance"])

        return sr_zones

    def _detect_chart_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect chart patterns relevant to breakout trading.

        Returns:
            List of patterns with their characteristics
        """
        patterns = []

        # Detect rectangle patterns
        rectangles = detect_rectangle_pattern(data, tolerance=0.02, min_touches=2)

        for rect in rectangles:
            patterns.append(
                {
                    "type": "rectangle",
                    "start_idx": rect["start_idx"],
                    "end_idx": rect["end_idx"],
                    "top": rect["top"],
                    "bottom": rect["bottom"],
                    "strength": rect["strength"],
                }
            )

        # Detect triangle patterns
        triangles = detect_triangle_pattern(data, min_points=5)

        for tri in triangles:
            patterns.append(
                {
                    "type": tri["pattern_type"],  # ascending, descending, symmetric
                    "start_idx": tri["start_idx"],
                    "end_idx": tri["end_idx"],
                    "upper_trendline": tri["upper_trendline"],
                    "lower_trendline": tri["lower_trendline"],
                    "apex_idx": tri["apex_idx"],
                    "strength": tri["strength"],
                }
            )

        # Get advanced chart patterns from ChartPattern detector
        advanced_patterns = self.chart_pattern_detector.detect_patterns(data)
        for pattern in advanced_patterns:
            patterns.append(
                {
                    "type": pattern["name"],
                    "start_idx": pattern["start_idx"],
                    "end_idx": pattern["end_idx"],
                    "critical_points": pattern["critical_points"],
                    "breakout_level": pattern["breakout_level"],
                    "target": pattern["target"],
                    "strength": pattern["reliability"],
                }
            )

        return patterns

    def _detect_breakouts(
        self,
        data: pd.DataFrame,
        consolidation_regions: List[Dict[str, Any]],
        sr_zones: Dict[str, List[float]],
        patterns: List[Dict[str, Any]],
        atr: float,
    ) -> List[Dict[str, Any]]:
        """
        Detect breakout conditions from various technical formations.

        Args:
            data: Market data
            consolidation_regions: Identified consolidation regions
            sr_zones: Support and resistance zones
            patterns: Chart patterns
            atr: Current ATR value

        Returns:
            List of potential breakout signals
        """
        breakout_signals = []

        # Get the most recent data points
        recent_data = data.iloc[-self.config.confirmation_periods:]
        latest_close = recent_data["close"].iloc[-1]
        latest_high = recent_data["high"].iloc[-1]
        latest_low = recent_data["low"].iloc[-1]
        latest_volume = recent_data["volume"].iloc[-1]
        avg_volume = data["volume"].rolling(20).mean().iloc[-1]

        # Check consolidation breakouts
        for region in consolidation_regions:
            # Only consider regions that have recently ended or are still active
            if region["end_idx"] >= data.index[-self.config.lookback_periods]:
                # Check for upside breakout
                if latest_close > region["high"] * (1 + self.config.price_confirmation_percentage / 100):
                    # Volume confirmation
                    if latest_volume > region["avg_volume"] * self.config.volume_confirmation_threshold:
                        # Check for volatility expansion
                        vol_expansion = volatility_expansion_indicator(data, window1=5, window2=20).iloc[-1]

                        if vol_expansion > self.config.volatility_expansion_threshold:
                            breakout_signals.append(
                                {
                                    "type": "consolidation_breakout",
                                    "direction": "up",
                                    "price_level": region["high"],
                                    "confidence": self._calculate_confidence("consolidation", latest_volume / region["avg_volume"], vol_expansion),
                                    "volatility": vol_expansion,
                                    "volume_surge": latest_volume / region["avg_volume"],
                                    "pattern_strength": region["duration"] / self.config.max_consolidation_periods,
                                }
                            )

                # Check for downside breakout
                elif latest_close < region["low"] * (1 - self.config.price_confirmation_percentage / 100):
                    # Volume confirmation
                    if latest_volume > region["avg_volume"] * self.config.volume_confirmation_threshold:
                        # Check for volatility expansion
                        vol_expansion = volatility_expansion_indicator(data, window1=5, window2=20).iloc[-1]

                        if vol_expansion > self.config.volatility_expansion_threshold:
                            breakout_signals.append(
                                {
                                    "type": "consolidation_breakout",
                                    "direction": "down",
                                    "price_level": region["low"],
                                    "confidence": self._calculate_confidence("consolidation", latest_volume / region["avg_volume"], vol_expansion),
                                    "volatility": vol_expansion,
                                    "volume_surge": latest_volume / region["avg_volume"],
                                    "pattern_strength": region["duration"] / self.config.max_consolidation_periods,
                                }
                            )

        # Check support/resistance breakouts
        for resistance in sr_zones["resistance"]:
            # Check for resistance breakout (with some buffer based on ATR)
            if latest_close > resistance and latest_high > resistance + (atr * self.config.false_breakout_atr_multiplier):
                # Check volume confirmation
                if latest_volume > avg_volume * self.config.volume_confirmation_threshold:
                    # Calculate distance from previous tests
                    previous_tests = self._count_previous_tests(data, resistance, "resistance")

                    breakout_signals.append(
                        {
                            "type": "resistance_breakout",
                            "direction": "up",
                            "price_level": resistance,
                            "confidence": self._calculate_confidence("resistance", latest_volume / avg_volume, previous_tests),
                            "volume_surge": latest_volume / avg_volume,
                            "previous_tests": previous_tests,
                        }
                    )

        for support in sr_zones["support"]:
            # Check for support breakdown
            if latest_close < support and latest_low < support - (atr * self.config.false_breakout_atr_multiplier):
                # Check volume confirmation
                if latest_volume > avg_volume * self.config.volume_confirmation_threshold:
                    # Calculate distance from previous tests
                    previous_tests = self._count_previous_tests(data, support, "support")

                    breakout_signals.append(
                        {
                            "type": "support_breakdown",
                            "direction": "down",
                            "price_level": support,
                            "confidence": self._calculate_confidence("support", latest_volume / avg_volume, previous_tests),
                            "volume_surge": latest_volume / avg_volume,
                            "previous_tests": previous_tests,
                        }
                    )

        # Check pattern breakouts
        for pattern in patterns:
            # Only consider patterns that have recently formed or are still developing
            if "end_idx" in pattern and pattern["end_idx"] >= data.index[-self.config.lookback_periods]:
                breakout_level = None
                direction = None

                # Pattern-specific breakout detection
                if pattern["type"] == "rectangle":
                    if latest_close > pattern["top"]:
                        breakout_level = pattern["top"]
                        direction = "up"
                    elif latest_close < pattern["bottom"]:
                        breakout_level = pattern["bottom"]
                        direction = "down"

                elif pattern["type"] in ["ascending_triangle", "descending_triangle", "symmetric_triangle"]:
                    # For triangles, calculate the current upper and lower trendline values
                    current_upper = None
                    current_lower = None

                    if "upper_trendline" in pattern:
                        current_upper = self._calculate_trendline_value(pattern["upper_trendline"], pattern["start_idx"], data.index[-1])

                    if "lower_trendline" in pattern:
                        current_lower = self._calculate_trendline_value(pattern["lower_trendline"], pattern["start_idx"], data.index[-1])

                    if current_upper and latest_close > current_upper:
                        breakout_level = current_upper
                        direction = "up"
                    elif current_lower and latest_close < current_lower:
                        breakout_level = current_lower
                        direction = "down"

                # Handle other pattern types
                elif "breakout_level" in pattern:
                    expected_direction = pattern.get("expected_direction", "up")

                    if expected_direction == "up" and latest_close > pattern["breakout_level"]:
                        breakout_level = pattern["breakout_level"]
                        direction = "up"
                    elif expected_direction == "down" and latest_close < pattern["breakout_level"]:
                        breakout_level = pattern["breakout_level"]
                        direction = "down"

                # If breakout detected, add to signals
                if breakout_level and direction:
                    # Check volume confirmation
                    if latest_volume > avg_volume * self.config.volume_confirmation_threshold:
                        pattern_strength = pattern.get("strength", 0.5)

                        breakout_signals.append(
                            {
                                "type": f'{pattern["type"]}_breakout',
                                "direction": direction,
                                "price_level": breakout_level,
                                "confidence": self._calculate_confidence(pattern["type"], latest_volume / avg_volume, pattern_strength),
                                "volume_surge": latest_volume / avg_volume,
                                "pattern_strength": pattern_strength,
                                "target": pattern.get("target", None),
                            }
                        )

        return breakout_signals

    def _apply_higher_timeframe_confirmation(self, signals: List[Dict[str, Any]], higher_tf_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Apply confirmation rules based on higher timeframe data.

        Args:
            signals: List of potential breakout signals
            higher_tf_data: Dictionary of higher timeframe data

        Returns:
            Filtered list of signals with confirmation status
        """
        confirmed_signals = []

        for signal in signals:
            is_confirmed = False

            # Loop through higher timeframes from lowest to highest
            for tf, tf_data in sorted(higher_tf_data.items()):
                if tf_data is None or len(tf_data) < 10:
                    continue

                # Get recent higher timeframe data
                recent_tf_data = tf_data.iloc[-5:]

                # Simplified trend detection on higher timeframe
                if signal["direction"] == "up":
                    # For upside breakouts, confirm if higher timeframe is trending up
                    ema20 = tf_data["close"].ewm(span=20).mean()
                    ema50 = tf_data["close"].ewm(span=50).mean()

                    # Current trend is up if latest close > EMAs and EMA20 > EMA50
                    tf_trend_up = recent_tf_data["close"].iloc[-1] > ema20.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]

                    if tf_trend_up:
                        is_confirmed = True
                        signal["confidence"] = min(SignalConfidence.VERY_HIGH, signal["confidence"] + 1)  # Increase confidence by one level
                        break

                elif signal["direction"] == "down":
                    # For downside breakouts, confirm if higher timeframe is trending down
                    ema20 = tf_data["close"].ewm(span=20).mean()
                    ema50 = tf_data["close"].ewm(span=50).mean()

                    # Current trend is down if latest close < EMAs and EMA20 < EMA50
                    tf_trend_down = recent_tf_data["close"].iloc[-1] < ema20.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]

                    if tf_trend_down:
                        is_confirmed = True
                        signal["confidence"] = min(SignalConfidence.VERY_HIGH, signal["confidence"] + 1)  # Increase confidence by one level
                        break

            # Add confirmation status to signal
            signal["higher_tf_confirmed"] = is_confirmed

            # Only keep signals that meet minimum confidence after higher TF evaluation
            if signal["confidence"] >= self.config.min_signal_confidence:
                confirmed_signals.append(signal)

        return confirmed_signals

    def _generate_complete_signal(self, signal: Dict[str, Any], data: pd.DataFrame, atr: float) -> Optional[SignalEvent]:
        """
        Generate a complete signal event with entry, stop loss, take profit levels.

        Args:
            signal: Signal information
            data: Market data
            atr: ATR value

        Returns:
            Complete SignalEvent or None if invalid
        """
        entry_price = data["close"].iloc[-1]
        direction = signal["direction"]
        signal_type = signal["type"]

        # Calculate stop loss based on signal type and direction
        stop_loss = None
        if direction == "up":
            # For upward breakouts, stop loss is below the breakout level
            buffer = atr * self.config.false_breakout_atr_multiplier
            stop_loss = signal["price_level"] - buffer

            # Additional logic for different pattern types
            if "triangle" in signal_type:
                # For triangles, use the most recent swing low
                swings = identify_swing_points(data.iloc[-30:], window=3)
                if swings["swing_lows"]:
                    recent_swing_low = data.loc[swings["swing_lows"][-1], "low"]
                    stop_loss = min(stop_loss, recent_swing_low)

        elif direction == "down":
            # For downward breakouts, stop loss is above the breakout level
            buffer = atr * self.config.false_breakout_atr_multiplier
            stop_loss = signal["price_level"] + buffer

            # Additional logic for different pattern types
            if "triangle" in signal_type:
                # For triangles, use the most recent swing high
                swings = identify_swing_points(data.iloc[-30:], window=3)
                if swings["swing_highs"]:
                    recent_swing_high = data.loc[swings["swing_highs"][-1], "high"]
                    stop_loss = max(stop_loss, recent_swing_high)

        if stop_loss is None:
            logger.warning(f"Could not calculate stop loss for signal: {signal}")
            return None

        # Calculate profit target
        take_profit = None
        risk = abs(entry_price - stop_loss)

        # Use pattern target if available, otherwise use risk:reward ratio
        if "target" in signal and signal["target"] is not None:
            take_profit = signal["target"]
        else:
            # Default risk:reward calculation
            if direction == "up":
                take_profit = entry_price + (risk * self.config.risk_reward_min)
            else:
                take_profit = entry_price - (risk * self.config.risk_reward_min)

        # Normalize prices to tick size for the asset
        asset = data.name if hasattr(data, "name") else "DEFAULT"
        tick_size = TICK_SIZE_MAPPING.get(asset, TICK_SIZE_MAPPING["DEFAULT"])
        entry_price, stop_loss, take_profit = normalize_price_series(
            pd.Series([entry_price, stop_loss, take_profit]), tick_size
        ).tolist()

        # Create the signal event
        return SignalEvent(
            timestamp=pd.Timestamp.now(),
            asset=data.name if hasattr(data, "name") else "unknown",
            strategy_name=self.__class__.__name__,
            signal_type=SignalType.ENTRY,
            direction=OrderSide.BUY if direction == "up" else OrderSide.SELL,
            confidence=signal["confidence"],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=abs(take_profit - entry_price) / risk if risk > 0 else None,
            metadata={
                "pattern_type": signal_type,
                "breakout_level": signal["price_level"],
                "volume_surge": signal.get("volume_surge", None),
                "pattern_strength": signal.get("pattern_strength", None),
                "higher_tf_confirmed": signal.get("higher_tf_confirmed", False),
                "volatility": signal.get("volatility", None),
            },
        )

    def _calculate_confidence(self, pattern_type: str, volume_surge: float, additional_factor: float) -> SignalConfidence:
        """
        Calculate signal confidence based on breakout characteristics.

        Args:
            pattern_type: Type of pattern or breakout
            volume_surge: Volume increase ratio
            additional_factor: Additional confidence factor (pattern strength, previous tests, etc.)

        Returns:
            SignalConfidence level
        """
        # Base score calculation
        base_score = 0

        # Adjust for pattern type
        if "resistance" in pattern_type or "support" in pattern_type:
            base_score += 1 * self.config.sr_importance_weight
        elif any(p in pattern_type for p in ["triangle", "head_and_shoulders", "double_top", "double_bottom"]):
            base_score += 1.5 * self.config.pattern_importance_weight
        elif "rectangle" in pattern_type:
            base_score += 1.3 * self.config.pattern_importance_weight
        else:  # consolidation or other patterns
            base_score += 1

        # Adjust for volume surge
        if volume_surge >= 2.5:
            base_score += 1.5
        elif volume_surge >= 1.8:
            base_score += 1
        elif volume_surge >= 1.5:
            base_score += 0.5

        # Adjust for additional factor (pattern strength, previous tests, etc.)
        if additional_factor >= 0.8:
            base_score += 1.5
        elif additional_factor >= 0.6:
            base_score += 1
        elif additional_factor >= 0.4:
            base_score += 0.5

        # Convert score to confidence level
        if base_score >= 4:
            return SignalConfidence.VERY_HIGH
        elif base_score >= 3:
            return SignalConfidence.HIGH
        elif base_score >= 2:
            return SignalConfidence.MEDIUM
        elif base_score >= 1:
            return SignalConfidence.LOW
        else:
            return SignalConfidence.VERY_LOW

    def _count_previous_tests(self, data: pd.DataFrame, level: float, level_type: str) -> int:
        """Count how many times a level has been tested previously."""
        count = 0
        price_buffer = level * 0.001  # 0.1% buffer

        # Last 50 candles should be sufficient
        window = min(50, len(data))
        recent_data = data.iloc[-window:]

        if level_type == "resistance":
            # Count number of times price approached resistance from below
            for i in range(1, len(recent_data)):
                if recent_data["high"].iloc[i] > level - price_buffer and recent_data["high"].iloc[i] < level + price_buffer:
                    count += 1
        else:  # support
            # Count number of times price approached support from above
            for i in range(1, len(recent_data)):
                if recent_data["low"].iloc[i] < level + price_buffer and recent_data["low"].iloc[i] > level - price_buffer:
                    count += 1

        return count

    def _calculate_trendline_value(self, trendline: Dict[str, Any], start_idx, current_idx) -> float:
        """Calculate the current value of a trendline at the given index."""
        slope = trendline["slope"]
        intercept = trendline["intercept"]

        # Convert indices to numeric for calculation
        start_num = pd.Index([start_idx]).get_indexer([start_idx])[0]
        current_num = pd.Index([current_idx]).get_indexer([current_idx])[0]

        # Calculate periods elapsed
        periods_elapsed = current_num - start_num

        # Calculate trendline value: y = mx + b
        return slope * periods_elapsed + intercept

    def _update_asset_metrics(self, data: pd.DataFrame, asset: str) -> None:
        """Update asset-specific metrics for adaptive parameters."""
        if not asset or not self.config.adaptive_parameters:
            return

        # Calculate and store volatility metrics
        atr = calculate_atr(data, period=14)
        atr_pct = atr.iloc[-1] / data["close"].iloc[-1]

        if asset not in self.asset_volatility_metrics:
            self.asset_volatility_metrics[asset] = []

        # Keep last 20 volatility readings
        self.asset_volatility_metrics[asset].append(atr_pct)
        if len(self.asset_volatility_metrics[asset]) > 20:
            self.asset_volatility_metrics[asset].pop(0)

        # Calculate average volatility for this asset
        avg_volatility = np.mean(self.asset_volatility_metrics[asset])

        # Adjust parameters based on volatility (example adjustments)
        if avg_volatility > 0.02:  # High volatility asset
            self.config.false_breakout_atr_multiplier = max(0.7, self.config.false_breakout_atr_multiplier)
            self.config.volatility_expansion_threshold = max(2.0, self.config.volatility_expansion_threshold)
        elif avg_volatility < 0.005:  # Low volatility asset
            self.config.false_breakout_atr_multiplier = min(0.3, self.config.false_breakout_atr_multiplier)
            self.config.volatility_expansion_threshold = min(1.5, self.config.volatility_expansion_threshold)

        logger.debug(f"Updated adaptive parameters for {asset} based on volatility: {avg_volatility}")

    async def on_signal_result(self, signal: SignalEvent, result: Dict[str, Any]) -> None:
        """
        Process the result of a signal (win/loss) to improve future predictions.

        Args:
            signal: The original signal event
            result: Trade result information
        """
        if not signal or not result:
            return

        # Extract relevant information
        asset = signal.asset
        pattern_type = signal.metadata.get("pattern_type", "unknown")
        is_win = result.get("is_win", False)
        profit_pct = result.get("profit_pct", 0)

        # Store information about this breakout
        if asset not in self.breakout_success_metrics:
            self.breakout_success_metrics[asset] = {}

        if pattern_type not in self.breakout_success_metrics[asset]:
            self.breakout_success_metrics[asset][pattern_type] = {"wins": 0, "total": 0, "avg_profit": []}

        # Update metrics
        self.breakout_success_metrics[asset][pattern_type]["total"] += 1
        if is_win:
            self.breakout_success_metrics[asset][pattern_type]["wins"] += 1

        # Store profit percentage
        self.breakout_success_metrics[asset][pattern_type]["avg_profit"].append(profit_pct)

        # If too many values, keep most recent 50
        if len(self.breakout_success_metrics[asset][pattern_type]["avg_profit"]) > 50:
            self.breakout_success_metrics[asset][pattern_type]["avg_profit"].pop(0)

        # For failed breakouts, store characteristics for learning
        if not is_win:
            if asset not in self.failed_breakout_patterns:
                self.failed_breakout_patterns[asset] = []

            # Store failed breakout characteristics
            self.failed_breakout_patterns[asset].append(
                {
                    "pattern_type": pattern_type,
                    "confidence": signal.confidence,
                    "volume_surge": signal.metadata.get("volume_surge", None),
                    "pattern_strength": signal.metadata.get("pattern_strength", None),
                    "higher_tf_confirmed": signal.metadata.get("higher_tf_confirmed", False),
                    "volatility": signal.metadata.get("volatility", None),
                    "timestamp": signal.timestamp,
                }
            )

            # Keep last 50 failed patterns
            if len(self.failed_breakout_patterns[asset]) > 50:
                self.failed_breakout_patterns[asset].pop(0)

        # Calculate success rate
        wins = self.breakout_success_metrics[asset][pattern_type]["wins"]
        total = self.breakout_success_metrics[asset][pattern_type]["total"]
        success_rate = wins / total if total > 0 else 0

        # Calculate average profit
        avg_profit = (
            np.mean(self.breakout_success_metrics[asset][pattern_type]["avg_profit"])
            if self.breakout_success_metrics[asset][pattern_type]["avg_profit"]
            else 0
        )

        logger.info(f"Breakout success rate for {asset} {pattern_type}: {success_rate:.2f} ({wins}/{total}) with avg profit: {avg_profit:.2f}%")

        # Adjust parameters based on historical performance
        if total >= 5 and self.config.adaptive_parameters:
            self._adapt_parameters_based_on_performance(asset, pattern_type, success_rate, avg_profit)

    def _adapt_parameters_based_on_performance(self, asset: str, pattern_type: str, success_rate: float, avg_profit: float) -> None:
        """
        Adapt strategy parameters based on historical performance.

        Args:
            asset: Asset symbol
            pattern_type: Type of breakout pattern
            success_rate: Historical success rate
            avg_profit: Average profit percentage
        """
        # Ensure asset-specific config exists
        if not self.config.asset_specific_config:
            self.config.asset_specific_config = {}

        if asset not in self.config.asset_specific_config:
            self.config.asset_specific_config[asset] = {}

        # Make parameter adjustments based on performance
        if success_rate < 0.4:  # Poor performance
            # Increase confirmation requirements
            self.config.asset_specific_config[asset]["volume_confirmation_threshold"] = min(2.0, self.config.volume_confirmation_threshold * 1.1)

            # Increase volatility requirements
            self.config.asset_specific_config[asset]["volatility_expansion_threshold"] = min(2.5, self.config.volatility_expansion_threshold * 1.1)

            # Increase false breakout protection
            self.config.asset_specific_config[asset]["false_breakout_atr_multiplier"] = min(1.0, self.config.false_breakout_atr_multiplier * 1.1)

            # Require higher minimum confidence
            self.config.asset_specific_config[asset]["min_signal_confidence"] = max(self.config.min_signal_confidence, SignalConfidence.MEDIUM)

        elif success_rate > 0.7:  # Excellent performance
            # Slightly relax parameters to take more trades
            self.config.asset_specific_config[asset]["volume_confirmation_threshold"] = max(1.3, self.config.volume_confirmation_threshold * 0.95)

            # Allow slightly less volatility expansion
            self.config.asset_specific_config[asset]["volatility_expansion_threshold"] = max(1.5, self.config.volatility_expansion_threshold * 0.95)

        # Update pattern-specific parameters
        if "consolidation" in pattern_type:
            if success_rate < 0.4:
                # Require longer consolidation for more reliable breakouts
                self.config.asset_specific_config[asset]["min_consolidation_periods"] = min(10, self.config.min_consolidation_periods + 1)
            elif success_rate > 0.7:
                # Allow shorter consolidation periods
                self.config.asset_specific_config[asset]["min_consolidation_periods"] = max(3, self.config.min_consolidation_periods - 1)

        logger.debug(f"Adapted parameters for {asset} based on {pattern_type} performance: {success_rate:.2f}")


# Factory function for strategy creation
def create_strategy(config: Dict[str, Any] = None) -> BreakoutBrain:
    """
    Factory function to create a BreakoutBrain strategy instance.

    Args:
        config: Optional strategy configuration

    Returns:
        Configured BreakoutBrain instance
    """
    return BreakoutBrain(config=config)
