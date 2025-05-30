#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Volatility Strategy Brain Implementation

This module implements a sophisticated volatility-based trading strategy
that capitalizes on changes in market volatility to identify profitable
trading opportunities.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from common.constants import (
    OrderSide,
    SignalConfidence,
    SignalType,
    TICK_SIZE_MAPPING,
)
from common.exceptions import InvalidDataError
from common.utils import normalize_price_series
from feature_service.features.technical import calculate_atr, calculate_bollinger_bands
from feature_service.features.volatility import calculate_historical_volatility
from strategy_brains.base_brain import SignalEvent, StrategyBrain, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class VolatilityConfig(StrategyConfig):
    """Configuration for the Volatility Brain strategy."""

    # Basic configuration
    lookback_periods: int = 100

    # Volatility measurement parameters
    atr_period: int = 14
    hv_period: int = 20
    bollinger_period: int = 20
    bollinger_std: float = 2.0

    # Volatility regime parameters
    low_vol_threshold: float = 0.3  # Percentile threshold for low volatility
    high_vol_threshold: float = 0.7  # Percentile threshold for high volatility
    regime_detection_window: int = 50

    # Signal generation parameters
    volatility_breakout_multiplier: float = 1.5
    vol_contraction_trade_threshold: float = 0.2  # Threshold for volatility contraction trades

    # Mean reversion parameters
    mean_reversion_overextension: float = 2.5  # Std devs for mean reversion signals
    mean_reversion_half_life: int = 5  # Expected return to mean half-life in periods

    # Bollinger band parameters
    bollinger_squeeze_percentile: float = 0.1  # Threshold for detecting bollinger squeeze

    # Risk management
    vol_adjusted_position_sizing: bool = True
    max_risk_per_trade_pct: float = 1.0

    # Signal generation
    min_signal_confidence: SignalConfidence = SignalConfidence.MEDIUM
    risk_reward_min: float = 1.5
    adaptive_parameters: bool = True

    # Asset-specific overrides
    asset_specific_config: Dict[str, Dict[str, Any]] = None


class VolatilityBrain(StrategyBrain):
    """
    Advanced implementation of a volatility-based trading strategy that identifies
    and exploits changes in market volatility through various techniques.

    Features:
    - Volatility regime detection and classification
    - Bollinger band squeeze breakout detection
    - Mean reversion during high volatility regimes
    - Volatility breakout anticipation and trading
    - Volatility-adjusted position sizing
    - Adaptive parameter optimization based on asset behavior
    """

    def __init__(self, config: Union[Dict[str, Any], VolatilityConfig] = None):
        """Initialize the Volatility Brain strategy with configuration."""
        super().__init__(config_class=VolatilityConfig, config=config)

        # Track volatility regimes and metrics
        self.volatility_history = {}  # Historical volatility by asset
        self.regime_history = {}  # Volatility regime history by asset
        self.successful_patterns = {}  # Store characteristics of successful trades

        # Adaptive parameter state
        self.asset_volatility_characteristics = {}  # Volatility characteristics by asset
        self.asset_mean_reversion_effectiveness = {}  # Mean reversion effectiveness by asset
        self.asset_breakout_effectiveness = {}  # Breakout effectiveness by asset

        logger.info("Volatility Brain initialized with configuration: %s", self.config)

    async def analyze(self, market_data: pd.DataFrame, context: Dict[str, Any] = None) -> List[SignalEvent]:
        """
        Analyze market data to identify volatility-based trading opportunities.

        Args:
            market_data: DataFrame containing OHLCV data
            context: Additional context data including asset info, market state, etc.

        Returns:
            List of signal events if volatility trade opportunities are detected
        """
        signals = []

        try:
            # Validate input data
            self._validate_data(market_data)

            # Apply asset-specific configuration if available
            asset = context.get("asset", "") if context else ""
            self._apply_asset_specific_config(asset)

            # Step 1: Calculate volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(market_data)

            # Step 2: Detect volatility regime
            volatility_regime = self._detect_volatility_regime(market_data, volatility_metrics, asset)

            # Step 3: Generate appropriate signals based on the volatility regime
            regime_signals = self._generate_regime_based_signals(market_data, volatility_metrics, volatility_regime)

            # Apply additional filters and risk management
            for signal in regime_signals:
                complete_signal = self._generate_complete_signal(signal, market_data, volatility_metrics["atr_latest"])
                if complete_signal:
                    signals.append(complete_signal)

            # Step 4: Update volatility history for this asset
            if asset:
                self._update_volatility_history(asset, volatility_metrics, volatility_regime)

                # Update adaptive parameters if enabled
                if self.config.adaptive_parameters:
                    self._update_adaptive_parameters(asset)

            logger.debug(f"Volatility analysis completed for {asset}. Found {len(signals)} signals.")

        except Exception as e:
            logger.error(f"Error in Volatility Brain analysis: {str(e)}", exc_info=True)

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
        if not asset or not self.config.asset_specific_config or asset not in self.config.asset_specific_config:
            return

        asset_config = self.config.asset_specific_config[asset]
        for key, value in asset_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Applied asset-specific config for {asset}: {key}={value}")

    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate various volatility metrics for analysis.

        Returns:
            Dictionary containing various volatility metrics
        """
        metrics = {}

        # Calculate ATR
        atr = calculate_atr(data, period=self.config.atr_period)
        metrics["atr"] = atr
        metrics["atr_latest"] = atr.iloc[-1]
        metrics["atr_pct"] = (atr / data["close"]).iloc[-1]  # ATR as percentage of price

        # Calculate historical volatility (standard deviation of returns)
        hv = calculate_historical_volatility(data, period=self.config.hv_period)
        metrics["hv"] = hv
        metrics["hv_latest"] = hv.iloc[-1]

        # Calculate HV percentile over lookback period
        metrics["hv_percentile"] = self._calculate_percentile(hv, self.config.lookback_periods)

        # Calculate Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(data["close"], period=self.config.bollinger_period, std_dev=self.config.bollinger_std)
        metrics["bollinger_upper"] = upper
        metrics["bollinger_middle"] = middle
        metrics["bollinger_lower"] = lower

        # Calculate Bollinger Band width
        bb_width = (upper - lower) / middle
        metrics["bb_width"] = bb_width
        metrics["bb_width_latest"] = bb_width.iloc[-1]

        # Calculate BB width percentile
        metrics["bb_width_percentile"] = self._calculate_percentile(bb_width, self.config.lookback_periods)

        # Detect Bollinger Band squeeze
        metrics["is_bb_squeeze"] = metrics["bb_width_percentile"] < self.config.bollinger_squeeze_percentile

        # Calculate price distance from Bollinger Bands
        latest_close = data["close"].iloc[-1]
        metrics["upper_band_distance"] = (metrics["bollinger_upper"].iloc[-1] - latest_close) / latest_close
        metrics["lower_band_distance"] = (latest_close - metrics["bollinger_lower"].iloc[-1]) / latest_close

        # Calculate recent close volatility (close-to-close)
        close_volatility = data["close"].pct_change().rolling(window=10).std().iloc[-1]
        metrics["close_volatility"] = close_volatility

        # Calculate intraday volatility (high-low range)
        intraday_volatility = ((data["high"] - data["low"]) / data["close"]).rolling(window=10).mean().iloc[-1]
        metrics["intraday_volatility"] = intraday_volatility

        return metrics

    def _detect_volatility_regime(self, data: pd.DataFrame, volatility_metrics: Dict[str, Any], asset: str) -> str:
        """
        Detect current volatility regime based on volatility metrics.

        Args:
            data: Market data
            volatility_metrics: Pre-calculated volatility metrics
            asset: Asset symbol

        Returns:
            String indicating volatility regime ('low', 'normal', 'high', 'increasing', 'decreasing')
        """
        # Get HV percentile as primary regime indicator
        hv_percentile = volatility_metrics["hv_percentile"]

        # Calculate volatility direction
        vol_direction = "stable"
        hv_series = volatility_metrics["hv"].dropna()
        if len(hv_series) >= 5:
            recent_hv = hv_series.iloc[-5:]
            slope = np.polyfit(range(len(recent_hv)), recent_hv.values, 1)[0]

            if slope > 0.001:
                vol_direction = "increasing"
            elif slope < -0.001:
                vol_direction = "decreasing"

        # Determine basic regime
        if hv_percentile <= self.config.low_vol_threshold:
            basic_regime = "low"
        elif hv_percentile >= self.config.high_vol_threshold:
            basic_regime = "high"
        else:
            basic_regime = "normal"

        # Enhance with directional information
        if basic_regime == "low" and vol_direction == "increasing":
            regime = "low_increasing"
        elif basic_regime == "high" and vol_direction == "decreasing":
            regime = "high_decreasing"
        elif vol_direction == "increasing":
            regime = "increasing"
        elif vol_direction == "decreasing":
            regime = "decreasing"
        else:
            regime = basic_regime

        # Check for Bollinger squeeze
        if volatility_metrics["is_bb_squeeze"]:
            regime = "squeeze"

        # Update regime history for this asset
        if asset:
            if asset not in self.regime_history:
                self.regime_history[asset] = []

            self.regime_history[asset].append(
                {
                    "timestamp": data.index[-1],
                    "regime": regime,
                    "hv_percentile": hv_percentile,
                    "bb_width_percentile": volatility_metrics["bb_width_percentile"],
                    "vol_direction": vol_direction,
                }
            )

            # Keep history manageable
            if len(self.regime_history[asset]) > 100:
                self.regime_history[asset].pop(0)

        logger.debug(f"Detected volatility regime for {asset}: {regime} (HV percentile: {hv_percentile:.2f})")
        return regime

    def _generate_regime_based_signals(self, data: pd.DataFrame, volatility_metrics: Dict[str, Any], volatility_regime: str) -> List[Dict[str, Any]]:
        """
        Generate appropriate trading signals based on the detected volatility regime.

        Args:
            data: Market data
            volatility_metrics: Pre-calculated volatility metrics
            volatility_regime: Detected volatility regime

        Returns:
            List of preliminary signal dictionaries
        """
        signals = []

        # Get latest close and ATR for reference
        latest_close = data["close"].iloc[-1]
        latest_atr = volatility_metrics["atr_latest"]

        # Strategy 1: Bollinger Band Squeeze Breakout
        if volatility_regime == "squeeze" or "low_increasing" in volatility_regime:
            # Detect breakout direction
            recent_data = data.iloc[-3:]

            # Check if price is moving out of the bands with increased momentum
            upper_band = volatility_metrics["bollinger_upper"].iloc[-1]
            lower_band = volatility_metrics["bollinger_lower"].iloc[-1]

            # Check for upward breakout
            if recent_data["close"].iloc[-1] > upper_band or recent_data["high"].iloc[-1] > upper_band:

                # Verify momentum with increasing volume
                if (
                    recent_data["volume"].iloc[-1] > data["volume"].iloc[-5:-1].mean() * 1.2
                    and recent_data["close"].iloc[-1] > recent_data["open"].iloc[-1]
                ):

                    signals.append(
                        {
                            "type": "squeeze_breakout",
                            "direction": "up",
                            "entry_price": latest_close,
                            "confidence": self._calculate_breakout_confidence(data, volatility_metrics, "up"),
                            "volatility_regime": volatility_regime,
                            "bb_width_percentile": volatility_metrics["bb_width_percentile"],
                        }
                    )

            # Check for downward breakout
            elif recent_data["close"].iloc[-1] < lower_band or recent_data["low"].iloc[-1] < lower_band:

                # Verify momentum with increasing volume
                if (
                    recent_data["volume"].iloc[-1] > data["volume"].iloc[-5:-1].mean() * 1.2
                    and recent_data["close"].iloc[-1] < recent_data["open"].iloc[-1]
                ):

                    signals.append(
                        {
                            "type": "squeeze_breakout",
                            "direction": "down",
                            "entry_price": latest_close,
                            "confidence": self._calculate_breakout_confidence(data, volatility_metrics, "down"),
                            "volatility_regime": volatility_regime,
                            "bb_width_percentile": volatility_metrics["bb_width_percentile"],
                        }
                    )

        # Strategy 2: Mean reversion during high volatility regimes
        if "high" in volatility_regime:
            # Check for mean reversion opportunities
            upper_band = volatility_metrics["bollinger_upper"].iloc[-1]
            lower_band = volatility_metrics["bollinger_lower"].iloc[-1]
            middle_band = volatility_metrics["bollinger_middle"].iloc[-1]

            # Check for overbought condition (price extended above upper band)
            upper_extension = (latest_close - upper_band) / latest_atr
            if upper_extension > 0:  # Price above upper band
                # Calculate confidence based on extension level
                extension_ratio = upper_extension / self.config.mean_reversion_overextension

                if extension_ratio > 0.8:  # Significant extension
                    signals.append(
                        {
                            "type": "mean_reversion",
                            "direction": "down",
                            "entry_price": latest_close,
                            "confidence": self._calculate_mean_reversion_confidence(data, volatility_metrics, "down"),
                            "volatility_regime": volatility_regime,
                            "extension_ratio": extension_ratio,
                            "target_price": middle_band,
                        }
                    )

            # Check for oversold condition (price extended below lower band)
            lower_extension = (lower_band - latest_close) / latest_atr
            if lower_extension > 0:  # Price below lower band
                # Calculate confidence based on extension level
                extension_ratio = lower_extension / self.config.mean_reversion_overextension

                if extension_ratio > 0.8:  # Significant extension
                    signals.append(
                        {
                            "type": "mean_reversion",
                            "direction": "up",
                            "entry_price": latest_close,
                            "confidence": self._calculate_mean_reversion_confidence(data, volatility_metrics, "up"),
                            "volatility_regime": volatility_regime,
                            "extension_ratio": extension_ratio,
                            "target_price": middle_band,
                        }
                    )

        # Strategy 3: Volatility trend following during directional volatility
        if volatility_regime in ["increasing", "decreasing"]:
            # Identify trend direction
            ema20 = data["close"].ewm(span=20).mean()
            ema50 = data["close"].ewm(span=50).mean()

            trend_up = latest_close > ema20.iloc[-1] > ema50.iloc[-1] and ema20.iloc[-1] > ema20.iloc[-5]

            trend_down = latest_close < ema20.iloc[-1] < ema50.iloc[-1] and ema20.iloc[-1] < ema20.iloc[-5]

            # In increasing volatility regime, go with the trend
            if volatility_regime == "increasing":
                if trend_up:
                    # Check for pullback to moving average as entry
                    if data["low"].iloc[-1] <= ema20.iloc[-1] <= data["high"].iloc[-1] or abs(latest_close - ema20.iloc[-1]) < latest_atr * 0.5:

                        signals.append(
                            {
                                "type": "volatility_trend",
                                "direction": "up",
                                "entry_price": latest_close,
                                "confidence": self._calculate_trend_confidence(data, volatility_metrics, "up"),
                                "volatility_regime": volatility_regime,
                                "atr_percentile": volatility_metrics["hv_percentile"],
                            }
                        )

                elif trend_down:
                    # Check for pullback to moving average as entry
                    if data["low"].iloc[-1] <= ema20.iloc[-1] <= data["high"].iloc[-1] or abs(latest_close - ema20.iloc[-1]) < latest_atr * 0.5:

                        signals.append(
                            {
                                "type": "volatility_trend",
                                "direction": "down",
                                "entry_price": latest_close,
                                "confidence": self._calculate_trend_confidence(data, volatility_metrics, "down"),
                                "volatility_regime": volatility_regime,
                                "atr_percentile": volatility_metrics["hv_percentile"],
                            }
                        )

        # Strategy 4: Volatility contraction entry
        if "low" in volatility_regime or volatility_regime == "decreasing":
            # Check if volatility is extremely low
            if volatility_metrics["hv_percentile"] < self.config.vol_contraction_trade_threshold:
                # Determine likely breakout direction based on recent price action
                price_series = data["close"].iloc[-20:]
                prices_above_middle = sum(price_series > volatility_metrics["bollinger_middle"].iloc[-20:])

                # Skew towards upper or lower half of range suggests probable breakout direction
                breakout_bias = "up" if prices_above_middle > 10 else "down"

                signals.append(
                    {
                        "type": "volatility_contraction",
                        "direction": breakout_bias,
                        "entry_price": latest_close,
                        "confidence": SignalConfidence.MEDIUM,  # Medium confidence due to predictive nature
                        "volatility_regime": volatility_regime,
                        "hv_percentile": volatility_metrics["hv_percentile"],
                    }
                )

        return signals

    def _generate_complete_signal(self, signal: Dict[str, Any], data: pd.DataFrame, atr: float) -> Optional[SignalEvent]:
        """
        Generate a complete signal event with entry, stop loss, take profit levels.

        Args:
            signal: Preliminary signal information
            data: Market data
            atr: ATR value

        Returns:
            Complete SignalEvent or None if invalid
        """
        entry_price = signal["entry_price"]
        direction = signal["direction"]
        signal_type = signal["type"]

        # Set appropriate stop loss based on strategy type and volatility
        stop_loss = None
        take_profit = None

        if signal_type == "squeeze_breakout":
            # For breakout strategies, use tighter stops
            if direction == "up":
                # Stop just below the breakout level
                lower_band = data["low"].iloc[-3:].min()
                stop_loss = max(lower_band, entry_price - (atr * 1.5))
            else:
                # Stop just above the breakout level
                upper_band = data["high"].iloc[-3:].max()
                stop_loss = min(upper_band, entry_price + (atr * 1.5))

            # Target is a multiple of ATR
            risk = abs(entry_price - stop_loss)
            take_profit = (
                entry_price + (risk * self.config.risk_reward_min) if direction == "up" else entry_price - (risk * self.config.risk_reward_min)
            )

        elif signal_type == "mean_reversion":
            # For mean reversion, stop is beyond the extension point
            if direction == "up":
                # Stop below recent low
                stop_loss = min(data["low"].iloc[-5:]) - (atr * 0.5)
            else:
                # Stop above recent high
                stop_loss = max(data["high"].iloc[-5:]) + (atr * 0.5)

            # Target is the mean (middle band)
            take_profit = signal.get("target_price")

        elif signal_type == "volatility_trend":
            # For trend strategies, wider stops
            if direction == "up":
                # Stop below recent swing low or support level
                stop_loss = min(data["low"].iloc[-10:])
            else:
                # Stop above recent swing high or resistance level
                stop_loss = max(data["high"].iloc[-10:])

            # Target is a multiple of risk
            risk = abs(entry_price - stop_loss)
            take_profit = (
                entry_price + (risk * self.config.risk_reward_min) if direction == "up" else entry_price - (risk * self.config.risk_reward_min)
            )

        elif signal_type == "volatility_contraction":
            # For volatility contraction, stop is volatility-adjusted
            volatility_factor = 2.0  # Wider stops for these setups
            if direction == "up":
                stop_loss = entry_price - (atr * volatility_factor)
            else:
                stop_loss = entry_price + (atr * volatility_factor)

            # Target is also volatility-adjusted
            risk = abs(entry_price - stop_loss)
            reward_factor = max(1.5, self.config.risk_reward_min)  # Higher RR for these setups
            take_profit = entry_price + (risk * reward_factor) if direction == "up" else entry_price - (risk * reward_factor)

        # Verify stop loss and take profit
        if stop_loss is None or take_profit is None:
            logger.warning(f"Could not calculate stop loss or take profit for signal: {signal}")
            return None

        # Normalize calculated prices to tick size for the asset
        asset = data.name if hasattr(data, "name") else "DEFAULT"
        tick_size = TICK_SIZE_MAPPING.get(asset, TICK_SIZE_MAPPING["DEFAULT"])
        entry_price, stop_loss, take_profit = normalize_price_series(
            pd.Series([entry_price, stop_loss, take_profit]), tick_size
        ).tolist()

        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(entry_price - take_profit)
        risk_reward_ratio = reward / risk if risk > 0 else 0

        # Filter out signals with inadequate risk-reward
        if risk_reward_ratio < self.config.risk_reward_min:
            logger.debug(f"Filtered out signal with insufficient risk-reward: {risk_reward_ratio:.2f}")
            return None

        # Filter out signals with insufficient confidence
        if signal["confidence"] < self.config.min_signal_confidence:
            logger.debug(f"Filtered out signal with insufficient confidence: {signal['confidence']}")
            return None

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
            risk_reward_ratio=risk_reward_ratio,
            metadata={
                "strategy_type": signal_type,
                "volatility_regime": signal.get("volatility_regime"),
                "hv_percentile": signal.get("hv_percentile"),
                "bb_width_percentile": signal.get("bb_width_percentile"),
                "extension_ratio": signal.get("extension_ratio"),
                "volatility_adjusted": self.config.vol_adjusted_position_sizing,
            },
        )

    def _calculate_breakout_confidence(self, data: pd.DataFrame, volatility_metrics: Dict[str, Any], direction: str) -> SignalConfidence:
        """
        Calculate confidence level for breakout signals.

        Args:
            data: Market data
            volatility_metrics: Volatility metrics
            direction: Breakout direction

        Returns:
            SignalConfidence level
        """
        # Start with base confidence
        confidence = SignalConfidence.MEDIUM

        # Check various factors that increase confidence

        # Factor 1: Volume confirmation
        recent_volume = data["volume"].iloc[-1]
        avg_volume = data["volume"].iloc[-10:-1].mean()
        volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1.0

        if volume_surge > 2.0:
            confidence = min(SignalConfidence.VERY_HIGH, confidence + 2)
        elif volume_surge > 1.5:
            confidence = min(SignalConfidence.HIGH, confidence + 1)

        # Factor 2: Extreme BB squeeze
        bb_width_percentile = volatility_metrics["bb_width_percentile"]
        if bb_width_percentile < 0.05:  # Extremely tight squeeze
            confidence = min(SignalConfidence.VERY_HIGH, confidence + 1)

        # Factor 3: Volatility increase confirmation
        hv_series = volatility_metrics["hv"].dropna()
        if len(hv_series) >= 3:
            vol_increasing = hv_series.iloc[-1] > hv_series.iloc[-2] > hv_series.iloc[-3]
            if vol_increasing:
                confidence = min(SignalConfidence.VERY_HIGH, confidence + 1)

        # Factor 4: Momentum confirmation
        if direction == "up":
            momentum_confirmed = data["close"].iloc[-1] > data["open"].iloc[-1] and data["close"].iloc[-2] > data["open"].iloc[-2]
        else:
            momentum_confirmed = data["close"].iloc[-1] < data["open"].iloc[-1] and data["close"].iloc[-2] < data["open"].iloc[-2]

        if momentum_confirmed:
            confidence = min(SignalConfidence.VERY_HIGH, confidence + 1)

        # Factor 5: False breakout risk
        if bb_width_percentile > 0.3:  # Not a true squeeze
            confidence = max(SignalConfidence.LOW, confidence - 1)

        return confidence

    def _calculate_mean_reversion_confidence(self, data: pd.DataFrame, volatility_metrics: Dict[str, Any], direction: str) -> SignalConfidence:
        """
        Calculate confidence level for mean reversion signals.

        Args:
            data: Market data
            volatility_metrics: Volatility metrics
            direction: Mean reversion direction

        Returns:
            SignalConfidence level
        """
        # Start with base confidence
        confidence = SignalConfidence.MEDIUM

        # Factor 1: Extension level
        if direction == "up":
            lower_band = volatility_metrics["bollinger_lower"].iloc[-1]
            extension = (lower_band - data["close"].iloc[-1]) / volatility_metrics["atr_latest"]
        else:
            upper_band = volatility_metrics["bollinger_upper"].iloc[-1]
            extension = (data["close"].iloc[-1] - upper_band) / volatility_metrics["atr_latest"]

        if extension > self.config.mean_reversion_overextension:
            confidence = min(SignalConfidence.VERY_HIGH, confidence + 1)

        # Factor 2: Reversing candle pattern
        if direction == "up":
            reversal_candle = (
                data["close"].iloc[-1] > data["open"].iloc[-1] and data["low"].iloc[-1] < data["low"].iloc[-2]  # Bullish candle  # New low
            )
        else:
            reversal_candle = (
                data["close"].iloc[-1] < data["open"].iloc[-1] and data["high"].iloc[-1] > data["high"].iloc[-2]  # Bearish candle  # New high
            )

        if reversal_candle:
            confidence = min(SignalConfidence.VERY_HIGH, confidence + 1)

        # Factor 3: Oversold/Overbought confirmation
        rsi = self._calculate_rsi(data["close"], 14)
        if direction == "up" and rsi < 30:  # Oversold
            confidence = min(SignalConfidence.HIGH, confidence + 1)
        elif direction == "down" and rsi > 70:  # Overbought
            confidence = min(SignalConfidence.HIGH, confidence + 1)

        # Factor 4: Volatility regime check
        if volatility_metrics["hv_percentile"] < 0.7:  # Not extreme high volatility
            confidence = max(SignalConfidence.LOW, confidence - 1)

        return confidence

    def _calculate_trend_confidence(self, data: pd.DataFrame, volatility_metrics: Dict[str, Any], direction: str) -> SignalConfidence:
        """
        Calculate confidence level for volatility trend following signals.

        Args:
            data: Market data
            volatility_metrics: Volatility metrics
            direction: Trend direction

        Returns:
            SignalConfidence level
        """
        # Start with base confidence
        confidence = SignalConfidence.MEDIUM

        # Factor 1: Trend strength
        ema20 = data["close"].ewm(span=20).mean()
        ema50 = data["close"].ewm(span=50).mean()

        if direction == "up":
            trend_strength = (ema20.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1]
            if trend_strength > 0.03:  # Strong uptrend
                confidence = min(SignalConfidence.HIGH, confidence + 1)
        else:
            trend_strength = (ema50.iloc[-1] - ema20.iloc[-1]) / ema50.iloc[-1]
            if trend_strength > 0.03:  # Strong downtrend
                confidence = min(SignalConfidence.HIGH, confidence + 1)

        # Factor 2: Rising volatility confirmation
        hv_series = volatility_metrics["hv"].dropna()
        if len(hv_series) >= 5:
            vol_slope = np.polyfit(range(5), hv_series.iloc[-5:].values, 1)[0]
            if vol_slope > 0.001:  # Clearly increasing volatility
                confidence = min(SignalConfidence.HIGH, confidence + 1)

        # Factor 3: Entry timing
        if direction == "up":
            good_entry = data["close"].iloc[-1] > ema20.iloc[-1] and data["low"].iloc[-1] <= ema20.iloc[-1] * 1.01  # Close to EMA
        else:
            good_entry = data["close"].iloc[-1] < ema20.iloc[-1] and data["high"].iloc[-1] >= ema20.iloc[-1] * 0.99  # Close to EMA

        if good_entry:
            confidence = min(SignalConfidence.HIGH, confidence + 1)

        # Factor 4: Volume confirmation
        recent_volume = data["volume"].iloc[-1]
        avg_volume = data["volume"].iloc[-10:-1].mean()
        volume_increasing = recent_volume > avg_volume * 1.2

        if volume_increasing:
            confidence = min(SignalConfidence.VERY_HIGH, confidence + 1)

        return confidence

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI for a given price series."""
        delta = prices.diff()

        gain = delta.copy()
        gain[gain < 0] = 0

        loss = delta.copy()
        loss[loss > 0] = 0
        loss = abs(loss)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]

    def _calculate_percentile(self, series: pd.Series, lookback: int) -> float:
        """
        Calculate the percentile of the latest value in the series.

        Args:
            series: Data series
            lookback: Lookback period for percentile calculation

        Returns:
            Percentile (0-1) of the latest value
        """
        if series is None or len(series) < lookback:
            return 0.5  # Default to median if insufficient data

        # Get recent data for percentile calculation
        recent_data = series.iloc[-lookback:].dropna()

        if len(recent_data) < 2:
            return 0.5

        latest_value = recent_data.iloc[-1]
        # Calculate percentile (percentage of values less than the latest)
        percentile = sum(recent_data < latest_value) / len(recent_data)

        return percentile

    def _update_volatility_history(self, asset: str, volatility_metrics: Dict[str, Any], volatility_regime: str) -> None:
        """
        Update historical volatility metrics for this asset.

        Args:
            asset: Asset symbol
            volatility_metrics: Current volatility metrics
            volatility_regime: Current volatility regime
        """
        if asset not in self.volatility_history:
            self.volatility_history[asset] = []

        # Store key volatility metrics
        self.volatility_history[asset].append(
            {
                "timestamp": pd.Timestamp.now(),
                "atr_pct": volatility_metrics["atr_pct"],
                "hv": volatility_metrics["hv_latest"],
                "hv_percentile": volatility_metrics["hv_percentile"],
                "bb_width": volatility_metrics["bb_width_latest"],
                "bb_width_percentile": volatility_metrics["bb_width_percentile"],
                "regime": volatility_regime,
            }
        )

        # Keep history manageable
        if len(self.volatility_history[asset]) > 100:
            self.volatility_history[asset].pop(0)

    def _update_adaptive_parameters(self, asset: str) -> None:
        """
        Update strategy parameters based on asset volatility characteristics.

        Args:
            asset: Asset symbol
        """
        if not asset or not self.config.adaptive_parameters:
            return

        if asset not in self.volatility_history or len(self.volatility_history[asset]) < 10:
            return

        # Calculate average and standard deviation of volatility
        atr_pct_values = [item["atr_pct"] for item in self.volatility_history[asset]]
        avg_atr_pct = np.mean(atr_pct_values)
        std_atr_pct = np.std(atr_pct_values)

        # Store asset volatility characteristics
        self.asset_volatility_characteristics[asset] = {"avg_atr_pct": avg_atr_pct, "std_atr_pct": std_atr_pct, "updated_at": pd.Timestamp.now()}

        # Adjust parameters based on volatility characteristics
        if avg_atr_pct > 0.02:  # High volatility asset
            if asset not in self.config.asset_specific_config:
                self.config.asset_specific_config[asset] = {}

            # Adjust mean reversion parameters
            self.config.asset_specific_config[asset]["mean_reversion_overextension"] = 3.0

            # Adjust breakout parameters
            self.config.asset_specific_config[asset]["volatility_breakout_multiplier"] = 2.0

            logger.debug(f"Adjusted parameters for high volatility asset {asset}: ATR={avg_atr_pct:.4f}")

        elif avg_atr_pct < 0.005:  # Low volatility asset
            if asset not in self.config.asset_specific_config:
                self.config.asset_specific_config[asset] = {}

            # Adjust mean reversion parameters
            self.config.asset_specific_config[asset]["mean_reversion_overextension"] = 2.0

            # Adjust breakout parameters
            self.config.asset_specific_config[asset]["volatility_breakout_multiplier"] = 1.2

            logger.debug(f"Adjusted parameters for low volatility asset {asset}: ATR={avg_atr_pct:.4f}")

        # Apply strategy effectiveness adjustments
        self._adjust_based_on_effectiveness(asset)

    def _adjust_based_on_effectiveness(self, asset: str) -> None:
        """
        Adjust strategy parameters based on historical effectiveness.

        Args:
            asset: Asset symbol
        """
        if asset not in self.asset_mean_reversion_effectiveness and asset not in self.asset_breakout_effectiveness:
            return

        # Adjust based on mean reversion effectiveness
        if asset in self.asset_mean_reversion_effectiveness:
            mr_effectiveness = self.asset_mean_reversion_effectiveness[asset]

            if mr_effectiveness.get("win_rate", 0) > 0.7:
                # Mean reversion has been very effective for this asset
                if asset not in self.config.asset_specific_config:
                    self.config.asset_specific_config[asset] = {}

                # Lower threshold to take more mean reversion trades
                self.config.asset_specific_config[asset]["mean_reversion_overextension"] = max(1.8, self.config.mean_reversion_overextension * 0.9)

                logger.debug(f"Adjusted mean reversion parameters for {asset} based on high effectiveness: {mr_effectiveness['win_rate']:.2f}")

            elif mr_effectiveness.get("win_rate", 0) < 0.3:
                # Mean reversion has been ineffective for this asset
                if asset not in self.config.asset_specific_config:
                    self.config.asset_specific_config[asset] = {}

                # Raise threshold to take fewer mean reversion trades
                self.config.asset_specific_config[asset]["mean_reversion_overextension"] = min(4.0, self.config.mean_reversion_overextension * 1.2)

                logger.debug(f"Adjusted mean reversion parameters for {asset} based on low effectiveness: {mr_effectiveness['win_rate']:.2f}")

        # Adjust based on breakout effectiveness
        if asset in self.asset_breakout_effectiveness:
            bo_effectiveness = self.asset_breakout_effectiveness[asset]

            if bo_effectiveness.get("win_rate", 0) > 0.7:
                # Breakout has been very effective for this asset
                if asset not in self.config.asset_specific_config:
                    self.config.asset_specific_config[asset] = {}

                # Lower threshold to take more breakout trades
                self.config.asset_specific_config[asset]["volatility_breakout_multiplier"] = max(
                    1.0, self.config.volatility_breakout_multiplier * 0.9
                )

                logger.debug(f"Adjusted breakout parameters for {asset} based on high effectiveness: {bo_effectiveness['win_rate']:.2f}")

            elif bo_effectiveness.get("win_rate", 0) < 0.3:
                # Breakout has been ineffective for this asset
                if asset not in self.config.asset_specific_config:
                    self.config.asset_specific_config[asset] = {}

                # Raise threshold to take fewer breakout trades
                self.config.asset_specific_config[asset]["volatility_breakout_multiplier"] = min(
                    2.5, self.config.volatility_breakout_multiplier * 1.2
                )

                logger.debug(f"Adjusted breakout parameters for {asset} based on low effectiveness: {bo_effectiveness['win_rate']:.2f}")

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
        strategy_type = signal.metadata.get("strategy_type", "unknown")
        is_win = result.get("is_win", False)
        profit_pct = result.get("profit_pct", 0)
        volatility_regime = signal.metadata.get("volatility_regime", "unknown")

        # Process based on strategy type
        if "mean_reversion" in strategy_type:
            # Update mean reversion effectiveness tracking
            if asset not in self.asset_mean_reversion_effectiveness:
                self.asset_mean_reversion_effectiveness[asset] = {"wins": 0, "total": 0, "profit_sum": 0, "win_rate": 0}

            stats = self.asset_mean_reversion_effectiveness[asset]
            stats["total"] += 1
            if is_win:
                stats["wins"] += 1
            stats["profit_sum"] += profit_pct
            stats["win_rate"] = stats["wins"] / stats["total"]

        elif "breakout" in strategy_type or "squeeze" in strategy_type:
            # Update breakout effectiveness tracking
            if asset not in self.asset_breakout_effectiveness:
                self.asset_breakout_effectiveness[asset] = {"wins": 0, "total": 0, "profit_sum": 0, "win_rate": 0}

            stats = self.asset_breakout_effectiveness[asset]
            stats["total"] += 1
            if is_win:
                stats["wins"] += 1
            stats["profit_sum"] += profit_pct
            stats["win_rate"] = stats["wins"] / stats["total"]

        # Store successful patterns for learning
        if is_win and profit_pct > 0.5:  # Significant win
            if asset not in self.successful_patterns:
                self.successful_patterns[asset] = []

            self.successful_patterns[asset].append(
                {
                    "strategy_type": strategy_type,
                    "volatility_regime": volatility_regime,
                    "profit_pct": profit_pct,
                    "hv_percentile": signal.metadata.get("hv_percentile"),
                    "bb_width_percentile": signal.metadata.get("bb_width_percentile"),
                    "timestamp": signal.timestamp,
                }
            )

            # Keep history manageable
            if len(self.successful_patterns[asset]) > 50:
                self.successful_patterns[asset].pop(0)

        # Log key information
        logger.info(
            "Volatility Brain signal result for %s (%s): Win=%s, Profit=%.2f%%, Regime=%s",
            asset,
            strategy_type,
            is_win,
            profit_pct,
            volatility_regime,
        )


# Factory function for strategy creation
def create_strategy(config: Dict[str, Any] = None) -> VolatilityBrain:
    """
    Factory function to create a VolatilityBrain strategy instance.

    Args:
        config: Optional strategy configuration

    Returns:
        Configured VolatilityBrain instance
    """
    return VolatilityBrain(config=config)
