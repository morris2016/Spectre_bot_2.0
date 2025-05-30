
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Pattern Brain Strategy Implementation

This module implements a trading strategy based on pattern recognition,
utilizing the system's advanced pattern detection capabilities to identify
and exploit recurring market patterns with high probability setups.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass

from common.utils import calculate_risk_reward, detect_market_condition
from common.logger import get_logger
from common.exceptions import StrategyError, PatternDetectionError

from strategy_brains.base_brain import BaseBrain, Signal, SignalStrength, SignalType
from intelligence.pattern_recognition.harmonic_patterns import HarmonicPatternDetector
from intelligence.pattern_recognition.candlestick_patterns import CandlestickPatternDetector
from intelligence.pattern_recognition.chart_patterns import ChartPatternDetector
from intelligence.pattern_recognition.volume_patterns import VolumePatternDetector
from feature_service.features.market_structure import MarketStructureAnalyzer

logger = get_logger(__name__)


@dataclass
class PatternMatch:
    """Data class for storing pattern detection results."""
    pattern_type: str
    confidence: float
    completion_price: float
    target_price: float
    stop_price: float
    formation_candles: List[int]
    timeframe: str
    timestamp: datetime
    risk_reward: float
    success_probability: float


class PatternBrain(BaseBrain):
    """
    Advanced pattern-based trading strategy utilizing multiple pattern detection
    technologies to identify high-probability trade setups.
    
    The PatternBrain combines different pattern types (harmonic, candlestick, chart, volume)
    across multiple timeframes to find confluence and generate high-confidence signals.
    """
    
    NAME = "pattern_brain"
    VERSION = "2.1.0"
    DESCRIPTION = "Advanced pattern recognition trading strategy"
    SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    SUPPORTED_ASSETS = ["*"]  # Supports all assets
    MIN_HISTORY_CANDLES = 200
    
    def __init__(self, 
                 config: Dict[str, Any],
                 asset_id: str,
                 exchange_id: str,
                 timeframe: str = "1h"):
        """
        Initialize the PatternBrain strategy.
        
        Args:
            config: Strategy configuration parameters
            asset_id: Identifier for the trading asset
            exchange_id: Identifier for the exchange
            timeframe: Primary trading timeframe
        """
        super().__init__(config, asset_id, exchange_id, timeframe)
        
        # Initialize pattern detectors
        self.harmonic_detector = HarmonicPatternDetector()
        self.candlestick_detector = CandlestickPatternDetector()
        self.chart_detector = ChartPatternDetector()
        self.volume_detector = VolumePatternDetector()
        self.structure_analyzer = MarketStructureAnalyzer()
        
        # Configure strategy parameters
        self.min_pattern_confidence = self.config.get("min_pattern_confidence", 0.7)
        self.min_risk_reward = self.config.get("min_risk_reward", 1.5)
        self.lookback_periods = self.config.get("lookback_periods", 100)
        self.pattern_confluence_threshold = self.config.get("pattern_confluence_threshold", 2)
        self.volume_confirmation_required = self.config.get("volume_confirmation_required", True)
        self.multi_timeframe_confirmation = self.config.get("multi_timeframe_confirmation", True)
        
        # Mapping of timeframes for multi-timeframe analysis
        self.higher_timeframe_map = {
            "1m": "15m",
            "5m": "1h",
            "15m": "4h",
            "30m": "4h",
            "1h": "1d",
            "4h": "1d",
            "1d": "1w"
        }
        
        # Historical pattern performance tracking
        self.pattern_performance = {}
        self.pattern_history = []
        
        # Load historical performance data if available
        self._load_performance_data()
        
        logger.info(f"PatternBrain initialized for {asset_id} on {exchange_id}, timeframe: {timeframe}")

    async def analyze(self, data: Dict[str, pd.DataFrame]) -> Signal:
        """
        Analyze market data to detect patterns and generate trading signals.
        
        Args:
            data: Dictionary of DataFrames containing OHLCV data for multiple timeframes
            
        Returns:
            Signal object containing the trading signal details
        """
        try:
            primary_data = data.get(self.timeframe)
            if primary_data is None or len(primary_data) < self.MIN_HISTORY_CANDLES:
                return Signal(
                    brain_id=self.id,
                    signal_type=SignalType.NEUTRAL,
                    strength=SignalStrength.WEAK,
                    metadata={"error": "Insufficient data for pattern analysis"}
                )
            
            # Detect market condition to adapt pattern recognition
            market_condition = detect_market_condition(primary_data)
            logger.debug(f"Market condition detected: {market_condition}")
            
            # Track detected patterns with their confidence scores
            detected_patterns = []
            
            # Perform pattern detection across all supported pattern types
            harmonic_patterns = await self._detect_harmonic_patterns(primary_data)
            candlestick_patterns = await self._detect_candlestick_patterns(primary_data)
            chart_patterns = await self._detect_chart_patterns(primary_data)
            volume_patterns = await self._detect_volume_patterns(primary_data)
            
            # Combine all detected patterns
            detected_patterns.extend(harmonic_patterns)
            detected_patterns.extend(candlestick_patterns)
            detected_patterns.extend(chart_patterns)
            detected_patterns.extend(volume_patterns)
            
            # Filter patterns by minimum confidence
            qualified_patterns = [
                p for p in detected_patterns 
                if p.confidence >= self.min_pattern_confidence and p.risk_reward >= self.min_risk_reward
            ]
            
            # Check for multi-timeframe confirmation if enabled
            if self.multi_timeframe_confirmation and self.higher_timeframe_map.get(self.timeframe):
                higher_tf = self.higher_timeframe_map[self.timeframe]
                higher_tf_data = data.get(higher_tf)
                
                if higher_tf_data is not None and len(higher_tf_data) >= 100:
                    higher_tf_trend = self._determine_higher_timeframe_trend(higher_tf_data)
                    qualified_patterns = self._filter_by_higher_timeframe(qualified_patterns, higher_tf_trend)
            
            # If no qualified patterns are found, return neutral signal
            if not qualified_patterns:
                return Signal(
                    brain_id=self.id,
                    signal_type=SignalType.NEUTRAL,
                    strength=SignalStrength.WEAK,
                    metadata={"patterns_detected": len(detected_patterns), "patterns_qualified": 0}
                )
            
            # Analyze confluence of patterns
            buy_patterns = [p for p in qualified_patterns if self._is_bullish_pattern(p)]
            sell_patterns = [p for p in qualified_patterns if self._is_bearish_pattern(p)]
            
            # Determine signal direction and strength based on pattern confluence
            signal_type, signal_strength, best_pattern = self._calculate_signal(
                buy_patterns, sell_patterns, market_condition
            )
            
            # Create signal with detailed metadata
            signal = Signal(
                brain_id=self.id,
                signal_type=signal_type,
                strength=signal_strength,
                metadata={
                    "pattern_type": best_pattern.pattern_type if best_pattern else None,
                    "confidence": best_pattern.confidence if best_pattern else None,
                    "target_price": best_pattern.target_price if best_pattern else None,
                    "stop_price": best_pattern.stop_price if best_pattern else None,
                    "risk_reward": best_pattern.risk_reward if best_pattern else None,
                    "success_probability": best_pattern.success_probability if best_pattern else None,
                    "market_condition": market_condition,
                    "patterns_detected": len(detected_patterns),
                    "patterns_qualified": len(qualified_patterns),
                    "buy_patterns": len(buy_patterns),
                    "sell_patterns": len(sell_patterns)
                }
            )
            
            # Log pattern detection details
            logger.info(
                f"PatternBrain signal: {signal_type} with strength {signal_strength}. "
                f"Detected {len(detected_patterns)} patterns, {len(qualified_patterns)} qualified."
            )
            
            return signal
            
        except PatternDetectionError as e:
            logger.error(f"Pattern detection error: {str(e)}")
            return Signal(
                brain_id=self.id,
                signal_type=SignalType.NEUTRAL,
                strength=SignalStrength.WEAK,
                metadata={"error": str(e)}
            )
        except Exception as e:
            logger.exception(f"Unexpected error in PatternBrain analysis: {str(e)}")
            return Signal(
                brain_id=self.id,
                signal_type=SignalType.NEUTRAL,
                strength=SignalStrength.WEAK,
                metadata={"error": f"Unexpected error: {str(e)}"}
            )

    async def _detect_harmonic_patterns(self, data: pd.DataFrame) -> List[PatternMatch]:
        """Detect harmonic patterns in the price data."""
        try:
            harmonic_results = await self.harmonic_detector.detect_patterns(data, self.lookback_periods)
            
            patterns = []
            for pattern in harmonic_results:
                # Adjust historical success rate if we have data
                success_rate = self._get_pattern_success_rate(pattern["pattern_name"], "harmonic")
                
                # Create pattern match object
                match = PatternMatch(
                    pattern_type=f"harmonic_{pattern['pattern_name']}",
                    confidence=pattern["confidence"],
                    completion_price=pattern["completion_price"],
                    target_price=pattern["target_price"],
                    stop_price=pattern["stop_price"],
                    formation_candles=pattern["formation_candles"],
                    timeframe=self.timeframe,
                    timestamp=data.index[-1],
                    risk_reward=pattern["risk_reward"],
                    success_probability=success_rate
                )
                patterns.append(match)
                
            return patterns
        except Exception as e:
            logger.error(f"Error detecting harmonic patterns: {str(e)}")
            return []

    async def _detect_candlestick_patterns(self, data: pd.DataFrame) -> List[PatternMatch]:
        """Detect candlestick patterns in the price data."""
        try:
            candlestick_results = await self.candlestick_detector.detect_patterns(data, self.lookback_periods)
            
            patterns = []
            for pattern in candlestick_results:
                # Calculate risk-reward and determine target/stop prices
                current_close = data['close'].iloc[-1]
                atr = data['atr'].iloc[-1] if 'atr' in data.columns else self._calculate_atr(data)
                
                if pattern["pattern_type"] == "bullish":
                    stop_price = current_close - (atr * 1.5)
                    target_price = current_close + (atr * 3.0)
                else:  # bearish
                    stop_price = current_close + (atr * 1.5)
                    target_price = current_close - (atr * 3.0)
                
                risk_reward = abs(target_price - current_close) / abs(stop_price - current_close)
                
                # Adjust historical success rate if we have data
                success_rate = self._get_pattern_success_rate(pattern["pattern_name"], "candlestick")
                
                # Create pattern match object
                match = PatternMatch(
                    pattern_type=f"candlestick_{pattern['pattern_name']}",
                    confidence=pattern["confidence"],
                    completion_price=current_close,
                    target_price=target_price,
                    stop_price=stop_price,
                    formation_candles=pattern["formation_candles"],
                    timeframe=self.timeframe,
                    timestamp=data.index[-1],
                    risk_reward=risk_reward,
                    success_probability=success_rate
                )
                patterns.append(match)
                
            return patterns
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns: {str(e)}")
            return []

    async def _detect_chart_patterns(self, data: pd.DataFrame) -> List[PatternMatch]:
        """Detect chart patterns in the price data."""
        try:
            chart_results = await self.chart_detector.detect_patterns(data, self.lookback_periods)
            
            patterns = []
            for pattern in chart_results:
                # Adjust historical success rate if we have data
                success_rate = self._get_pattern_success_rate(pattern["pattern_name"], "chart")
                
                # Create pattern match object
                match = PatternMatch(
                    pattern_type=f"chart_{pattern['pattern_name']}",
                    confidence=pattern["confidence"],
                    completion_price=pattern["completion_price"],
                    target_price=pattern["target_price"],
                    stop_price=pattern["stop_price"],
                    formation_candles=pattern["formation_candles"],
                    timeframe=self.timeframe,
                    timestamp=data.index[-1],
                    risk_reward=pattern["risk_reward"],
                    success_probability=success_rate
                )
                patterns.append(match)
                
            return patterns
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {str(e)}")
            return []

    async def _detect_volume_patterns(self, data: pd.DataFrame) -> List[PatternMatch]:
        """Detect volume patterns in the price data."""
        try:
            volume_results = await self.volume_detector.detect_patterns(data, self.lookback_periods)
            
            patterns = []
            for pattern in volume_results:
                # For volume patterns, we often need to estimate target/stop prices
                current_close = data['close'].iloc[-1]
                atr = data['atr'].iloc[-1] if 'atr' in data.columns else self._calculate_atr(data)
                
                # Different multipliers based on pattern type
                multiplier = 2.0
                if "climax" in pattern["pattern_name"].lower():
                    multiplier = 3.0
                elif "divergence" in pattern["pattern_name"].lower():
                    multiplier = 2.5
                
                if pattern["pattern_type"] == "bullish":
                    stop_price = current_close - (atr * 1.0)
                    target_price = current_close + (atr * multiplier)
                else:  # bearish
                    stop_price = current_close + (atr * 1.0)
                    target_price = current_close - (atr * multiplier)
                
                risk_reward = abs(target_price - current_close) / abs(stop_price - current_close)
                
                # Adjust historical success rate if we have data
                success_rate = self._get_pattern_success_rate(pattern["pattern_name"], "volume")
                
                # Create pattern match object
                match = PatternMatch(
                    pattern_type=f"volume_{pattern['pattern_name']}",
                    confidence=pattern["confidence"],
                    completion_price=current_close,
                    target_price=target_price,
                    stop_price=stop_price,
                    formation_candles=pattern["formation_candles"],
                    timeframe=self.timeframe,
                    timestamp=data.index[-1],
                    risk_reward=risk_reward,
                    success_probability=success_rate
                )
                patterns.append(match)
                
            return patterns
        except Exception as e:
            logger.error(f"Error detecting volume patterns: {str(e)}")
            return []

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range if not available in the data."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.vstack([tr1, tr2, tr3])
        true_range = np.max(tr, axis=0)
        atr = np.mean(true_range[-period:])
        
        return float(atr)

    def _determine_higher_timeframe_trend(self, data: pd.DataFrame) -> str:
        """Determine the trend in the higher timeframe."""
        # Use market structure analysis for trend determination
        try:
            structure = self.structure_analyzer.analyze_market_structure(data)
            return structure.get("trend", "neutral")
        except Exception as e:
            logger.error(f"Error determining higher timeframe trend: {str(e)}")
            return "neutral"

    def _filter_by_higher_timeframe(self, patterns: List[PatternMatch], higher_tf_trend: str) -> List[PatternMatch]:
        """Filter patterns based on higher timeframe trend alignment."""
        filtered_patterns = []
        
        for pattern in patterns:
            pattern_direction = "bullish" if self._is_bullish_pattern(pattern) else "bearish"
            
            # Check if pattern aligns with higher timeframe trend
            if (higher_tf_trend == "uptrend" and pattern_direction == "bullish") or \
               (higher_tf_trend == "downtrend" and pattern_direction == "bearish") or \
               (higher_tf_trend == "neutral"):
                # Increase confidence for aligned patterns
                if (higher_tf_trend == "uptrend" and pattern_direction == "bullish") or \
                   (higher_tf_trend == "downtrend" and pattern_direction == "bearish"):
                    pattern.confidence = min(pattern.confidence * 1.2, 1.0)
                
                filtered_patterns.append(pattern)
        
        return filtered_patterns

    def _is_bullish_pattern(self, pattern: PatternMatch) -> bool:
        """Determine if a pattern is bullish based on target and stop prices."""
        return pattern.target_price > pattern.completion_price

    def _is_bearish_pattern(self, pattern: PatternMatch) -> bool:
        """Determine if a pattern is bearish based on target and stop prices."""
        return pattern.target_price < pattern.completion_price

    def _calculate_signal(self, 
                         buy_patterns: List[PatternMatch], 
                         sell_patterns: List[PatternMatch],
                         market_condition: str) -> Tuple[SignalType, SignalStrength, Optional[PatternMatch]]:
        """
        Calculate the final signal type and strength based on pattern confluence.
        
        Returns:
            Tuple of (SignalType, SignalStrength, best_pattern)
        """
        # Default to neutral signal
        signal_type = SignalType.NEUTRAL
        signal_strength = SignalStrength.WEAK
        best_pattern = None
        
        # Count patterns and calculate average confidence
        buy_count = len(buy_patterns)
        sell_count = len(sell_patterns)
        
        # No patterns detected
        if buy_count == 0 and sell_count == 0:
            return signal_type, signal_strength, best_pattern
            
        # Calculate average confidences
        avg_buy_conf = np.mean([p.confidence for p in buy_patterns]) if buy_patterns else 0
        avg_sell_conf = np.mean([p.confidence for p in sell_patterns]) if sell_patterns else 0
        
        # Sort patterns by confidence * success_probability * risk_reward for finding best pattern
        buy_patterns.sort(key=lambda p: p.confidence * p.success_probability * p.risk_reward, reverse=True)
        sell_patterns.sort(key=lambda p: p.confidence * p.success_probability * p.risk_reward, reverse=True)
        
        # Determine signal direction
        if buy_count > sell_count and avg_buy_conf > avg_sell_conf:
            signal_type = SignalType.BUY
            best_pattern = buy_patterns[0] if buy_patterns else None
        elif sell_count > buy_count and avg_sell_conf > avg_buy_conf:
            signal_type = SignalType.SELL
            best_pattern = sell_patterns[0] if sell_patterns else None
        
        # Neutral if there's balance or no clear signal
        else:
            # Check if any pattern has exceptionally high confidence
            best_buy = buy_patterns[0] if buy_patterns else None
            best_sell = sell_patterns[0] if sell_patterns else None
            
            if best_buy and best_sell:
                buy_score = best_buy.confidence * best_buy.success_probability * best_buy.risk_reward
                sell_score = best_sell.confidence * best_sell.success_probability * best_sell.risk_reward
                
                if buy_score > sell_score * 1.5:  # Significantly stronger buy signal
                    signal_type = SignalType.BUY
                    best_pattern = best_buy
                elif sell_score > buy_score * 1.5:  # Significantly stronger sell signal
                    signal_type = SignalType.SELL
                    best_pattern = best_sell
            
        # Determine signal strength based on pattern confidence, counts and market condition
        if best_pattern:
            # Base strength on best pattern confidence
            strength_value = best_pattern.confidence
            
            # Adjust for pattern confluence
            if signal_type == SignalType.BUY and buy_count >= self.pattern_confluence_threshold:
                strength_value += 0.1 * min(buy_count - 1, 3)  # Up to 0.3 bonus for multiple patterns
            elif signal_type == SignalType.SELL and sell_count >= self.pattern_confluence_threshold:
                strength_value += 0.1 * min(sell_count - 1, 3)  # Up to 0.3 bonus for multiple patterns
            
            # Adjust for market condition alignment
            if (signal_type == SignalType.BUY and market_condition == "uptrend") or \
               (signal_type == SignalType.SELL and market_condition == "downtrend"):
                strength_value += 0.1  # Bonus for trend alignment
                
            # Adjust for success probability
            strength_value += (best_pattern.success_probability - 0.5) * 0.2  # Up to 0.1 bonus
            
            # Adjust for risk-reward
            if best_pattern.risk_reward >= 2.0:
                strength_value += 0.1
            
            # Convert to enum
            if strength_value >= 0.8:
                signal_strength = SignalStrength.VERY_STRONG
            elif strength_value >= 0.7:
                signal_strength = SignalStrength.STRONG
            elif strength_value >= 0.6:
                signal_strength = SignalStrength.MODERATE
            else:
                signal_strength = SignalStrength.WEAK
        
        return signal_type, signal_strength, best_pattern

    def _get_pattern_success_rate(self, pattern_name: str, pattern_category: str) -> float:
        """
        Get the historical success rate for a specific pattern.
        Returns default values if no historical data available.
        """
        pattern_key = f"{pattern_category}_{pattern_name}"
        
        if pattern_key in self.pattern_performance:
            return self.pattern_performance[pattern_key]["success_rate"]
        
        # Default success rates by category if no historical data
        default_rates = {
            "harmonic": 0.65,
            "candlestick": 0.55,
            "chart": 0.60,
            "volume": 0.58
        }
        
        return default_rates.get(pattern_category, 0.5)

    def _load_performance_data(self):
        """Load historical pattern performance data from database."""
        try:
            # This would normally load from a database
            # For demonstration, we initialize with some reasonable defaults
            # In a real implementation, this would fetch actual historical performance
            
            self.pattern_performance = {
                "harmonic_gartley": {"success_rate": 0.68, "trades": 120},
                "harmonic_butterfly": {"success_rate": 0.72, "trades": 95},
                "harmonic_bat": {"success_rate": 0.67, "trades": 110},
                "harmonic_crab": {"success_rate": 0.71, "trades": 85},
                "chart_head_shoulders": {"success_rate": 0.63, "trades": 150},
                "chart_double_top": {"success_rate": 0.65, "trades": 130},
                "chart_double_bottom": {"success_rate": 0.67, "trades": 125},
                "chart_triangle": {"success_rate": 0.59, "trades": 160},
                "candlestick_doji": {"success_rate": 0.52, "trades": 200},
                "candlestick_engulfing": {"success_rate": 0.62, "trades": 180},
                "candlestick_hammer": {"success_rate": 0.59, "trades": 170},
                "volume_climax": {"success_rate": 0.64, "trades": 90},
                "volume_divergence": {"success_rate": 0.61, "trades": 110}
            }
            
            logger.info(f"Loaded performance data for {len(self.pattern_performance)} pattern types")
            
        except Exception as e:
            logger.error(f"Error loading pattern performance data: {str(e)}")
            # Initialize empty performance data
            self.pattern_performance = {}

    async def on_trade_completed(self, trade_result: Dict[str, Any]):
        """
        Update pattern performance statistics when a trade is completed.
        This is called by the system when a trade based on this strategy completes.
        
        Args:
            trade_result: Details about the completed trade
        """
        try:
            # Extract pattern info from trade metadata
            if "pattern_type" not in trade_result:
                return
                
            pattern_type = trade_result["pattern_type"]
            was_successful = trade_result.get("successful", False)
            profit_loss = trade_result.get("profit_loss", 0.0)
            
            # Update pattern statistics
            if pattern_type in self.pattern_performance:
                stats = self.pattern_performance[pattern_type]
                total_trades = stats["trades"] + 1
                successful_trades = int(stats["success_rate"] * stats["trades"]) + (1 if was_successful else 0)
                new_success_rate = successful_trades / total_trades
                
                self.pattern_performance[pattern_type] = {
                    "success_rate": new_success_rate,
                    "trades": total_trades
                }
                
                logger.info(
                    f"Updated pattern {pattern_type} performance: "
                    f"{new_success_rate:.2f} success rate over {total_trades} trades"
                )
            else:
                # First trade with this pattern
                self.pattern_performance[pattern_type] = {
                    "success_rate": 1.0 if was_successful else 0.0,
                    "trades": 1
                }
            
            # Add to pattern history
            self.pattern_history.append({
                "pattern_type": pattern_type,
                "timestamp": datetime.now(),
                "successful": was_successful,
                "profit_loss": profit_loss
            })
            
            # In a full implementation, we would persist this data to database
            
        except Exception as e:
            logger.error(f"Error updating pattern performance data: {str(e)}")

    async def adapt(self, performance_metrics: Dict[str, Any]):
        """
        Adapt the strategy based on recent performance.
        This is called periodically by the system to allow the strategy to adapt.
        
        Args:
            performance_metrics: Recent performance metrics for this strategy
        """
        try:
            # Extract relevant metrics
            win_rate = performance_metrics.get("win_rate", 0.5)
            avg_risk_reward = performance_metrics.get("avg_risk_reward", 1.0)
            profit_factor = performance_metrics.get("profit_factor", 1.0)
            
            # Adapt strategy parameters based on performance
            if win_rate < 0.4:
                # If win rate is too low, increase minimum confidence requirement
                self.min_pattern_confidence = min(self.min_pattern_confidence + 0.05, 0.9)
                logger.info(f"Adapting: Increased min pattern confidence to {self.min_pattern_confidence:.2f}")
                
            if avg_risk_reward < 1.2:
                # If risk-reward is too low, increase minimum requirement
                self.min_risk_reward = min(self.min_risk_reward + 0.1, 2.5)
                logger.info(f"Adapting: Increased min risk-reward to {self.min_risk_reward:.2f}")
                
            if profit_factor < 1.0:
                # If losing money overall, require multi-timeframe confirmation
                self.multi_timeframe_confirmation = True
                logger.info(f"Adapting: Enabled multi-timeframe confirmation")
                
            # If performance is good, we can be slightly less restrictive
            if win_rate > 0.6 and profit_factor > 1.5:
                self.min_pattern_confidence = max(self.min_pattern_confidence - 0.03, 0.65)
                logger.info(f"Adapting: Decreased min pattern confidence to {self.min_pattern_confidence:.2f}")
                
            # Log adaptation
            logger.info(
                f"Strategy adapted based on performance metrics: "
                f"win_rate={win_rate:.2f}, avg_risk_reward={avg_risk_reward:.2f}, "
                f"profit_factor={profit_factor:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error in strategy adaptation: {str(e)}")
