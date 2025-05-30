#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Momentum Strategy Brain Implementation

This module implements a sophisticated momentum-based trading strategy with adaptive parameters,
multiple timeframe analysis, and advanced signal generation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field

# Internal imports
from strategy_brains.base_brain import BaseBrain, BrainConfig, SignalStrength, TradeDirection, AssetType
from common.constants import TIMEFRAMES
from common.utils import calculate_success_rate, calculate_expected_value
from feature_service.features.technical import (
    calculate_rsi, calculate_macd, calculate_adx, calculate_stochastic, 
    calculate_obv, detect_divergence
)
from feature_service.features.volatility import calculate_atr, calculate_bollinger_bands
from feature_service.features.volume import calculate_volume_profile, detect_volume_climax
from feature_service.features.pattern import detect_patterns
from ml_models.prediction import predict_momentum_score

logger = logging.getLogger(__name__)

@dataclass
class MomentumBrainConfig(BrainConfig):
    """Configuration for Momentum Strategy Brain"""
    # Timeframes to analyze (in minutes)
    timeframes: List[int] = field(default_factory=lambda: [5, 15, 60, 240])
    
    # RSI parameters
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_weight: float = 0.2
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_weight: float = 0.3
    
    # ADX parameters
    adx_period: int = 14
    adx_threshold: float = 25.0
    adx_weight: float = 0.15
    
    # Volume parameters
    volume_lookback: int = 20
    volume_threshold: float = 1.5
    volume_weight: float = 0.2
    
    # Price action parameters
    price_action_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    price_action_weight: float = 0.15
    
    # Divergence detection
    divergence_lookback: int = 40
    
    # Adaptive parameter settings
    adaptive_parameters: bool = True
    parameter_update_frequency: int = 24  # hours
    
    # ML model integration
    use_ml_models: bool = True
    ml_prediction_weight: float = 0.4
    
    # Capital allocation
    max_position_size: float = 0.05  # Max 5% of capital per trade
    
    # Risk management
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    trailing_stop_activation: float = 0.01  # 1% in profit
    trailing_stop_distance: float = 0.005  # 0.5% trailing
    
    # Multi-timeframe consensus
    timeframe_weights: Dict[int, float] = field(default_factory=lambda: {
        5: 0.15,    # 5-minute weight
        15: 0.20,   # 15-minute weight
        60: 0.30,   # 1-hour weight
        240: 0.35   # 4-hour weight
    })


class MomentumBrain(BaseBrain):
    """
    Momentum Strategy Brain Implementation
    
    This class implements a sophisticated momentum-based trading strategy that:
    - Analyzes multiple timeframes for consensus
    - Detects trend strength and continuation patterns
    - Incorporates volume analysis for confirmation
    - Uses ML models for momentum prediction enhancement
    - Adapts parameters based on market conditions
    - Implements advanced position sizing and risk management
    """
    
    def __init__(self, config: Optional[MomentumBrainConfig] = None):
        """
        Initialize the Momentum Brain strategy
        
        Args:
            config: Configuration for the momentum strategy (optional)
        """
        self.config = config or MomentumBrainConfig()
        super().__init__(self.config)
        
        # Recent performance metrics
        self.recent_trades = []
        self.success_rate = 0.0
        self.expected_value = 0.0
        
        # Strategy state
        self.last_parameter_update = datetime.now()
        self.asset_momentum_cache = {}  # Cache for asset-specific momentum scores
        self.divergence_alerts = {}  # Track divergences by asset
        
        # Performance tracking by market regime
        self.regime_performance = {
            "trending": {"trades": 0, "wins": 0},
            "ranging": {"trades": 0, "wins": 0},
            "volatile": {"trades": 0, "wins": 0}
        }
        
        logger.info(f"Momentum Brain initialized with {len(self.config.timeframes)} timeframes")
    
    def initialize(self, assets: List[str], platform: str) -> None:
        """
        Initialize the strategy for specific assets and platform
        
        Args:
            assets: List of asset symbols to trade
            platform: Trading platform (e.g., 'binance', 'deriv')
        """
        super().initialize(assets, platform)
        
        # Initialize asset-specific configurations
        for asset in assets:
            # Load asset-specific optimized parameters if available
            self._load_asset_specific_parameters(asset, platform)
            
            # Initialize momentum cache for each asset
            self.asset_momentum_cache[asset] = {
                tf: {"score": 0, "timestamp": datetime.now() - timedelta(days=1)} 
                for tf in self.config.timeframes
            }
            
            # Initialize divergence tracking
            self.divergence_alerts[asset] = {
                "rsi": {"active": False, "timestamp": None},
                "macd": {"active": False, "timestamp": None},
                "obv": {"active": False, "timestamp": None}
            }
        
        logger.info(f"Momentum Brain initialized for {len(assets)} assets on {platform}")
    
    def _load_asset_specific_parameters(self, asset: str, platform: str) -> None:
        """
        Load asset-specific optimized parameters from database
        
        Args:
            asset: Asset symbol
            platform: Trading platform
        """
        try:
            # This would typically load from a database
            # For now, we'll adjust parameters based on asset volatility class
            asset_volatility = self._get_asset_volatility_class(asset, platform)
            
            if asset_volatility == "high":
                # Adjust parameters for high volatility assets
                self.config.rsi_overbought = 75.0
                self.config.rsi_oversold = 25.0
                self.config.stop_loss_atr_multiplier = 2.5
                self.config.take_profit_atr_multiplier = 3.5
            elif asset_volatility == "low":
                # Adjust parameters for low volatility assets
                self.config.rsi_overbought = 65.0
                self.config.rsi_oversold = 35.0
                self.config.stop_loss_atr_multiplier = 1.5
                self.config.take_profit_atr_multiplier = 2.5
            
            logger.debug(f"Loaded parameters for {asset} on {platform} (volatility: {asset_volatility})")
        except Exception as e:
            logger.warning(f"Failed to load asset-specific parameters for {asset}: {str(e)}")
    
    def _get_asset_volatility_class(self, asset: str, platform: str) -> str:
        """
        Determine the volatility class of an asset
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            
        Returns:
            Volatility class ("high", "medium", "low")
        """
        # This would typically query a database or calculate based on historical data
        # For demonstration, we'll classify based on asset type
        
        # Cryptocurrency assets are typically higher volatility
        if platform == "binance" or asset.endswith("USD") or asset.endswith("USDT"):
            return "high"
        # Forex majors typically have lower volatility
        elif any(forex in asset for forex in ["EUR", "USD", "JPY", "GBP"]):
            return "low"
        # Default to medium volatility
        return "medium"
    
    def analyze(self, asset: str, data: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze market data for momentum signals
        
        Args:
            asset: Asset symbol
            data: Dictionary mapping timeframes to dataframes of OHLCV data
            
        Returns:
            Analysis results including momentum scores, signals, and key levels
        """
        momentum_scores = {}
        signals = {}
        key_levels = {}
        current_regime = self._detect_market_regime(data)
        
        # Only analyze timeframes for which we have data
        available_timeframes = set(data.keys()).intersection(self.config.timeframes)
        
        # Calculate consolidated momentum score across timeframes
        for tf in available_timeframes:
            df = data[tf].copy()
            
            # Skip if not enough data
            if len(df) < max(self.config.rsi_period, self.config.macd_slow) + 10:
                logger.warning(f"Not enough data for {asset} on {tf} timeframe")
                continue
            
            # Calculate technical indicators
            df['rsi'] = calculate_rsi(df['close'], self.config.rsi_period)
            df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(
                df['close'], 
                self.config.macd_fast, 
                self.config.macd_slow, 
                self.config.macd_signal
            )
            df['adx'] = calculate_adx(df['high'], df['low'], df['close'], self.config.adx_period)
            df['obv'] = calculate_obv(df['close'], df['volume'])
            df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
            df['upper_band'], df['middle_band'], df['lower_band'] = calculate_bollinger_bands(df['close'], 20, 2)
            df['stoch_k'], df['stoch_d'] = calculate_stochastic(df['high'], df['low'], df['close'], 14, 3)
            
            # Detect patterns
            patterns = detect_patterns(df)
            
            # Detect divergences
            divergences = detect_divergence(df['close'], df['rsi'], df['macd'], df['obv'], self.config.divergence_lookback)
            
            # Calculate momentum score components
            rsi_score = self._calculate_rsi_score(df['rsi'].iloc[-1])
            macd_score = self._calculate_macd_score(df['macd'].iloc[-1], df['macd_signal'].iloc[-1], df['macd_hist'].iloc[-3:])
            adx_score = self._calculate_adx_score(df['adx'].iloc[-1])
            volume_score = self._calculate_volume_score(df, tf)
            price_action_score = self._calculate_price_action_score(df)
            pattern_score = self._calculate_pattern_score(patterns)
            divergence_score = self._calculate_divergence_score(divergences)
            
            # Calculate ML-based prediction if enabled
            ml_score = 0
            if self.config.use_ml_models:
                ml_features = self._extract_ml_features(df)
                ml_score = predict_momentum_score(asset, tf, ml_features)
            
            # Weight individual scores based on market regime
            regime_weights = self._get_regime_specific_weights(current_regime)
            
            # Calculate combined momentum score
            weighted_score = (
                rsi_score * regime_weights.get('rsi', self.config.rsi_weight) +
                macd_score * regime_weights.get('macd', self.config.macd_weight) +
                adx_score * regime_weights.get('adx', self.config.adx_weight) +
                volume_score * regime_weights.get('volume', self.config.volume_weight) +
                price_action_score * regime_weights.get('price_action', self.config.price_action_weight) +
                pattern_score * regime_weights.get('pattern', 0.1) +
                divergence_score * regime_weights.get('divergence', 0.1)
            )
            
            # Incorporate ML prediction if enabled
            if self.config.use_ml_models:
                weighted_score = (weighted_score * (1 - self.config.ml_prediction_weight) + 
                               ml_score * self.config.ml_prediction_weight)
            
            # Store momentum score
            momentum_scores[tf] = weighted_score
            
            # Determine signal for this timeframe
            signal = self._determine_signal(df, weighted_score, divergences, patterns)
            signals[tf] = signal
            
            # Identify key price levels
            key_levels[tf] = self._identify_key_levels(df)
            
            # Update momentum cache
            self.asset_momentum_cache[asset][tf] = {
                "score": weighted_score,
                "timestamp": datetime.now()
            }
        
        # Combine timeframe signals with weighted consensus
        combined_signal = self._combine_timeframe_signals(signals, available_timeframes)
        
        # Update divergence alerts
        self._update_divergence_alerts(asset, divergences)
        
        return {
            "asset": asset,
            "momentum_scores": momentum_scores,
            "signals": signals,
            "combined_signal": combined_signal,
            "key_levels": key_levels,
            "market_regime": current_regime,
            "entry_price": self._calculate_optimal_entry(data, combined_signal['direction']),
            "stop_loss": self._calculate_stop_loss(data, combined_signal['direction']),
            "take_profit": self._calculate_take_profit(data, combined_signal['direction']),
            "confidence": combined_signal['strength'],
            "position_size": self._calculate_position_size(asset, combined_signal['strength']),
            "timestamp": datetime.now()
        }
    
    def _detect_market_regime(self, data: Dict[int, pd.DataFrame]) -> str:
        """
        Detect the current market regime (trending, ranging, volatile)
        
        Args:
            data: Market data for different timeframes
            
        Returns:
            Market regime classification
        """
        # Use the 1-hour timeframe if available, otherwise use the highest available
        tf = 60 if 60 in data else max(data.keys())
        df = data[tf].copy()
        
        # Calculate ADX for trend strength
        adx = calculate_adx(df['high'], df['low'], df['close'], 14).iloc[-1]
        
        # Calculate volatility using ATR relative to price
        atr = calculate_atr(df['high'], df['low'], df['close'], 14).iloc[-1]
        price = df['close'].iloc[-1]
        relative_volatility = atr / price * 100  # as percentage of price
        
        # Calculate price range over last 20 periods
        high_low_range = (df['high'].rolling(20).max() - df['low'].rolling(20).min()).iloc[-1]
        price_range_pct = high_low_range / price * 100
        
        # Classify regime
        if adx > 30:  # Strong trend
            return "trending"
        elif relative_volatility > 2.5:  # High volatility
            return "volatile"
        else:  # Range-bound
            return "ranging"
    
    def _get_regime_specific_weights(self, regime: str) -> Dict[str, float]:
        """
        Get regime-specific weights for different signal components
        
        Args:
            regime: Market regime classification
            
        Returns:
            Dictionary of weights for different components
        """
        if regime == "trending":
            return {
                'rsi': 0.15,
                'macd': 0.35,
                'adx': 0.25,
                'volume': 0.15,
                'price_action': 0.1,
                'pattern': 0.05,
                'divergence': 0.15
            }
        elif regime == "ranging":
            return {
                'rsi': 0.3,
                'macd': 0.15,
                'adx': 0.05,
                'volume': 0.2,
                'price_action': 0.15,
                'pattern': 0.1,
                'divergence': 0.05
            }
        elif regime == "volatile":
            return {
                'rsi': 0.1,
                'macd': 0.2,
                'adx': 0.1,
                'volume': 0.3,
                'price_action': 0.2,
                'pattern': 0.15,
                'divergence': 0.05
            }
        else:
            # Default weights
            return {
                'rsi': self.config.rsi_weight,
                'macd': self.config.macd_weight,
                'adx': self.config.adx_weight,
                'volume': self.config.volume_weight,
                'price_action': self.config.price_action_weight,
                'pattern': 0.1,
                'divergence': 0.1
            }
    
    def _calculate_rsi_score(self, rsi: float) -> float:
        """
        Calculate momentum score component from RSI
        
        Args:
            rsi: Current RSI value
            
        Returns:
            Momentum score component from -1.0 to 1.0
        """
        if rsi > self.config.rsi_overbought:
            # Extremely overbought
            if rsi > 85:
                return -0.8  # Strong sell signal (potential reversal)
            else:
                return 0.8  # Strong buy signal (momentum)
        elif rsi < self.config.rsi_oversold:
            # Extremely oversold
            if rsi < 15:
                return 0.8  # Strong buy signal (potential reversal)
            else:
                return -0.8  # Strong sell signal (momentum)
        else:
            # Normalize RSI from 0-100 to -1.0 to 1.0
            normalized = (rsi - 50) / 50
            return normalized
    
    def _calculate_macd_score(self, macd: float, signal: float, hist: pd.Series) -> float:
        """
        Calculate momentum score component from MACD
        
        Args:
            macd: Current MACD value
            signal: Current MACD signal line value
            hist: Last 3 MACD histogram values
            
        Returns:
            Momentum score component from -1.0 to 1.0
        """
        # MACD line crossing above signal line (recent crossover)
        if macd > signal and macd - signal < 0.05 * abs(signal):
            return 0.8
        # MACD line crossing below signal line (recent crossover)
        elif macd < signal and signal - macd < 0.05 * abs(signal):
            return -0.8
        # MACD above signal line (strong momentum)
        elif macd > signal:
            return 0.5
        # MACD below signal line (weak momentum)
        elif macd < signal:
            return -0.5
        
        # Check for increasing/decreasing histogram (acceleration)
        if len(hist) >= 3:
            if hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3]:
                return 0.3  # Increasing histogram (accelerating)
            elif hist.iloc[-1] < hist.iloc[-2] < hist.iloc[-3]:
                return -0.3  # Decreasing histogram (decelerating)
        
        return 0.0
    
    def _calculate_adx_score(self, adx: float) -> float:
        """
        Calculate momentum score component from ADX
        
        Args:
            adx: Current ADX value
            
        Returns:
            Momentum score component from 0.0 to 1.0 (ADX is direction-agnostic)
        """
        if adx >= 50:
            return 1.0  # Very strong trend
        elif adx >= 35:
            return 0.8  # Strong trend
        elif adx >= self.config.adx_threshold:
            return 0.5  # Moderate trend
        else:
            return 0.0  # Weak or no trend
    
    def _calculate_volume_score(self, df: pd.DataFrame, timeframe: int) -> float:
        """
        Calculate momentum score component from volume analysis
        
        Args:
            df: DataFrame with market data
            timeframe: Current timeframe
            
        Returns:
            Momentum score component from -1.0 to 1.0
        """
        # Get recent volume
        recent_volume = df['volume'].iloc[-self.config.volume_lookback:]
        avg_volume = recent_volume.mean()
        latest_volume = recent_volume.iloc[-1]
        latest_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # Calculate normalized volume (compared to average)
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume spike detection
        volume_spike = volume_ratio > self.config.volume_threshold
        
        # Volume confirmation (high volume with price movement)
        price_change = (latest_close - prev_close) / prev_close
        
        if volume_spike:
            if price_change > 0:
                return 0.8  # High volume with price increase
            elif price_change < 0:
                return -0.8  # High volume with price decrease
            return 0.0  # High volume but no price movement
        else:
            # Check for volume trend
            volume_trend = np.polyfit(range(len(recent_volume)), recent_volume, 1)[0]
            normalized_trend = max(min(volume_trend / avg_volume, 1.0), -1.0)
            
            # Combine with price direction
            if price_change > 0:
                return 0.2 + (normalized_trend * 0.3)  # Moderate buy signal
            elif price_change < 0:
                return -0.2 + (normalized_trend * -0.3)  # Moderate sell signal
            return 0.0
    
    def _calculate_price_action_score(self, df: pd.DataFrame) -> float:
        """
        Calculate momentum score component from price action
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Momentum score component from -1.0 to 1.0
        """
        # Calculate short-term, medium-term, and long-term trends
        trends = []
        
        for period in self.config.price_action_periods:
            if len(df) < period + 1:
                continue
                
            # Linear regression slope for price trend
            y = df['close'].iloc[-period:].values
            x = np.array(range(period))
            slope = np.polyfit(x, y, 1)[0]
            
            # Normalize slope by average price
            avg_price = df['close'].iloc[-period:].mean()
            normalized_slope = slope / avg_price if avg_price > 0 else 0
            
            # Add trend to list
            trends.append(normalized_slope)
        
        if not trends:
            return 0.0
        
        # Weight recent trends more heavily
        weights = np.array([1.5, 1.0, 0.5][:len(trends)])
        weights = weights / weights.sum()  # Normalize weights
        
        # Calculate weighted trend score
        weighted_trend = np.sum(np.array(trends) * weights)
        
        # Scale to -1.0 to 1.0
        return max(min(weighted_trend * 100, 1.0), -1.0)
    
    def _calculate_pattern_score(self, patterns: List[Dict[str, Any]]) -> float:
        """
        Calculate momentum score component from detected patterns
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Momentum score component from -1.0 to 1.0
        """
        if not patterns:
            return 0.0
        
        pattern_scores = []
        
        for pattern in patterns:
            pattern_type = pattern.get('type', '')
            pattern_strength = pattern.get('strength', 0.5)
            
            # Bullish patterns
            if any(p in pattern_type.lower() for p in ['bullish', 'bottom', 'support', 'hammer']):
                pattern_scores.append(pattern_strength)
            # Bearish patterns
            elif any(p in pattern_type.lower() for p in ['bearish', 'top', 'resistance', 'shooting star']):
                pattern_scores.append(-pattern_strength)
        
        if not pattern_scores:
            return 0.0
        
        # Return average of pattern scores
        return sum(pattern_scores) / len(pattern_scores)
    
    def _calculate_divergence_score(self, divergences: Dict[str, Any]) -> float:
        """
        Calculate momentum score component from detected divergences
        
        Args:
            divergences: Dictionary of detected divergences
            
        Returns:
            Momentum score component from -1.0 to 1.0
        """
        if not divergences:
            return 0.0
        
        divergence_scores = []
        
        for indicator, div_data in divergences.items():
            div_type = div_data.get('type', '')
            div_strength = div_data.get('strength', 0.5)
            
            if div_type == 'bullish':
                divergence_scores.append(div_strength)
            elif div_type == 'bearish':
                divergence_scores.append(-div_strength)
        
        if not divergence_scores:
            return 0.0
        
        # Return average of divergence scores
        return sum(divergence_scores) / len(divergence_scores)
    
    def _extract_ml_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features for ML prediction
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary of features for ML model
        """
        # Calculate additional features for ML model
        features = {}
        
        # Basic OHLCV features
        features['close'] = df['close'].iloc[-1]
        features['volume'] = df['volume'].iloc[-1]
        
        # Technical indicators
        features['rsi'] = df['rsi'].iloc[-1] if 'rsi' in df else 50.0
        features['macd'] = df['macd'].iloc[-1] if 'macd' in df else 0.0
        features['macd_signal'] = df['macd_signal'].iloc[-1] if 'macd_signal' in df else 0.0
        features['adx'] = df['adx'].iloc[-1] if 'adx' in df else 20.0
        
        # Price action features
        if len(df) >= 20:
            features['returns_1'] = df['close'].pct_change(1).iloc[-1]
            features['returns_5'] = df['close'].pct_change(5).iloc[-1]
            features['returns_10'] = df['close'].pct_change(10).iloc[-1]
            features['returns_20'] = df['close'].pct_change(20).iloc[-1]
            
            # Volatility features
            features['volatility_5'] = df['close'].pct_change().rolling(5).std().iloc[-1]
            features['volatility_20'] = df['close'].pct_change().rolling(20).std().iloc[-1]
            
            # Range features
            features['high_low_ratio_5'] = (df['high'].rolling(5).max() / df['low'].rolling(5).min()).iloc[-1]
            features['high_low_ratio_20'] = (df['high'].rolling(20).max() / df['low'].rolling(20).min()).iloc[-1]
        
        return features
    
    def _determine_signal(self, df: pd.DataFrame, momentum_score: float, 
                          divergences: Dict[str, Any], patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine trading signal based on momentum score and other factors
        
        Args:
            df: DataFrame with market data
            momentum_score: Calculated momentum score
            divergences: Detected divergences
            patterns: Detected patterns
            
        Returns:
            Signal dictionary with direction and strength
        """
        # Determine direction based on momentum score
        if momentum_score > 0.7:
            direction = TradeDirection.STRONG_BUY
            strength = SignalStrength.VERY_HIGH
        elif momentum_score > 0.4:
            direction = TradeDirection.BUY
            strength = SignalStrength.HIGH
        elif momentum_score > 0.2:
            direction = TradeDirection.WEAK_BUY
            strength = SignalStrength.MEDIUM
        elif momentum_score < -0.7:
            direction = TradeDirection.STRONG_SELL
            strength = SignalStrength.VERY_HIGH
        elif momentum_score < -0.4:
            direction = TradeDirection.SELL
            strength = SignalStrength.HIGH
        elif momentum_score < -0.2:
            direction = TradeDirection.WEAK_SELL
            strength = SignalStrength.MEDIUM
        else:
            direction = TradeDirection.NEUTRAL
            strength = SignalStrength.LOW
        
        # Adjust signal based on patterns and divergences
        pattern_adjustment = self._adjust_signal_for_patterns(patterns)
        divergence_adjustment = self._adjust_signal_for_divergences(divergences)
        
        # Apply adjustments
        adjusted_strength = min(
            SignalStrength.VERY_HIGH,
            max(SignalStrength.VERY_LOW, strength + pattern_adjustment + divergence_adjustment)
        )
        
        # Calculate success probability based on historical performance
        success_prob = self._calculate_signal_success_probability(direction, adjusted_strength)
        
        return {
            "direction": direction,
            "strength": adjusted_strength,
            "original_score": momentum_score,
            "adjusted_score": momentum_score + (pattern_adjustment + divergence_adjustment) * 0.1,
            "explanation": self._generate_signal_explanation(direction, momentum_score, patterns, divergences),
            "success_probability": success_prob
        }
    
    def _adjust_signal_for_patterns(self, patterns: List[Dict[str, Any]]) -> int:
        """
        Adjust signal strength based on detected patterns
        
        Args:
            patterns: Detected patterns
            
        Returns:
            Adjustment value for signal strength (-2 to +2)
        """
        if not patterns:
            return 0
        
        # Count bullish and bearish patterns by strength
        strong_bullish = 0
        weak_bullish = 0
        strong_bearish = 0
        weak_bearish = 0
        
        for pattern in patterns:
            pattern_type = pattern.get('type', '').lower()
            pattern_strength = pattern.get('strength', 0.5)
            
            if 'bullish' in pattern_type or 'bottom' in pattern_type:
                if pattern_strength >= 0.7:
                    strong_bullish += 1
                else:
                    weak_bullish += 1
            elif 'bearish' in pattern_type or 'top' in pattern_type:
                if pattern_strength >= 0.7:
                    strong_bearish += 1
                else:
                    weak_bearish += 1
        
        # Calculate net adjustment
        adjustment = (strong_bullish * 2 + weak_bullish) - (strong_bearish * 2 + weak_bearish)
        
        # Limit adjustment range
        return max(-2, min(2, adjustment))
    
    def _adjust_signal_for_divergences(self, divergences: Dict[str, Any]) -> int:
        """
        Adjust signal strength based on detected divergences
        
        Args:
            divergences: Detected divergences
            
        Returns:
            Adjustment value for signal strength (-2 to +2)
        """
        if not divergences:
            return 0
        
        # Count bullish and bearish divergences by strength
        strong_bullish = 0
        weak_bullish = 0
        strong_bearish = 0
        weak_bearish = 0
        
        for indicator, div_data in divergences.items():
            div_type = div_data.get('type', '')
            div_strength = div_data.get('strength', 0.5)
            
            if div_type == 'bullish':
                if div_strength >= 0.7:
                    strong_bullish += 1
                else:
                    weak_bullish += 1
            elif div_type == 'bearish':
                if div_strength >= 0.7:
                    strong_bearish += 1
                else:
                    weak_bearish += 1
        
        # Calculate net adjustment
        adjustment = (strong_bullish * 2 + weak_bullish) - (strong_bearish * 2 + weak_bearish)
        
        # Limit adjustment range
        return max(-2, min(2, adjustment))
    
    def _calculate_signal_success_probability(self, direction: TradeDirection, 
                                            strength: SignalStrength) -> float:
        """
        Calculate probability of success for a given signal based on historical performance
        
        Args:
            direction: Signal direction
            strength: Signal strength
            
        Returns:
            Probability of success (0.0 to 1.0)
        """
        # Base success rates by signal strength
        base_rates = {
            SignalStrength.VERY_HIGH: 0.85,
            SignalStrength.HIGH: 0.75,
            SignalStrength.MEDIUM: 0.65,
            SignalStrength.LOW: 0.55,
            SignalStrength.VERY_LOW: 0.45
        }
        
        # Adjust based on recent performance
        if self.recent_trades and len(self.recent_trades) >= 10:
            # Separate by direction
            direction_trades = [trade for trade in self.recent_trades 
                              if trade.get('direction') == direction]
            
            if direction_trades and len(direction_trades) >= 5:
                # Calculate success rate for this direction
                successful = sum(1 for trade in direction_trades if trade.get('result') == 'win')
                direction_success_rate = successful / len(direction_trades)
                
                # Weight recent performance
                performance_weight = min(0.5, len(direction_trades) / 20)  # Max 0.5 weight for recent performance
                
                # Blend base rate with recent performance
                return base_rates[strength] * (1 - performance_weight) + direction_success_rate * performance_weight
        
        # Return base rate if no significant historical data
        return base_rates[strength]
    
    def _generate_signal_explanation(self, direction: TradeDirection, score: float,
                                    patterns: List[Dict[str, Any]], divergences: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation for the signal
        
        Args:
            direction: Signal direction
            score: Momentum score
            patterns: Detected patterns
            divergences: Detected divergences
            
        Returns:
            Explanation string
        """
        explanation = []
        
        # Direction explanation
        if direction in [TradeDirection.STRONG_BUY, TradeDirection.BUY, TradeDirection.WEAK_BUY]:
            explanation.append(f"Bullish momentum detected (score: {score:.2f})")
        elif direction in [TradeDirection.STRONG_SELL, TradeDirection.SELL, TradeDirection.WEAK_SELL]:
            explanation.append(f"Bearish momentum detected (score: {score:.2f})")
        else:
            explanation.append(f"Neutral momentum detected (score: {score:.2f})")
        
        # Add pattern information
        if patterns:
            pattern_strs = []
            for pattern in patterns[:3]:  # Limit to 3 patterns
                pattern_strs.append(f"{pattern.get('type', 'Unknown')} (strength: {pattern.get('strength', 0):.2f})")
            explanation.append(f"Patterns: {', '.join(pattern_strs)}")
        
        # Add divergence information
        if divergences:
            divergence_strs = []
            for indicator, div_data in list(divergences.items())[:3]:  # Limit to 3 divergences
                divergence_strs.append(
                    f"{indicator} {div_data.get('type', 'unknown')} (strength: {div_data.get('strength', 0):.2f})"
                )
            explanation.append(f"Divergences: {', '.join(divergence_strs)}")
        
        return ". ".join(explanation)
    
    def _identify_key_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Identify key price levels (support, resistance, targets)
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary of key price levels
        """
        # Calculate key levels
        latest_close = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'] - df['low']).mean()
        
        # Identify recent swing points
        recent_df = df.iloc[-100:] if len(df) >= 100 else df
        
        # Find swing highs and lows (simple algorithm)
        highs = []
        lows = []
        
        for i in range(2, len(recent_df) - 2):
            # Swing high
            if (recent_df['high'].iloc[i] > recent_df['high'].iloc[i-1] and 
                recent_df['high'].iloc[i] > recent_df['high'].iloc[i-2] and
                recent_df['high'].iloc[i] > recent_df['high'].iloc[i+1] and
                recent_df['high'].iloc[i] > recent_df['high'].iloc[i+2]):
                highs.append(recent_df['high'].iloc[i])
            
            # Swing low
            if (recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and 
                recent_df['low'].iloc[i] < recent_df['low'].iloc[i-2] and
                recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1] and
                recent_df['low'].iloc[i] < recent_df['low'].iloc[i+2]):
                lows.append(recent_df['low'].iloc[i])
        
        # Identify closest support and resistance
        support_levels = [low for low in lows if low < latest_close]
        resistance_levels = [high for high in highs if high > latest_close]
        
        support = max(support_levels) if support_levels else (latest_close - 2 * atr)
        resistance = min(resistance_levels) if resistance_levels else (latest_close + 2 * atr)
        
        # Include Fibonacci levels if applicable
        fib_levels = {}
        if len(df) >= 20:
            # Use recent swing high and low for Fibonacci levels
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            range_size = recent_high - recent_low
            
            # Add Fibonacci retracement levels if we have a clear trend
            if range_size > 5 * atr:  # Significant range
                if latest_close < (recent_high + recent_low) / 2:  # In downtrend
                    fib_levels = {
                        "fib_0.236": recent_low + 0.236 * range_size,
                        "fib_0.382": recent_low + 0.382 * range_size,
                        "fib_0.5": recent_low + 0.5 * range_size,
                        "fib_0.618": recent_low + 0.618 * range_size,
                        "fib_0.786": recent_low + 0.786 * range_size
                    }
                else:  # In uptrend
                    fib_levels = {
                        "fib_0.236": recent_high - 0.236 * range_size,
                        "fib_0.382": recent_high - 0.382 * range_size,
                        "fib_0.5": recent_high - 0.5 * range_size,
                        "fib_0.618": recent_high - 0.618 * range_size,
                        "fib_0.786": recent_high - 0.786 * range_size
                    }
        
        return {
            "current_price": latest_close,
            "support": support,
            "resistance": resistance,
            "daily_high": df['high'].iloc[-1],
            "daily_low": df['low'].iloc[-1],
            "atr": atr,
            **fib_levels
        }
    
    def _combine_timeframe_signals(self, signals: Dict[int, Dict[str, Any]], 
                                 timeframes: List[int]) -> Dict[str, Any]:
        """
        Combine signals from multiple timeframes with weighted consensus
        
        Args:
            signals: Dictionary of signals by timeframe
            timeframes: List of available timeframes
            
        Returns:
            Combined signal with direction and strength
        """
        if not signals or not timeframes:
            return {
                "direction": TradeDirection.NEUTRAL,
                "strength": SignalStrength.VERY_LOW,
                "score": 0,
                "explanation": "Insufficient data for signal generation"
            }
        
        # Convert directions to numeric scores
        direction_scores = {
            TradeDirection.STRONG_BUY: 2.0,
            TradeDirection.BUY: 1.0,
            TradeDirection.WEAK_BUY: 0.5,
            TradeDirection.NEUTRAL: 0.0,
            TradeDirection.WEAK_SELL: -0.5,
            TradeDirection.SELL: -1.0,
            TradeDirection.STRONG_SELL: -2.0
        }
        
        # Calculate weighted score
        total_weight = 0
        weighted_score = 0
        
        for tf in timeframes:
            if tf not in signals:
                continue
                
            signal = signals[tf]
            weight = self.config.timeframe_weights.get(tf, 0.25)  # Default to 0.25 if not specified
            
            # Get numeric score for direction
            score = direction_scores.get(signal['direction'], 0.0)
            
            # Weight by signal strength
            strength_multiplier = {
                SignalStrength.VERY_HIGH: 1.0,
                SignalStrength.HIGH: 0.8,
                SignalStrength.MEDIUM: 0.6,
                SignalStrength.LOW: 0.4,
                SignalStrength.VERY_LOW: 0.2
            }.get(signal['strength'], 0.5)
            
            weighted_score += score * weight * strength_multiplier
            total_weight += weight
        
        # Normalize score
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine direction and strength from final score
        if final_score > 1.5:
            direction = TradeDirection.STRONG_BUY
            strength = SignalStrength.VERY_HIGH
        elif final_score > 0.8:
            direction = TradeDirection.BUY
            strength = SignalStrength.HIGH
        elif final_score > 0.3:
            direction = TradeDirection.WEAK_BUY
            strength = SignalStrength.MEDIUM
        elif final_score < -1.5:
            direction = TradeDirection.STRONG_SELL
            strength = SignalStrength.VERY_HIGH
        elif final_score < -0.8:
            direction = TradeDirection.SELL
            strength = SignalStrength.HIGH
        elif final_score < -0.3:
            direction = TradeDirection.WEAK_SELL
            strength = SignalStrength.MEDIUM
        else:
            direction = TradeDirection.NEUTRAL
            strength = SignalStrength.LOW
        
        # Generate explanation
        explanation = f"Combined signal from {len(timeframes)} timeframes: {final_score:.2f}"
        
        # Calculate success probability
        success_prob = self._calculate_signal_success_probability(direction, strength)
        
        return {
            "direction": direction,
            "strength": strength,
            "score": final_score,
            "explanation": explanation,
            "timeframes_analyzed": list(timeframes),
            "success_probability": success_prob
        }
    
    def _calculate_optimal_entry(self, data: Dict[int, pd.DataFrame], direction: TradeDirection) -> float:
        """
        Calculate optimal entry price based on current market conditions
        
        Args:
            data: Market data for different timeframes
            direction: Trade direction
            
        Returns:
            Optimal entry price
        """
        # Use the smallest timeframe for precise entry
        tf = min(data.keys())
        df = data[tf].copy()
        
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'] - df['low']).mean()
        
        # Adjust entry based on direction
        if direction in [TradeDirection.STRONG_BUY, TradeDirection.BUY]:
            # For strong signals, enter at market
            return current_price
        elif direction == TradeDirection.WEAK_BUY:
            # For weak buy signals, try to get a slightly better entry
            return max(current_price - 0.3 * atr, df['low'].iloc[-1])
        elif direction in [TradeDirection.STRONG_SELL, TradeDirection.SELL]:
            # For strong sell signals, enter at market
            return current_price
        elif direction == TradeDirection.WEAK_SELL:
            # For weak sell signals, try to get a slightly better entry
            return min(current_price + 0.3 * atr, df['high'].iloc[-1])
        else:
            # Neutral - no trade
            return current_price
    
    def _calculate_stop_loss(self, data: Dict[int, pd.DataFrame], direction: TradeDirection) -> float:
        """
        Calculate appropriate stop loss level based on market conditions
        
        Args:
            data: Market data for different timeframes
            direction: Trade direction
            
        Returns:
            Stop loss price level
        """
        # Use a larger timeframe for stop loss to avoid noise
        tf = max(min(data.keys(), key=lambda x: abs(x - 60)), data.keys())  # Prefer 1-hour if available
        df = data[tf].copy()
        
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'] - df['low']).mean()
        
        # Identify recent swing points
        if len(df) >= 20:
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
        else:
            recent_high = df['high'].max()
            recent_low = df['low'].min()
        
        # Set stop loss based on direction and volatility
        if direction in [TradeDirection.STRONG_BUY, TradeDirection.BUY, TradeDirection.WEAK_BUY]:
            # For buy trades, place stop below recent low or ATR-based level
            technical_stop = current_price - self.config.stop_loss_atr_multiplier * atr
            swing_stop = recent_low - 0.1 * atr  # Slightly below recent low
            
            # Use the more conservative stop (higher of the two)
            return max(technical_stop, swing_stop)
        elif direction in [TradeDirection.STRONG_SELL, TradeDirection.SELL, TradeDirection.WEAK_SELL]:
            # For sell trades, place stop above recent high or ATR-based level
            technical_stop = current_price + self.config.stop_loss_atr_multiplier * atr
            swing_stop = recent_high + 0.1 * atr  # Slightly above recent high
            
            # Use the more conservative stop (lower of the two)
            return min(technical_stop, swing_stop)
        else:
            # Neutral - no trade
            return current_price
    
    def _calculate_take_profit(self, data: Dict[int, pd.DataFrame], direction: TradeDirection) -> float:
        """
        Calculate appropriate take profit level based on market conditions
        
        Args:
            data: Market data for different timeframes
            direction: Trade direction
            
        Returns:
            Take profit price level
        """
        # Use a larger timeframe for take profit to capture larger moves
        tf = max(data.keys())
        df = data[tf].copy()
        
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'] - df['low']).mean()
        
        # Calculate risk (distance to stop)
        stop_loss = self._calculate_stop_loss(data, direction)
        risk = abs(current_price - stop_loss)
        
        # Set reward:risk ratio based on signal strength
        # Use a larger timeframe for signal evaluation
        tf_signal = max(min(data.keys(), key=lambda x: abs(x - 60)), data.keys())  # Prefer 1-hour if available
        signal = self._determine_signal(
            data[tf_signal],
            self.asset_momentum_cache.get(self.current_asset, {}).get(tf_signal, {}).get("score", 0),
            {},  # Divergences not needed here
            []   # Patterns not needed here
        )
        
        # Set reward:risk ratio based on signal strength
        reward_risk_ratio = {
            SignalStrength.VERY_HIGH: 3.0,
            SignalStrength.HIGH: 2.5,
            SignalStrength.MEDIUM: 2.0,
            SignalStrength.LOW: 1.5,
            SignalStrength.VERY_LOW: 1.0
        }.get(signal['strength'], 2.0)
        
        # Calculate take profit based on reward:risk ratio
        if direction in [TradeDirection.STRONG_BUY, TradeDirection.BUY, TradeDirection.WEAK_BUY]:
            return current_price + (risk * reward_risk_ratio)
        elif direction in [TradeDirection.STRONG_SELL, TradeDirection.SELL, TradeDirection.WEAK_SELL]:
            return current_price - (risk * reward_risk_ratio)
        else:
            # Neutral - no trade
            return current_price
    
    def _calculate_position_size(self, asset: str, signal_strength: SignalStrength) -> float:
        """
        Calculate appropriate position size based on signal confidence and capital
        
        Args:
            asset: Asset being traded
            signal_strength: Strength of the trading signal
            
        Returns:
            Position size as fraction of available capital (0.0 to 1.0)
        """
        # Base position size on signal strength
        base_size = {
            SignalStrength.VERY_HIGH: 1.0,
            SignalStrength.HIGH: 0.75,
            SignalStrength.MEDIUM: 0.5,
            SignalStrength.LOW: 0.25,
            SignalStrength.VERY_LOW: 0.1
        }.get(signal_strength, 0.5)
        
        # Scale by maximum position size
        position_size = base_size * self.config.max_position_size
        
        # Adjust based on recent performance for this asset
        asset_performance = self._get_asset_performance(asset)
        
        if asset_performance > 0.7:  # >70% win rate
            position_size *= 1.2  # Increase by 20%
        elif asset_performance < 0.3:  # <30% win rate
            position_size *= 0.8  # Decrease by 20%
        
        # Ensure within limits
        return min(self.config.max_position_size, max(0.01, position_size))
    
    def _get_asset_performance(self, asset: str) -> float:
        """
        Get recent performance metrics for a specific asset
        
        Args:
            asset: Asset symbol
            
        Returns:
            Win rate for recent trades (0.0 to 1.0)
        """
        # Filter recent trades for this asset
        asset_trades = [trade for trade in self.recent_trades if trade.get('asset') == asset]
        
        if not asset_trades or len(asset_trades) < 5:
            return 0.5  # Default to 50% if insufficient data
        
        # Calculate win rate
        wins = sum(1 for trade in asset_trades if trade.get('result') == 'win')
        return wins / len(asset_trades)
    
    def _update_divergence_alerts(self, asset: str, divergences: Dict[str, Any]) -> None:
        """
        Update divergence alert tracking for an asset
        
        Args:
            asset: Asset symbol
            divergences: Detected divergences
        """
        if asset not in self.divergence_alerts:
            self.divergence_alerts[asset] = {
                "rsi": {"active": False, "timestamp": None},
                "macd": {"active": False, "timestamp": None},
                "obv": {"active": False, "timestamp": None}
            }
        
        # Update alerts for each indicator
        for indicator, div_data in divergences.items():
            if indicator in self.divergence_alerts[asset]:
                self.divergence_alerts[asset][indicator] = {
                    "active": True,
                    "timestamp": datetime.now(),
                    "type": div_data.get('type', 'unknown'),
                    "strength": div_data.get('strength', 0.5)
                }
    
    def generate_signal(self, asset: str, data: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate trading signal for a specific asset
        
        Args:
            asset: Asset symbol
            data: Market data for different timeframes
            
        Returns:
            Trading signal with entry/exit points and position sizing
        """
        # Set current asset (for internal methods)
        self.current_asset = asset
        
        # Update parameters if needed
        if self.config.adaptive_parameters:
            self._update_strategy_parameters()
        
        # Perform market analysis
        analysis = self.analyze(asset, data)
        
        # Extract signal information
        direction = analysis['combined_signal']['direction']
        strength = analysis['combined_signal']['strength']
        entry_price = analysis['entry_price']
        stop_loss = analysis['stop_loss']
        take_profit = analysis['take_profit']
        confidence = analysis['confidence']
        
        # Calculate position size
        position_size = self._calculate_position_size(asset, strength)
        
        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Only generate trade if direction is not neutral and risk-reward ratio is favorable
        if direction != TradeDirection.NEUTRAL and risk_reward_ratio >= 1.5:
            signal = {
                "asset": asset,
                "platform": self.platform,
                "timestamp": datetime.now(),
                "direction": direction,
                "strength": strength,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": risk_reward_ratio,
                "position_size": position_size,
                "confidence": confidence,
                "success_probability": analysis['combined_signal'].get('success_probability', 0.5),
                "explanation": analysis['combined_signal'].get('explanation', ''),
                "key_levels": analysis['key_levels'],
                "trade_id": f"momentum_{asset}_{int(datetime.now().timestamp())}",
                "brain_type": "momentum",
                "market_regime": analysis['market_regime']
            }
            
            # Log signal generation
            logger.info(f"Generated {direction} signal for {asset} with {strength} strength")
            return signal
        else:
            # No trade signal
            logger.debug(f"No trade signal for {asset} (direction: {direction}, R:R: {risk_reward_ratio:.2f})")
            return {
                "asset": asset,
                "platform": self.platform,
                "timestamp": datetime.now(),
                "direction": TradeDirection.NEUTRAL,
                "strength": SignalStrength.VERY_LOW,
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "risk_reward_ratio": 0,
                "position_size": 0,
                "confidence": SignalStrength.VERY_LOW,
                "explanation": "No tradeable setup detected",
                "brain_type": "momentum",
                "market_regime": analysis['market_regime']
            }
    
    def _update_strategy_parameters(self) -> None:
        """Update strategy parameters based on recent performance and market conditions"""
        # Check if it's time to update parameters
        time_since_update = (datetime.now() - self.last_parameter_update).total_seconds() / 3600
        if time_since_update < self.config.parameter_update_frequency:
            return
        
        # Update based on recent performance
        if self.recent_trades and len(self.recent_trades) >= 20:
            self.success_rate = calculate_success_rate(self.recent_trades)
            self.expected_value = calculate_expected_value(self.recent_trades)
            
            # Adjust parameters based on performance
            if self.success_rate < 0.4:  # Poor performance
                # More conservative settings
                self.config.rsi_overbought -= 5
                self.config.rsi_oversold += 5
                self.config.adx_threshold += 5
                self.config.stop_loss_atr_multiplier += 0.5
                self.config.take_profit_atr_multiplier -= 0.5
                logger.info("Adjusting to more conservative parameters due to low success rate")
            elif self.success_rate > 0.7:  # Very good performance
                # More aggressive settings
                self.config.rsi_overbought += 5
                self.config.rsi_oversold -= 5
                self.config.adx_threshold -= 5
                self.config.stop_loss_atr_multiplier -= 0.2
                self.config.take_profit_atr_multiplier += 0.2
                logger.info("Adjusting to more aggressive parameters due to high success rate")
            
            # Adjust market regime weights
            for regime, perf in self.regime_performance.items():
                if perf['trades'] >= 10:
                    regime_success = perf['wins'] / perf['trades']
                    if regime_success > 0.7:
                        # Increase weights for this regime
                        regime_weights = self._get_regime_specific_weights(regime)
                        logger.info(f"Optimizing weights for {regime} regime (success rate: {regime_success:.2f})")
            
            # Reset performance tracking for next period
            self.last_parameter_update = datetime.now()
    
    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """
        Update strategy performance with a completed trade result
        
        Args:
            trade_result: Dictionary with trade result information
        """
        # Add to recent trades (maintain last 100)
        self.recent_trades.append(trade_result)
        if len(self.recent_trades) > 100:
            self.recent_trades.pop(0)
        
        # Update regime-specific performance
        regime = trade_result.get('market_regime', 'trending')
        if regime in self.regime_performance:
            self.regime_performance[regime]['trades'] += 1
            if trade_result.get('result') == 'win':
                self.regime_performance[regime]['wins'] += 1
        
        # Log performance update
        logger.info(f"Updated momentum brain performance with {trade_result.get('result')} trade")
        
        # Update asset-specific metrics
        asset = trade_result.get('asset')
        if asset:
            self._update_asset_specific_metrics(asset, trade_result)
    
    def _update_asset_specific_metrics(self, asset: str, trade_result: Dict[str, Any]) -> None:
        """
        Update asset-specific performance metrics
        
        Args:
            asset: Asset symbol
            trade_result: Trade result information
        """
        # This would typically update a database with asset-specific performance metrics
        # For now, we'll just log the result
        result = trade_result.get('result', 'unknown')
        direction = trade_result.get('direction', TradeDirection.NEUTRAL)
        
        logger.info(f"Updated metrics for {asset}: {direction} trade {result}")
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the brain's state for persistence
        
        Returns:
            Dictionary containing state information
        """
        return {
            "config": self.config.__dict__,
            "recent_trades": self.recent_trades,
            "success_rate": self.success_rate,
            "expected_value": self.expected_value,
            "last_parameter_update": self.last_parameter_update.isoformat(),
            "asset_momentum_cache": {
                asset: {
                    tf: {
                        "score": data["score"],
                        "timestamp": data["timestamp"].isoformat()
                    } for tf, data in timeframes.items()
                } for asset, timeframes in self.asset_momentum_cache.items()
            },
            "divergence_alerts": self.divergence_alerts,
            "regime_performance": self.regime_performance
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load the brain's state from persistence
        
        Args:
            state: Dictionary containing state information
        """
        if state:
            # Load configuration
            if "config" in state:
                self.config = MomentumBrainConfig(**state["config"])
            
            # Load metrics
            self.recent_trades = state.get("recent_trades", [])
            self.success_rate = state.get("success_rate", 0.0)
            self.expected_value = state.get("expected_value", 0.0)
            
            # Load last parameter update timestamp
            if "last_parameter_update" in state:
                self.last_parameter_update = datetime.fromisoformat(state["last_parameter_update"])
            
            # Load momentum cache
            if "asset_momentum_cache" in state:
                self.asset_momentum_cache = {}
                for asset, timeframes in state["asset_momentum_cache"].items():
                    self.asset_momentum_cache[asset] = {}
                    for tf, data in timeframes.items():
                        self.asset_momentum_cache[asset][int(tf)] = {
                            "score": data["score"],
                            "timestamp": datetime.fromisoformat(data["timestamp"])
                        }
            
            # Load divergence alerts
            self.divergence_alerts = state.get("divergence_alerts", {})
            
            # Load regime performance
            self.regime_performance = state.get("regime_performance", {
                "trending": {"trades": 0, "wins": 0},
                "ranging": {"trades": 0, "wins": 0},
                "volatile": {"trades": 0, "wins": 0}
            })
            
            logger.info(f"Loaded momentum brain state with {len(self.recent_trades)} historical trades")
