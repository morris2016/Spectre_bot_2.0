"""
QuantumSpectre Elite Trading System
Mean Reversion Strategy Brain Implementation

This module implements a sophisticated mean-reversion trading strategy that
identifies overbought/oversold conditions and mean-reverting setups across
multiple timeframes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass, field
import uuid

# Internal imports
from strategy_brains.base_brain import BaseBrain, BrainConfig, SignalStrength, TradeDirection, AssetType
from common.constants import TIMEFRAMES
from common.utils import calculate_success_rate, calculate_expected_value
from feature_service.features.technical import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_stochastic,
    detect_divergence, calculate_atr
)
from feature_service.features.volatility import (
    calculate_historical_volatility, calculate_volatility_ratio
)
from feature_service.features.market_structure import (
    find_support_resistance_levels, classify_market_structure
)
from feature_service.features.pattern import detect_patterns
from ml_models.prediction import predict_mean_reversion_probability

logger = logging.getLogger(__name__)

@dataclass
class MeanReversionBrainConfig(BrainConfig):
    """Configuration for Mean Reversion Strategy Brain"""
    # Timeframes to analyze (in minutes)
    timeframes: List[int] = field(default_factory=lambda: [15, 60, 240, 1440])
    
    # RSI parameters
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_weight: float = 0.25
    
    # Bollinger Bands parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0
    bb_weight: float = 0.25
    
    # Stochastic parameters
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0
    stoch_weight: float = 0.2
    
    # Mean reversion parameters
    mean_lookback: int = 50
    std_dev_threshold: float = 2.0
    mean_weight: float = 0.3
    
    # Volume parameters
    volume_lookback: int = 20
    volume_threshold: float = 1.5
    volume_weight: float = 0.15
    
    # Volatility parameters
    volatility_lookback: int = 20
    volatility_threshold: float = 1.8
    volatility_weight: float = 0.15
    
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
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 2.0
    trailing_stop_activation: float = 0.01  # 1% in profit
    trailing_stop_distance: float = 0.005  # 0.5% trailing
    
    # Multi-timeframe consensus weights
    timeframe_weights: Dict[int, float] = field(default_factory=lambda: {
        15: 0.15,    # 15-minute weight
        60: 0.25,    # 1-hour weight
        240: 0.30,   # 4-hour weight
        1440: 0.30   # Daily weight
    })
    
    # Mean reversion exit parameters
    mean_reversion_exit_threshold: float = 0.5  # Exit when price reverts to 50% of deviation
    
    # Regime-specific adjustments
    ranging_market_boost: float = 1.5  # Boost signal in ranging markets
    trending_market_penalty: float = 0.7  # Reduce signal in trending markets
    
    # Counter-trend protection
    adr_limit: float = 2.0  # Limit trades to 2x average daily range


class MeanReversionBrain(BaseBrain):
    """
    Mean Reversion Strategy Brain Implementation
    
    This class implements a sophisticated mean-reversion trading strategy that:
    - Identifies overbought/oversold conditions
    - Detects price deviations from historical means
    - Analyzes multiple timeframes for consensus
    - Uses Bollinger Bands for volatility-adjusted mean reversion
    - Incorporates volume and volatility analysis for confirmation
    - Uses ML models for reversal probability enhancement
    - Adapts parameters based on market conditions
    - Implements advanced position sizing and risk management
    """
    
    def __init__(self, config: Optional[MeanReversionBrainConfig] = None):
        """
        Initialize the Mean Reversion Brain strategy
        
        Args:
            config: Configuration for the mean reversion strategy (optional)
        """
        self.config = config or MeanReversionBrainConfig()
        super().__init__(self.config)
        
        # Recent performance metrics
        self.recent_trades = []
        self.success_rate = 0.0
        self.expected_value = 0.0
        
        # Strategy state
        self.last_parameter_update = datetime.now()
        self.asset_mean_reversion_cache = {}  # Cache for asset-specific reversion scores
        self.active_signals = {}  # Track active signals by asset
        
        # Performance tracking by market regime
        self.regime_performance = {
            "trending": {"trades": 0, "wins": 0},
            "ranging": {"trades": 0, "wins": 0},
            "volatile": {"trades": 0, "wins": 0}
        }
        
        # Asset-specific metrics
        self.asset_metrics = {}
        
        logger.info(f"Mean Reversion Brain initialized with {len(self.config.timeframes)} timeframes")
    
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
            
            # Initialize mean reversion cache for each asset
            self.asset_mean_reversion_cache[asset] = {
                tf: {"score": 0, "timestamp": datetime.now() - timedelta(days=1)} 
                for tf in self.config.timeframes
            }
            
            # Initialize active signals
            self.active_signals[asset] = {
                "active": False,
                "direction": TradeDirection.NEUTRAL,
                "signal_id": None,
                "timestamp": None
            }
            
            # Initialize asset metrics
            self.asset_metrics[asset] = {
                "average_daily_range": None,
                "historical_volatility": None,
                "mean_reversion_success_rate": 0.5,
                "optimal_holding_period": 12,  # Default 12 hours
                "best_performing_setup": None
            }
        
        logger.info(f"Mean Reversion Brain initialized for {len(assets)} assets on {platform}")
    
    def _load_asset_specific_parameters(self, asset: str, platform: str) -> None:
        """
        Load asset-specific optimized parameters from database
        
        Args:
            asset: Asset symbol
            platform: Trading platform
        """
        try:
            # This would typically load from a database
            # For now, we'll adjust parameters based on asset characteristics
            asset_volatility = self._get_asset_volatility_class(asset, platform)
            asset_type = self._get_asset_type(asset, platform)
            
            if asset_volatility == "high":
                # Adjust parameters for high volatility assets
                self.config.rsi_overbought = 75.0
                self.config.rsi_oversold = 25.0
                self.config.bb_std_dev = 2.5
                self.config.stop_loss_atr_multiplier = 2.0
                self.config.std_dev_threshold = 2.5
            elif asset_volatility == "low":
                # Adjust parameters for low volatility assets
                self.config.rsi_overbought = 65.0
                self.config.rsi_oversold = 35.0
                self.config.bb_std_dev = 1.5
                self.config.stop_loss_atr_multiplier = 1.2
                self.config.std_dev_threshold = 1.5
            
            # Further adjust based on asset type
            if asset_type == AssetType.CRYPTO:
                self.config.mean_reversion_exit_threshold = 0.6  # More aggressive exit
                self.config.ranging_market_boost = 1.7  # Stronger boost
            elif asset_type == AssetType.FOREX:
                self.config.mean_reversion_exit_threshold = 0.4  # More conservative exit
                self.config.ranging_market_boost = 1.3  # Milder boost
            
            logger.debug(f"Loaded parameters for {asset} on {platform} (volatility: {asset_volatility}, type: {asset_type})")
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
    
    def _get_asset_type(self, asset: str, platform: str) -> AssetType:
        """
        Determine the asset type
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            
        Returns:
            Asset type
        """
        if platform == "binance":
            return AssetType.CRYPTO
        elif platform == "deriv":
            if any(forex in asset for forex in ["EUR", "USD", "JPY", "GBP"]):
                return AssetType.FOREX
            elif any(index in asset for forex in ["OTC_", "IDX"]):
                return AssetType.INDEX
            else:
                return AssetType.COMMODITY
        
        return AssetType.OTHER
    
    def analyze(self, asset: str, data: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze market data for mean reversion signals
        
        Args:
            asset: Asset symbol
            data: Dictionary mapping timeframes to dataframes of OHLCV data
            
        Returns:
            Analysis results including mean reversion scores, signals, and key levels
        """
        reversion_scores = {}
        signals = {}
        key_levels = {}
        current_regime = self._detect_market_regime(data)
        
        # Update asset metrics if needed
        self._update_asset_metrics(asset, data)
        
        # Only analyze timeframes for which we have data
        available_timeframes = set(data.keys()).intersection(self.config.timeframes)
        
        # Calculate mean reversion score across timeframes
        for tf in available_timeframes:
            df = data[tf].copy()
            
            # Skip if not enough data
            min_data_points = max(
                self.config.rsi_period, 
                self.config.bb_period, 
                self.config.mean_lookback
            ) + 10
            
            if len(df) < min_data_points:
                logger.warning(f"Not enough data for {asset} on {tf} timeframe")
                continue
            
            # Calculate technical indicators
            df['rsi'] = calculate_rsi(df['close'], self.config.rsi_period)
            df['stoch_k'], df['stoch_d'] = calculate_stochastic(
                df['high'], df['low'], df['close'], 
                self.config.stoch_k_period, 
                self.config.stoch_d_period
            )
            df['upper_band'], df['middle_band'], df['lower_band'] = calculate_bollinger_bands(
                df['close'], 
                self.config.bb_period, 
                self.config.bb_std_dev
            )
            df['atr'] = calculate_atr(df['high'], df['low'], df['close'], 14)
            
            # Calculate price deviation from mean
            df['mean_20'] = df['close'].rolling(20).mean()
            df['mean_50'] = df['close'].rolling(50).mean()
            df['std_20'] = df['close'].rolling(20).std()
            df['distance_from_mean'] = (df['close'] - df['mean_50']) / df['std_20']
            
            # Calculate historical volatility
            df['hist_vol'] = calculate_historical_volatility(df['close'], 20)
            df['vol_ratio'] = calculate_volatility_ratio(df['close'], 10, 30)
            
            # Detect patterns
            patterns = detect_patterns(df)
            
            # Detect divergences
            divergences = detect_divergence(df['close'], df['rsi'], None, None, self.config.divergence_lookback)
            
            # Calculate mean reversion score components
            rsi_score = self._calculate_rsi_score(df['rsi'].iloc[-1])
            stoch_score = self._calculate_stochastic_score(df['stoch_k'].iloc[-1], df['stoch_d'].iloc[-1])
            bb_score = self._calculate_bollinger_score(
                df['close'].iloc[-1],
                df['upper_band'].iloc[-1],
                df['middle_band'].iloc[-1],
                df['lower_band'].iloc[-1]
            )
            mean_score = self._calculate_mean_deviation_score(df['distance_from_mean'].iloc[-1])
            volume_score = self._calculate_volume_score(df, tf)
            volatility_score = self._calculate_volatility_score(df['hist_vol'].iloc[-1], df['vol_ratio'].iloc[-1])
            pattern_score = self._calculate_pattern_score(patterns, "reversal")
            divergence_score = self._calculate_divergence_score(divergences)
            
            # Calculate ML-based prediction if enabled
            ml_score = 0
            if self.config.use_ml_models:
                ml_features = self._extract_ml_features(df)
                ml_score = predict_mean_reversion_probability(asset, tf, ml_features)
            
            # Weight individual scores based on market regime
            regime_weights = self._get_regime_specific_weights(current_regime)
            
            # Calculate combined mean reversion score
            weighted_score = (
                rsi_score * regime_weights.get('rsi', self.config.rsi_weight) +
                stoch_score * regime_weights.get('stoch', self.config.stoch_weight) +
                bb_score * regime_weights.get('bb', self.config.bb_weight) +
                mean_score * regime_weights.get('mean', self.config.mean_weight) +
                volume_score * regime_weights.get('volume', self.config.volume_weight) +
                volatility_score * regime_weights.get('volatility', self.config.volatility_weight) +
                pattern_score * regime_weights.get('pattern', 0.1) +
                divergence_score * regime_weights.get('divergence', 0.1)
            )
            
            # Incorporate ML prediction if enabled
            if self.config.use_ml_models:
                weighted_score = (
                    weighted_score * (1 - self.config.ml_prediction_weight) + 
                    ml_score * self.config.ml_prediction_weight
                )
            
            # Apply regime-specific adjustments
            if current_regime == "ranging":
                weighted_score *= self.config.ranging_market_boost
            elif current_regime == "trending":
                weighted_score *= self.config.trending_market_penalty
            
            # Store mean reversion score
            reversion_scores[tf] = weighted_score
            
            # Determine signal for this timeframe
            signal = self._determine_signal(df, weighted_score, divergences, patterns)
            signals[tf] = signal
            
            # Identify key price levels
            key_levels[tf] = self._identify_key_levels(df)
            
            # Update mean reversion cache
            self.asset_mean_reversion_cache[asset][tf] = {
                "score": weighted_score,
                "timestamp": datetime.now()
            }
        
        # Combine timeframe signals with weighted consensus
        combined_signal = self._combine_timeframe_signals(signals, available_timeframes)
        
        # Update active signals
        if combined_signal['direction'] != TradeDirection.NEUTRAL:
            self._update_active_signals(asset, combined_signal)
        
        return {
            "asset": asset,
            "reversion_scores": reversion_scores,
            "signals": signals,
            "combined_signal": combined_signal,
            "key_levels": key_levels,
            "market_regime": current_regime,
            "entry_price": self._calculate_optimal_entry(data, combined_signal['direction']),
            "stop_loss": self._calculate_stop_loss(data, combined_signal['direction']),
            "take_profit": self._calculate_take_profit(data, combined_signal['direction']),
            "confidence": combined_signal['strength'],
            "position_size": self._calculate_position_size(asset, combined_signal['strength'], data),
            "timestamp": datetime.now(),
            "asset_metrics": self.asset_metrics[asset]
        }
    
    def _update_asset_metrics(self, asset: str, data: Dict[int, pd.DataFrame]) -> None:
        """
        Update asset-specific metrics
        
        Args:
            asset: Asset symbol
            data: Market data for different timeframes
        """
        # Use daily timeframe if available
        if 1440 in data:
            df = data[1440].copy()
            # Calculate average daily range
            df['daily_range'] = df['high'] - df['low']
            self.asset_metrics[asset]['average_daily_range'] = df['daily_range'].mean()
            
            # Calculate historical volatility
            if len(df) >= 20:
                self.asset_metrics[asset]['historical_volatility'] = calculate_historical_volatility(
                    df['close'], 20
                ).iloc[-1]
        
        # If daily not available, use 4h data to approximate
        elif 240 in data:
            df = data[240].copy()
            # Estimate daily range from 4h data
            df['range'] = df['high'] - df['low']
            estimated_daily_range = df['range'].mean() * 6  # 6 4h candles in a day
            self.asset_metrics[asset]['average_daily_range'] = estimated_daily_range
            
            # Calculate historical volatility
            if len(df) >= 30:
                self.asset_metrics[asset]['historical_volatility'] = calculate_historical_volatility(
                    df['close'], 30
                ).iloc[-1]
    
    def _detect_market_regime(self, data: Dict[int, pd.DataFrame]) -> str:
        """
        Detect the current market regime (trending, ranging, volatile)
        
        Args:
            data: Market data for different timeframes
            
        Returns:
            Market regime classification
        """
        # Use the 4-hour timeframe if available, otherwise use the highest available
        tf = 240 if 240 in data else max(data.keys())
        df = data[tf].copy()
        
        # Calculate key metrics
        if len(df) < 30:
            return "unknown"  # Not enough data for regime detection
        
        # Calculate recent volatility
        recent_volatility = calculate_historical_volatility(df['close'], 20).iloc[-1]
        
        # Calculate Bollinger Bandwidth (indicator of range vs trend)
        upper, middle, lower = calculate_bollinger_bands(df['close'], 20, 2.0)
        bandwidth = (upper - lower) / middle
        bandwidth_percentile = np.percentile(bandwidth, 80)  # Get 80th percentile
        
        # Calculate trend strength using linear regression slope
        x = np.array(range(20))
        y = df['close'].iloc[-20:].values
        slope, _ = np.polyfit(x, y, 1)
        normalized_slope = slope / df['close'].mean()
        
        # Calculate price range relative to average range
        price_range = (df['high'].rolling(20).max() - df['low'].rolling(20).min()).iloc[-1]
        avg_range = (df['high'] - df['low']).rolling(20).mean().iloc[-1]
        range_ratio = price_range / (avg_range * 20) if avg_range > 0 else 1.0
        
        # Classify regime
        if bandwidth.iloc[-1] > bandwidth_percentile and abs(normalized_slope) > 0.001:
            return "trending"  # Wide bands with significant slope
        elif recent_volatility > 0.03 and range_ratio > 2.0:  # 3% volatility and 2x normal range
            return "volatile"
        else:
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
                'rsi': 0.2,
                'stoch': 0.15,
                'bb': 0.15,
                'mean': 0.2,
                'volume': 0.1,
                'volatility': 0.1,
                'pattern': 0.1,
                'divergence': 0.2
            }
        elif regime == "ranging":
            return {
                'rsi': 0.25,
                'stoch': 0.2,
                'bb': 0.3,
                'mean': 0.25,
                'volume': 0.15,
                'volatility': 0.1,
                'pattern': 0.05,
                'divergence': 0.1
            }
        elif regime == "volatile":
            return {
                'rsi': 0.15,
                'stoch': 0.1,
                'bb': 0.2,
                'mean': 0.15,
                'volume': 0.2,
                'volatility': 0.25,
                'pattern': 0.15,
                'divergence': 0.1
            }
        else:
            # Default weights
            return {
                'rsi': self.config.rsi_weight,
                'stoch': self.config.stoch_weight,
                'bb': self.config.bb_weight,
                'mean': self.config.mean_weight,
                'volume': self.config.volume_weight,
                'volatility': self.config.volatility_weight,
                'pattern': 0.1,
                'divergence': 0.1
            }
    
    def _calculate_rsi_score(self, rsi: float) -> float:
        """
        Calculate mean reversion score component from RSI
        
        Args:
            rsi: Current RSI value
            
        Returns:
            Mean reversion score component from -1.0 to 1.0
            Positive values indicate bullish reversal, negative indicate bearish reversal
        """
        if rsi < self.config.rsi_oversold:
            # Extremely oversold - bullish reversal potential
            if rsi < 20:
                return 0.9  # Strong bullish reversal signal
            else:
                return 0.7  # Moderate bullish reversal signal
        elif rsi > self.config.rsi_overbought:
            # Extremely overbought - bearish reversal potential
            if rsi > 80:
                return -0.9  # Strong bearish reversal signal
            else:
                return -0.7  # Moderate bearish reversal signal
        elif rsi < 40:
            # Approaching oversold - mild bullish bias
            return 0.3
        elif rsi > 60:
            # Approaching overbought - mild bearish bias
            return -0.3
        else:
            # Neutral zone - weak or no reversal signal
            return 0.0
    
    def _calculate_stochastic_score(self, k: float, d: float) -> float:
        """
        Calculate mean reversion score component from Stochastic oscillator
        
        Args:
            k: Current %K value
            d: Current %D value
            
        Returns:
            Mean reversion score component from -1.0 to 1.0
        """
        # Check for oversold conditions (bullish reversal)
        if k < self.config.stoch_oversold and d < self.config.stoch_oversold:
            if k > d:  # %K crossing above %D (bullish)
                return 0.8
            else:
                return 0.6
        # Check for overbought conditions (bearish reversal)
        elif k > self.config.stoch_overbought and d > self.config.stoch_overbought:
            if k < d:  # %K crossing below %D (bearish)
                return -0.8
            else:
                return -0.6
        # Check for %K crossing %D in neutral zone
        elif 40 <= k <= 60 and 40 <= d <= 60:
            if k > d:  # Bullish crossover
                return 0.2
            else:  # Bearish crossover
                return -0.2
        # Check for potential reversals from extreme conditions
        elif k < 30 or d < 30:
            return 0.4  # Potential bullish reversal
        elif k > 70 or d > 70:
            return -0.4  # Potential bearish reversal
        else:
            # No clear signal
            return 0.0
    
    def _calculate_bollinger_score(self, price: float, upper: float, middle: float, lower: float) -> float:
        """
        Calculate mean reversion score component from Bollinger Bands
        
        Args:
            price: Current price
            upper: Upper Bollinger Band
            middle: Middle Bollinger Band (SMA)
            lower: Lower Bollinger Band
            
        Returns:
            Mean reversion score component from -1.0 to 1.0
        """
        # Calculate how far price is from bands (as percentage of band width)
        band_width = upper - lower
        
        if band_width == 0:  # Avoid division by zero
            return 0.0
        
        # Calculate distance from middle band (normalized by band width)
        middle_distance = (price - middle) / (0.5 * band_width)
        
        # Calculate distance from each band (normalized)
        upper_distance = (price - upper) / band_width
        lower_distance = (lower - price) / band_width
        
        # Price touching or outside lower band (bullish reversal potential)
        if price <= lower:
            return 0.8 + (lower_distance * 0.2)  # Stronger signal the further outside
        # Price touching or outside upper band (bearish reversal potential)
        elif price >= upper:
            return -0.8 - (upper_distance * 0.2)  # Stronger signal the further outside
        # Price closer to lower band (mild bullish bias)
        elif price < middle:
            # Linear scale from 0 at middle to 0.6 near lower band
            return 0.6 * (middle - price) / (middle - lower)
        # Price closer to upper band (mild bearish bias)
        elif price > middle:
            # Linear scale from 0 at middle to -0.6 near upper band
            return -0.6 * (price - middle) / (upper - middle)
        else:
            # Price at middle band - neutral
            return 0.0
    
    def _calculate_mean_deviation_score(self, distance_from_mean: float) -> float:
        """
        Calculate mean reversion score component from deviation from mean
        
        Args:
            distance_from_mean: Distance from mean in standard deviations
            
        Returns:
            Mean reversion score component from -1.0 to 1.0
        """
        # Normalize and cap extreme values
        normalized_distance = max(min(distance_from_mean, 4.0), -4.0)
        
        # Score based on deviation magnitude and direction
        if normalized_distance <= -self.config.std_dev_threshold:
            # Significantly below mean - bullish reversal potential
            return 0.7 + min((abs(normalized_distance) - self.config.std_dev_threshold) * 0.15, 0.3)
        elif normalized_distance >= self.config.std_dev_threshold:
            # Significantly above mean - bearish reversal potential
            return -0.7 - min((normalized_distance - self.config.std_dev_threshold) * 0.15, 0.3)
        elif normalized_distance < 0:
            # Somewhat below mean - mild bullish bias
            return 0.4 * (normalized_distance / -self.config.std_dev_threshold)
        elif normalized_distance > 0:
            # Somewhat above mean - mild bearish bias
            return -0.4 * (normalized_distance / self.config.std_dev_threshold)
        else:
            # At mean - neutral
            return 0.0
    
    def _calculate_volume_score(self, df: pd.DataFrame, timeframe: int) -> float:
        """
        Calculate mean reversion score component from volume analysis
        
        Args:
            df: DataFrame with market data
            timeframe: Current timeframe
            
        Returns:
            Volume score component from -1.0 to 1.0
        """
        # Get recent volume
        if len(df) < self.config.volume_lookback + 1:
            return 0.0  # Not enough data
            
        recent_volume = df['volume'].iloc[-self.config.volume_lookback:]
        avg_volume = recent_volume.mean()
        latest_volume = recent_volume.iloc[-1]
        latest_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # Calculate normalized volume (compared to average)
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume spike detection
        volume_spike = volume_ratio > self.config.volume_threshold
        
        # Analyze volume with price movement
        price_change = (latest_close - prev_close) / prev_close
        
        # Check for mean reversion patterns in volume
        if volume_spike:
            # High volume with price drop - potential reversal (bullish)
            if price_change < -0.01:
                return 0.7
            # High volume with price rise - potential reversal (bearish)
            elif price_change > 0.01:
                return -0.7
            # High volume with minimal price movement - indecision
            return 0.0
        else:
            # Check for climax volume patterns (several periods)
            if len(df) >= self.config.volume_lookback + 5:
                # Check for declining volume during price movement
                recent_price_trend = np.polyfit(
                    range(5), df['close'].iloc[-5:].values, 1
                )[0]
                recent_volume_trend = np.polyfit(
                    range(5), df['volume'].iloc[-5:].values, 1
                )[0]
                
                # Declining volume during price drop (exhaustion)
                if recent_price_trend < 0 and recent_volume_trend < 0:
                    return 0.4  # Bullish signal (exhausted selling)
                # Declining volume during price rise (exhaustion)
                elif recent_price_trend > 0 and recent_volume_trend < 0:
                    return -0.4  # Bearish signal (exhausted buying)
            
            # No strong volume signal
            return 0.0
    
    def _calculate_volatility_score(self, hist_vol: float, vol_ratio: float) -> float:
        """
        Calculate mean reversion score component from volatility analysis
        
        Args:
            hist_vol: Historical volatility
            vol_ratio: Ratio of short-term to long-term volatility
            
        Returns:
            Volatility score component from -1.0 to 1.0
        """
        # Volatility spike detection (short-term vs long-term)
        if vol_ratio > self.config.volatility_threshold:
            return 0.6  # Volatility spike often precedes reversal
        elif vol_ratio < 1.0 / self.config.volatility_threshold:
            return -0.4  # Volatility contraction may indicate continuation
        
        # High absolute volatility often coincides with reversal points
        if hist_vol > 0.03:  # 3% daily volatility is high
            return 0.5
        elif hist_vol < 0.01:  # 1% daily volatility is low
            return -0.3  # Low volatility often precedes continuation
        
        # No strong volatility signal
        return 0.0
    
    def _calculate_pattern_score(self, patterns: List[Dict[str, Any]], pattern_type: str = "reversal") -> float:
        """
        Calculate mean reversion score component from detected patterns
        
        Args:
            patterns: List of detected patterns
            pattern_type: Type of patterns to focus on ("reversal" or "all")
            
        Returns:
            Pattern score component from -1.0 to 1.0
        """
        if not patterns:
            return 0.0
        
        pattern_scores = []
        
        for pattern in patterns:
            pattern_name = pattern.get('type', '').lower()
            pattern_strength = pattern.get('strength', 0.5)
            
            # Filter for reversal patterns if requested
            if pattern_type == "reversal" and not any(rev in pattern_name for rev in 
                   ['reversal', 'hammer', 'engulfing', 'doji', 'star', 'harami', 'pinbar']):
                continue
            
            # Bullish reversal patterns
            if any(bull in pattern_name for bull in ['bullish', 'hammer', 'morning']):
                pattern_scores.append(pattern_strength)
            # Bearish reversal patterns
            elif any(bear in pattern_name for bear in ['bearish', 'hanging', 'evening']):
                pattern_scores.append(-pattern_strength)
        
        if not pattern_scores:
            return 0.0
        
        # Return average of pattern scores
        return sum(pattern_scores) / len(pattern_scores)
    
    def _calculate_divergence_score(self, divergences: Dict[str, Any]) -> float:
        """
        Calculate mean reversion score component from detected divergences
        
        Args:
            divergences: Dictionary of detected divergences
            
        Returns:
            Divergence score component from -1.0 to 1.0
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
        features['stoch_k'] = df['stoch_k'].iloc[-1] if 'stoch_k' in df else 50.0
        features['stoch_d'] = df['stoch_d'].iloc[-1] if 'stoch_d' in df else 50.0
        
        # Bollinger Band features
        if all(col in df for col in ['upper_band', 'middle_band', 'lower_band']):
            current_price = df['close'].iloc[-1]
            upper = df['upper_band'].iloc[-1]
            middle = df['middle_band'].iloc[-1]
            lower = df['lower_band'].iloc[-1]
            
            features['bb_width'] = (upper - lower) / middle if middle > 0 else 0
            features['bb_position'] = (current_price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
        
        # Mean reversion features
        features['distance_from_mean'] = df['distance_from_mean'].iloc[-1] if 'distance_from_mean' in df else 0.0
        
        # Price action features
        if len(df) >= 20:
            features['returns_1'] = df['close'].pct_change(1).iloc[-1]
            features['returns_5'] = df['close'].pct_change(5).iloc[-1]
            features['returns_10'] = df['close'].pct_change(10).iloc[-1]
            
            # Volatility features
            features['hist_vol'] = df['hist_vol'].iloc[-1] if 'hist_vol' in df else 0.02
            features['vol_ratio'] = df['vol_ratio'].iloc[-1] if 'vol_ratio' in df else 1.0
            
            # Range features
            features['daily_range_ratio'] = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
        
        return features
    
    def _determine_signal(self, df: pd.DataFrame, reversion_score: float, 
                          divergences: Dict[str, Any], patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine trading signal based on mean reversion score and other factors
        
        Args:
            df: DataFrame with market data
            reversion_score: Calculated mean reversion score
            divergences: Detected divergences
            patterns: Detected patterns
            
        Returns:
            Signal dictionary with direction and strength
        """
        # Determine direction based on reversion score
        # Positive score indicates bullish reversal (oversold condition)
        # Negative score indicates bearish reversal (overbought condition)
        if reversion_score > 0.7:
            direction = TradeDirection.STRONG_BUY
            strength = SignalStrength.VERY_HIGH
        elif reversion_score > 0.4:
            direction = TradeDirection.BUY
            strength = SignalStrength.HIGH
        elif reversion_score > 0.2:
            direction = TradeDirection.WEAK_BUY
            strength = SignalStrength.MEDIUM
        elif reversion_score < -0.7:
            direction = TradeDirection.STRONG_SELL
            strength = SignalStrength.VERY_HIGH
        elif reversion_score < -0.4:
            direction = TradeDirection.SELL
            strength = SignalStrength.HIGH
        elif reversion_score < -0.2:
            direction = TradeDirection.WEAK_SELL
            strength = SignalStrength.MEDIUM
        else:
            direction = TradeDirection.NEUTRAL
            strength = SignalStrength.LOW
        
        # Adjust signal based on patterns and divergences
        pattern_adjustment = self._adjust_signal_for_patterns(patterns, "reversal")
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
            "original_score": reversion_score,
            "adjusted_score": reversion_score + (pattern_adjustment + divergence_adjustment) * 0.1,
            "explanation": self._generate_signal_explanation(direction, reversion_score, patterns, divergences),
            "success_probability": success_prob
        }
    
    def _adjust_signal_for_patterns(self, patterns: List[Dict[str, Any]], pattern_filter: str = "reversal") -> int:
        """
        Adjust signal strength based on detected patterns
        
        Args:
            patterns: Detected patterns
            pattern_filter: Type of patterns to consider
            
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
            
            # Filter for reversal patterns if requested
            if pattern_filter == "reversal" and not any(rev in pattern_type for rev in 
                   ['reversal', 'hammer', 'engulfing', 'doji', 'star', 'harami', 'pinbar']):
                continue
            
            if 'bullish' in pattern_type or 'hammer' in pattern_type or 'morning' in pattern_type:
                if pattern_strength >= 0.7:
                    strong_bullish += 1
                else:
                    weak_bullish += 1
            elif 'bearish' in pattern_type or 'hanging' in pattern_type or 'evening' in pattern_type:
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
            score: Mean reversion score
            patterns: Detected patterns
            divergences: Detected divergences
            
        Returns:
            Explanation string
        """
        explanation = []
        
        # Direction explanation
        if direction in [TradeDirection.STRONG_BUY, TradeDirection.BUY, TradeDirection.WEAK_BUY]:
            explanation.append(f"Bullish reversal potential detected (score: {score:.2f})")
        elif direction in [TradeDirection.STRONG_SELL, TradeDirection.SELL, TradeDirection.WEAK_SELL]:
            explanation.append(f"Bearish reversal potential detected (score: {score:.2f})")
        else:
            explanation.append(f"No clear reversal signal (score: {score:.2f})")
        
        # Add pattern information
        if patterns:
            reversal_patterns = [p for p in patterns if any(rev in p.get('type', '').lower() for rev in 
                              ['reversal', 'hammer', 'engulfing', 'doji', 'star', 'harami', 'pinbar'])]
            
            if reversal_patterns:
                pattern_strs = []
                for pattern in reversal_patterns[:3]:  # Limit to 3 patterns
                    pattern_strs.append(f"{pattern.get('type', 'Unknown')} (strength: {pattern.get('strength', 0):.2f})")
                explanation.append(f"Reversal patterns: {', '.join(pattern_strs)}")
        
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
        
        # Extract Bollinger Bands
        middle_band = df['middle_band'].iloc[-1] if 'middle_band' in df.columns else df['close'].rolling(20).mean().iloc[-1]
        upper_band = df['upper_band'].iloc[-1] if 'upper_band' in df.columns else middle_band + 2 * df['close'].rolling(20).std().iloc[-1]
        lower_band = df['lower_band'].iloc[-1] if 'lower_band' in df.columns else middle_band - 2 * df['close'].rolling(20).std().iloc[-1]
        
        # Find support and resistance levels
        support_resistance = find_support_resistance_levels(
            df['high'].values, df['low'].values, df['close'].values, num_levels=3
        )
        
        support_levels = [level for level in support_resistance['support'] if level < latest_close]
        resistance_levels = [level for level in support_resistance['resistance'] if level > latest_close]
        
        nearest_support = max(support_levels) if support_levels else (latest_close - 2 * atr)
        nearest_resistance = min(resistance_levels) if resistance_levels else (latest_close + 2 * atr)
        
        # Calculate reversion targets
        if latest_close > middle_band:
            # Price above mean - potential bearish reversion
            reversion_target_50 = middle_band + (latest_close - middle_band) * (1 - self.config.mean_reversion_exit_threshold)
            reversion_target_100 = middle_band
        else:
            # Price below mean - potential bullish reversion
            reversion_target_50 = middle_band - (middle_band - latest_close) * (1 - self.config.mean_reversion_exit_threshold)
            reversion_target_100 = middle_band
        
        return {
            "current_price": latest_close,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "middle_band": middle_band,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "atr": atr,
            "reversion_target_50": reversion_target_50,
            "reversion_target_100": reversion_target_100
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
        explanation = f"Combined reversal signal from {len(timeframes)} timeframes: {final_score:.2f}"
        
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
    
    def _update_active_signals(self, asset: str, signal: Dict[str, Any]) -> None:
        """
        Update tracking of active signals for an asset
        
        Args:
            asset: Asset symbol
            signal: Generated signal
        """
        if asset not in self.active_signals:
            self.active_signals[asset] = {
                "active": False,
                "direction": TradeDirection.NEUTRAL,
                "signal_id": None,
                "timestamp": None
            }
        
        direction = signal.get('direction', TradeDirection.NEUTRAL)
        strength = signal.get('strength', SignalStrength.VERY_LOW)

        # Only update if signal is strong enough
        if direction != TradeDirection.NEUTRAL and strength >= SignalStrength.MEDIUM:
            self.active_signals[asset]["active"] = True
            self.active_signals[asset]["direction"] = direction
            self.active_signals[asset]["signal_id"] = signal.get("signal_id", str(uuid.uuid4()))
            self.active_signals[asset]["timestamp"] = datetime.now()
        else:
            self.active_signals[asset]["active"] = False
            self.active_signals[asset]["direction"] = TradeDirection.NEUTRAL
            self.active_signals[asset]["signal_id"] = None
            self.active_signals[asset]["timestamp"] = None
