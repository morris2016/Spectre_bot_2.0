"""
Feature Extraction for the QuantumSpectre Elite Trading System.

This module extracts tradable features from processed market data for use
by the trading strategies and machine learning models.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import ta
from common import ta_candles
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

from common.config import settings
from common.logger import get_logger
from common.metrics import calculate_timing
from common.models.feature import FeatureSet, Feature
from common.exceptions import FeatureExtractionError

logger = get_logger("feature_extraction")

class FeatureExtractor:
    """
    Feature extractor for market data.
    
    This class extracts tradable features from market data for use by trading
    strategies and machine learning models. It includes a wide range of technical
    indicators, market microstructure metrics, and other features.
    """
    
    def __init__(self, load_scalers: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            load_scalers: Whether to load pre-trained scalers
        """
        # Feature extraction configuration
        self.config = settings.FEATURE_EXTRACTION
        
        # Cache for intermediate calculations
        self.cache: Dict[str, Dict[str, Any]] = {
            'candles': {},  # Cached candle data
            'features': {}  # Cached feature values
        }
        
        # Scalers for normalizing features
        self.scalers: Dict[str, Any] = {}
        
        # Performance tracking
        self.timing_stats: Dict[str, List[float]] = {}
        
        # Load pre-trained scalers if enabled
        if load_scalers:
            self._load_scalers()
        
        logger.info(f"FeatureExtractor initialized with {len(self.config.ENABLED_FEATURES)} enabled features")
    
    def _load_scalers(self) -> None:
        """Load pre-trained scalers for feature normalization."""
        scaler_dir = os.path.join(settings.MODEL_DIR, 'scalers')
        
        if not os.path.exists(scaler_dir):
            logger.warning(f"Scaler directory not found: {scaler_dir}")
            return
            
        try:
            # Load each scaler
            for feature in self.config.ENABLED_FEATURES:
                scaler_path = os.path.join(scaler_dir, f"{feature}_scaler.pkl")
                if os.path.exists(scaler_path):
                    self.scalers[feature] = joblib.load(scaler_path)
                    logger.debug(f"Loaded scaler for feature: {feature}")
                    
            logger.info(f"Loaded {len(self.scalers)} feature scalers")
            
        except Exception as e:
            logger.error(f"Error loading scalers: {str(e)}")
    
    def _save_scalers(self) -> None:
        """Save trained scalers for feature normalization."""
        scaler_dir = os.path.join(settings.MODEL_DIR, 'scalers')
        
        # Create directory if it doesn't exist
        os.makedirs(scaler_dir, exist_ok=True)
        
        try:
            # Save each scaler
            for feature, scaler in self.scalers.items():
                scaler_path = os.path.join(scaler_dir, f"{feature}_scaler.pkl")
                joblib.dump(scaler, scaler_path)
                
            logger.info(f"Saved {len(self.scalers)} feature scalers")
            
        except Exception as e:
            logger.error(f"Error saving scalers: {str(e)}")
    
    @calculate_timing
    def extract_features(
        self, 
        candles: pd.DataFrame, 
        orderbook: Optional[Dict[str, Any]] = None,
        trades: Optional[pd.DataFrame] = None,
        ticker: Optional[Dict[str, Any]] = None,
        symbol: str = "",
        exchange: str = ""
    ) -> FeatureSet:
        """
        Extract features from market data.
        
        Args:
            candles: Candlestick data
            orderbook: Latest orderbook snapshot
            trades: Recent trades
            ticker: Latest ticker data
            symbol: Trading symbol
            exchange: Exchange name
            
        Returns:
            FeatureSet object with extracted features
        """
        try:
            if candles.empty:
                raise FeatureExtractionError("Empty candle data provided")
                
            # Ensure candles are sorted by time
            candles = candles.sort_values('timestamp').reset_index(drop=True)
            
            # Initialize feature dictionary
            feature_dict: Dict[str, Feature] = {}
            
            # Calculate features based on configuration
            for feature_name in self.config.ENABLED_FEATURES:
                if feature_name.startswith('ta_'):
                    # Technical analysis feature
                    value = self._calculate_ta_feature(feature_name, candles)
                elif feature_name.startswith('ob_'):
                    # Orderbook feature
                    value = self._calculate_orderbook_feature(feature_name, orderbook)
                elif feature_name.startswith('trade_'):
                    # Trade feature
                    value = self._calculate_trade_feature(feature_name, trades)
                elif feature_name.startswith('volatility_'):
                    # Volatility feature
                    value = self._calculate_volatility_feature(feature_name, candles)
                elif feature_name.startswith('pattern_'):
                    # Pattern recognition feature
                    value = self._calculate_pattern_feature(feature_name, candles)
                elif feature_name.startswith('trend_'):
                    # Trend feature
                    value = self._calculate_trend_feature(feature_name, candles)
                elif feature_name.startswith('momentum_'):
                    # Momentum feature
                    value = self._calculate_momentum_feature(feature_name, candles)
                elif feature_name.startswith('custom_'):
                    # Custom feature
                    value = self._calculate_custom_feature(feature_name, candles, orderbook, trades, ticker)
                else:
                    # Unknown feature
                    logger.warning(f"Unknown feature type: {feature_name}")
                    continue
                
                # Skip if feature calculation failed
                if value is None:
                    continue
                
                # Normalize feature if scaler exists
                normalized_value = self._normalize_feature(feature_name, value)
                
                # Create Feature object
                feature = Feature(
                    name=feature_name,
                    value=value,
                    normalized_value=normalized_value,
                    timestamp=int(time.time() * 1000)
                )
                
                # Add to feature dictionary
                feature_dict[feature_name] = feature
            
            # Create FeatureSet
            feature_set = FeatureSet(
                symbol=symbol,
                exchange=exchange,
                timestamp=int(time.time() * 1000),
                features=feature_dict,
                metadata={
                    'candle_count': len(candles),
                    'latest_candle_time': int(candles.iloc[-1]['timestamp']),
                    'feature_count': len(feature_dict)
                }
            )
            
            # Cache latest feature set for this symbol
            cache_key = f"{exchange}:{symbol}"
            self.cache['features'][cache_key] = feature_set
            
            return feature_set
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            # Return an empty feature set on error
            return FeatureSet(
                symbol=symbol,
                exchange=exchange,
                timestamp=int(time.time() * 1000),
                features={},
                metadata={'error': str(e)}
            )
    
    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """
        Normalize a feature value using its scaler.
        
        Args:
            feature_name: Feature name
            value: Raw feature value
            
        Returns:
            Normalized feature value
        """
        # Check if scaler exists
        if feature_name in self.scalers:
            try:
                # Reshape value for scaler
                reshaped_value = np.array([[value]])
                normalized = self.scalers[feature_name].transform(reshaped_value)[0][0]
                return normalized
            except Exception as e:
                logger.error(f"Error normalizing feature {feature_name}: {str(e)}")
                return value
        else:
            # Simple min-max normalization based on typical ranges
            if feature_name.startswith('ta_rsi'):
                # RSI is typically 0-100
                return value / 100.0
            elif feature_name.startswith('ta_macd'):
                # MACD can be positive or negative, normalize to [-1, 1]
                return max(min(value / 5.0, 1.0), -1.0)  # Assuming typical range of -5 to 5
            elif feature_name.startswith('ob_imbalance'):
                # Order book imbalance is typically -1 to 1
                return value
            else:
                # Default normalization - clip to [-1, 1]
                return max(min(value, 1.0), -1.0)
    
    def _train_scalers(self, feature_history: Dict[str, List[float]]) -> None:
        """
        Train scalers for feature normalization.
        
        Args:
            feature_history: Dictionary of feature values by name
        """
        for feature_name, values in feature_history.items():
            if len(values) < 100:
                logger.warning(f"Not enough data to train scaler for {feature_name}: {len(values)} values")
                continue
                
            try:
                # Create and fit scaler
                scaler = MinMaxScaler(feature_range=(-1, 1))
                
                # Reshape values
                reshaped_values = np.array(values).reshape(-1, 1)
                
                # Fit scaler
                scaler.fit(reshaped_values)
                
                # Save scaler
                self.scalers[feature_name] = scaler
                
                logger.debug(f"Trained scaler for feature: {feature_name}")
                
            except Exception as e:
                logger.error(f"Error training scaler for {feature_name}: {str(e)}")
        
        # Save scalers to disk
        self._save_scalers()
    
    def _calculate_ta_feature(self, feature_name: str, candles: pd.DataFrame) -> Optional[float]:
        """
        Calculate a technical analysis feature.
        
        Args:
            feature_name: Feature name
            candles: Candlestick data
            
        Returns:
            Feature value or None if calculation fails
        """
        try:
            # Extract price and volume arrays
            open_price = candles['open'].values
            high_price = candles['high'].values
            low_price = candles['low'].values
            close_price = candles['close'].values
            volume = candles['volume'].values if 'volume' in candles.columns else None
            
            # Calculate feature based on name
            if feature_name == 'ta_rsi_14':
                # 14-period RSI
                rsi = ta.momentum.rsi(pd.Series(close_price), window=14)
                return rsi.iloc[-1]
                
            elif feature_name == 'ta_rsi_7':
                # 7-period RSI
                rsi = ta.momentum.rsi(pd.Series(close_price), window=7)
                return rsi.iloc[-1]
                
            elif feature_name == 'ta_macd':
                # MACD
                macd = ta.trend.macd(pd.Series(close_price), window_fast=12, window_slow=26, window_sign=9)
                return macd.iloc[-1]
                
            elif feature_name == 'ta_macd_signal':
                # MACD signal line
                signal = ta.trend.macd_signal(pd.Series(close_price), window_fast=12, window_slow=26, window_sign=9)
                return signal.iloc[-1]
                
            elif feature_name == 'ta_macd_hist':
                # MACD histogram
                hist = ta.trend.macd_diff(pd.Series(close_price), window_fast=12, window_slow=26, window_sign=9)
                return hist.iloc[-1]
                
            elif feature_name == 'ta_sma_10':
                # 10-period SMA
                sma = ta.trend.sma_indicator(pd.Series(close_price), window=10)
                return sma.iloc[-1]
                
            elif feature_name == 'ta_sma_20':
                # 20-period SMA
                sma = ta.trend.sma_indicator(pd.Series(close_price), window=20)
                return sma.iloc[-1]
                
            elif feature_name == 'ta_sma_50':
                # 50-period SMA
                sma = ta.trend.sma_indicator(pd.Series(close_price), window=50)
                return sma.iloc[-1]
                
            elif feature_name == 'ta_sma_200':
                # 200-period SMA
                sma = ta.trend.sma_indicator(pd.Series(close_price), window=200)
                return sma.iloc[-1]
                
            elif feature_name == 'ta_ema_10':
                # 10-period EMA
                ema = ta.trend.ema_indicator(pd.Series(close_price), window=10)
                return ema.iloc[-1]
                
            elif feature_name == 'ta_ema_20':
                # 20-period EMA
                ema = ta.trend.ema_indicator(pd.Series(close_price), window=20)
                return ema.iloc[-1]
                
            elif feature_name == 'ta_ema_50':
                # 50-period EMA
                ema = ta.trend.ema_indicator(pd.Series(close_price), window=50)
                return ema.iloc[-1]
                
            elif feature_name == 'ta_ema_200':
                # 200-period EMA
                ema = ta.trend.ema_indicator(pd.Series(close_price), window=200)
                return ema.iloc[-1]
                
            elif feature_name == 'ta_bollinger_upper':
                # Bollinger Band Upper (20, 2)
                upper = ta.volatility.bollinger_hband(pd.Series(close_price), window=20, window_dev=2)
                return upper.iloc[-1]
                
            elif feature_name == 'ta_bollinger_middle':
                # Bollinger Band Middle (20, 2)
                middle = ta.volatility.bollinger_mavg(pd.Series(close_price), window=20)
                return middle.iloc[-1]
                
            elif feature_name == 'ta_bollinger_lower':
                # Bollinger Band Lower (20, 2)
                lower = ta.volatility.bollinger_lband(pd.Series(close_price), window=20, window_dev=2)
                return lower.iloc[-1]
                
            elif feature_name == 'ta_bollinger_width':
                # Bollinger Band Width (20, 2)
                upper = ta.volatility.bollinger_hband(pd.Series(close_price), window=20, window_dev=2)
                lower = ta.volatility.bollinger_lband(pd.Series(close_price), window=20, window_dev=2)
                middle = ta.volatility.bollinger_mavg(pd.Series(close_price), window=20)
                return (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
                
            elif feature_name == 'ta_bollinger_pct':
                # Position within Bollinger Bands as percentage
                upper = ta.volatility.bollinger_hband(pd.Series(close_price), window=20, window_dev=2)
                lower = ta.volatility.bollinger_lband(pd.Series(close_price), window=20, window_dev=2)
                band_width = upper.iloc[-1] - lower.iloc[-1]
                if band_width > 0:
                    return (close_price[-1] - lower.iloc[-1]) / band_width
                else:
                    return 0.5
                    
            elif feature_name == 'ta_stoch_k':
                # Stochastic %K (14, 3, 3)
                k = ta.momentum.stoch(
                    pd.Series(high_price),
                    pd.Series(low_price),
                    pd.Series(close_price),
                    k=14,
                    d=3,
                    smooth_k=3,
                )
                return k.iloc[-1]
                
            elif feature_name == 'ta_stoch_d':
                # Stochastic %D (14, 3, 3)
                d = ta.momentum.stoch_signal(
                    pd.Series(high_price),
                    pd.Series(low_price),
                    pd.Series(close_price),
                    k=14,
                    d=3,
                    smooth_k=3,
                )
                return d.iloc[-1]
                
            elif feature_name == 'ta_adx':
                # ADX (14)
                adx = ta.trend.adx(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return adx.iloc[-1]
                
            elif feature_name == 'ta_adx_di_plus':
                # +DI (14)
                plus_di = ta.trend.adx_pos(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return plus_di.iloc[-1]
                
            elif feature_name == 'ta_adx_di_minus':
                # -DI (14)
                minus_di = ta.trend.adx_neg(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return minus_di.iloc[-1]
                
            elif feature_name == 'ta_atr':
                # ATR (14)
                atr = ta.volatility.average_true_range(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return atr.iloc[-1]
                
            elif feature_name == 'ta_atr_percent':
                # ATR as percentage of price
                atr = ta.volatility.average_true_range(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return atr.iloc[-1] / close_price[-1] * 100 if close_price[-1] > 0 else 0
                
            elif feature_name == 'ta_cci':
                # CCI (14)
                cci = ta.trend.cci(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return cci.iloc[-1]
                
            elif feature_name == 'ta_obv':
                # OBV
                if volume is not None:
                    obv = ta.volume.on_balance_volume(pd.Series(close_price), pd.Series(volume))
                    return obv.iloc[-1]
                else:
                    logger.warning("Volume data not available for OBV calculation")
                    return None
                    
            elif feature_name == 'ta_roc':
                # Rate of Change (10)
                roc = ta.momentum.roc(pd.Series(close_price), window=10)
                return roc.iloc[-1]
                
            elif feature_name == 'ta_roc_5':
                # Rate of Change (5)
                roc = ta.momentum.roc(pd.Series(close_price), window=5)
                return roc.iloc[-1]
                
            elif feature_name == 'ta_roc_21':
                # Rate of Change (21)
                roc = ta.momentum.roc(pd.Series(close_price), window=21)
                return roc.iloc[-1]
                
            elif feature_name == 'ta_willr':
                # Williams %R (14)
                willr = ta.momentum.williams_r(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), lbp=14)
                return willr.iloc[-1]
                
            elif feature_name == 'ta_mom':
                # Momentum (10)
                mom = pd.Series(close_price).diff(10)
                return mom.iloc[-1]
                
            elif feature_name == 'ta_mom_5':
                # Momentum (5)
                mom = pd.Series(close_price).diff(5)
                return mom.iloc[-1]
                
            elif feature_name == 'ta_mom_21':
                # Momentum (21)
                mom = pd.Series(close_price).diff(21)
                return mom.iloc[-1]
                
            elif feature_name == 'ta_trix':
                # TRIX (30)
                trix = ta.trend.trix(pd.Series(close_price), window=30)
                return trix.iloc[-1]
                
            elif feature_name == 'ta_ultosc':
                # Ultimate Oscillator (7, 14, 28)
                ultosc = ta.momentum.uo(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), s=7, m=14, l=28)
                return ultosc.iloc[-1]
                
            elif feature_name == 'ta_natr':
                # Normalized ATR (14)
                natr = ta.volatility.natr(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return natr.iloc[-1]
                
            elif feature_name == 'ta_kama':
                # Kaufman Adaptive Moving Average (30)
                kama = ta.trend.kama(pd.Series(close_price), window=30)
                return kama.iloc[-1]
                
            elif feature_name == 'ta_tema':
                # Triple Exponential Moving Average (30)
                tema = ta.trend.tema(pd.Series(close_price), window=30)
                return tema.iloc[-1]
                
            else:
                # Unknown technical feature
                logger.warning(f"Unknown technical feature: {feature_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating technical feature {feature_name}: {str(e)}")
            return None
    
    def _calculate_orderbook_feature(self, feature_name: str, orderbook: Optional[Dict[str, Any]]) -> Optional[float]:
        """
        Calculate an orderbook feature.
        
        Args:
            feature_name: Feature name
            orderbook: Orderbook data
            
        Returns:
            Feature value or None if calculation fails
        """
        if orderbook is None:
            return None
            
        try:
            # Calculate feature based on name
            if feature_name == 'ob_imbalance':
                # Order book imbalance
                if 'book_imbalance' in orderbook:
                    return orderbook['book_imbalance']
                    
                # Calculate if not already present
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if not bids or not asks:
                    return 0.0
                    
                # Calculate total value on each side (price * quantity)
                bid_value = sum(bid[0] * bid[1] for bid in bids[:10])
                ask_value = sum(ask[0] * ask[1] for ask in asks[:10])
                
                total_value = bid_value + ask_value
                if total_value > 0:
                    imbalance = (bid_value - ask_value) / total_value
                    return imbalance
                else:
                    return 0.0
                    
            elif feature_name == 'ob_spread':
                # Bid-ask spread
                if 'spread' in orderbook:
                    return orderbook['spread']
                    
                # Calculate if not already present
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if not bids or not asks:
                    return 0.0
                    
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                
                return best_ask - best_bid
                
            elif feature_name == 'ob_spread_pct':
                # Bid-ask spread as percentage of price
                if 'spread_pct' in orderbook:
                    return orderbook['spread_pct']
                    
                # Calculate if not already present
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if not bids or not asks:
                    return 0.0
                    
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                
                if mid_price > 0:
                    return spread / mid_price * 100
                else:
                    return 0.0
                    
            elif feature_name == 'ob_mid_price':
                # Mid price
                if 'mid_price' in orderbook:
                    return orderbook['mid_price']
                    
                # Calculate if not already present
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if not bids or not asks:
                    return 0.0
                    
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                
                return (best_bid + best_ask) / 2
                
            elif feature_name == 'ob_depth_ratio':
                # Ratio of bid depth to ask depth
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if not bids or not asks:
                    return 0.0
                    
                # Calculate total quantity on each side
                bid_quantity = sum(bid[1] for bid in bids[:10])
                ask_quantity = sum(ask[1] for ask in asks[:10])
                
                if ask_quantity > 0:
                    return bid_quantity / ask_quantity
                else:
                    return 1.0
                    
            elif feature_name == 'ob_bid_quantity':
                # Total bid quantity
                bids = orderbook.get('bids', [])
                
                if not bids:
                    return 0.0
                    
                return sum(bid[1] for bid in bids[:10])
                
            elif feature_name == 'ob_ask_quantity':
                # Total ask quantity
                asks = orderbook.get('asks', [])
                
                if not asks:
                    return 0.0
                    
                return sum(ask[1] for ask in asks[:10])
                
            elif feature_name == 'ob_bid_levels':
                # Number of bid price levels
                bids = orderbook.get('bids', [])
                
                if not bids:
                    return 0.0
                    
                return len(bids)
                
            elif feature_name == 'ob_ask_levels':
                # Number of ask price levels
                asks = orderbook.get('asks', [])
                
                if not asks:
                    return 0.0
                    
                return len(asks)
                
            elif feature_name == 'ob_best_bid':
                # Best bid price
                bids = orderbook.get('bids', [])
                
                if not bids:
                    return 0.0
                    
                return bids[0][0]
                
            elif feature_name == 'ob_best_ask':
                # Best ask price
                asks = orderbook.get('asks', [])
                
                if not asks:
                    return 0.0
                    
                return asks[0][0]
                
            elif feature_name == 'ob_weighted_mid_price':
                # Weighted mid price based on order book quantities
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if not bids or not asks:
                    return 0.0
                    
                bid_value = sum(bid[0] * bid[1] for bid in bids[:5])
                ask_value = sum(ask[0] * ask[1] for ask in asks[:5])
                total_bid_qty = sum(bid[1] for bid in bids[:5])
                total_ask_qty = sum(ask[1] for ask in asks[:5])
                
                if total_bid_qty + total_ask_qty > 0:
                    weighted_mid = (bid_value + ask_value) / (total_bid_qty + total_ask_qty)
                    return weighted_mid
                else:
                    return 0.0
                    
            elif feature_name == 'ob_liquidity_ratio':
                # Ratio of liquidity within 1% of mid price
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if not bids or not asks:
                    return 0.0
                    
                mid_price = (bids[0][0] + asks[0][0]) / 2
                bid_threshold = mid_price * 0.99
                ask_threshold = mid_price * 1.01
                
                near_bid_qty = sum(bid[1] for bid in bids if bid[0] >= bid_threshold)
                near_ask_qty = sum(ask[1] for ask in asks if ask[0] <= ask_threshold)
                
                total_bid_qty = sum(bid[1] for bid in bids)
                total_ask_qty = sum(ask[1] for ask in asks)
                
                if total_bid_qty + total_ask_qty > 0:
                    return (near_bid_qty + near_ask_qty) / (total_bid_qty + total_ask_qty)
                else:
                    return 0.0
                    
            else:
                # Unknown orderbook feature
                logger.warning(f"Unknown orderbook feature: {feature_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating orderbook feature {feature_name}: {str(e)}")
            return None
    
    def _calculate_trade_feature(self, feature_name: str, trades: Optional[pd.DataFrame]) -> Optional[float]:
        """
        Calculate a trade feature.
        
        Args:
            feature_name: Feature name
            trades: Recent trades data
            
        Returns:
            Feature value or None if calculation fails
        """
        if trades is None or trades.empty:
            return None
            
        try:
            # Calculate feature based on name
            if feature_name == 'trade_buy_volume_ratio':
                # Ratio of buy volume to total volume
                buys = trades[trades['side'] == 'buy']
                total_volume = trades['quantity'].sum()
                
                if total_volume > 0:
                    buy_volume = buys['quantity'].sum()
                    return buy_volume / total_volume
                else:
                    return 0.5
                    
            elif feature_name == 'trade_sell_volume_ratio':
                # Ratio of sell volume to total volume
                sells = trades[trades['side'] == 'sell']
                total_volume = trades['quantity'].sum()
                
                if total_volume > 0:
                    sell_volume = sells['quantity'].sum()
                    return sell_volume / total_volume
                else:
                    return 0.5
                    
            elif feature_name == 'trade_buy_count_ratio':
                # Ratio of buy trades to total trades
                buys = trades[trades['side'] == 'buy']
                total_trades = len(trades)
                
                if total_trades > 0:
                    buy_count = len(buys)
                    return buy_count / total_trades
                else:
                    return 0.5
                    
            elif feature_name == 'trade_sell_count_ratio':
                # Ratio of sell trades to total trades
                sells = trades[trades['side'] == 'sell']
                total_trades = len(trades)
                
                if total_trades > 0:
                    sell_count = len(sells)
                    return sell_count / total_trades
                else:
                    return 0.5
                    
            elif feature_name == 'trade_avg_size':
                # Average trade size
                return trades['quantity'].mean()
                
            elif feature_name == 'trade_med_size':
                # Median trade size
                return trades['quantity'].median()
                
            elif feature_name == 'trade_max_size':
                # Maximum trade size
                return trades['quantity'].max()
                
            elif feature_name == 'trade_min_size':
                # Minimum trade size
                return trades['quantity'].min()
                
            elif feature_name == 'trade_volume':
                # Total volume
                return trades['quantity'].sum()
                
            elif feature_name == 'trade_count':
                # Number of trades
                return len(trades)
                
            elif feature_name == 'trade_vwap':
                # Volume-weighted average price
                if 'value' in trades.columns:
                    total_value = trades['value'].sum()
                    total_quantity = trades['quantity'].sum()
                    
                    if total_quantity > 0:
                        return total_value / total_quantity
                    else:
                        return 0.0
                else:
                    # Calculate value as price * quantity
                    total_value = (trades['price'] * trades['quantity']).sum()
                    total_quantity = trades['quantity'].sum()
                    
                    if total_quantity > 0:
                        return total_value / total_quantity
                    else:
                        return 0.0
                        
            elif feature_name == 'trade_price_std':
                # Standard deviation of trade prices
                return trades['price'].std()
                
            elif feature_name == 'trade_price_skew':
                # Skewness of trade prices
                return trades['price'].skew()
                
            elif feature_name == 'trade_price_kurt':
                # Kurtosis of trade prices
                return trades['price'].kurtosis()
                
            elif feature_name == 'trade_price_range':
                # Range of trade prices
                return trades['price'].max() - trades['price'].min()
                
            elif feature_name == 'trade_price_range_pct':
                # Range of trade prices as percentage of mean
                price_mean = trades['price'].mean()
                price_range = trades['price'].max() - trades['price'].min()
                
                if price_mean > 0:
                    return price_range / price_mean * 100
                else:
                    return 0.0
                    
            elif feature_name == 'trade_time_weighted_price':
                # Time-weighted average price
                # More recent trades have higher weight
                trades = trades.sort_values('timestamp')
                weights = np.linspace(0.5, 1.0, len(trades))
                weighted_price = (trades['price'] * weights).sum() / weights.sum()
                
                return weighted_price
                
            elif feature_name == 'trade_buy_pressure':
                # Buy pressure (ratio of buy volume * price to total)
                buys = trades[trades['side'] == 'buy']
                
                if 'value' in trades.columns:
                    total_value = trades['value'].sum()
                    
                    if total_value > 0:
                        buy_value = buys['value'].sum()
                        return buy_value / total_value
                    else:
                        return 0.5
                else:
                    # Calculate value as price * quantity
                    total_value = (trades['price'] * trades['quantity']).sum()
                    
                    if total_value > 0:
                        buy_value = (buys['price'] * buys['quantity']).sum()
                        return buy_value / total_value
                    else:
                        return 0.5
                        
            else:
                # Unknown trade feature
                logger.warning(f"Unknown trade feature: {feature_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating trade feature {feature_name}: {str(e)}")
            return None
    
    def _calculate_volatility_feature(self, feature_name: str, candles: pd.DataFrame) -> Optional[float]:
        """
        Calculate a volatility feature.
        
        Args:
            feature_name: Feature name
            candles: Candlestick data
            
        Returns:
            Feature value or None if calculation fails
        """
        try:
            # Calculate feature based on name
            if feature_name == 'volatility_atr_14':
                # 14-period ATR
                high_price = candles['high'].values
                low_price = candles['low'].values
                close_price = candles['close'].values

                atr = ta.volatility.average_true_range(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return atr.iloc[-1]
                
            elif feature_name == 'volatility_atr_14_pct':
                # 14-period ATR as percentage of price
                high_price = candles['high'].values
                low_price = candles['low'].values
                close_price = candles['close'].values

                atr = ta.volatility.average_true_range(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return atr.iloc[-1] / close_price[-1] * 100 if close_price[-1] > 0 else 0
                
            elif feature_name == 'volatility_natr_14':
                # 14-period Normalized ATR
                high_price = candles['high'].values
                low_price = candles['low'].values
                close_price = candles['close'].values

                natr = ta.volatility.natr(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return natr.iloc[-1]
                
            elif feature_name == 'volatility_stddev_20':
                # 20-period standard deviation of returns
                close_price = candles['close'].values
                returns = np.diff(close_price) / close_price[:-1]
                
                if len(returns) >= 20:
                    return np.std(returns[-20:]) * 100  # As percentage
                else:
                    return np.std(returns) * 100
                    
            elif feature_name == 'volatility_stddev_20_annualized':
                # Annualized 20-period standard deviation of returns
                close_price = candles['close'].values
                returns = np.diff(close_price) / close_price[:-1]
                
                if len(returns) >= 20:
                    daily_std = np.std(returns[-20:])
                    # Assuming candles are daily
                    annualized_std = daily_std * np.sqrt(252)
                    return annualized_std * 100  # As percentage
                else:
                    daily_std = np.std(returns)
                    annualized_std = daily_std * np.sqrt(252)
                    return annualized_std * 100
                    
            elif feature_name == 'volatility_parkinson':
                # Parkinson volatility estimator
                high_price = candles['high'].values
                low_price = candles['low'].values
                
                # Calculate log high/low ratio
                log_hl_ratio = np.log(high_price / low_price)
                
                # Parkinson estimator
                n = 20  # Using 20 periods
                if len(log_hl_ratio) >= n:
                    parkinson = np.sqrt(1 / (4 * n * np.log(2)) * np.sum(log_hl_ratio[-n:] ** 2))
                    return parkinson * 100  # As percentage
                else:
                    parkinson = np.sqrt(1 / (4 * len(log_hl_ratio) * np.log(2)) * np.sum(log_hl_ratio ** 2))
                    return parkinson * 100
                    
            elif feature_name == 'volatility_gk':
                # Garman-Klass volatility estimator
                high_price = candles['high'].values
                low_price = candles['low'].values
                open_price = candles['open'].values
                close_price = candles['close'].values
                
                # Calculate log returns and ranges
                log_hl = np.log(high_price / low_price)
                log_co = np.log(close_price / open_price)
                
                # Garman-Klass estimator
                n = 20  # Using 20 periods
                if len(log_hl) >= n:
                    gk = np.sqrt(1 / n * np.sum(0.5 * log_hl[-n:] ** 2 - (2 * np.log(2) - 1) * log_co[-n:] ** 2))
                    return gk * 100  # As percentage
                else:
                    gk = np.sqrt(1 / len(log_hl) * np.sum(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2))
                    return gk * 100
                    
            elif feature_name == 'volatility_yang_zhang':
                # Yang-Zhang volatility estimator
                high_price = candles['high'].values
                low_price = candles['low'].values
                open_price = candles['open'].values
                close_price = candles['close'].values
                
                # Need to calculate overnight returns and intraday returns
                # Shift close price to get previous close
                prev_close = np.roll(close_price, 1)
                prev_close[0] = open_price[0]  # First element has no previous close
                
                # Calculate overnight returns (open / prev close)
                overnight_returns = np.log(open_price / prev_close)
                
                # Calculate intraday returns (close / open)
                intraday_returns = np.log(close_price / open_price)
                
                # Calculate Rogers-Satchell estimator
                rs = np.log(high_price / close_price) * np.log(high_price / open_price) + np.log(low_price / close_price) * np.log(low_price / open_price)
                
                # Yang-Zhang estimator
                n = 20  # Using 20 periods
                k = 0.34 / (1.34 + (n + 1) / (n - 1))
                
                if len(rs) >= n:
                    overnight_var = np.sum(overnight_returns[-n:] ** 2) / (n - 1)
                    intraday_var = np.sum(intraday_returns[-n:] ** 2) / (n - 1)
                    rs_var = np.sum(rs[-n:]) / n
                    
                    yz = np.sqrt(overnight_var + k * intraday_var + (1 - k) * rs_var)
                    return yz * 100  # As percentage
                else:
                    overnight_var = np.sum(overnight_returns ** 2) / (len(overnight_returns) - 1) if len(overnight_returns) > 1 else 0
                    intraday_var = np.sum(intraday_returns ** 2) / (len(intraday_returns) - 1) if len(intraday_returns) > 1 else 0
                    rs_var = np.sum(rs) / len(rs) if len(rs) > 0 else 0
                    
                    yz = np.sqrt(overnight_var + k * intraday_var + (1 - k) * rs_var)
                    return yz * 100
                    
            elif feature_name == 'volatility_bollinger_width':
                # Bollinger Band Width (20, 2)
                close_price = candles['close'].values

                upper = ta.volatility.bollinger_hband(pd.Series(close_price), window=20, window_dev=2)
                lower = ta.volatility.bollinger_lband(pd.Series(close_price), window=20, window_dev=2)
                middle = ta.volatility.bollinger_mavg(pd.Series(close_price), window=20)
                return (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1] * 100  # As percentage
                
            elif feature_name == 'volatility_chaikin':
                # Chaikin Volatility
                high_price = candles['high'].values
                low_price = candles['low'].values
                
                # Calculate high-low range
                hl_range = high_price - low_price
                
                # Calculate EMA of range
                ema10 = ta.trend.ema_indicator(pd.Series(hl_range), window=10)
                ema1 = ta.trend.ema_indicator(pd.Series(hl_range), window=1)
                
                # Chaikin Volatility
                chaikin = (ema10[-1] - ema1[-1]) / ema1[-1] * 100 if ema1[-1] > 0 else 0
                return chaikin
                
            else:
                # Unknown volatility feature
                logger.warning(f"Unknown volatility feature: {feature_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating volatility feature {feature_name}: {str(e)}")
            return None
    
    def _calculate_pattern_feature(self, feature_name: str, candles: pd.DataFrame) -> Optional[float]:
        """
        Calculate a pattern recognition feature.
        
        Args:
            feature_name: Feature name
            candles: Candlestick data
            
        Returns:
            Feature value or None if calculation fails
        """
        try:
            # Extract price arrays
            open_price = candles['open'].values
            high_price = candles['high'].values
            low_price = candles['low'].values
            close_price = candles['close'].values
            
            # Calculate feature based on name
            if feature_name == 'pattern_doji':
                # Doji pattern (open close nearly equal)
                recent_candles = candles.iloc[-5:]
                doji_score = 0
                
                for _, candle in recent_candles.iterrows():
                    body_size = abs(candle['close'] - candle['open'])
                    candle_range = candle['high'] - candle['low']
                    
                    if candle_range > 0:
                        body_ratio = body_size / candle_range
                        if body_ratio < 0.1:  # Body is less than 10% of range
                            doji_score += 1
                
                return doji_score / 5.0  # Normalize to 0-1 range
                
            elif feature_name == 'pattern_engulfing':
                # Bullish/Bearish Engulfing pattern
                if len(candles) < 2:
                    return 0.0
                    
                current = candles.iloc[-1]
                previous = candles.iloc[-2]
                
                current_body = abs(current['close'] - current['open'])
                previous_body = abs(previous['close'] - previous['open'])
                
                # Check if current body engulfs previous body
                current_bullish = current['close'] > current['open']
                previous_bullish = previous['close'] > previous['open']
                
                # Bullish engulfing
                if (not previous_bullish) and current_bullish:
                    if (current['open'] <= previous['close']) and (current['close'] >= previous['open']):
                        return 1.0  # Bullish signal
                
                # Bearish engulfing
                if previous_bullish and (not current_bullish):
                    if (current['open'] >= previous['close']) and (current['close'] <= previous['open']):
                        return -1.0  # Bearish signal
                
                return 0.0  # No engulfing pattern
                
            elif feature_name == 'pattern_hammer':
                # Hammer/Hanging Man pattern
                recent_candles = candles.iloc[-3:]
                hammer_score = 0
                
                for _, candle in recent_candles.iterrows():
                    body_size = abs(candle['close'] - candle['open'])
                    candle_range = candle['high'] - candle['low']
                    
                    if candle_range > 0 and body_size > 0:
                        # Check for long lower shadow
                        lower_shadow = min(candle['open'], candle['close']) - candle['low']
                        lower_ratio = lower_shadow / body_size
                        
                        # Check for little or no upper shadow
                        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
                        upper_ratio = upper_shadow / body_size
                        
                        if lower_ratio > 2.0 and upper_ratio < 0.5:
                            # Bullish if hammer forms after downtrend
                            if candle['close'] > candle['open']:
                                hammer_score += 1  # Bullish hammer
                            else:
                                hammer_score -= 1  # Potential hanging man (bearish)
                
                return hammer_score / 3.0  # Normalize to -1 to 1 range
                
            elif feature_name == 'pattern_shooting_star':
                # Shooting Star pattern
                recent_candles = candles.iloc[-3:]
                star_score = 0
                
                for _, candle in recent_candles.iterrows():
                    body_size = abs(candle['close'] - candle['open'])
                    candle_range = candle['high'] - candle['low']
                    
                    if candle_range > 0 and body_size > 0:
                        # Check for long upper shadow
                        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
                        upper_ratio = upper_shadow / body_size
                        
                        # Check for little or no lower shadow
                        lower_shadow = min(candle['open'], candle['close']) - candle['low']
                        lower_ratio = lower_shadow / body_size
                        
                        if upper_ratio > 2.0 and lower_ratio < 0.5:
                            star_score -= 1  # Bearish shooting star
                
                return star_score / 3.0  # Normalize to -1 to 0 range
                
            elif feature_name == 'pattern_marubozu':
                # Marubozu pattern (candle with no or very small shadows)
                recent_candles = candles.iloc[-3:]
                marubozu_score = 0
                
                for _, candle in recent_candles.iterrows():
                    body_size = abs(candle['close'] - candle['open'])
                    candle_range = candle['high'] - candle['low']
                    
                    if candle_range > 0 and body_size > 0:
                        body_ratio = body_size / candle_range
                        
                        if body_ratio > 0.95:  # Body is at least 95% of range
                            if candle['close'] > candle['open']:
                                marubozu_score += 1  # Bullish marubozu
                            else:
                                marubozu_score -= 1  # Bearish marubozu
                
                return marubozu_score / 3.0  # Normalize to -1 to 1 range
                
            elif feature_name == 'pattern_harami':
                # Harami pattern
                if len(candles) < 2:
                    return 0.0
                    
                current = candles.iloc[-1]
                previous = candles.iloc[-2]
                
                current_body = abs(current['close'] - current['open'])
                previous_body = abs(previous['close'] - previous['open'])
                
                if current_body < previous_body * 0.5:  # Current body is less than half of previous
                    current_bullish = current['close'] > current['open']
                    previous_bullish = previous['close'] > previous['open']
                    
                    # Check if current body is inside previous body
                    if previous_bullish:
                        if (current['open'] < previous['close']) and (current['close'] > previous['open']):
                            return -1.0  # Bearish harami
                    else:
                        if (current['open'] > previous['close']) and (current['close'] < previous['open']):
                            return 1.0  # Bullish harami
                
                return 0.0  # No harami pattern
                
            elif feature_name == 'pattern_morning_star':
                # Morning Star pattern
                if len(candles) < 3:
                    return 0.0
                    
                first = candles.iloc[-3]
                middle = candles.iloc[-2]
                last = candles.iloc[-1]
                
                # First day: long bearish candle
                first_bearish = first['close'] < first['open']
                first_body = abs(first['close'] - first['open'])
                
                # Second day: small body with gap down
                middle_body = abs(middle['close'] - middle['open'])
                gap_down = middle['high'] < first['close']
                
                # Third day: bullish candle that closes above middle of first candle
                last_bullish = last['close'] > last['open']
                first_midpoint = (first['open'] + first['close']) / 2
                closes_above_midpoint = last['close'] > first_midpoint
                
                if first_bearish and (middle_body < first_body * 0.5) and gap_down and last_bullish and closes_above_midpoint:
                    return 1.0  # Bullish morning star
                
                return 0.0  # No morning star pattern
                
            elif feature_name == 'pattern_evening_star':
                # Evening Star pattern
                if len(candles) < 3:
                    return 0.0
                    
                first = candles.iloc[-3]
                middle = candles.iloc[-2]
                last = candles.iloc[-1]
                
                # First day: long bullish candle
                first_bullish = first['close'] > first['open']
                first_body = abs(first['close'] - first['open'])
                
                # Second day: small body with gap up
                middle_body = abs(middle['close'] - middle['open'])
                gap_up = middle['low'] > first['close']
                
                # Third day: bearish candle that closes below middle of first candle
                last_bearish = last['close'] < last['open']
                first_midpoint = (first['open'] + first['close']) / 2
                closes_below_midpoint = last['close'] < first_midpoint
                
                if first_bullish and (middle_body < first_body * 0.5) and gap_up and last_bearish and closes_below_midpoint:
                    return -1.0  # Bearish evening star
                
                return 0.0  # No evening star pattern
                
            elif feature_name == 'pattern_piercing_line':
                # Piercing Line pattern
                if len(candles) < 2:
                    return 0.0
                    
                first = candles.iloc[-2]
                second = candles.iloc[-1]
                
                # First day: bearish candle
                first_bearish = first['close'] < first['open']
                
                # Second day: bullish candle that opens below first's low
                # and closes above middle of first candle
                second_bullish = second['close'] > second['open']
                opens_below = second['open'] < first['low']
                first_midpoint = (first['open'] + first['close']) / 2
                closes_above_midpoint = second['close'] > first_midpoint
                
                if first_bearish and second_bullish and opens_below and closes_above_midpoint:
                    return 1.0  # Bullish piercing line
                
                return 0.0  # No piercing line pattern
                
            elif feature_name == 'pattern_dark_cloud_cover':
                # Dark Cloud Cover pattern
                if len(candles) < 2:
                    return 0.0
                    
                first = candles.iloc[-2]
                second = candles.iloc[-1]
                
                # First day: bullish candle
                first_bullish = first['close'] > first['open']
                
                # Second day: bearish candle that opens above first's high
                # and closes below middle of first candle
                second_bearish = second['close'] < second['open']
                opens_above = second['open'] > first['high']
                first_midpoint = (first['open'] + first['close']) / 2
                closes_below_midpoint = second['close'] < first_midpoint
                
                if first_bullish and second_bearish and opens_above and closes_below_midpoint:
                    return -1.0  # Bearish dark cloud cover
                
                return 0.0  # No dark cloud cover pattern
                
            elif feature_name.startswith('pattern_cdl_'):
                pattern_name = feature_name[len('pattern_cdl_'):]
                try:
                    result = ta_candles.cdl_pattern(open_=open_price, high=high_price, low=low_price,
                                                  close=close_price, name=pattern_name)
                    value = result.iloc[-1] / 100.0
                    return value
                except Exception:
                    return 0.0
                else:
                    logger.warning(f"Unknown TALib pattern: CDL{pattern_name}")
                    return None
                    
            else:
                # Unknown pattern feature
                logger.warning(f"Unknown pattern feature: {feature_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating pattern feature {feature_name}: {str(e)}")
            return None
    
    def _calculate_trend_feature(self, feature_name: str, candles: pd.DataFrame) -> Optional[float]:
        """
        Calculate a trend feature.
        
        Args:
            feature_name: Feature name
            candles: Candlestick data
            
        Returns:
            Feature value or None if calculation fails
        """
        try:
            # Extract price arrays
            close_price = candles['close'].values
            high_price = candles['high'].values
            low_price = candles['low'].values
            
            # Calculate feature based on name
            if feature_name == 'trend_adx':
                # ADX (14)
                adx = ta.trend.adx(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return adx.iloc[-1]
                
            elif feature_name == 'trend_adx_slope':
                # Slope of ADX (14)
                adx = ta.trend.adx(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                
                if len(adx) >= 5:
                    slope = (adx[-1] - adx[-5]) / 5.0
                    return slope
                else:
                    return 0.0
                    
            elif feature_name == 'trend_aroon_up':
                # Aroon Up (14)
                aroon_up = ta.trend.aroon_up(pd.Series(close_price), window=14)
                return aroon_up.iloc[-1]
                
            elif feature_name == 'trend_aroon_down':
                # Aroon Down (14)
                aroon_down = ta.trend.aroon_down(pd.Series(close_price), window=14)
                return aroon_down.iloc[-1]
                
            elif feature_name == 'trend_aroon_oscillator':
                # Aroon Oscillator (14)
                aroon_up = ta.trend.aroon_up(pd.Series(close_price), window=14)
                aroon_down = ta.trend.aroon_down(pd.Series(close_price), window=14)
                aroon_osc = aroon_up - aroon_down
                return aroon_osc.iloc[-1]
                
            elif feature_name == 'trend_di_plus':
                # +DI (14)
                plus_di = ta.trend.adx_pos(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return plus_di.iloc[-1]
                
            elif feature_name == 'trend_di_minus':
                # -DI (14)
                minus_di = ta.trend.adx_neg(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return minus_di.iloc[-1]
                
            elif feature_name == 'trend_di_diff':
                # Difference between +DI and -DI
                plus_di = ta.trend.adx_pos(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                minus_di = ta.trend.adx_neg(pd.Series(high_price), pd.Series(low_price), pd.Series(close_price), window=14)
                return plus_di.iloc[-1] - minus_di.iloc[-1]
                
            elif feature_name == 'trend_ma_10_20_cross':
                # Crossover between 10 and 20 period MA
                ma10 = ta.trend.sma_indicator(pd.Series(close_price), window=10)
                ma20 = ta.trend.sma_indicator(pd.Series(close_price), window=20)
                
                if len(ma10) >= 2 and len(ma20) >= 2:
                    current_diff = ma10[-1] - ma20[-1]
                    previous_diff = ma10[-2] - ma20[-2]
                    
                    if current_diff > 0 and previous_diff <= 0:
                        return 1.0  # Bullish crossover
                    elif current_diff < 0 and previous_diff >= 0:
                        return -1.0  # Bearish crossover
                
                return 0.0  # No crossover
                
            elif feature_name == 'trend_ma_20_50_cross':
                # Crossover between 20 and 50 period MA
                ma20 = ta.trend.sma_indicator(pd.Series(close_price), window=20)
                ma50 = ta.trend.sma_indicator(pd.Series(close_price), window=50)
                
                if len(ma20) >= 2 and len(ma50) >= 2:
                    current_diff = ma20[-1] - ma50[-1]
                    previous_diff = ma20[-2] - ma50[-2]
                    
                    if current_diff > 0 and previous_diff <= 0:
                        return 1.0  # Bullish crossover
                    elif current_diff < 0 and previous_diff >= 0:
                        return -1.0  # Bearish crossover
                
                return 0.0  # No crossover
                
            elif feature_name == 'trend_ma_50_200_cross':
                # Crossover between 50 and 200 period MA (golden/death cross)
                ma50 = ta.trend.sma_indicator(pd.Series(close_price), window=50)
                ma200 = ta.trend.sma_indicator(pd.Series(close_price), window=200)
                
                if len(ma50) >= 2 and len(ma200) >= 2:
                    current_diff = ma50[-1] - ma200[-1]
                    previous_diff = ma50[-2] - ma200[-2]
                    
                    if current_diff > 0 and previous_diff <= 0:
                        return 1.0  # Bullish crossover (golden cross)
                    elif current_diff < 0 and previous_diff >= 0:
                        return -1.0  # Bearish crossover (death cross)
                
                return 0.0  # No crossover
                
            elif feature_name == 'trend_price_ma_20':
                # Price relative to 20-period MA
                ma20 = ta.trend.sma_indicator(pd.Series(close_price), window=20)
                return close_price[-1] / ma20.iloc[-1] - 1.0 if ma20.iloc[-1] > 0 else 0.0
                
            elif feature_name == 'trend_price_ma_50':
                # Price relative to 50-period MA
                ma50 = ta.trend.sma_indicator(pd.Series(close_price), window=50)
                return close_price[-1] / ma50.iloc[-1] - 1.0 if ma50.iloc[-1] > 0 else 0.0
                
            elif feature_name == 'trend_price_ma_200':
                # Price relative to 200-period MA
                ma200 = ta.trend.sma_indicator(pd.Series(close_price), window=200)
                return close_price[-1] / ma200.iloc[-1] - 1.0 if ma200.iloc[-1] > 0 else 0.0
                
            elif feature_name == 'trend_macd_histogram':
                # MACD histogram
                hist = ta.trend.macd_diff(pd.Series(close_price), window_fast=12, window_slow=26, window_sign=9)
                return hist.iloc[-1]
                
            elif feature_name == 'trend_macd_cross':
                # MACD line crossing Signal line
                macd = ta.trend.macd(pd.Series(close_price), window_fast=12, window_slow=26, window_sign=9)
                signal = ta.trend.macd_signal(pd.Series(close_price), window_fast=12, window_slow=26, window_sign=9)

                if len(macd) >= 2 and len(signal) >= 2:
                    current_diff = macd.iloc[-1] - signal.iloc[-1]
                    previous_diff = macd.iloc[-2] - signal.iloc[-2]

                    if current_diff > 0 and previous_diff <= 0:
                        return 1.0  # Bullish MACD crossover
                    elif current_diff < 0 and previous_diff >= 0:
                        return -1.0  # Bearish MACD crossover

                return 0.0  # No crossover

            else:
                # Unknown trend feature
                logger.warning(f"Unknown trend feature: {feature_name}")
                return None

        except Exception as e:
            logger.error(f"Error calculating trend feature {feature_name}: {str(e)}")
            return None
