#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Technical Indicators Feature Module

This module implements advanced technical indicators with optimized calculation
and adaptive parameterization for effective pattern recognition and signal generation.
"""

import numpy as np
import pandas as pd
from common.logger import get_logger
try:
    import ta  # type: ignore
    TA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ta = None  # type: ignore
    TA_AVAILABLE = False
    get_logger(__name__).warning(
        "ta library not available; technical indicators degraded"
    )
from numba import jit, cuda
from typing import Dict, List, Union, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import warnings

from common.utils import timeit, numpy_rolling_window
from common.exceptions import FeatureCalculationError
from common.constants import TECHNICAL_INDICATOR_PARAMS
from feature_service.features.base_feature import BaseFeature
from feature_service.features.volatility import (
    calculate_bollinger_bands as _volatility_bbands,
    calculate_atr as _volatility_atr,
)

logger = get_logger(__name__)


class TechnicalFeatures(BaseFeature):
    """Advanced Technical Indicators with optimized calculation and adaptive parameterization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the technical features generator with configurable parameters.
        
        Args:
            config: Configuration dictionary with optional parameters for indicators
        """
        super().__init__(config)
        self.config = config or {}
        self._validate_config()
        self._setup_indicator_params()
        self._initialize_calculation_engine()
        logger.info("Technical features module initialized with %d indicators", 
                   len(self.enabled_indicators))
    
    def _validate_config(self):
        """Validate the provided configuration"""
        if not isinstance(self.config.get('indicators', {}), dict):
            raise ValueError("Indicators configuration must be a dictionary")
            
        # Set defaults if not provided
        if 'indicators' not in self.config:
            self.config['indicators'] = TECHNICAL_INDICATOR_PARAMS
            
        if 'use_gpu' not in self.config:
            # Auto-detect GPU availability
            try:
                if cuda.is_available():
                    self.config['use_gpu'] = True
                    logger.info("GPU acceleration enabled for technical indicators")
                else:
                    self.config['use_gpu'] = False
                    logger.info("GPU not available, using CPU calculation")
            except:
                self.config['use_gpu'] = False
                logger.info("CUDA not configured, using CPU calculation")
    
    def _setup_indicator_params(self):
        """Configure indicator parameters from config"""
        self.indicator_params = self.config.get('indicators', {})
        self.enabled_indicators = [
            indicator for indicator, params in self.indicator_params.items()
            if params.get('enabled', True)
        ]
        
        # Ensure essential indicators are enabled
        essential_indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS']
        for indicator in essential_indicators:
            if indicator not in self.enabled_indicators:
                self.enabled_indicators.append(indicator)
                if indicator not in self.indicator_params:
                    self.indicator_params[indicator] = {'enabled': True}
                else:
                    self.indicator_params[indicator]['enabled'] = True
                    
                logger.info(f"Essential indicator {indicator} was auto-enabled")
    
    def _initialize_calculation_engine(self):
        """Initialize the calculation engine based on available hardware"""
        self.use_gpu = self.config.get('use_gpu', False)
        self.max_workers = self.config.get('max_workers', 
                                          min(32, (os.cpu_count() or 4) + 4))
        
        # Set up the thread pool for parallel calculation
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
    
    @timeit
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators based on input data.
        
        Args:
            data: DataFrame with OHLCV data
                Must contain 'open', 'high', 'low', 'close', 'volume' columns
                
        Returns:
            DataFrame with all calculated technical indicators
        """
        try:
            self._validate_input_data(data)
            
            # Create a copy to avoid modifying the original
            result = data.copy()
            
            # Calculate indicators in parallel when possible
            if len(self.enabled_indicators) > 1:
                # Group indicators by dependency to avoid calculation order issues
                independent_indicators = self._get_independent_indicators()
                dependent_indicators = [ind for ind in self.enabled_indicators 
                                      if ind not in independent_indicators]
                
                # Calculate independent indicators in parallel
                futures = {}
                for indicator in independent_indicators:
                    method_name = f"_calculate_{indicator.lower()}"
                    if hasattr(self, method_name):
                        futures[indicator] = self.executor.submit(
                            getattr(self, method_name), data
                        )
                
                # Collect results from independent indicators
                for indicator, future in futures.items():
                    try:
                        indicator_result = future.result()
                        if indicator_result is not None:
                            result = pd.concat([result, indicator_result], axis=1)
                    except Exception as e:
                        logger.error(f"Error calculating {indicator}: {str(e)}")
                        
                # Calculate dependent indicators sequentially
                for indicator in dependent_indicators:
                    method_name = f"_calculate_{indicator.lower()}"
                    if hasattr(self, method_name):
                        try:
                            indicator_result = getattr(self, method_name)(result)
                            if indicator_result is not None:
                                result = pd.concat([result, indicator_result], axis=1)
                        except Exception as e:
                            logger.error(f"Error calculating {indicator}: {str(e)}")
            else:
                # Calculate all indicators sequentially
                for indicator in self.enabled_indicators:
                    method_name = f"_calculate_{indicator.lower()}"
                    if hasattr(self, method_name):
                        try:
                            indicator_result = getattr(self, method_name)(result)
                            if indicator_result is not None:
                                result = pd.concat([result, indicator_result], axis=1)
                        except Exception as e:
                            logger.error(f"Error calculating {indicator}: {str(e)}")
            
            # Calculate derived meta-indicators
            self._calculate_meta_indicators(result)
            
            # Remove NaN values if configured
            if self.config.get('remove_nan', False):
                result.dropna(inplace=True)
                
            return result
            
        except Exception as e:
            logger.exception("Error calculating technical features: %s", str(e))
            raise FeatureCalculationError(f"Failed to calculate technical features: {str(e)}")
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate that input data has required columns"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Input data missing required columns: {', '.join(missing_columns)}")
        
        if len(data) < 50:
            logger.warning("Input data contains less than 50 rows, some indicators may not be reliable")
    
    def _get_independent_indicators(self) -> List[str]:
        """Get list of indicators that can be calculated independently"""
        # These indicators only depend on OHLCV data and not on other indicators
        independent_indicators = [
            'SMA', 'EMA', 'RSI', 'BBANDS', 'ATR', 'ADX', 'STOCH', 'CCI', 
            'MOM', 'ROC', 'WILLR', 'ULTOSC', 'SAR', 'TRANGE'
        ]
        return [ind for ind in independent_indicators if ind in self.enabled_indicators]
    
    def _calculate_meta_indicators(self, data: pd.DataFrame) -> None:
        """Calculate meta-indicators derived from multiple base indicators"""
        try:
            # Example: Calculate TTM Squeeze
            if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'keltner_upper', 'keltner_lower']):
                data['squeeze_on'] = (data['bb_upper'] <= data['keltner_upper']) & \
                                     (data['bb_lower'] >= data['keltner_lower'])
                data['squeeze_off'] = ~data['squeeze_on']
                
            # Example: Calculate Ichimoku Cloud relation
            if all(col in data.columns for col in ['ichimoku_a', 'ichimoku_b']):
                data['ichimoku_cloud_green'] = data['ichimoku_a'] > data['ichimoku_b']
                data['ichimoku_cloud_red'] = data['ichimoku_a'] < data['ichimoku_b']
                
            # Example: Indicator convergence/divergence
            if all(col in data.columns for col in ['rsi', 'macd']):
                # Normalized versions for comparison
                data['rsi_norm'] = (data['rsi'] - 50) / 50
                data['macd_norm'] = (data['macd'] - data['macd'].min()) / \
                                    (data['macd'].max() - data['macd'].min() + 1e-10) * 2 - 1
                data['rsi_macd_convergence'] = -(data['rsi_norm'] - data['macd_norm']).abs()
                
            # Add more meta-indicators as needed
            
        except Exception as e:
            logger.error(f"Error calculating meta-indicators: {str(e)}")
    
    #### Individual Indicator Calculation Methods ####
    
    def _calculate_sma(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Simple Moving Average with multiple periods"""
        try:
            result = pd.DataFrame(index=data.index)
            periods = self.indicator_params.get('SMA', {}).get('periods', [5, 10, 20, 50, 200])
            
            for period in periods:
                if period > 0:
                    col_name = f'sma_{period}'
                    result[col_name] = ta.sma(data['close'], length=period)
            
            return result
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_ema(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Exponential Moving Average with multiple periods"""
        try:
            result = pd.DataFrame(index=data.index)
            periods = self.indicator_params.get('EMA', {}).get('periods', [5, 10, 20, 50, 200])
            
            for period in periods:
                if period > 0:
                    col_name = f'ema_{period}'
                    result[col_name] = ta.ema(data['close'], length=period)
            
            return result
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('MACD', {})
            
            fastperiod = params.get('fastperiod', 12)
            slowperiod = params.get('slowperiod', 26)
            signalperiod = params.get('signalperiod', 9)
            
            macd_df = ta.macd(data['close'], fast=fastperiod, slow=slowperiod, signal=signalperiod)
            macd = macd_df.iloc[:, 0]
            macdsignal = macd_df.iloc[:, 2]
            macdhist = macd_df.iloc[:, 1]

            result['macd'] = macd
            result['macdsignal'] = macdsignal
            result['macdhist'] = macdhist
            
            # Add MACD crossover signals
            result['macd_bullish'] = (macd > macdsignal) & (macd.shift(1) <= macdsignal.shift(1))
            result['macd_bearish'] = (macd < macdsignal) & (macd.shift(1) >= macdsignal.shift(1))
            
            # Add MACD zero line crossover
            result['macd_bull_zero'] = (macd > 0) & (macd.shift(1) <= 0)
            result['macd_bear_zero'] = (macd < 0) & (macd.shift(1) >= 0)
            
            return result
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI (Relative Strength Index) with multiple periods"""
        try:
            result = pd.DataFrame(index=data.index)
            periods = self.indicator_params.get('RSI', {}).get('periods', [6, 14, 21])
            
            for period in periods:
                if period > 0:
                    col_name = f'rsi_{period}'
                    result[col_name] = ta.rsi(data['close'], length=period)
            
            # Set the default period as main RSI
            default_period = self.indicator_params.get('RSI', {}).get('default_period', 14)
            result['rsi'] = result.get(f'rsi_{default_period}', result[f'rsi_{periods[0]}'])
            
            # Add RSI overbought/oversold indicators
            overbought = self.indicator_params.get('RSI', {}).get('overbought', 70)
            oversold = self.indicator_params.get('RSI', {}).get('oversold', 30)
            
            result['rsi_overbought'] = result['rsi'] > overbought
            result['rsi_oversold'] = result['rsi'] < oversold
            
            # Add RSI divergence detection (price/RSI divergence)
            self._add_rsi_divergence(result, data)
            
            return result
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _add_rsi_divergence(self, result: pd.DataFrame, data: pd.DataFrame) -> None:
        """Add RSI divergence detection"""
        try:
            # Get the main RSI column
            rsi_col = 'rsi'
            if rsi_col not in result.columns:
                return
                
            # Find local extrema in price
            price_highs = self._find_local_extrema(data['close'], True)
            price_lows = self._find_local_extrema(data['close'], False)
            
            # Find local extrema in RSI
            rsi_highs = self._find_local_extrema(result[rsi_col], True)
            rsi_lows = self._find_local_extrema(result[rsi_col], False)
            
            # Initialize divergence columns
            result['rsi_bull_div'] = False
            result['rsi_bear_div'] = False
            
            # Bullish divergence: Price making lower lows but RSI making higher lows
            for i in range(1, len(price_lows)):
                if i >= len(price_lows) or i >= len(rsi_lows):
                    continue
                    
                curr_idx = price_lows[i]
                prev_idx = price_lows[i-1]
                
                if curr_idx in rsi_lows and prev_idx in rsi_lows:
                    # Check if price made lower low
                    price_lower = data.loc[curr_idx, 'close'] < data.loc[prev_idx, 'close']
                    # Check if RSI made higher low
                    rsi_higher = result.loc[curr_idx, rsi_col] > result.loc[prev_idx, rsi_col]
                    
                    if price_lower and rsi_higher:
                        result.loc[curr_idx, 'rsi_bull_div'] = True
            
            # Bearish divergence: Price making higher highs but RSI making lower highs
            for i in range(1, len(price_highs)):
                if i >= len(price_highs) or i >= len(rsi_highs):
                    continue
                    
                curr_idx = price_highs[i]
                prev_idx = price_highs[i-1]
                
                if curr_idx in rsi_highs and prev_idx in rsi_highs:
                    # Check if price made higher high
                    price_higher = data.loc[curr_idx, 'close'] > data.loc[prev_idx, 'close']
                    # Check if RSI made lower high
                    rsi_lower = result.loc[curr_idx, rsi_col] < result.loc[prev_idx, rsi_col]
                    
                    if price_higher and rsi_lower:
                        result.loc[curr_idx, 'rsi_bear_div'] = True
                        
        except Exception as e:
            logger.error(f"Error calculating RSI divergence: {str(e)}")
    
    def _find_local_extrema(self, series: pd.Series, find_max: bool = True, window: int = 5) -> List[int]:
        """Find local extrema (maxima or minima) in a time series"""
        indices = []
        
        if len(series) < 2 * window + 1:
            return indices
            
        for i in range(window, len(series) - window):
            # Extract window of values centered at current point
            values = series.iloc[i - window: i + window + 1]
            center_value = values.iloc[window]
            
            # Check if center is a local extrema
            if find_max:
                if center_value == values.max():
                    indices.append(series.index[i])
            else:
                if center_value == values.min():
                    indices.append(series.index[i])
                    
        return indices
    
    def _calculate_bbands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('BBANDS', {})
            
            period = params.get('period', 20)
            nbdevup = params.get('nbdevup', 2)
            nbdevdn = params.get('nbdevdn', 2)
            matype = params.get('matype', 0)  # 0=SMA
            
            bb = ta.bbands(data['close'], length=period, std=nbdevup)
            upperband = bb.iloc[:, 0]
            middleband = bb.iloc[:, 1]
            lowerband = bb.iloc[:, 2]
            
            result['bb_upper'] = upperband
            result['bb_middle'] = middleband
            result['bb_lower'] = lowerband
            
            # Calculate bandwidth and %B
            result['bb_bandwidth'] = (upperband - lowerband) / middleband
            result['bb_percent_b'] = (data['close'] - lowerband) / (upperband - lowerband)
            
            # Bollinger Band squeeze and expansion signals
            result['bb_squeeze'] = result['bb_bandwidth'] < result['bb_bandwidth'].rolling(window=50).quantile(0.2)
            result['bb_expansion'] = result['bb_bandwidth'] > result['bb_bandwidth'].rolling(window=50).quantile(0.8)
            
            # Price relation to bands
            result['close_above_upper'] = data['close'] > upperband
            result['close_below_lower'] = data['close'] < lowerband
            result['close_between_bands'] = (data['close'] >= lowerband) & (data['close'] <= upperband)
            
            return result
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range"""
        try:
            result = pd.DataFrame(index=data.index)
            periods = self.indicator_params.get('ATR', {}).get('periods', [14, 21])
            
            for period in periods:
                if period > 0:
                    col_name = f'atr_{period}'
                    result[col_name] = ta.atr(high=data['high'], low=data['low'], close=data['close'], length=period)
            
            # Set the default period as main ATR
            default_period = self.indicator_params.get('ATR', {}).get('default_period', 14)
            result['atr'] = result.get(f'atr_{default_period}', result[f'atr_{periods[0]}'])
            
            # Normalize ATR as percentage of price
            result['atr_percent'] = result['atr'] / data['close'] * 100
            
            # Volatility regime classification
            result['volatility_regime'] = pd.cut(
                result['atr_percent'],
                bins=[-float('inf'), 1, 2, 3, float('inf')],
                labels=['low', 'normal', 'high', 'extreme']
            )
            
            return result
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_adx(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('ADX', {})
            
            period = params.get('period', 14)
            
            adx_df = ta.adx(high=data['high'], low=data['low'], close=data['close'], length=period)
            result['adx'] = adx_df['ADX_{}'.format(period)]
            result['plus_di'] = adx_df['DMP_{}'.format(period)]
            result['minus_di'] = adx_df['DMN_{}'.format(period)]
            
            # Trend strength classification
            result['trend_strength'] = pd.cut(
                result['adx'],
                bins=[-float('inf'), 20, 40, 60, float('inf')],
                labels=['weak', 'moderate', 'strong', 'extreme']
            )
            
            # ADX crossover signals
            result['di_bullish'] = (result['plus_di'] > result['minus_di']) & (result['plus_di'].shift(1) <= result['minus_di'].shift(1))
            result['di_bearish'] = (result['plus_di'] < result['minus_di']) & (result['plus_di'].shift(1) >= result['minus_di'].shift(1))
            
            return result
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_stoch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('STOCH', {})
            
            fastk_period = params.get('fastk_period', 5)
            slowk_period = params.get('slowk_period', 3)
            slowk_matype = params.get('slowk_matype', 0)
            slowd_period = params.get('slowd_period', 3)
            slowd_matype = params.get('slowd_matype', 0)
            
            stoch_df = ta.stoch(high=data['high'], low=data['low'], close=data['close'],
                                k=fastk_period, d=slowd_period, smooth_k=slowk_period)
            slowk = stoch_df.iloc[:, 0]
            slowd = stoch_df.iloc[:, 1]
            
            result['stoch_k'] = slowk
            result['stoch_d'] = slowd
            
            # Overbought/oversold signals
            overbought = params.get('overbought', 80)
            oversold = params.get('oversold', 20)
            
            result['stoch_overbought'] = (result['stoch_k'] > overbought) & (result['stoch_d'] > overbought)
            result['stoch_oversold'] = (result['stoch_k'] < oversold) & (result['stoch_d'] < oversold)
            
            # Stochastic crossover signals
            result['stoch_bullish'] = (result['stoch_k'] > result['stoch_d']) & (result['stoch_k'].shift(1) <= result['stoch_d'].shift(1))
            result['stoch_bearish'] = (result['stoch_k'] < result['stoch_d']) & (result['stoch_k'].shift(1) >= result['stoch_d'].shift(1))
            
            # Stochastic hook patterns
            result['stoch_bull_hook'] = (result['stoch_k'] < oversold) & (result['stoch_k'] > result['stoch_k'].shift(1)) & (result['stoch_k'].shift(1) < result['stoch_k'].shift(2))
            result['stoch_bear_hook'] = (result['stoch_k'] > overbought) & (result['stoch_k'] < result['stoch_k'].shift(1)) & (result['stoch_k'].shift(1) > result['stoch_k'].shift(2))
            
            return result
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_cci(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Commodity Channel Index"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('CCI', {})
            
            period = params.get('period', 14)
            
            result['cci'] = ta.cci(high=data['high'], low=data['low'], close=data['close'], length=period)
            
            # Overbought/oversold signals
            overbought = params.get('overbought', 100)
            oversold = params.get('oversold', -100)
            extreme_overbought = params.get('extreme_overbought', 200)
            extreme_oversold = params.get('extreme_oversold', -200)
            
            result['cci_overbought'] = result['cci'] > overbought
            result['cci_oversold'] = result['cci'] < oversold
            result['cci_extreme_overbought'] = result['cci'] > extreme_overbought
            result['cci_extreme_oversold'] = result['cci'] < extreme_oversold
            
            # CCI zero line crossover
            result['cci_bull_zero'] = (result['cci'] > 0) & (result['cci'].shift(1) <= 0)
            result['cci_bear_zero'] = (result['cci'] < 0) & (result['cci'].shift(1) >= 0)
            
            return result
        except Exception as e:
            logger.error(f"Error calculating CCI: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_ichimoku(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('ICHIMOKU', {})
            
            tenkan_period = params.get('tenkan', 9)
            kijun_period = params.get('kijun', 26)
            senkou_span_b_period = params.get('senkou_span_b', 52)
            displacement = params.get('displacement', 26)
            
            # Calculate Tenkan-sen (Conversion Line)
            high_tenkan = data['high'].rolling(window=tenkan_period).max()
            low_tenkan = data['low'].rolling(window=tenkan_period).min()
            result['ichimoku_tenkan'] = (high_tenkan + low_tenkan) / 2
            
            # Calculate Kijun-sen (Base Line)
            high_kijun = data['high'].rolling(window=kijun_period).max()
            low_kijun = data['low'].rolling(window=kijun_period).min()
            result['ichimoku_kijun'] = (high_kijun + low_kijun) / 2
            
            # Calculate Senkou Span A (Leading Span A)
            result['ichimoku_senkou_a'] = ((result['ichimoku_tenkan'] + result['ichimoku_kijun']) / 2).shift(displacement)
            
            # Calculate Senkou Span B (Leading Span B)
            high_senkou = data['high'].rolling(window=senkou_span_b_period).max()
            low_senkou = data['low'].rolling(window=senkou_span_b_period).min()
            result['ichimoku_senkou_b'] = ((high_senkou + low_senkou) / 2).shift(displacement)
            
            # Calculate Chikou Span (Lagging Span)
            result['ichimoku_chikou'] = data['close'].shift(-displacement)
            
            # Set more convenient names for cloud components
            result['ichimoku_a'] = result['ichimoku_senkou_a']
            result['ichimoku_b'] = result['ichimoku_senkou_b']
            
            # Signal generation
            # Tenkan-Kijun Cross (TK Cross)
            result['ichimoku_tk_bullish'] = (result['ichimoku_tenkan'] > result['ichimoku_kijun']) & (result['ichimoku_tenkan'].shift(1) <= result['ichimoku_kijun'].shift(1))
            result['ichimoku_tk_bearish'] = (result['ichimoku_tenkan'] < result['ichimoku_kijun']) & (result['ichimoku_tenkan'].shift(1) >= result['ichimoku_kijun'].shift(1))
            
            # Price relative to cloud
            result['price_above_cloud'] = (data['close'] > result['ichimoku_a']) & (data['close'] > result['ichimoku_b'])
            result['price_below_cloud'] = (data['close'] < result['ichimoku_a']) & (data['close'] < result['ichimoku_b'])
            result['price_in_cloud'] = ~(result['price_above_cloud'] | result['price_below_cloud'])
            
            # Cloud color (green when A > B, red when B > A)
            result['green_cloud'] = result['ichimoku_a'] > result['ichimoku_b']
            result['red_cloud'] = result['ichimoku_a'] < result['ichimoku_b']
            
            return result
        except Exception as e:
            logger.error(f"Error calculating Ichimoku: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price"""
        try:
            result = pd.DataFrame(index=data.index)
            
            # Check if we have the required data
            if not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
                logger.warning("Missing required columns for VWAP calculation")
                return result
                
            # Calculate typical price
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            
            # Calculate cumulative values
            cumulative_tp_vol = (typical_price * data['volume']).cumsum()
            cumulative_vol = data['volume'].cumsum()
            
            # VWAP calculation
            result['vwap'] = cumulative_tp_vol / cumulative_vol
            
            # Calculate VWAP bands
            std_dev = (typical_price - result['vwap']).rolling(window=14).std()
            result['vwap_upper_1'] = result['vwap'] + std_dev
            result['vwap_lower_1'] = result['vwap'] - std_dev
            result['vwap_upper_2'] = result['vwap'] + 2 * std_dev
            result['vwap_lower_2'] = result['vwap'] - 2 * std_dev
            
            # Price relationship to VWAP
            result['close_above_vwap'] = data['close'] > result['vwap']
            result['close_below_vwap'] = data['close'] < result['vwap']
            
            # VWAP crossovers
            result['cross_above_vwap'] = (data['close'] > result['vwap']) & (data['close'].shift(1) <= result['vwap'].shift(1))
            result['cross_below_vwap'] = (data['close'] < result['vwap']) & (data['close'].shift(1) >= result['vwap'].shift(1))
            
            return result
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_supertrend(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate SuperTrend indicator"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('SUPERTREND', {})
            
            period = params.get('period', 10)
            multiplier = params.get('multiplier', 3)
            
            # Ensure we have ATR available
            if 'atr' not in data.columns:
                # Calculate ATR if not already present
                atr = ta.atr(high=data['high'], low=data['low'], close=data['close'], length=period)
            else:
                atr = data['atr'].values
            
            # Basic bands
            basic_upper = (data['high'] + data['low']) / 2 + multiplier * atr
            basic_lower = (data['high'] + data['low']) / 2 - multiplier * atr
            
            # Initialize SuperTrend columns
            result['supertrend'] = 0.0
            result['supertrend_upper'] = basic_upper
            result['supertrend_lower'] = basic_lower
            result['supertrend_direction'] = 0  # 1 for up, -1 for down
            
            # SuperTrend calculation - first value initialization
            if data['close'].iloc[0] <= basic_upper.iloc[0]:
                result.loc[data.index[0], 'supertrend'] = basic_upper.iloc[0]
                result.loc[data.index[0], 'supertrend_direction'] = -1
            else:
                result.loc[data.index[0], 'supertrend'] = basic_lower.iloc[0]
                result.loc[data.index[0], 'supertrend_direction'] = 1
            
            # Iterative calculation
            for i in range(1, len(data)):
                curr_idx = data.index[i]
                prev_idx = data.index[i-1]
                
                # Direction based on previous value
                prev_direction = result.loc[prev_idx, 'supertrend_direction']
                
                # Upper band
                if basic_upper.iloc[i] < result.loc[prev_idx, 'supertrend_upper'] or data['close'].iloc[i-1] > result.loc[prev_idx, 'supertrend_upper']:
                    result.loc[curr_idx, 'supertrend_upper'] = basic_upper.iloc[i]
                else:
                    result.loc[curr_idx, 'supertrend_upper'] = result.loc[prev_idx, 'supertrend_upper']
                
                # Lower band
                if basic_lower.iloc[i] > result.loc[prev_idx, 'supertrend_lower'] or data['close'].iloc[i-1] < result.loc[prev_idx, 'supertrend_lower']:
                    result.loc[curr_idx, 'supertrend_lower'] = basic_lower.iloc[i]
                else:
                    result.loc[curr_idx, 'supertrend_lower'] = result.loc[prev_idx, 'supertrend_lower']
                
                # SuperTrend value
                if prev_direction == 1 and data['close'].iloc[i] <= result.loc[curr_idx, 'supertrend_upper']:
                    result.loc[curr_idx, 'supertrend'] = result.loc[curr_idx, 'supertrend_upper']
                    result.loc[curr_idx, 'supertrend_direction'] = -1
                elif prev_direction == -1 and data['close'].iloc[i] >= result.loc[curr_idx, 'supertrend_lower']:
                    result.loc[curr_idx, 'supertrend'] = result.loc[curr_idx, 'supertrend_lower']
                    result.loc[curr_idx, 'supertrend_direction'] = 1
                elif prev_direction == 1:
                    result.loc[curr_idx, 'supertrend'] = result.loc[curr_idx, 'supertrend_lower']
                    result.loc[curr_idx, 'supertrend_direction'] = 1
                else:
                    result.loc[curr_idx, 'supertrend'] = result.loc[curr_idx, 'supertrend_upper']
                    result.loc[curr_idx, 'supertrend_direction'] = -1
            
            # Signal generation
            result['supertrend_uptrend'] = result['supertrend_direction'] == 1
            result['supertrend_downtrend'] = result['supertrend_direction'] == -1
            
            # Trend change signals
            result['supertrend_buy'] = (result['supertrend_direction'] == 1) & (result['supertrend_direction'].shift(1) == -1)
            result['supertrend_sell'] = (result['supertrend_direction'] == -1) & (result['supertrend_direction'].shift(1) == 1)
            
            return result
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_keltner(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channels"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('KELTNER', {})
            
            period = params.get('period', 20)
            multiplier = params.get('multiplier', 2)
            
            # Calculate EMA
            ema = ta.ema(data['close'], length=period)
            
            # Ensure we have ATR
            if 'atr' not in data.columns:
                atr = ta.atr(high=data['high'], low=data['low'], close=data['close'], length=period)
            else:
                atr = data['atr'].values
            
            # Calculate Keltner Channels
            result['keltner_middle'] = ema
            result['keltner_upper'] = ema + multiplier * atr
            result['keltner_lower'] = ema - multiplier * atr
            
            # Price position relative to channels
            result['price_above_keltner'] = data['close'] > result['keltner_upper']
            result['price_below_keltner'] = data['close'] < result['keltner_lower']
            result['price_in_keltner'] = (data['close'] >= result['keltner_lower']) & (data['close'] <= result['keltner_upper'])
            
            # Channel width as volatility measure
            result['keltner_width'] = (result['keltner_upper'] - result['keltner_lower']) / result['keltner_middle']

            return result
        except Exception as e:
            logger.error(f"Error calculating Keltner Channels: {str(e)}")
            return pd.DataFrame(index=data.index)

    def _calculate_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        try:
            result = pd.DataFrame(index=data.index)
            result["obv"] = calculate_obv(data["close"], data["volume"])
            return result
        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            return pd.DataFrame(index=data.index)

    # Add more indicator calculation methods as needed

# Helper functions that might be implemented if needed by the technical indicators
@jit(nopython=True)
def _numba_rolling_max(arr, window):
    """Numba-optimized rolling window maximum calculation"""
    n = len(arr)
    result = np.empty(n)
    result[:] = np.nan
    
    for i in range(window - 1, n):
        result[i] = np.max(arr[i - window + 1:i + 1])
    
    return result

@jit(nopython=True)
def _numba_rolling_min(arr, window):
    """Numba-optimized rolling window minimum calculation"""
    n = len(arr)
    result = np.empty(n)
    result[:] = np.nan
    
    for i in range(window - 1, n):
        result[i] = np.min(arr[i - window + 1:i + 1])
    
    return result

def calculate_technical_features(data, config=None):
    """
    Convenience function to calculate technical features from OHLCV data.
    
    Args:
        data: DataFrame with OHLCV data
        config: Optional configuration for technical features
        
    Returns:
        DataFrame with calculated technical indicators
    """
    calculator = TechnicalFeatures(config)
    return calculator.calculate_features(data)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standalone RSI calculation."""
    return ta.rsi(series, length=period)


def calculate_macd(series: pd.Series,
                   fastperiod: int = 12,
                   slowperiod: int = 26,
                   signalperiod: int = 9) -> pd.DataFrame:
    """Standalone MACD calculation."""
    df = ta.macd(series, fast=fastperiod, slow=slowperiod, signal=signalperiod)
    df.columns = ['macd', 'macdhist', 'macdsignal']
    return df


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Standalone ADX calculation."""
    adx_df = ta.adx(high=high, low=low, close=close, length=period)
    adx_df.columns = [f"ADX_{period}", f"DMP_{period}", f"DMN_{period}"]
    return adx_df


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wrapper for ATR calculation using the volatility module."""
    return _volatility_atr(high, low, close, period)

  
def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Standalone Stochastic Oscillator calculation."""

    k = ta.stoch(high, low, close, k_period, d_period)
    df = pd.DataFrame({
        'stoch_k': k,
        'stoch_d': k.rolling(d_period).mean()
    })
    return df


def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Wrapper for Bollinger Bands using the volatility module."""
    upper, middle, lower = _volatility_bbands(close, window=period, num_std_dev=std_dev)
    return pd.DataFrame({'bb_upper': upper, 'bb_middle': middle, 'bb_lower': lower})


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""

    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = 0
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    return obv


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Standalone Bollinger Band calculation."""
    bb = ta.bbands(close, length=period, std=std)
    return bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Standalone Average True Range calculation."""
    return ta.atr(high=high, low=low, close=close, length=period)


def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 14) -> pd.Series:

    """Basic divergence detection between price and an indicator."""
    diff_price = price.diff()
    diff_ind = indicator.diff()
    divergence = (diff_price * diff_ind) < 0
    return divergence.rolling(lookback).sum() > 0


__all__ = [
    'TechnicalFeatures',
    'calculate_technical_features',
    'calculate_rsi',
    'calculate_macd',
    'calculate_adx',
    'calculate_stochastic',
    'calculate_obv',
    'calculate_bollinger_bands',
    'calculate_atr',
    'detect_divergence',

]

# Module initialization
import os
if __name__ == "__main__":
    from common.logger import setup_logging
    setup_logging()
    
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    n = 1000
    dates = pd.date_range('2020-01-01', periods=n, freq='1h')
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, n).cumsum(),
        'high': np.random.normal(100, 10, n).cumsum() + np.random.normal(1, 0.5, n),
        'low': np.random.normal(100, 10, n).cumsum() - np.random.normal(1, 0.5, n),
        'close': np.random.normal(100, 10, n).cumsum(),
        'volume': np.random.normal(1000, 500, n)
    }, index=dates)
    
    # Make sure high, low are correctly ordered
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Calculate technical features
    tech_features = calculate_technical_features(data)
    
    # Log first few rows with key indicators for debugging
    logger.info(tech_features.iloc[-5:][['sma_20', 'ema_20', 'rsi', 'macd', 'bb_upper', 'bb_lower']])
