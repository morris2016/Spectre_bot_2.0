#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Volatility Features Module

This module implements advanced volatility indicators and metrics with optimized calculation
and adaptive parameterization for effective market regime detection and risk management.
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
        "ta library not available; volatility indicators degraded"
    )
from numba import jit
from typing import Dict, List, Union, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import warnings
import math


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Average True Range."""
    return ta.atr(high=high, low=low, close=close, length=period)


def calculate_bollinger_bands(prices: pd.Series,
                              period: int = 20,
                              std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return Bollinger Bands."""
    bb = ta.bbands(prices, length=period, std=std)
    return bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]




from common.utils import timeit, numpy_rolling_window
from common.exceptions import FeatureCalculationError
from common.constants import VOLATILITY_INDICATOR_PARAMS
from feature_service.features.base_feature import BaseFeature


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Simple Average True Range calculation."""
    return ta.atr(high=high, low=low, close=close, length=period)


def calculate_bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    bb = ta.bbands(close, length=period, std=std)
    return pd.DataFrame({
        'bb_lower': bb.iloc[:, 0],
        'bb_middle': bb.iloc[:, 1],
        'bb_upper': bb.iloc[:, 2],
    })


def calculate_historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate historical volatility of a price series."""
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(window=period).std()
    return vol * np.sqrt(252) * 100


def calculate_volatility_ratio(close: pd.Series, short_period: int = 5, long_period: int = 20) -> pd.Series:
    """Calculate ratio of short-term to long-term volatility."""
    log_returns = np.log(close / close.shift(1))
    short_vol = log_returns.rolling(window=short_period).std()
    long_vol = log_returns.rolling(window=long_period).std()
    return short_vol / long_vol


def volatility_expansion_indicator(close: pd.Series, period: int = 20) -> pd.Series:
    """Simple volatility expansion indicator based on ATR changes."""
    atr = calculate_atr(close, close, close, period)
    atr_mean = atr.rolling(window=period).mean()
    return atr / atr_mean
def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Simple ATR calculation used by strategies."""
    return ta.atr(high=high, low=low, close=close, length=period)


def calculate_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return Bollinger Bands (upper, middle, lower)."""
    bands = ta.volatility.BollingerBands(close=close, window=window, window_dev=num_std_dev)
    return bands.bollinger_hband(), bands.bollinger_mavg(), bands.bollinger_lband()


def calculate_historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Standalone historical volatility as annualized standard deviation."""
    log_returns = np.log(close / close.shift(1))
    vol = log_returns.rolling(window=period).std()
    return vol * np.sqrt(252) * 100


def calculate_volatility_ratio(
    close: pd.Series, short_period: int = 5, long_period: int = 20
) -> pd.Series:
    """Return the ratio of short-term to long-term volatility."""
    log_returns = np.log(close / close.shift(1))
    short_vol = log_returns.rolling(window=short_period).std()
    long_vol = log_returns.rolling(window=long_period).std()
    return short_vol / long_vol

logger = get_logger(__name__)


class VolatilityFeatures(BaseFeature):
    """Advanced Volatility Features with optimized calculation for market regime detection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the volatility features generator with configurable parameters.
        
        Args:
            config: Configuration dictionary with optional parameters for volatility indicators
        """
        super().__init__(config)
        self.config = config or {}
        self._validate_config()
        self._setup_indicator_params()
        self._initialize_calculation_engine()
        logger.info("Volatility features module initialized with %d indicators", 
                   len(self.enabled_indicators))
    
    def _validate_config(self):
        """Validate the provided configuration"""
        if not isinstance(self.config.get('indicators', {}), dict):
            raise ValueError("Indicators configuration must be a dictionary")
            
        # Set defaults if not provided
        if 'indicators' not in self.config:
            self.config['indicators'] = VOLATILITY_INDICATOR_PARAMS
            
        if 'use_gpu' not in self.config:
            # Auto-detect GPU availability
            try:
                from numba import cuda
                if cuda.is_available():
                    self.config['use_gpu'] = True
                    logger.info("GPU acceleration enabled for volatility indicators")
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
        essential_indicators = ['ATR', 'HISTORICAL_VOL', 'VOLATILITY_RATIO', 'BOLLINGER_WIDTH']
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
        Calculate all volatility indicators based on input data.
        
        Args:
            data: DataFrame with OHLCV data
                Must contain 'open', 'high', 'low', 'close', 'volume' columns
                
        Returns:
            DataFrame with all calculated volatility indicators
        """
        try:
            self._validate_input_data(data)
            
            # Create a copy to avoid modifying the original
            result = pd.DataFrame(index=data.index)
            
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
                            indicator_result = getattr(self, method_name)(result, data)
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
                            indicator_result = getattr(self, method_name)(data)
                            if indicator_result is not None:
                                result = pd.concat([result, indicator_result], axis=1)
                        except Exception as e:
                            logger.error(f"Error calculating {indicator}: {str(e)}")
            
            # Calculate market regime based on volatility metrics
            self._calculate_market_regime(result)
            
            # Remove NaN values if configured
            if self.config.get('remove_nan', False):
                result.dropna(inplace=True)
                
            return result
            
        except Exception as e:
            logger.exception("Error calculating volatility features: %s", str(e))
            raise FeatureCalculationError(f"Failed to calculate volatility features: {str(e)}")
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate that input data has required columns"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Input data missing required columns: {', '.join(missing_columns)}")
        
        if len(data) < 50:
            logger.warning("Input data contains less than 50 rows, some volatility indicators may not be reliable")
    
    def _get_independent_indicators(self) -> List[str]:
        """Get list of indicators that can be calculated independently"""
        # These indicators only depend on OHLCV data and not on other indicators
        independent_indicators = [
            'ATR', 'HISTORICAL_VOL', 'TRUE_RANGE', 'PARKINSONS_VOL',
            'GARMAN_KLASS_VOL', 'YANG_ZHANG_VOL', 'BOLLINGER_WIDTH'
        ]
        return [ind for ind in independent_indicators if ind in self.enabled_indicators]
    
    def _calculate_market_regime(self, result: pd.DataFrame) -> None:
        """Calculate market regime based on volatility indicators"""
        try:
            # Requires multiple indicators to be present
            required_indicators = ['historical_vol', 'atr_percent', 'bollinger_width']
            if not all(ind in result.columns for ind in required_indicators):
                logger.warning("Not all required indicators available for market regime calculation")
                return
                
            # Create a composite volatility score
            result['volatility_composite'] = 0.0
            
            # Add historical volatility component (normalized)
            vol_lookback = 100  # days/periods for percentile
            if len(result) > vol_lookback:
                vol_series = result['historical_vol'].dropna()
                vol_percentiles = vol_series.rolling(window=vol_lookback).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
                result['vol_percentile'] = vol_percentiles
                result['volatility_composite'] += result['vol_percentile'].fillna(0.5)
            
            # Add ATR percent component
            if 'atr_percent' in result.columns:
                atr_lookback = 100  # days/periods for percentile
                if len(result) > atr_lookback:
                    atr_series = result['atr_percent'].dropna()
                    atr_percentiles = atr_series.rolling(window=atr_lookback).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                    )
                    result['atr_percentile'] = atr_percentiles
                    result['volatility_composite'] += result['atr_percentile'].fillna(0.5)
            
            # Add Bollinger width component
            if 'bollinger_width' in result.columns:
                bbw_lookback = 100  # days/periods for percentile
                if len(result) > bbw_lookback:
                    bbw_series = result['bollinger_width'].dropna()
                    bbw_percentiles = bbw_series.rolling(window=bbw_lookback).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                    )
                    result['bbw_percentile'] = bbw_percentiles
                    result['volatility_composite'] += result['bbw_percentile'].fillna(0.5)
            
            # Normalize the composite score
            n_indicators = sum([1 for col in ['vol_percentile', 'atr_percentile', 'bbw_percentile'] 
                               if col in result.columns])
            if n_indicators > 0:
                result['volatility_composite'] /= n_indicators
            
            # Classify the volatility regime
            result['volatility_regime'] = pd.cut(
                result['volatility_composite'],
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
            
            # Add volatility trend (increasing/decreasing)
            if 'volatility_composite' in result.columns:
                vol_comp = result['volatility_composite']
                result['volatility_trend'] = np.where(
                    vol_comp > vol_comp.shift(5),
                    'increasing',
                    np.where(vol_comp < vol_comp.shift(5), 'decreasing', 'stable')
                )
            
        except Exception as e:
            logger.error(f"Error calculating market regime: {str(e)}")
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('ATR', {})
            
            periods = params.get('periods', [5, 14, 21])
            
            for period in periods:
                if period > 0:
                    col_name = f'atr_{period}'
                    result[col_name] = ta.atr(high=data['high'], low=data['low'], close=data['close'], length=period)
            
            # Set default period as main ATR
            default_period = params.get('default_period', 14)
            default_col = f'atr_{default_period}'
            if default_col in result.columns:
                result['atr'] = result[default_col]
            else:
                result['atr'] = result[f'atr_{periods[0]}']
            
            # Calculate ATR as percentage of price
            result['atr_percent'] = result['atr'] / data['close'] * 100
            
            # ATR expansion/contraction
            result['atr_expanding'] = result['atr'] > result['atr'].shift(1)
            result['atr_contracting'] = result['atr'] < result['atr'].shift(1)
            
            # ATR trend over multiple periods
            atr_short_ma = result['atr'].rolling(window=5).mean()
            atr_long_ma = result['atr'].rolling(window=20).mean()
            result['atr_trend'] = atr_short_ma - atr_long_ma
            
            # Classify ATR trend
            result['atr_trend_regime'] = np.where(
                result['atr_trend'] > 0.05 * result['atr'],
                'expanding',
                np.where(
                    result['atr_trend'] < -0.05 * result['atr'],
                    'contracting',
                    'neutral'
                )
            )
            
            return result
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_historical_vol(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate historical volatility (standard deviation of returns)"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('HISTORICAL_VOL', {})
            
            periods = params.get('periods', [5, 10, 20, 50, 100])
            close = data['close']
            
            # Calculate log returns
            log_returns = np.log(close / close.shift(1))
            
            for period in periods:
                if period > 0:
                    # Standard deviation of log returns
                    vol = log_returns.rolling(window=period).std()
                    
                    # Annualize the volatility (sqrt of time)
                    timeframe_multiplier = self._get_timeframe_multiplier(data)
                    annualized_vol = vol * np.sqrt(timeframe_multiplier)
                    
                    col_name = f'historical_vol_{period}'
                    result[col_name] = annualized_vol
            
            # Set default period as main historical volatility
            default_period = params.get('default_period', 20)
            default_col = f'historical_vol_{default_period}'
            if default_col in result.columns:
                result['historical_vol'] = result[default_col]
            else:
                result['historical_vol'] = result[f'historical_vol_{periods[0]}']
            
            # Express as percentage
            for col in result.columns:
                if col.startswith('historical_vol'):
                    result[col] = result[col] * 100
            
            # Volatility of volatility (meta-volatility)
            if 'historical_vol' in result.columns:
                result['vol_of_vol'] = result['historical_vol'].rolling(window=20).std()
            
            return result
        except Exception as e:
            logger.error(f"Error calculating historical volatility: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _get_timeframe_multiplier(self, data: pd.DataFrame) -> float:
        """
        Determine the annualization factor based on the data's timeframe
        Returns the multiplier to annualize volatility
        """
        # Default to daily timeframe
        multiplier = 252  # Trading days in a year
        
        # Try to detect the timeframe from the index
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            # Calculate the median time difference
            diff = pd.Series(data.index).diff().median()
            
            if isinstance(diff, pd.Timedelta):
                seconds = diff.total_seconds()
                
                # Determine timeframe
                if seconds <= 60:
                    # 1-minute data
                    multiplier = 252 * 6.5 * 60  # ~98,280
                elif seconds <= 300:
                    # 5-minute data
                    multiplier = 252 * 6.5 * 12  # ~19,656
                elif seconds <= 900:
                    # 15-minute data
                    multiplier = 252 * 6.5 * 4  # ~6,552
                elif seconds <= 3600:
                    # Hourly data
                    multiplier = 252 * 6.5  # ~1,638
                elif seconds <= 86400:
                    # Daily data
                    multiplier = 252
                elif seconds <= 604800:
                    # Weekly data
                    multiplier = 52
                else:
                    # Monthly data or larger
                    multiplier = 12
        
        return multiplier
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate True Range"""
        try:
            result = pd.DataFrame(index=data.index)
            
            # True Range is the greatest of:
            # 1. Current high - current low
            # 2. Absolute value of current high - previous close
            # 3. Absolute value of current low - previous close
            result['true_range'] = ta.true_range(high=data['high'], low=data['low'], close=data['close'])
            
            # True Range as percentage of close price
            result['true_range_percent'] = result['true_range'] / data['close'] * 100
            
            # True Range expansions/contractions
            tr_ma5 = result['true_range'].rolling(window=5).mean()
            tr_ma20 = result['true_range'].rolling(window=20).mean()
            result['tr_expansion'] = tr_ma5 > tr_ma20
            
            return result
        except Exception as e:
            logger.error(f"Error calculating True Range: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_parkinsons_vol(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Parkinson's Volatility"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('PARKINSONS_VOL', {})
            
            periods = params.get('periods', [5, 10, 20, 50])
            
            # Parkinson's formula incorporates high-low range
            hl_ratio = np.log(data['high'] / data['low'])
            hl_square = hl_ratio ** 2
            
            for period in periods:
                if period > 0:
                    # Parkinson's volatility formula
                    vol = np.sqrt(
                        1 / (4 * np.log(2)) * hl_square.rolling(window=period).mean()
                    )
                    
                    # Annualize the volatility
                    timeframe_multiplier = self._get_timeframe_multiplier(data)
                    annualized_vol = vol * np.sqrt(timeframe_multiplier)
                    
                    col_name = f'parkinsons_vol_{period}'
                    result[col_name] = annualized_vol * 100  # Convert to percentage
            
            # Set default period as main Parkinson's volatility
            default_period = params.get('default_period', 20)
            default_col = f'parkinsons_vol_{default_period}'
            if default_col in result.columns:
                result['parkinsons_vol'] = result[default_col]
            else:
                result['parkinsons_vol'] = result[f'parkinsons_vol_{periods[0]}']
            
            return result
        except Exception as e:
            logger.error(f"Error calculating Parkinson's Volatility: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_garman_klass_vol(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Garman-Klass Volatility"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('GARMAN_KLASS_VOL', {})
            
            periods = params.get('periods', [5, 10, 20, 50])
            
            # Get log values needed for the formula
            log_hl = np.log(data['high'] / data['low']) ** 2
            log_co = np.log(data['close'] / data['open']) ** 2
            
            for period in periods:
                if period > 0:
                    # Garman-Klass volatility formula
                    vol = np.sqrt(
                        0.5 * log_hl.rolling(window=period).mean() - 
                        (2 * np.log(2) - 1) * log_co.rolling(window=period).mean()
                    )
                    
                    # Annualize the volatility
                    timeframe_multiplier = self._get_timeframe_multiplier(data)
                    annualized_vol = vol * np.sqrt(timeframe_multiplier)
                    
                    col_name = f'garman_klass_vol_{period}'
                    result[col_name] = annualized_vol * 100  # Convert to percentage
            
            # Set default period as main Garman-Klass volatility
            default_period = params.get('default_period', 20)
            default_col = f'garman_klass_vol_{default_period}'
            if default_col in result.columns:
                result['garman_klass_vol'] = result[default_col]
            else:
                result['garman_klass_vol'] = result[f'garman_klass_vol_{periods[0]}']
            
            return result
        except Exception as e:
            logger.error(f"Error calculating Garman-Klass Volatility: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_yang_zhang_vol(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Yang-Zhang Volatility"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('YANG_ZHANG_VOL', {})
            
            periods = params.get('periods', [5, 10, 20, 50])
            
            # Calculate components needed for Yang-Zhang formula
            close_prev = data['close'].shift(1)
            open_to_prev_close = np.log(data['open'] / close_prev) ** 2
            
            # Handle the first row which has no previous close
            open_to_prev_close.iloc[0] = 0
            
            high_to_open = np.log(data['high'] / data['open']) ** 2
            low_to_open = np.log(data['low'] / data['open']) ** 2
            close_to_open = np.log(data['close'] / data['open']) ** 2
            
            for period in periods:
                if period > 0:
                    # Overnight volatility (close to open)
                    overnight_vol = open_to_prev_close.rolling(window=period).mean()
                    
                    # Open to close volatility
                    open_close_vol = close_to_open.rolling(window=period).mean()
                    
                    # Rogers-Satchell volatility
                    rs_vol = (high_to_open * (high_to_open - close_to_open) + 
                              low_to_open * (low_to_open - close_to_open)).rolling(window=period).mean()
                    
                    # Yang-Zhang volatility formula (k is typically 0.34)
                    k = 0.34
                    vol = np.sqrt(
                        overnight_vol + k * open_close_vol + (1 - k) * rs_vol
                    )
                    
                    # Annualize the volatility
                    timeframe_multiplier = self._get_timeframe_multiplier(data)
                    annualized_vol = vol * np.sqrt(timeframe_multiplier)
                    
                    col_name = f'yang_zhang_vol_{period}'
                    result[col_name] = annualized_vol * 100  # Convert to percentage
            
            # Set default period as main Yang-Zhang volatility
            default_period = params.get('default_period', 20)
            default_col = f'yang_zhang_vol_{default_period}'
            if default_col in result.columns:
                result['yang_zhang_vol'] = result[default_col]
            else:
                result['yang_zhang_vol'] = result[f'yang_zhang_vol_{periods[0]}']
            
            return result
        except Exception as e:
            logger.error(f"Error calculating Yang-Zhang Volatility: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_bollinger_width(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Band Width (volatility measure)"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('BOLLINGER_WIDTH', {})
            
            periods = params.get('periods', [20])
            std_devs = params.get('std_devs', [2])
            
            for period in periods:
                for std_dev in std_devs:
                    # Calculate Bollinger Bands
                    ma = data['close'].rolling(window=period).mean()
                    std = data['close'].rolling(window=period).std()
                    upper = ma + std_dev * std
                    lower = ma - std_dev * std
                    
                    # Calculate Bollinger Band Width
                    width = (upper - lower) / ma
                    col_name = f'bollinger_width_{period}_{std_dev}'
                    result[col_name] = width
            
            # Set default as main Bollinger Width
            default_period = params.get('default_period', 20)
            default_std_dev = params.get('default_std_dev', 2)
            default_col = f'bollinger_width_{default_period}_{default_std_dev}'
            
            if default_col in result.columns:
                result['bollinger_width'] = result[default_col]
            else:
                result['bollinger_width'] = result[list(result.columns)[0]]
            
            # Bollinger Width percentile (historical context)
            lookback = 100
            if len(result) > lookback:
                result['bollinger_width_percentile'] = result['bollinger_width'].rolling(window=lookback).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
            
            # Bollinger Width regime
            result['bollinger_width_regime'] = pd.cut(
                result['bollinger_width'],
                bins=[0, result['bollinger_width'].quantile(0.2), 
                     result['bollinger_width'].quantile(0.8), float('inf')],
                labels=['tight', 'normal', 'wide']
            )
            
            return result
        except Exception as e:
            logger.error(f"Error calculating Bollinger Width: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_volatility_ratio(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility ratio (short-term vs long-term volatility)"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('VOLATILITY_RATIO', {})
            
            short_period = params.get('short_period', 5)
            long_period = params.get('long_period', 20)
            
            # Calculate log returns
            log_returns = np.log(data['close'] / data['close'].shift(1))
            
            # Short-term and long-term volatility
            short_vol = log_returns.rolling(window=short_period).std()
            long_vol = log_returns.rolling(window=long_period).std()
            
            # Volatility ratio
            result['volatility_ratio'] = short_vol / long_vol
            
            # Regime based on volatility ratio
            result['volatility_ratio_regime'] = np.where(
                result['volatility_ratio'] > 1.2,
                'increasing',
                np.where(
                    result['volatility_ratio'] < 0.8,
                    'decreasing',
                    'stable'
                )
            )
            
            return result
        except Exception as e:
            logger.error(f"Error calculating Volatility Ratio: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_keltner_width(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Keltner Channel Width (volatility measure)"""
        try:
            output = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('KELTNER_WIDTH', {})
            
            periods = params.get('periods', [20])
            multipliers = params.get('multipliers', [2])
            
            # Calculate ATR if not already in the result
            if 'atr' not in result.columns:
                atr = ta.atr(high=data['high'], low=data['low'], close=data['close'], length=20)
            else:
                atr = result['atr'].values
            
            for period in periods:
                for multiplier in multipliers:
                    # Calculate EMA
                    ema = ta.ema(data['close'], length=period)
                    
                    # Calculate Keltner Channels
                    upper = ema + multiplier * atr
                    lower = ema - multiplier * atr
                    
                    # Calculate Keltner Channel Width
                    width = (upper - lower) / ema
                    col_name = f'keltner_width_{period}_{multiplier}'
                    output[col_name] = width
            
            # Set default as main Keltner Width
            default_period = params.get('default_period', 20)
            default_multiplier = params.get('default_multiplier', 2)
            default_col = f'keltner_width_{default_period}_{default_multiplier}'
            
            if default_col in output.columns:
                output['keltner_width'] = output[default_col]
            else:
                output['keltner_width'] = output[list(output.columns)[0]]
            
            return output
        except Exception as e:
            logger.error(f"Error calculating Keltner Width: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_donchian_width(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Donchian Channel Width (volatility measure)"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('DONCHIAN_WIDTH', {})
            
            periods = params.get('periods', [20])
            
            for period in periods:
                # Calculate Donchian Channels
                upper = data['high'].rolling(window=period).max()
                lower = data['low'].rolling(window=period).min()
                middle = (upper + lower) / 2
                
                # Calculate Donchian Channel Width
                width = (upper - lower) / middle
                col_name = f'donchian_width_{period}'
                result[col_name] = width
            
            # Set default as main Donchian Width
            default_period = params.get('default_period', 20)
            default_col = f'donchian_width_{default_period}'
            
            if default_col in result.columns:
                result['donchian_width'] = result[default_col]
            else:
                result['donchian_width'] = result[list(result.columns)[0]]
            
            # Donchian Width percentile (historical context)
            lookback = 100
            if len(result) > lookback:
                result['donchian_width_percentile'] = result['donchian_width'].rolling(window=lookback).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1]
                )
            
            return result
        except Exception as e:
            logger.error(f"Error calculating Donchian Width: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_hurst_exponent(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Hurst Exponent for trend/mean-reversion detection"""
        try:
            result = pd.DataFrame(index=data.index)
            params = self.indicator_params.get('HURST_EXPONENT', {})
            
            # Lookback window for calculation
            lookback = params.get('lookback', 100)
            min_window = 10
            max_window = 100
            
            # Create empty array for results
            hurst = np.full(len(data), np.nan)
            
            # Get log returns
            log_returns = np.log(data['close'] / data['close'].shift(1)).fillna(0).values
            
            # Calculate Hurst exponent for each point with sufficient history
            for i in range(lookback, len(log_returns)):
                returns_window = log_returns[i-lookback:i]
                hurst[i] = self._calculate_hurst(returns_window, min_window, max_window)
            
            result['hurst_exponent'] = hurst
            
            # Classify market type based on Hurst exponent
            # H > 0.5: trending, H < 0.5: mean-reverting, H ≈ 0.5: random walk
            result['market_type'] = np.where(
                result['hurst_exponent'] > 0.6,
                'trending',
                np.where(
                    result['hurst_exponent'] < 0.4,
                    'mean_reverting',
                    'random_walk'
                )
            )
            
            return result
        except Exception as e:
            logger.error(f"Error calculating Hurst Exponent: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_hurst(self, returns: np.ndarray, min_window: int, max_window: int) -> float:
        """Helper method to calculate Hurst exponent using R/S analysis"""
        # Ensure we have enough data
        if len(returns) < max_window:
            return np.nan
            
        window_sizes = range(min_window, max_window + 1, 10)
        rs_values = []
        
        for w in window_sizes:
            rs_values.append(self._calculate_rs(returns, w))
            
        if not rs_values:
            return np.nan
            
        # Linear regression of log(R/S) vs log(window)
        y = np.log(rs_values)
        x = np.log(window_sizes)
            
        # Simple linear regression
        slope, intercept = np.polyfit(x, y, 1)
            
        return slope
    
    def _calculate_rs(self, returns: np.ndarray, window: int) -> float:
        """Calculate R/S value for a given window size"""
        # Split returns into windows
        n_windows = len(returns) // window
        if n_windows == 0:
            return np.nan
            
        rs_values = []
        
        for i in range(n_windows):
            window_returns = returns[i * window:(i + 1) * window]
            
            # Convert returns to price series (starting at 1)
            prices = np.exp(np.cumsum(window_returns))
            
            # Mean and standard deviation
            mean = np.mean(prices)
            std = np.std(prices)
            
            if std == 0:
                continue
                
            # Calculate cumulative deviations
            deviations = prices - mean
            cumulative = np.cumsum(deviations)
            
            # R/S value
            r = max(cumulative) - min(cumulative)
            s = std
            
            if s > 0:
                rs_values.append(r / s)
        
        # Return average R/S value
        if not rs_values:
            return np.nan
            
        return np.mean(rs_values)
    
    def _calculate_garch_vol(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate GARCH volatility forecast"""
        try:
            # This is a simplified version without using the ARCH package
            # A full implementation would use arch.univariate.GARCH
            
            result = pd.DataFrame(index=data.index)
            
            # Get log returns
            log_returns = np.log(data['close'] / data['close'].shift(1)).fillna(0)
            
            # Use a simple EWMA model as a proxy for GARCH
            # EWMA is similar to GARCH(1,1) with specific constraints
            lambda_param = 0.94  # Standard RiskMetrics lambda
            
            # Calculate EWMA variance
            ewma_var = np.zeros(len(log_returns))
            ewma_var[0] = log_returns.iloc[0] ** 2
            
            for t in range(1, len(log_returns)):
                ewma_var[t] = (1 - lambda_param) * log_returns.iloc[t-1] ** 2 + lambda_param * ewma_var[t-1]
            
            # Convert variance to volatility (standard deviation)
            ewma_vol = np.sqrt(ewma_var)
            
            # Annualize
            timeframe_multiplier = self._get_timeframe_multiplier(data)
            annualized_vol = ewma_vol * np.sqrt(timeframe_multiplier)
            
            result['garch_vol'] = annualized_vol * 100  # Convert to percentage
            
            return result
        except Exception as e:
            logger.error(f"Error calculating GARCH volatility: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_realized_vol(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate realized volatility using intraday data"""
        try:
            result = pd.DataFrame(index=data.index)
            
            # For this to work properly, we need intraday data
            # This is a simplified version that works with OHLC data
            
            # Use Garman-Klass or Yang-Zhang as proxy for realized volatility
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                # Garman-Klass volatility
                log_hl = np.log(data['high'] / data['low']) ** 2
                log_co = np.log(data['close'] / data['open']) ** 2
                
                # Realized volatility calculation
                rv = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
                
                # Rolling window for stability
                periods = [5, 10, 20]
                for period in periods:
                    result[f'realized_vol_{period}'] = rv.rolling(window=period).mean()
                
                # Annualize the volatility
                timeframe_multiplier = self._get_timeframe_multiplier(data)
                for col in result.columns:
                    if col.startswith('realized_vol'):
                        result[col] = result[col] * np.sqrt(timeframe_multiplier) * 100  # Convert to percentage
                
                # Set default period as main realized volatility
                result['realized_vol'] = result['realized_vol_10']
            
            return result
        except Exception as e:
            logger.error(f"Error calculating realized volatility: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_vix_proxy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate a VIX-like index using implied volatility proxy"""
        try:
            result = pd.DataFrame(index=data.index)
            
            # This is a simplified proxy that estimates VIX-like behavior
            # Real VIX calculation requires options data
            
            # Use Bollinger Band Width as a proxy for implied volatility
            bb_width = self._calculate_bollinger_width(data)
            if 'bollinger_width' in bb_width.columns:
                # Normalize using historical range
                lookback = 100
                if len(data) > lookback:
                    rolling_max = bb_width['bollinger_width'].rolling(window=lookback).max()
                    rolling_min = bb_width['bollinger_width'].rolling(window=lookback).min()
                    normalized = (bb_width['bollinger_width'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
                    
                    # Scale to a VIX-like range (typically 10-40)
                    result['vix_proxy'] = normalized * 30 + 10
                    
                    # VIX regime classification
                    result['vix_regime'] = pd.cut(
                        result['vix_proxy'],
                        bins=[0, 15, 25, 35, float('inf')],
                        labels=['low_fear', 'normal', 'high_fear', 'extreme_fear']
                    )
            
            return result
        except Exception as e:
            logger.error(f"Error calculating VIX proxy: {str(e)}")
            return pd.DataFrame(index=data.index)
    
    # Add more volatility indicator calculation methods as needed


class VolatilityAnalyzer:
    """Lightweight wrapper exposing common volatility indicators."""

    def atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        return ta.atr(high=data['high'], low=data['low'], close=data['close'], length=period)


class VolatilityCalculator:
    """Simplified calculator that delegates to :class:`VolatilityFeatures`."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._features = VolatilityFeatures(config)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with volatility indicators."""
        return self._features.calculate_features(data)


def calculate_atr(high: Union[pd.Series, List[float]], low: Union[pd.Series, List[float]],
                  close: Union[pd.Series, List[float]], period: int = 14) -> pd.Series:
    """Standalone Average True Range calculation."""
    return ta.atr(high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), length=period)


def calculate_bollinger_bands(close: Union[pd.Series, List[float]], period: int = 20,
                              std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Standalone Bollinger Bands calculation."""
    series = pd.Series(close)
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return upper, ma, lower

def calculate_volatility_features(data, config=None):
    """
    Convenience function to calculate volatility features from OHLCV data.
    
    Args:
        data: DataFrame with OHLCV data
        config: Optional configuration for volatility features
        
    Returns:
        DataFrame with calculated volatility indicators
    """
    calculator = VolatilityFeatures(config)
    return calculator.calculate_features(data)


__all__ = [
    "VolatilityFeatures",
    "VolatilityAnalyzer",
    "VolatilityCalculator",
    "calculate_volatility_features",
    "calculate_atr",
    "calculate_bollinger_bands",
    "calculate_historical_volatility",
    "calculate_volatility_ratio",
    "volatility_expansion_indicator",
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
    
    # Calculate volatility features
    vol_features = calculate_volatility_features(data)
    
    # Log first few rows with key indicators for debugging
    logger.info(vol_features.iloc[-5:][['historical_vol', 'atr_percent', 'volatility_regime']])
