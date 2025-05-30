#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Feature Service - Feature Extraction

This module provides the core functionality for extracting features from market data.
It implements a comprehensive set of technical, statistical, and machine learning
features optimized for trading pattern recognition and market analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import inspect
from functools import wraps, lru_cache
import time
try:
    import ta  # type: ignore
    TA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ta = None  # type: ignore
    TA_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "ta library not found; using pandas implementations for indicators"
    )
from scipy import stats, signal
try:
    import pywt  # type: ignore
    PYWT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pywt = None  # type: ignore
    PYWT_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "pywt not available; wavelet features disabled"
    )
from sklearn import preprocessing
try:
    import statsmodels.api as sm  # type: ignore
    STATSMODELS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    sm = None  # type: ignore
    STATSMODELS_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "statsmodels not available; econometrics features disabled"
    )
from empyrical import max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio
from feature_service.processor_utils import cudf, HAS_GPU
from numba import cuda, jit, vectorize
import bottleneck as bn
from feature_service.features.cross_asset import (
    compute_pair_correlation,
    cointegration_score,
)

from common.logger import get_logger
from common.metrics import MetricsCollector
# Exported helper functions
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wrapper for ATR calculation."""
    return ta.atr(high=high, low=low, close=close, length=period)


def fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Compute Fibonacci retracement levels."""
    diff = high - low
    return {
        'level_23.6': high - diff * 0.236,
        'level_38.2': high - diff * 0.382,
        'level_50.0': high - diff * 0.5,
        'level_61.8': high - diff * 0.618,
        'level_78.6': high - diff * 0.786,
    }
# from common.utils import profile_execution, chunk_dataframe # Removed unused imports
from common.constants import DEFAULT_FEATURE_PARAMS
# Removed unused FEATURE_CACHE_SIZE
from common.exceptions import (
    FeatureCalculationError, InvalidFeatureDefinitionError
)


def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simple wrapper to calculate Average True Range."""
    return ta.atr(high=data['high'], low=data['low'], close=data['close'], length=period)


def fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Return key Fibonacci retracement levels."""
    ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    diff = high - low
    levels = {'0': high, '1': low}
    for r in ratios:
        levels[str(r)] = high - diff * r
    return levels


__all__ = ['atr', 'fibonacci_levels']
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Simple ATR calculation used by tests."""
    return ta.atr(high=high, low=low, close=close, length=period)


def fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Compute basic Fibonacci retracement levels."""
    diff = high - low
    return {
        '0.0%': high,
        '23.6%': high - diff * 0.236,
        '38.2%': high - diff * 0.382,
        '50.0%': high - diff * 0.5,
        '61.8%': high - diff * 0.618,
        '100%': low,
    }

logger = get_logger(__name__)
metrics = MetricsCollector.get_instance("feature_service.extractor")


# Feature calculation decorator for tracking and error handling
def feature_calculation(f):
    """Decorator for feature calculation functions with error handling and metrics."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        feature_name = f.__name__
        
        try:
            # Call the feature calculation function
            result = f(*args, **kwargs)
            
            # Record metrics
            duration = (time.time() - start_time) * 1000
            metrics.timing(f"feature.calculation.{feature_name}", duration)
            
            if isinstance(result, pd.Series) and result.isna().mean() > 0.5:
                logger.warning(f"Feature {feature_name} has more than 50% NaN values")
                metrics.increment(f"feature.calculation.{feature_name}.high_nan_rate")
            
            return result
            
        except Exception as e:
            # Log error and record metric
            logger.error(f"Error calculating feature {feature_name}: {str(e)}")
            metrics.increment(f"feature.calculation.{feature_name}.error")
            
            # Return a series of NaN with same index as input
            if len(args) > 0 and isinstance(args[0], pd.DataFrame):
                return pd.Series(np.nan, index=args[0].index, name=feature_name)
            
            # Re-raise the exception
            raise FeatureCalculationError(f"Error calculating {feature_name}: {str(e)}") from e
    
    # Store original function signature for inspection
    wrapper.__signature__ = inspect.signature(f)
    return wrapper


def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simple Average True Range calculation."""
    high = data['high']
    low = data['low']
    close = data['close']
    prev_close = close.shift()
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate basic Fibonacci retracement levels."""
    diff = high - low
    return {
        '0.236': high - diff * 0.236,
        '0.382': high - diff * 0.382,
        '0.500': high - diff * 0.5,
        '0.618': high - diff * 0.618,
        '0.786': high - diff * 0.786,
    }


class FeatureExtractor:
    """
    Advanced feature extraction engine for market data analysis.
    
    This class provides a comprehensive set of technical, statistical, pattern-based,
    and machine learning features for market analysis and trading strategy development.
    Features can be calculated efficiently on both CPU and GPU.
    """
    
    def __init__(self, features: List[str], use_gpu: bool = False):
        """
        Initialize the feature extractor.
        
        Args:
            features: List of feature names to extract
            use_gpu: Whether to use GPU acceleration when available
        """
        self.features = features
        self.use_gpu = use_gpu
        
        # Register all available feature calculation methods
        self._register_features()
        
        # Validate requested features
        self._validate_features()
        
        logger.debug(f"FeatureExtractor initialized with {len(features)} features, GPU: {use_gpu}")
    
    def _register_features(self):
        """Register all available feature calculation methods."""
        self.available_features = {}
        
        # Find all methods with feature_calculation decorator
        for name, method in inspect.getmembers(self):
            if name.startswith('_'):
                continue
                
            if hasattr(method, '__wrapped__'):
                # This is a feature calculation method
                self.available_features[name] = {
                    'method': method,
                    'signature': inspect.signature(method),
                    'doc': inspect.getdoc(method),
                    'category': getattr(method, 'category', 'unknown')
                }
        
        logger.debug(f"Registered {len(self.available_features)} available features")
    
    def _validate_features(self):
        """Validate that all requested features are available."""
        invalid_features = [f for f in self.features if f not in self.available_features]
        if invalid_features:
            raise InvalidFeatureDefinitionError(f"Invalid features requested: {invalid_features}")
    
    def get_all_feature_info(self) -> Dict[str, Any]:
        """
        Get information about all available features.
        
        Returns:
            Dictionary containing feature metadata
        """
        return {
            name: {
                'signature': str(info['signature']),
                'doc': info['doc'],
                'category': info['category']
            }
            for name, info in self.available_features.items()
        }
    
    def extract_features(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Extract requested features from market data.
        
        Args:
            data: DataFrame containing OHLCV market data
            params: Additional parameters for feature calculation
            
        Returns:
            DataFrame containing calculated features
        """
        start_time = time.time()
        metrics.increment("feature.extraction.started")
        
        # Ensure data has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Input data missing required columns: {missing_columns}")
        
        # Initialize parameters with defaults
        params = params or {}
        for param, default in DEFAULT_FEATURE_PARAMS.items():
            params.setdefault(param, default)
        
        # Calculate each requested feature
        results = {}
        
        for feature in self.features:
            if feature in self.available_features:
                method = self.available_features[feature]['method']
                try:
                    start_feature_time = time.time()
                    results[feature] = method(data, params)
                    metrics.timing(f"feature.{feature}.time", (time.time() - start_feature_time) * 1000)
                except Exception as e:
                    logger.error(f"Error calculating feature {feature}: {str(e)}")
                    results[feature] = pd.Series(np.nan, index=data.index, name=feature)
                    metrics.increment(f"feature.{feature}.error")
            else:
                logger.warning(f"Requested feature not found: {feature}")
                results[feature] = pd.Series(np.nan, index=data.index, name=feature)
        
        # Combine results into a DataFrame
        result_df = pd.DataFrame(results, index=data.index)
        
        # Calculate metrics
        elapsed_ms = (time.time() - start_time) * 1000
        features_per_second = len(self.features) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        
        metrics.timing("feature.extraction.time", elapsed_ms)
        metrics.gauge("feature.extraction.features_per_second", features_per_second)
        metrics.increment("feature.extraction.completed")
        
        return result_df
    
    def extract_features_gpu(
        self,
        data: cudf.DataFrame,
        params: Optional[Dict[str, Any]] = None
    ) -> cudf.DataFrame:
        """
        Extract requested features using GPU acceleration.
        
        Args:
            data: cuDF DataFrame containing OHLCV market data
            params: Additional parameters for feature calculation
            
        Returns:
            cuDF DataFrame containing calculated features
        """
        if not self.use_gpu:
            raise ValueError("GPU acceleration not enabled for this extractor")
        
        start_time = time.time()
        metrics.increment("feature.extraction.gpu.started")
        
        # Ensure data has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Input data missing required columns: {missing_columns}")
        
        # Initialize parameters with defaults
        params = params or {}
        for param, default in DEFAULT_FEATURE_PARAMS.items():
            params.setdefault(param, default)
        
        # Calculate each requested feature
        results = {}
        
        for feature in self.features:
            if feature in self.available_features:
                # Check if GPU version exists
                gpu_method_name = f"{feature}_gpu"
                if hasattr(self, gpu_method_name):
                    method = getattr(self, gpu_method_name)
                    try:
                        start_feature_time = time.time()
                        results[feature] = method(data, params)
                        metrics.timing(f"feature.{feature}.gpu.time", (time.time() - start_feature_time) * 1000)
                    except Exception as e:
                        logger.error(f"Error calculating GPU feature {feature}: {str(e)}")
                        results[feature] = cudf.Series(np.nan, index=data.index, name=feature)
                        metrics.increment(f"feature.{feature}.gpu.error")
                else:
                    # Fall back to CPU implementation with pandas conversion
                    logger.debug(f"No GPU implementation for {feature}, falling back to CPU")
                    try:
                        # Convert to pandas, calculate, convert back
                        pd_data = data.to_pandas()
                        method = self.available_features[feature]['method']
                        
                        start_feature_time = time.time()
                        pd_result = method(pd_data, params)
                        result_series = cudf.Series.from_pandas(pd_result)
                        result_series.index = data.index
                        results[feature] = result_series
                        
                        metrics.timing(f"feature.{feature}.gpu_fallback.time", 
                                     (time.time() - start_feature_time) * 1000)
                    except Exception as e:
                        logger.error(f"Error in GPU fallback for {feature}: {str(e)}")
                        results[feature] = cudf.Series(np.nan, index=data.index, name=feature)
                        metrics.increment(f"feature.{feature}.gpu_fallback.error")
            else:
                logger.warning(f"Requested feature not found: {feature}")
                results[feature] = cudf.Series(np.nan, index=data.index, name=feature)
        
        # Combine results into a DataFrame
        result_df = cudf.DataFrame(results, index=data.index)
        
        # Calculate metrics
        elapsed_ms = (time.time() - start_time) * 1000
        features_per_second = len(self.features) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        
        metrics.timing("feature.extraction.gpu.time", elapsed_ms)
        metrics.gauge("feature.extraction.gpu.features_per_second", features_per_second)
        metrics.increment("feature.extraction.gpu.completed")
        
        return result_df
    
    #
    # Technical Indicator Features
    #
    
    @feature_calculation
    def sma(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Simple Moving Average
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - sma_period: Period for SMA calculation (default: 14)
        
        Returns:
            Series containing SMA values
        """
        period = params.get('sma_period', 14)
        if TA_AVAILABLE:
            return ta.sma(data['close'], length=period)
        return data['close'].rolling(period).mean()
    
    def sma_gpu(self, data: cudf.DataFrame, params: Dict[str, Any]) -> cudf.Series:
        """GPU-accelerated SMA calculation."""
        period = params.get('sma_period', 14)
        return data['close'].rolling(period).mean()
    
    @feature_calculation
    def ema(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Exponential Moving Average
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - ema_period: Period for EMA calculation (default: 14)
        
        Returns:
            Series containing EMA values
        """
        period = params.get('ema_period', 14)
        if TA_AVAILABLE:
            try:
                return ta.ema(data['close'], length=period)
            except Exception:
                pass
        return data['close'].ewm(span=period, adjust=False).mean()
    
    def ema_gpu(self, data: cudf.DataFrame, params: Dict[str, Any]) -> cudf.Series:
        """GPU-accelerated EMA calculation."""
        period = params.get('ema_period', 14)
        return data['close'].ewm(span=period, adjust=False).mean()
    
    @feature_calculation
    def rsi(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - rsi_period: Period for RSI calculation (default: 14)
        
        Returns:
            Series containing RSI values
        """
        period = params.get('rsi_period', 14)
        if TA_AVAILABLE:
            try:
                return ta.rsi(data['close'], length=period)
            except Exception:
                pass

        delta = data['close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(span=period - 1, adjust=False).mean()
        ma_down = down.ewm(span=period - 1, adjust=False).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @feature_calculation
    def macd(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Moving Average Convergence Divergence
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - macd_fast_period: Fast period (default: 12)
                - macd_slow_period: Slow period (default: 26)
                - macd_signal_period: Signal period (default: 9)
        
        Returns:
            Series containing MACD line values
        """
        fast_period = params.get('macd_fast_period', 12)
        slow_period = params.get('macd_slow_period', 26)
        signal_period = params.get('macd_signal_period', 9)
        
        if TA_AVAILABLE:
            try:
                macd_df = ta.macd(
                    data['close'], fast=fast_period, slow=slow_period, signal=signal_period
                )
                return macd_df.iloc[:, 0]
            except Exception:
                pass

        ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return macd_line
    
    @feature_calculation
    def macd_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        MACD Signal Line
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - macd_fast_period: Fast period (default: 12)
                - macd_slow_period: Slow period (default: 26)
                - macd_signal_period: Signal period (default: 9)
        
        Returns:
            Series containing MACD signal line values
        """
        fast_period = params.get('macd_fast_period', 12)
        slow_period = params.get('macd_slow_period', 26)
        signal_period = params.get('macd_signal_period', 9)
        
        if TA_AVAILABLE:
            try:
                macd_df = ta.macd(
                    data['close'], fast=fast_period, slow=slow_period, signal=signal_period
                )
                return macd_df.iloc[:, 2]
            except Exception:
                pass

        ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        return signal_line
    
    @feature_calculation
    def macd_hist(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        MACD Histogram
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - macd_fast_period: Fast period (default: 12)
                - macd_slow_period: Slow period (default: 26)
                - macd_signal_period: Signal period (default: 9)
        
        Returns:
            Series containing MACD histogram values
        """
        fast_period = params.get('macd_fast_period', 12)
        slow_period = params.get('macd_slow_period', 26)
        signal_period = params.get('macd_signal_period', 9)
        
        if TA_AVAILABLE:
            try:
                macd_df = ta.macd(
                    data['close'], fast=fast_period, slow=slow_period, signal=signal_period
                )
                return macd_df.iloc[:, 1]
            except Exception:
                pass

        ema_fast = data['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        hist = macd_line - signal_line
        return hist
    
    @feature_calculation
    def bollinger_upper(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Bollinger Bands - Upper Band
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - bb_period: Period for BB calculation (default: 20)
                - bb_std_dev: Standard deviation multiplier (default: 2.0)
        
        Returns:
            Series containing upper Bollinger Band values
        """
        period = params.get('bb_period', 20)
        std_dev = params.get('bb_std_dev', 2.0)
        
        bb = ta.bbands(data['close'], length=period, std=std_dev)
        return bb.iloc[:, 0]
    
    @feature_calculation
    def bollinger_middle(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Bollinger Bands - Middle Band
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - bb_period: Period for BB calculation (default: 20)
                - bb_std_dev: Standard deviation multiplier (default: 2.0)
        
        Returns:
            Series containing middle Bollinger Band values
        """
        period = params.get('bb_period', 20)
        std_dev = params.get('bb_std_dev', 2.0)
        
        bb = ta.bbands(data['close'], length=period, std=std_dev)
        return bb.iloc[:, 1]
    
    @feature_calculation
    def bollinger_lower(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Bollinger Bands - Lower Band
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - bb_period: Period for BB calculation (default: 20)
                - bb_std_dev: Standard deviation multiplier (default: 2.0)
        
        Returns:
            Series containing lower Bollinger Band values
        """
        period = params.get('bb_period', 20)
        std_dev = params.get('bb_std_dev', 2.0)
        
        bb = ta.bbands(data['close'], length=period, std=std_dev)
        return bb.iloc[:, 2]
    
    @feature_calculation
    def bollinger_width(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Bollinger Bandwidth
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - bb_period: Period for BB calculation (default: 20)
                - bb_std_dev: Standard deviation multiplier (default: 2.0)
        
        Returns:
            Series containing Bollinger bandwidth values
        """
        period = params.get('bb_period', 20)
        std_dev = params.get('bb_std_dev', 2.0)
        
        bb = ta.bbands(data['close'], length=period, std=std_dev)
        bandwidth = (bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1]
        return bandwidth
    
    @feature_calculation
    def stoch_k(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Stochastic Oscillator %K
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - stoch_k_period: %K period (default: 14)
                - stoch_d_period: %D period (default: 3)
                - stoch_slowing: Slowing period (default: 3)
        
        Returns:
            Series containing Stochastic %K values
        """
        k_period = params.get('stoch_k_period', 14)
        d_period = params.get('stoch_d_period', 3)
        slowing = params.get('stoch_slowing', 3)
        
        stoch = ta.stoch(high=data['high'], low=data['low'], close=data['close'],
                          k=k_period, d=d_period, smooth_k=slowing)
        k = stoch.iloc[:, 0]
        d = stoch.iloc[:, 1]
        
        return k
    
    @feature_calculation
    def stoch_d(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Stochastic Oscillator %D
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - stoch_k_period: %K period (default: 14)
                - stoch_d_period: %D period (default: 3)
                - stoch_slowing: Slowing period (default: 3)
        
        Returns:
            Series containing Stochastic %D values
        """
        k_period = params.get('stoch_k_period', 14)
        d_period = params.get('stoch_d_period', 3)
        slowing = params.get('stoch_slowing', 3)
        
        stoch = ta.stoch(high=data['high'], low=data['low'], close=data['close'],
                          k=k_period, d=d_period, smooth_k=slowing)
        return stoch.iloc[:, 1]
    
    @feature_calculation
    def atr(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Average True Range
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - atr_period: Period for ATR calculation (default: 14)
        
        Returns:
            Series containing ATR values
        """
        period = params.get('atr_period', 14)
        if TA_AVAILABLE:
            try:
                return ta.atr(
                    high=data['high'], low=data['low'], close=data['close'], length=period
                )
            except Exception:
                pass

        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift()
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @feature_calculation
    def adx(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Average Directional Index
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - adx_period: Period for ADX calculation (default: 14)
        
        Returns:
            Series containing ADX values
        """
        period = params.get('adx_period', 14)
        return ta.adx(high=data['high'], low=data['low'], close=data['close'], length=period)
    
    @feature_calculation
    def plus_di(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Plus Directional Indicator (+DI)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - di_period: Period for DI calculation (default: 14)
        
        Returns:
            Series containing +DI values
        """
        period = params.get('di_period', 14)
        return ta.adx(high=data['high'], low=data['low'], close=data['close'], length=period)['DMP_{}_{}'.format(period, period)]
    
    @feature_calculation
    def minus_di(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Minus Directional Indicator (-DI)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - di_period: Period for DI calculation (default: 14)
        
        Returns:
            Series containing -DI values
        """
        period = params.get('di_period', 14)
        return ta.adx(high=data['high'], low=data['low'], close=data['close'], length=period)['DMN_{}_{}'.format(period, period)]


# Standalone helper wrappers
def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range from OHLCV data."""
    return ta.atr(high=data['high'], low=data['low'], close=data['close'], length=period)


def fibonacci_levels(data: pd.DataFrame) -> Dict[str, float]:
    """Compute basic Fibonacci retracement levels."""
    high = data['high'].max()
    low = data['low'].min()
    diff = high - low
    return {
        '0.0%': high,
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff,
        '61.8%': high - 0.618 * diff,
        '100.0%': low,
    }
    
    @feature_calculation
    def obv(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        On-Balance Volume
        
        Args:
            data: OHLCV DataFrame
            params: Not used for OBV
        
        Returns:
            Series containing OBV values
        """
        return ta.obv(close=data['close'], volume=data['volume'])
    
    @feature_calculation
    def cci(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Commodity Channel Index
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - cci_period: Period for CCI calculation (default: 14)
        
        Returns:
            Series containing CCI values
        """
        period = params.get('cci_period', 14)
        return ta.cci(high=data['high'], low=data['low'], close=data['close'], length=period)
    
    @feature_calculation
    def mfi(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Money Flow Index
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - mfi_period: Period for MFI calculation (default: 14)
        
        Returns:
            Series containing MFI values
        """
        period = params.get('mfi_period', 14)
        return ta.mfi(high=data['high'], low=data['low'], close=data['close'], volume=data['volume'], length=period)
    
    @feature_calculation
    def williams_r(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Williams %R
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - willr_period: Period for Williams %R calculation (default: 14)
        
        Returns:
            Series containing Williams %R values
        """
        period = params.get('willr_period', 14)
        return ta.willr(high=data['high'], low=data['low'], close=data['close'], length=period)
    
    @feature_calculation
    def ichimoku_conversion(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Ichimoku Conversion Line (Tenkan-sen)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - ichimoku_conversion_period: Conversion period (default: 9)
        
        Returns:
            Series containing Conversion Line values
        """
        conversion_period = params.get('ichimoku_conversion_period', 9)
        
        # Calculate (high + low) / 2 for the specified period
        high_values = data['high'].rolling(window=conversion_period).max()
        low_values = data['low'].rolling(window=conversion_period).min()
        conversion_line = (high_values + low_values) / 2
        
        return conversion_line
    
    @feature_calculation
    def ichimoku_base(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Ichimoku Base Line (Kijun-sen)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - ichimoku_base_period: Base period (default: 26)
        
        Returns:
            Series containing Base Line values
        """
        base_period = params.get('ichimoku_base_period', 26)
        
        # Calculate (high + low) / 2 for the specified period
        high_values = data['high'].rolling(window=base_period).max()
        low_values = data['low'].rolling(window=base_period).min()
        base_line = (high_values + low_values) / 2
        
        return base_line
    
    @feature_calculation
    def keltner_upper(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Keltner Channel - Upper Band
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - keltner_period: Period for EMA calculation (default: 20)
                - keltner_atr_period: Period for ATR calculation (default: 10)
                - keltner_multiplier: ATR multiplier (default: 2.0)
        
        Returns:
            Series containing upper Keltner Channel values
        """
        period = params.get('keltner_period', 20)
        atr_period = params.get('keltner_atr_period', 10)
        multiplier = params.get('keltner_multiplier', 2.0)
        
        # Calculate middle line (EMA of typical price)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        middle = typical_price.ewm(span=period, adjust=False).mean()
        
        # Calculate ATR
        atr = ta.atr(high=data['high'], low=data['low'], close=data['close'], length=atr_period)
        
        # Calculate upper band
        upper = middle + (multiplier * atr)
        
        return upper
    
    @feature_calculation
    def keltner_middle(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Keltner Channel - Middle Band
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - keltner_period: Period for EMA calculation (default: 20)
        
        Returns:
            Series containing middle Keltner Channel values
        """
        period = params.get('keltner_period', 20)
        
        # Calculate middle line (EMA of typical price)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        middle = typical_price.ewm(span=period, adjust=False).mean()
        
        return middle
    
    @feature_calculation
    def keltner_lower(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Keltner Channel - Lower Band
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - keltner_period: Period for EMA calculation (default: 20)
                - keltner_atr_period: Period for ATR calculation (default: 10)
                - keltner_multiplier: ATR multiplier (default: 2.0)
        
        Returns:
            Series containing lower Keltner Channel values
        """
        period = params.get('keltner_period', 20)
        atr_period = params.get('keltner_atr_period', 10)
        multiplier = params.get('keltner_multiplier', 2.0)
        
        # Calculate middle line (EMA of typical price)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        middle = typical_price.ewm(span=period, adjust=False).mean()
        
        # Calculate ATR
        atr = ta.atr(high=data['high'], low=data['low'], close=data['close'], length=atr_period)
        
        # Calculate lower band
        lower = middle - (multiplier * atr)
        
        return lower
    
    #
    # Momentum and Trend Features
    #
    
    @feature_calculation
    def price_momentum(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Price Momentum (percent change)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - momentum_period: Period for momentum calculation (default: 10)
        
        Returns:
            Series containing momentum values
        """
        period = params.get('momentum_period', 10)
        return data['close'].pct_change(period)
    
    @feature_calculation
    def momentum_relative(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Relative Momentum (price / SMA)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - momentum_period: Period for momentum calculation (default: 10)
        
        Returns:
            Series containing relative momentum values
        """
        period = params.get('momentum_period', 10)
        sma = data['close'].rolling(period).mean()
        return data['close'] / sma
    
    @feature_calculation
    def trix(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Triple Exponential Average (TRIX)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - trix_period: Period for TRIX calculation (default: 14)
        
        Returns:
            Series containing TRIX values
        """
        period = params.get('trix_period', 14)
        return ta.trix(data['close'], length=period)
    
    @feature_calculation
    def rate_of_change(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Rate of Change (ROC)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - roc_period: Period for ROC calculation (default: 10)
        
        Returns:
            Series containing ROC values
        """
        period = params.get('roc_period', 10)
        return ta.roc(data['close'], length=period)
    
    @feature_calculation
    def awesome_oscillator(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Awesome Oscillator
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - ao_fast_period: Fast period (default: 5)
                - ao_slow_period: Slow period (default: 34)
        
        Returns:
            Series containing Awesome Oscillator values
        """
        fast_period = params.get('ao_fast_period', 5)
        slow_period = params.get('ao_slow_period', 34)
        
        # Calculate median price
        median_price = (data['high'] + data['low']) / 2
        
        # Calculate SMAs
        fast_sma = median_price.rolling(fast_period).mean()
        slow_sma = median_price.rolling(slow_period).mean()
        
        # Calculate AO
        ao = fast_sma - slow_sma
        
        return ao
    
    @feature_calculation
    def trend_strength(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Trend Strength Indicator (calculated from ADX)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - adx_period: Period for ADX calculation (default: 14)
                - trend_threshold: Threshold for strong trend (default: 25)
        
        Returns:
            Series containing trend strength values (0-1)
        """
        period = params.get('adx_period', 14)
        threshold = params.get('trend_threshold', 25)
        
        # Calculate ADX
        adx = ta.adx(high=data['high'], low=data['low'], close=data['close'], length=period)
        
        # Normalize to 0-1 range with threshold
        trend_strength = (adx - threshold).clip(lower=0) / (100 - threshold)
        
        return trend_strength
    
    @feature_calculation
    def supertrend(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        SuperTrend Indicator
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - supertrend_period: Period for ATR calculation (default: 10)
                - supertrend_multiplier: ATR multiplier (default: 3.0)
        
        Returns:
            Series containing SuperTrend values
        """
        period = params.get('supertrend_period', 10)
        multiplier = params.get('supertrend_multiplier', 3.0)
        
        # Calculate ATR
        atr = ta.atr(high=data['high'], low=data['low'], close=data['close'], length=period)
        
        # Calculate basic upper and lower bands
        hl2 = (data['high'] + data['low']) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        # Initialize SuperTrend series
        supertrend = pd.Series(0.0, index=data.index)
        direction = pd.Series(1, index=data.index)  # 1 for uptrend, -1 for downtrend
        
        # Calculate SuperTrend
        for i in range(period, len(data)):
            # Adjust upper and lower bands based on previous values
            if basic_upper.iloc[i] < supertrend.iloc[i-1] or data['close'].iloc[i-1] > supertrend.iloc[i-1]:
                upper = basic_upper.iloc[i]
            else:
                upper = supertrend.iloc[i-1]
                
            if basic_lower.iloc[i] > supertrend.iloc[i-1] or data['close'].iloc[i-1] < supertrend.iloc[i-1]:
                lower = basic_lower.iloc[i]
            else:
                lower = supertrend.iloc[i-1]
            
            # Determine trend direction
            if supertrend.iloc[i-1] == upper:
                # Previous trend was down
                if data['close'].iloc[i] > upper:
                    # Trend reversal to up
                    supertrend.iloc[i] = lower
                    direction.iloc[i] = 1
                else:
                    # Trend continues down
                    supertrend.iloc[i] = upper
                    direction.iloc[i] = -1
            else:
                # Previous trend was up
                if data['close'].iloc[i] < lower:
                    # Trend reversal to down
                    supertrend.iloc[i] = upper
                    direction.iloc[i] = -1
                else:
                    # Trend continues up
                    supertrend.iloc[i] = lower
                    direction.iloc[i] = 1
        
        return supertrend
    
    @feature_calculation
    def supertrend_direction(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        SuperTrend Direction
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - supertrend_period: Period for ATR calculation (default: 10)
                - supertrend_multiplier: ATR multiplier (default: 3.0)
        
        Returns:
            Series containing SuperTrend direction values (1 for uptrend, -1 for downtrend)
        """
        period = params.get('supertrend_period', 10)
        multiplier = params.get('supertrend_multiplier', 3.0)
        
        # Calculate SuperTrend
        supertrend = self.supertrend(data, params)
        
        # Determine direction (1 for uptrend, -1 for downtrend)
        direction = pd.Series(0, index=data.index)
        direction[data['close'] > supertrend] = 1
        direction[data['close'] <= supertrend] = -1
        
        return direction
    
    #
    # Volatility Features
    #
    
    @feature_calculation
    def volatility(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Price Volatility (standard deviation of returns)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - volatility_period: Period for volatility calculation (default: 20)
                - volatility_scale: Annualization factor (default: 1.0)
        
        Returns:
            Series containing volatility values
        """
        period = params.get('volatility_period', 20)
        scale = params.get('volatility_scale', 1.0)
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Calculate rolling standard deviation
        volatility = returns.rolling(window=period).std() * np.sqrt(scale)
        
        return volatility
    
    @feature_calculation
    def normalized_volatility(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Normalized Volatility (relative to historical levels)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - volatility_period: Period for volatility calculation (default: 20)
                - normalization_period: Period for normalization (default: 100)
        
        Returns:
            Series containing normalized volatility values
        """
        period = params.get('volatility_period', 20)
        norm_period = params.get('normalization_period', 100)
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Calculate rolling standard deviation
        vol = returns.rolling(window=period).std()
        
        # Calculate rolling mean and std of volatility for normalization
        vol_mean = vol.rolling(window=norm_period).mean()
        vol_std = vol.rolling(window=norm_period).std()
        
        # Normalize volatility
        norm_vol = (vol - vol_mean) / vol_std
        
        return norm_vol
    
    @feature_calculation
    def parkinson_volatility(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Parkinson Volatility (using high-low range)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - parkinson_period: Period for volatility calculation (default: 20)
                - parkinson_scale: Annualization factor (default: 252)
        
        Returns:
            Series containing Parkinson volatility values
        """
        period = params.get('parkinson_period', 20)
        scale = params.get('parkinson_scale', 252)
        
        # Calculate log high/low ratio squared
        log_hl_ratio = np.log(data['high'] / data['low'])
        log_hl_ratio_sq = log_hl_ratio ** 2
        
        # Calculate Parkinson volatility
        parkinsons = np.sqrt(
            (1 / (4 * np.log(2))) * 
            log_hl_ratio_sq.rolling(window=period).mean() * 
            scale
        )
        
        return parkinsons
    
    @feature_calculation
    def garman_klass_volatility(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Garman-Klass Volatility
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - gk_period: Period for volatility calculation (default: 20)
                - gk_scale: Annualization factor (default: 252)
        
        Returns:
            Series containing Garman-Klass volatility values
        """
        period = params.get('gk_period', 20)
        scale = params.get('gk_scale', 252)
        
        # Calculate log values
        log_hl = np.log(data['high'] / data['low']) ** 2
        log_co = np.log(data['close'] / data['open']) ** 2
        
        # Calculate Garman-Klass estimator components
        gk_components = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        
        # Calculate volatility
        gk_vol = np.sqrt(gk_components.rolling(window=period).mean() * scale)
        
        return gk_vol
    
    #
    # Volume Features
    #
    
    @feature_calculation
    def volume_profile(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Volume Profile (normalized relative to recent volume)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - volume_profile_period: Period for volume normalization (default: 20)
        
        Returns:
            Series containing normalized volume values
        """
        period = params.get('volume_profile_period', 20)
        
        # Calculate rolling average volume
        avg_volume = data['volume'].rolling(window=period).mean()
        
        # Calculate normalized volume
        norm_volume = data['volume'] / avg_volume
        
        return norm_volume
    
    @feature_calculation
    def volume_oscillator(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Volume Oscillator
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - vol_osc_fast_period: Fast period (default: 5)
                - vol_osc_slow_period: Slow period (default: 20)
        
        Returns:
            Series containing volume oscillator values
        """
        fast_period = params.get('vol_osc_fast_period', 5)
        slow_period = params.get('vol_osc_slow_period', 20)
        
        # Calculate fast and slow EMAs of volume
        fast_ema = data['volume'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['volume'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate oscillator
        oscillator = ((fast_ema - slow_ema) / slow_ema) * 100
        
        return oscillator
    
    @feature_calculation
    def volume_zone_oscillator(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Volume Zone Oscillator (VZO)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - vzo_period: Period for VZO calculation (default: 14)
                - vzo_signal_period: Signal line period (default: 9)
        
        Returns:
            Series containing VZO values
        """
        period = params.get('vzo_period', 14)
        signal_period = params.get('vzo_signal_period', 9)
        
        # Determine positive/negative volume
        price_change = data['close'].diff()
        positive_volume = np.where(price_change > 0, data['volume'], 0)
        negative_volume = np.where(price_change < 0, data['volume'], 0)
        
        # Create DataFrame for volume series
        vol_df = pd.DataFrame({
            'positive': positive_volume,
            'negative': negative_volume,
            'total': data['volume']
        }, index=data.index)
        
        # Calculate EMAs
        positive_ema = vol_df['positive'].ewm(span=period, adjust=False).mean()
        negative_ema = vol_df['negative'].ewm(span=period, adjust=False).mean()
        total_ema = vol_df['total'].ewm(span=period, adjust=False).mean()
        
        # Calculate VZO
        vzo = ((positive_ema - negative_ema) / total_ema) * 100
        
        return vzo
    
    @feature_calculation
    def money_flow_index(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Money Flow Index (MFI)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - mfi_period: Period for MFI calculation (default: 14)
        
        Returns:
            Series containing MFI values
        """
        period = params.get('mfi_period', 14)
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * data['volume']
        
        # Determine positive and negative money flow
        price_change = typical_price.diff()
        positive_flow = np.where(price_change > 0, raw_money_flow, 0)
        negative_flow = np.where(price_change < 0, raw_money_flow, 0)
        
        # Create DataFrame for flows
        flow_df = pd.DataFrame({
            'positive': positive_flow,
            'negative': negative_flow
        }, index=data.index)
        
        # Calculate rolling sums
        positive_sum = flow_df['positive'].rolling(window=period).sum()
        negative_sum = flow_df['negative'].rolling(window=period).sum()
        
        # Calculate money flow ratio and index
        money_flow_ratio = positive_sum / negative_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    #
    # Pattern Features
    #
    
    @feature_calculation
    def zigzag_pattern(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        ZigZag Pattern Detection
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - zigzag_pct: Percentage for zigzag identification (default: 5.0)
                - zigzag_backstep: Backstep lookback (default: 3)
        
        Returns:
            Series with zigzag pattern points (1: peak, -1: trough, 0: none)
        """
        threshold_pct = params.get('zigzag_pct', 5.0) / 100.0
        backstep = params.get('zigzag_backstep', 3)
        
        # Initialize series
        zigzag = pd.Series(0, index=data.index)
        
        # Price high and low
        price_high = data['high']
        price_low = data['low']
        
        # Initial values
        last_peak_idx = 0
        last_trough_idx = 0
        last_peak_price = price_high.iloc[0]
        last_trough_price = price_low.iloc[0]
        
        # Current direction (1: looking for peak, -1: looking for trough)
        direction = 1
        
        # Iterate through prices
        for i in range(backstep, len(data)):
            if direction == 1:  # Looking for a peak
                # New peak found
                if price_high.iloc[i] > last_peak_price:
                    last_peak_idx = i
                    last_peak_price = price_high.iloc[i]
                # Significant drop from peak, confirming a peak
                elif price_low.iloc[i] < (last_peak_price * (1 - threshold_pct)):
                    # Mark the peak
                    zigzag.iloc[last_peak_idx] = 1
                    # Switch direction to look for trough
                    direction = -1
                    last_trough_idx = i
                    last_trough_price = price_low.iloc[i]
            else:  # Looking for a trough
                # New trough found
                if price_low.iloc[i] < last_trough_price:
                    last_trough_idx = i
                    last_trough_price = price_low.iloc[i]
                # Significant rise from trough, confirming a trough
                elif price_high.iloc[i] > (last_trough_price * (1 + threshold_pct)):
                    # Mark the trough
                    zigzag.iloc[last_trough_idx] = -1
                    # Switch direction to look for peak
                    direction = 1
                    last_peak_idx = i
                    last_peak_price = price_high.iloc[i]
        
        return zigzag
    
    @feature_calculation
    def divergence_rsi(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        RSI Divergence Detection
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - rsi_period: Period for RSI calculation (default: 14)
                - divergence_lookback: Lookback period for extremes (default: 20)
        
        Returns:
            Series with divergence signals (1: bullish, -1: bearish, 0: none)
        """
        rsi_period = params.get('rsi_period', 14)
        lookback = params.get('divergence_lookback', 20)
        
        # Calculate RSI
        rsi = ta.rsi(data['close'], length=rsi_period)
        
        # Initialize divergence series
        divergence = pd.Series(0, index=data.index)
        
        # Need at least 2*lookback data points
        if len(data) < 2 * lookback:
            return divergence
        
        # Iterate through data points, starting after we have enough data
        for i in range(2 * lookback, len(data)):
            # Check the window
            window = slice(i - lookback, i + 1)
            
            # Find local price extremes
            price_max_idx = data['high'].iloc[window].idxmax()
            price_min_idx = data['low'].iloc[window].idxmin()
            
            # Find RSI extremes
            rsi_max_idx = rsi.iloc[window].idxmax()
            rsi_min_idx = rsi.iloc[window].idxmin()
            
            # Check for bullish divergence (price makes lower low but RSI makes higher low)
            if price_min_idx == data.index[i] and rsi_min_idx != price_min_idx:
                # Confirm price made lower low but RSI did not
                prev_window = slice(i - 2*lookback, i - lookback + 1)
                prev_price_min_idx = data['low'].iloc[prev_window].idxmin()
                prev_rsi_min_idx = rsi.iloc[prev_window].idxmin()
                
                if (data['low'].loc[price_min_idx] < data['low'].loc[prev_price_min_idx] and 
                    rsi.loc[rsi_min_idx] > rsi.loc[prev_rsi_min_idx]):
                    divergence.iloc[i] = 1  # Bullish divergence
            
            # Check for bearish divergence (price makes higher high but RSI makes lower high)
            if price_max_idx == data.index[i] and rsi_max_idx != price_max_idx:
                # Confirm price made higher high but RSI did not
                prev_window = slice(i - 2*lookback, i - lookback + 1)
                prev_price_max_idx = data['high'].iloc[prev_window].idxmax()
                prev_rsi_max_idx = rsi.iloc[prev_window].idxmax()
                
                if (data['high'].loc[price_max_idx] > data['high'].loc[prev_price_max_idx] and 
                    rsi.loc[rsi_max_idx] < rsi.loc[prev_rsi_max_idx]):
                    divergence.iloc[i] = -1  # Bearish divergence
        
        return divergence
    
    @feature_calculation
    def elliott_wave_degree(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Elliott Wave Degree Estimation
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - elliott_pct: Percentage for wave identification (default: 3.0)
                - elliott_smoothing: Smoothing period (default: 5)
        
        Returns:
            Series with estimated Elliott Wave degree
        """
        threshold_pct = params.get('elliott_pct', 3.0) / 100.0
        smoothing = params.get('elliott_smoothing', 5)
        
        # Smooth the price
        smooth_price = data['close'].rolling(window=smoothing).mean()
        
        # Initialize wave counting
        wave_degree = pd.Series(0, index=data.index)
        
        # Need enough data for waves
        if len(data) < 5 * smoothing:
            return wave_degree
        
        # Detect pivots using rate of change
        smooth_roc = smooth_price.diff(smoothing).rolling(window=smoothing).mean()
        pivot_points = (smooth_roc.shift(1) * smooth_roc < 0).astype(int)
        
        # Count sequence of pivots to estimate wave degree
        pivot_count = pivot_points.rolling(window=13, min_periods=1).sum()
        
        # Scale to 1-5 for typical Elliott Wave degrees
        # 1-2: Initial waves, 3: Strongest, 4-5: Corrective
        wave_degree = (pivot_count % 8) + 1
        
        return wave_degree
    
    #
    # Specialized Market Features
    #
    
    @feature_calculation
    def liquidity_ratio(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Liquidity Ratio (volume / price range)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - liquidity_period: Period for calculation (default: 14)
        
        Returns:
            Series with liquidity ratio values
        """
        period = params.get('liquidity_period', 14)
        
        # Calculate price range
        price_range = data['high'] - data['low']
        
        # Calculate ratio of volume to price range
        liquidity = data['volume'] / (price_range + 1e-10)  # Avoid division by zero
        
        # Smooth the ratio
        smoothed_liquidity = liquidity.rolling(window=period).mean()
        
        return smoothed_liquidity
    
    @feature_calculation
    def market_facilitation_index(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Market Facilitation Index (MFI) by Bill Williams
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - mfi_scaling: Scaling factor (default: 100.0)
        
        Returns:
            Series with MFI values
        """
        scaling = params.get('mfi_scaling', 100.0)
        
        # Calculate MFI: (high - low) / volume
        price_range = data['high'] - data['low']
        mfi = (price_range / (data['volume'] + 1e-10)) * scaling
        
        return mfi
    
    @feature_calculation
    def relative_volume(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Relative Volume (compared to average)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - rel_volume_period: Period for average calculation (default: 20)
        
        Returns:
            Series with relative volume values
        """
        period = params.get('rel_volume_period', 20)
        
        # Calculate average volume
        avg_volume = data['volume'].rolling(window=period).mean()
        
        # Calculate relative volume
        rel_volume = data['volume'] / avg_volume
        
        return rel_volume
    
    @feature_calculation
    def vwap(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP)
        
        Args:
            data: OHLCV DataFrame
            params: Parameters including:
                - vwap_period: Period for VWAP calculation (default: 'day')
        
        Returns:
            Series with VWAP values
        """
        # For simplicity, calculate a rolling VWAP
        period = params.get('vwap_period', 24)  # Default to one day (24 hours for crypto)
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate VWAP components
        price_volume = typical_price * data['volume']
        
        # Calculate cumulative values
        cum_price_volume = price_volume.rolling(window=period).sum()
        cum_volume = data['volume'].rolling(window=period).sum()
        
        # Calculate VWAP
        vwap = cum_price_volume / (cum_volume + 1e-10)  # Avoid division by zero

        return vwap



def atr(high: Union[pd.Series, List[float]],
        low: Union[pd.Series, List[float]],
        close: Union[pd.Series, List[float]],
        period: int = 14) -> pd.Series:
    """Standalone ATR calculator for easy reuse."""
    data = pd.DataFrame({'high': high, 'low': low, 'close': close})
    return ta.atr(high=data['high'], low=data['low'], close=data['close'], length=period)


def fibonacci_levels(high: float, low: float) -> List[float]:
    """Calculate Fibonacci retracement levels."""
    levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    diff = high - low
    return [high - diff * lvl for lvl in levels]
# Convenience wrapper functions

def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR directly."""
    return ta.atr(high=data['high'], low=data['low'], close=data['close'], length=period)


def fibonacci_levels(price: float, ratios: Optional[List[float]] = None) -> Dict[str, float]:
    """Generate basic Fibonacci retracement levels."""
    if ratios is None:
        ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    return {str(r): price * r for r in ratios}

def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Convenience wrapper for ATR calculation."""
    return ta.atr(high=data['high'], low=data['low'], close=data['close'], length=period)


def fibonacci_levels(high: float, low: float) -> List[float]:
    """Simple Fibonacci retracement level calculator."""
    diff = high - low
    return [high - diff * r for r in [0.236, 0.382, 0.5, 0.618, 0.786]]

def fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate common Fibonacci retracement levels."""
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
    diff = high - low
    return {str(r): high - diff * r for r in ratios}


__all__ = [
    'FeatureExtractor', 'atr', 'fibonacci_levels'
]

#
# Cross-Asset Features
#

@feature_calculation
def pair_correlation(
    self,
    data: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.Series:
    """Rolling correlation between two assets."""
    other = params.get("other_series")
    window = params.get("corr_window", 30)
    if other is None:
        raise ValueError("other_series parameter is required for pair correlation")
    return compute_pair_correlation(data["close"], other, window=window)

@feature_calculation
def cointegration_pvalue(
    self,
    data: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.Series:
    """Cointegration test p-value between two assets."""
    other = params.get("other_series")
    if other is None:
        raise ValueError("other_series parameter is required for cointegration")
    pval = cointegration_score(data["close"], other)
    return pd.Series([pval] * len(data), index=data.index, name="cointegration_pvalue")
@feature_calculation
def pair_correlation(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Correlation between this asset and another."""
    pair_data = params.get("pair_data")
    column = params.get("cross_column", "close")
    if pair_data is None:
        raise ValueError("pair_data parameter required for pair_correlation")
    """Correlation between this asset and a paired asset."""
    pair_data = params.get("pair_data")
    column = params.get("pair_column", "close")
    if pair_data is None:
        raise ValueError("pair_data parameter is required for pair_correlation")
    corr = compute_pair_correlation(data, pair_data, column=column)
    return pd.Series(corr, index=data.index, name="pair_correlation")

@feature_calculation
def cointegration_pvalue(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """Cointegration test p-value with another asset."""
    pair_data = params.get("pair_data")
    column = params.get("cross_column", "close")
    if pair_data is None:
        raise ValueError("pair_data parameter required for cointegration_pvalue")
    pvalue = cointegration_score(data, pair_data, column=column)
    return pd.Series(pvalue, index=data.index, name="cointegration_pvalue")

    """Engle-Granger cointegration p-value between two assets."""
    pair_data = params.get("pair_data")
    column = params.get("pair_column", "close")
    if pair_data is None:
        raise ValueError("pair_data parameter is required for cointegration_pvalue")
    pval = cointegration_score(data, pair_data, column=column)
    return pd.Series(pval, index=data.index, name="cointegration_pvalue")

__all__ = ['atr', 'fibonacci_levels']
