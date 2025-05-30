

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Advanced Data Aggregation and Transformation

This module provides sophisticated aggregation functions for time series data
transformation, enabling complex pattern recognition and feature extraction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Callable, Optional, Tuple, Any
from functools import partial
import numba
from numba import jit, cuda
import bottleneck as bn
import scipy.signal as signal
import scipy.stats as stats
from dataclasses import dataclass

from common.logger import get_logger
from common.utils import (
    optimize_numeric_dtypes, parallel_apply, calculate_memory_usage,
    TimeseriesValidator
)
from common.constants import TIMEFRAMES, DEFAULT_WINDOW_SIZES
from feature_service.transformers.normalizers import normalize_series

logger = get_logger(__name__)


@dataclass
class AggregationResult:
    """Container for aggregation results with metadata."""
    data: Union[pd.DataFrame, pd.Series]
    computation_time: float
    memory_usage: float
    aggregation_type: str
    parameters: Dict
    input_shape: Tuple
    output_shape: Tuple


class TimeseriesAggregator:
    """
    Advanced time series aggregation engine with GPU acceleration capabilities.
    
    This class provides high-performance data aggregation functions with 
    automatic hardware acceleration detection and optimization.
    """
    
    def __init__(
        self, 
        use_gpu: bool = True,
        precision: str = 'float32',
        parallel_threshold: int = 100000,
        memory_optimized: bool = True
    ):
        """
        Initialize the TimeseriesAggregator.
        
        Args:
            use_gpu: Whether to use GPU acceleration when available
            precision: Numeric precision for calculations
            parallel_threshold: Minimum size for parallel processing
            memory_optimized: Whether to optimize memory usage
        """
        self.use_gpu = use_gpu and self._check_gpu_availability()
        self.precision = precision
        self.parallel_threshold = parallel_threshold
        self.memory_optimized = memory_optimized
        self.validator = TimeseriesValidator()
        logger.info(f"TimeseriesAggregator initialized with GPU support: {self.use_gpu}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            gpu_count = len(cuda.gpus)
            return gpu_count > 0
        except Exception:
            return False
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _numba_rolling_apply(values, window, func):
        """JIT-compiled rolling window calculation."""
        result = np.empty_like(values)
        result[:window-1] = np.nan
        
        for i in range(window-1, len(values)):
            result[i] = func(values[i-window+1:i+1])
            
        return result
    
    def rolling_apply(
        self, 
        data: Union[pd.Series, np.ndarray], 
        window: int, 
        func: Callable,
        center: bool = False,
        min_periods: Optional[int] = None,
        cuda_optimized: bool = False
    ) -> np.ndarray:
        """
        Apply a function over a rolling window with optimization.
        
        Args:
            data: Input time series data
            window: Window size for rolling calculation
            func: Function to apply to each window
            center: Whether to center the window
            min_periods: Minimum number of observations required
            cuda_optimized: Whether the function has a CUDA implementation
            
        Returns:
            Array of results from rolling calculation
        """
        if min_periods is None:
            min_periods = window
            
        data_values = data.values if isinstance(data, (pd.Series, pd.DataFrame)) else data
        data_size = len(data_values)
        
        # For small datasets, use standard rolling
        if data_size < self.parallel_threshold:
            if isinstance(data, pd.Series):
                return data.rolling(window=window, center=center, min_periods=min_periods).apply(
                    func, raw=True
                ).values
            else:
                return pd.Series(data_values).rolling(
                    window=window, center=center, min_periods=min_periods
                ).apply(func, raw=True).values
        
        # Use GPU if available and function is optimized for it
        if self.use_gpu and cuda_optimized:
            return self._cuda_rolling_apply(data_values, window, func, center, min_periods)
        
        # Use Numba JIT compilation for CPU optimization
        return self._numba_rolling_apply(data_values, window, func)
    
    def _cuda_rolling_apply(self, values, window, func, center, min_periods):
        """CUDA-accelerated rolling window calculation."""
        # Implementation depends on specific CUDA kernels for the function
        # This is a placeholder for function-specific CUDA implementations
        raise NotImplementedError("Specific CUDA implementation required")
    
    def volume_weighted_average(
        self, 
        price: pd.Series, 
        volume: pd.Series, 
        window: int = 20
    ) -> pd.Series:
        """
        Calculate volume-weighted average price over a window.
        
        Args:
            price: Price series
            volume: Volume series
            window: Window size for calculation
            
        Returns:
            Volume-weighted average price series
        """
        self.validator.validate_equal_length(price, volume)
        vwap = (price * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
        return vwap
    
    def time_weighted_average(
        self, 
        data: pd.Series, 
        window: int = 20,
        decay_factor: float = 0.94
    ) -> pd.Series:
        """
        Calculate time-weighted average with exponential decay.
        
        Args:
            data: Input time series
            window: Window size for calculation
            decay_factor: Exponential decay factor (between 0 and 1)
            
        Returns:
            Time-weighted average series
        """
        # Create weights with exponential decay
        weights = np.array([decay_factor ** i for i in range(window)])
        weights = weights[::-1] / weights.sum()
        
        # Calculate weighted rolling average
        twap = data.rolling(window=window).apply(
            lambda x: np.sum(weights[:len(x)] * x) / np.sum(weights[:len(x)]),
            raw=True
        )
        return twap
    
    def adaptive_moving_average(
        self, 
        data: pd.Series, 
        fast_period: int = 9,
        slow_period: int = 30,
        volatility_window: int = 20
    ) -> pd.Series:
        """
        Calculate adaptive moving average that adjusts to volatility.
        
        Args:
            data: Input time series
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
            volatility_window: Window for volatility calculation
            
        Returns:
            Adaptive moving average series
        """
        # Calculate volatility as rolling standard deviation
        volatility = data.rolling(window=volatility_window).std()
        normalized_volatility = volatility / volatility.rolling(window=volatility_window).max()
        
        # Calculate fast and slow MAs
        fast_ma = data.rolling(window=fast_period).mean()
        slow_ma = data.rolling(window=slow_period).mean()
        
        # Blend based on volatility (higher volatility = more weight to fast MA)
        adaptive_ma = (
            normalized_volatility * fast_ma + 
            (1 - normalized_volatility) * slow_ma
        )
        
        return adaptive_ma
    
    def hull_moving_average(
        self, 
        data: pd.Series, 
        period: int = 20
    ) -> pd.Series:
        """
        Calculate Hull Moving Average for reduced lag.
        
        Args:
            data: Input time series
            period: Period for calculation
            
        Returns:
            Hull moving average series
        """
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # Calculate weighted moving averages
        wma1 = self.weighted_moving_average(data, half_period)
        wma2 = self.weighted_moving_average(data, period)
        
        # Calculate raw HMA
        raw_hma = 2 * wma1 - wma2
        
        # Calculate final HMA
        hma = self.weighted_moving_average(raw_hma, sqrt_period)
        
        return hma
    
    def weighted_moving_average(
        self, 
        data: pd.Series, 
        period: int = 20
    ) -> pd.Series:
        """
        Calculate linearly weighted moving average.
        
        Args:
            data: Input time series
            period: Period for calculation
            
        Returns:
            Weighted moving average series
        """
        weights = np.arange(1, period + 1)
        wma = data.rolling(window=period).apply(
            lambda x: np.sum(weights[:len(x)] * x) / np.sum(weights[:len(x)]),
            raw=True
        )
        return wma
    
    def zero_lag_moving_average(
        self, 
        data: pd.Series, 
        period: int = 20, 
        alpha: float = 0.7
    ) -> pd.Series:
        """
        Calculate Zero Lag Exponential Moving Average.
        
        Args:
            data: Input time series
            period: Period for calculation
            alpha: Smoothing factor
            
        Returns:
            Zero lag EMA series
        """
        ema = data.ewm(span=period, adjust=False).mean()
        
        # Calculate error correction
        error = data - ema
        
        # Apply error correction to remove lag
        zlema = ema + alpha * error
        
        return zlema
    
    def variable_index_dynamic_average(
        self, 
        data: pd.Series, 
        short_period: int = 7,
        long_period: int = 14,
        alpha: float = 0.3
    ) -> pd.Series:
        """
        Calculate Variable Index Dynamic Average (VIDYA).
        
        Args:
            data: Input time series
            short_period: Short period for CMO calculation
            long_period: Long period for VIDYA
            alpha: Smoothing factor
            
        Returns:
            VIDYA series
        """
        # Calculate Chande Momentum Oscillator
        ups = np.zeros(len(data))
        downs = np.zeros(len(data))
        
        diff = data.diff().fillna(0).values
        ups[diff > 0] = diff[diff > 0]
        downs[diff < 0] = -diff[diff < 0]
        
        up_sum = pd.Series(ups).rolling(window=short_period).sum()
        down_sum = pd.Series(downs).rolling(window=short_period).sum()
        
        # CMO calculation
        cmo = ((up_sum - down_sum) / (up_sum + down_sum)).abs().fillna(0)
        
        # Calculate VIDYA
        vidya = pd.Series(index=data.index, dtype='float64')
        vidya.iloc[long_period-1] = data.iloc[:long_period].mean()
        
        # Iterative calculation
        for i in range(long_period, len(data)):
            k = alpha * cmo.iloc[i]
            vidya.iloc[i] = k * data.iloc[i] + (1 - k) * vidya.iloc[i-1]
            
        return vidya
    
    def fractal_adaptive_moving_average(
        self, 
        data: pd.Series, 
        period: int = 20,
        fc: float = 0.5
    ) -> pd.Series:
        """
        Calculate Fractal Adaptive Moving Average (FRAMA).
        
        Args:
            data: Input time series
            period: Period for calculation
            fc: Fractal constant
            
        Returns:
            FRAMA series
        """
        window = period * 2  # We need double the period for calculation
        
        # Initialize result series
        frama = pd.Series(index=data.index, dtype='float64')
        frama.iloc[window-1] = data.iloc[:window].mean()
        
        # Calculate fractal dimension for each window
        for i in range(window, len(data)):
            # Split the data into two halves
            segment = data.iloc[i-window:i]
            half_len = len(segment) // 2
            first_half = segment.iloc[:half_len]
            second_half = segment.iloc[half_len:]
            
            # Calculate price ranges
            high1 = first_half.max()
            low1 = first_half.min()
            high2 = second_half.max()
            low2 = second_half.min()
            high = segment.max()
            low = segment.min()
            
            # Calculate fractal dimension
            n3 = (high - low) / period
            n1 = (high1 - low1) / (period / 2)
            n2 = (high2 - low2) / (period / 2)
            
            if n1 > 0 and n2 > 0 and n3 > 0:
                dim = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
            else:
                dim = 1.0
                
            # Calculate alpha based on fractal dimension
            alpha = np.exp(-4.6 * (dim - 1))
            alpha = max(min(alpha, 1.0), 0.01)  # Bound between 0.01 and 1.0
            
            # Calculate FRAMA
            frama.iloc[i] = alpha * data.iloc[i] + (1 - alpha) * frama.iloc[i-1]
            
        return frama
    
    def kaufman_adaptive_moving_average(
        self, 
        data: pd.Series, 
        period: int = 20,
        fast_ef: float = 0.67,
        slow_ef: float = 0.05
    ) -> pd.Series:
        """
        Calculate Kaufman's Adaptive Moving Average (KAMA).
        
        Args:
            data: Input time series
            period: Period for efficiency ratio calculation
            fast_ef: Fast efficiency factor
            slow_ef: Slow efficiency factor
            
        Returns:
            KAMA series
        """
        # Calculate direction and volatility
        change = (data - data.shift(period)).abs()
        volatility = data.diff().abs().rolling(window=period).sum()
        
        # Calculate efficiency ratio
        er = pd.Series(index=data.index, dtype='float64')
        er[volatility > 0] = change[volatility > 0] / volatility[volatility > 0]
        er[volatility == 0] = 1.0
        er = er.fillna(0)
        
        # Calculate smoothing constant
        sc = (er * (fast_ef - slow_ef) + slow_ef) ** 2
        
        # Initialize KAMA
        kama = pd.Series(index=data.index, dtype='float64')
        kama.iloc[period-1] = data.iloc[period-1]
        
        # Calculate KAMA
        for i in range(period, len(data)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (data.iloc[i] - kama.iloc[i-1])
            
        return kama
    
    def triple_exponential_moving_average(
        self, 
        data: pd.Series, 
        period: int = 20
    ) -> pd.Series:
        """
        Calculate Triple Exponential Moving Average (TEMA).
        
        Args:
            data: Input time series
            period: Period for calculation
            
        Returns:
            TEMA series
        """
        # Calculate EMAs
        ema1 = data.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        
        # Calculate TEMA
        tema = 3 * ema1 - 3 * ema2 + ema3
        
        return tema
    
    def median_adaptive_moving_average(
        self, 
        data: pd.Series, 
        period: int = 20,
        fast_period: int = 5,
        slow_period: int = 30
    ) -> pd.Series:
        """
        Calculate Median Adaptive Moving Average.
        
        Args:
            data: Input time series
            period: Main calculation period
            fast_period: Fast median period
            slow_period: Slow median period
            
        Returns:
            Median adaptive moving average series
        """
        # Calculate volatility signal
        volatility = data.rolling(window=period).std() / data.rolling(window=period).mean()
        normalized_vol = volatility / volatility.rolling(window=period).max().fillna(0)
        
        # Calculate fast and slow medians
        fast_median = data.rolling(window=fast_period).median()
        slow_median = data.rolling(window=slow_period).median()
        
        # Blend based on volatility
        mama = normalized_vol * fast_median + (1 - normalized_vol) * slow_median
        
        return mama
    
    def correlation_weighted_average(
        self, 
        primary: pd.Series, 
        reference: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate correlation-weighted average based on correlation
        with a reference series.
        
        Args:
            primary: Primary time series
            reference: Reference time series
            window: Correlation window
            
        Returns:
            Correlation-weighted average series
        """
        self.validator.validate_equal_length(primary, reference)
        
        # Initialize result series
        cwa = pd.Series(index=primary.index, dtype='float64')
        
        # Calculate rolling correlation and transformed weight
        roll_corr = primary.rolling(window=window).corr(reference).fillna(0)
        
        # Transform correlation to weight (higher correlation = higher weight)
        # Square and normalize to 0.2-1.0 range to avoid extreme values
        weights = (0.8 * (roll_corr ** 2) + 0.2).clip(0.2, 1.0)
        
        # Calculate SMA as the base
        sma = primary.rolling(window=window).mean()
        
        # Apply correlation-based adjustments
        adjustment = weights * (primary - sma)
        cwa = sma + adjustment * 0.5  # Scale adjustment by 0.5 to moderate effect
        
        return cwa
    
    def multi_timeframe_aggregation(
        self, 
        data: pd.Series, 
        timeframes: List[str] = None,
        aggregation_func: Callable = np.mean,
        weights: List[float] = None
    ) -> pd.Series:
        """
        Perform multi-timeframe aggregation with optional weighting.
        
        Args:
            data: Input high-frequency time series
            timeframes: List of timeframes to aggregate to
            aggregation_func: Function to use for aggregation
            weights: Optional weights for each timeframe
            
        Returns:
            Aggregated and resampled back to original series
        """
        if timeframes is None:
            timeframes = ['5min', '15min', '1H', '4H']
            
        if weights is None:
            # Default to equal weights
            weights = [1.0 / len(timeframes)] * len(timeframes)
        
        # Ensure weights sum to 1
        weights = np.array(weights) / np.sum(weights)
        
        # Store original index for reindexing
        original_index = data.index
        
        # Create aggregated versions for each timeframe
        aggregated_data = []
        
        for tf in timeframes:
            # Resample to larger timeframe
            resampled = data.resample(tf).apply(aggregation_func)
            
            # Forward fill to original frequency and reindex
            filled = resampled.reindex(original_index, method='ffill')
            
            aggregated_data.append(filled)
            
        # Combine with weights
        result = pd.Series(0.0, index=original_index)
        
        for i, agg_data in enumerate(aggregated_data):
            result += weights[i] * agg_data
            
        return result
    
    def adaptive_threshold_aggregation(
        self, 
        data: pd.Series, 
        base_window: int = 20,
        threshold_factor: float = 1.5
    ) -> pd.Series:
        """
        Aggregate data with adaptive threshold based on volatility.
        
        Args:
            data: Input time series
            base_window: Base window size
            threshold_factor: Factor to multiply volatility by for threshold
            
        Returns:
            Adaptively aggregated series
        """
        # Calculate volatility
        volatility = data.rolling(window=base_window).std()
        
        # Calculate adaptive window size (larger during high volatility)
        adaptive_window = (base_window * (1 + volatility / data.rolling(window=base_window).mean() * threshold_factor)).fillna(base_window).astype(int)
        adaptive_window = adaptive_window.clip(lower=base_window // 2, upper=base_window * 3)
        
        # Apply adaptive window
        result = pd.Series(index=data.index, dtype='float64')
        
        for i in range(len(data)):
            if i < base_window:
                result.iloc[i] = data.iloc[:i+1].mean()
            else:
                window = min(adaptive_window.iloc[i], i+1)
                result.iloc[i] = data.iloc[i-window+1:i+1].mean()
                
        return result

    def regime_weighted_aggregation(
        self, 
        data: pd.Series, 
        regime_indicator: pd.Series,
        regime_windows: Dict[int, int] = None
    ) -> pd.Series:
        """
        Aggregate data with different windows based on regime.
        
        Args:
            data: Input time series
            regime_indicator: Series indicating market regime (integers)
            regime_windows: Dict mapping regime ID to window size
            
        Returns:
            Regime-adaptive aggregated series
        """
        self.validator.validate_equal_length(data, regime_indicator)
        
        if regime_windows is None:
            # Default windows for different regimes
            # 1: Trending, 2: Ranging, 3: Volatile
            regime_windows = {1: 10, 2: 20, 3: 5}
            
        # Apply regime-specific windows
        result = pd.Series(index=data.index, dtype='float64')
        
        for regime, window in regime_windows.items():
            # Get mask for this regime
            mask = regime_indicator == regime
            
            # Calculate MA for this regime with appropriate window
            if mask.any():
                regime_data = data.copy()
                ma = regime_data.rolling(window=window).mean()
                
                # Apply only to relevant regime
                result[mask] = ma[mask]
                
        # Fill any remaining values with default window
        default_window = int(np.median(list(regime_windows.values())))
        missing_mask = result.isna()
        
        if missing_mask.any():
            result[missing_mask] = data.rolling(window=default_window).mean()[missing_mask]
            
        return result
    
    def threshold_aggregation(
        self, 
        data: pd.Series, 
        threshold: float,
        upward_window: int = 10,
        downward_window: int = 20
    ) -> pd.Series:
        """
        Apply different aggregation windows based on threshold crossings.
        
        Args:
            data: Input time series
            threshold: Threshold value for crossing detection
            upward_window: Window size for upward crosses
            downward_window: Window size for downward crosses
            
        Returns:
            Threshold-adaptive aggregated series
        """
        # Detect threshold crossings
        above_threshold = data > threshold
        upward_cross = above_threshold & ~above_threshold.shift(1).fillna(False)
        downward_cross = ~above_threshold & above_threshold.shift(1).fillna(False)
        
        # Calculate states
        state = pd.Series(0, index=data.index)
        
        # Initialize with upward state where we start above threshold
        if above_threshold.iloc[0]:
            state.iloc[0] = 1
            
        # Process all crossings
        for i in range(1, len(data)):
            if upward_cross.iloc[i]:
                state.iloc[i:] = 1
            elif downward_cross.iloc[i]:
                state.iloc[i:] = -1
            else:
                state.iloc[i] = state.iloc[i-1]
                
        # Apply appropriate windows
        result = pd.Series(index=data.index, dtype='float64')
        
        # Upward state values
        upward_mask = state == 1
        if upward_mask.any():
            result[upward_mask] = data.rolling(window=upward_window).mean()[upward_mask]
            
        # Downward state values
        downward_mask = state == -1
        if downward_mask.any():
            result[downward_mask] = data.rolling(window=downward_window).mean()[downward_mask]
            
        # Initial state
        initial_mask = state == 0
        if initial_mask.any():
            # For initial state, use average of both windows
            avg_window = (upward_window + downward_window) // 2
            result[initial_mask] = data.rolling(window=avg_window).mean()[initial_mask]
            
        return result
    
    def piecewise_aggregation(
        self, 
        data: pd.Series,
        change_points: List[int] = None,
        min_segment_size: int = 20,
        max_change_points: int = 5
    ) -> pd.Series:
        """
        Apply piecewise aggregation with auto-detected change points.
        
        Args:
            data: Input time series
            change_points: Optional pre-calculated change points
            min_segment_size: Minimum size of any segment
            max_change_points: Maximum number of change points to detect
            
        Returns:
            Piecewise aggregated series
        """
        # Detect change points if not provided
        if change_points is None:
            change_points = self._detect_change_points(
                data, min_segment_size, max_change_points
            )
            
        # Add start and end points
        all_points = [0] + change_points + [len(data)]
        all_points.sort()
        
        # Calculate segment means
        result = pd.Series(index=data.index, dtype='float64')
        
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i+1]
            
            # Calculate segment mean
            segment_mean = data.iloc[start:end].mean()
            
            # Apply to this segment
            result.iloc[start:end] = segment_mean
            
        return result
    
    def _detect_change_points(
        self, 
        data: pd.Series,
        min_segment_size: int,
        max_change_points: int
    ) -> List[int]:
        """
        Detect change points in time series using binary segmentation.
        
        Args:
            data: Input time series
            min_segment_size: Minimum size of any segment
            max_change_points: Maximum number of change points
            
        Returns:
            List of change point indices
        """
        # Use binary segmentation algorithm for change point detection
        try:
            # Use scipy.signal.find_peaks for simple implementation
            # In a production environment, replace with a more robust
            # change point detection algorithm
            diff = np.abs(data.diff().fillna(0))
            peaks, _ = signal.find_peaks(diff, distance=min_segment_size)
            
            # Sort by peak height and take top n
            peak_heights = diff.iloc[peaks].values
            sorted_indices = np.argsort(-peak_heights)  # Descending
            
            # Get top change points
            top_peaks = peaks[sorted_indices[:max_change_points]]
            top_peaks.sort()  # Sort by position
            
            return list(top_peaks)
        except Exception as e:
            logger.warning(f"Error in change point detection: {e}")
            return []
    
    def volatility_adjusted_aggregation(
        self, 
        data: pd.Series,
        base_window: int = 20,
        vol_window: int = 50,
        min_window: int = 5,
        max_window: int = 50
    ) -> pd.Series:
        """
        Apply volatility-adjusted window sizes for aggregation.
        
        Args:
            data: Input time series
            base_window: Base window size
            vol_window: Window for volatility calculation
            min_window: Minimum window size
            max_window: Maximum window size
            
        Returns:
            Volatility-adjusted aggregated series
        """
        # Calculate normalized volatility
        volatility = data.rolling(window=vol_window).std() / data.rolling(window=vol_window).mean()
        normalized_vol = volatility / volatility.rolling(window=vol_window).max().fillna(0)
        
        # Higher volatility = shorter window
        # Lower volatility = longer window
        window_size = ((1 - normalized_vol) * (max_window - min_window) + min_window).fillna(base_window)
        window_size = window_size.astype(int).clip(min_window, max_window)
        
        # Apply dynamic windows
        result = pd.Series(index=data.index, dtype='float64')
        
        for i in range(len(data)):
            if i < min_window:
                result.iloc[i] = data.iloc[:i+1].mean()
            else:
                window = min(window_size.iloc[i], i+1)
                result.iloc[i] = data.iloc[i-window+1:i+1].mean()
                
        return result
    
    def cross_sectional_aggregation(
        self, 
        data_frame: pd.DataFrame,
        window: int = 20,
        min_correlation: float = 0.3,
        use_stddev_weights: bool = True
    ) -> pd.DataFrame:
        """
        Aggregate multiple series with cross-sectional analysis.
        
        Args:
            data_frame: DataFrame with multiple related series
            window: Rolling window for calculations
            min_correlation: Minimum correlation to include in aggregation
            use_stddev_weights: Whether to weight by inverse std dev
            
        Returns:
            Cross-sectionally aggregated DataFrame
        """
        # Result will match input shape
        result = pd.DataFrame(index=data_frame.index, columns=data_frame.columns)
        
        # Calculate rolling correlations between all columns
        for col in data_frame.columns:
            # Series to aggregate
            series = data_frame[col]
            
            # Initialize with standard MA
            ma = series.rolling(window=window).mean()
            
            # Find correlated columns
            correlated_columns = []
            
            for other_col in data_frame.columns:
                if other_col == col:
                    continue
                    
                # Calculate correlation
                corr = series.rolling(window=window).corr(data_frame[other_col])
                
                # If average correlation is above threshold, include
                if corr.mean() > min_correlation:
                    correlated_columns.append(other_col)
                    
            # If we found correlated columns, use them for adjustment
            if correlated_columns:
                # Calculate weights
                if use_stddev_weights:
                    # Inverse volatility weighting
                    weights = {}
                    for c in [col] + correlated_columns:
                        std = data_frame[c].rolling(window=window).std()
                        weights[c] = 1 / std
                        
                    # Normalize weights
                    weight_sum = sum(w.fillna(0) for w in weights.values())
                    normalized_weights = {c: w.fillna(0) / weight_sum for c, w in weights.items()}
                else:
                    # Equal weights
                    weight = 1.0 / (len(correlated_columns) + 1)
                    normalized_weights = {c: pd.Series(weight, index=data_frame.index) 
                                        for c in [col] + correlated_columns}
                
                # Calculate weighted average
                weighted_avg = pd.Series(0.0, index=data_frame.index)
                
                for c in [col] + correlated_columns:
                    c_ma = data_frame[c].rolling(window=window).mean()
                    weighted_avg += normalized_weights[c] * c_ma
                    
                # Use the weighted average
                ma = weighted_avg
                
            # Store result
            result[col] = ma
            
        return result
    
    def entropy_weighted_aggregation(
        self, 
        data: pd.Series,
        window: int = 20,
        entropy_window: int = 50
    ) -> pd.Series:
        """
        Apply entropy-weighted aggregation for adaptive averaging.
        
        Args:
            data: Input time series
            window: Base window size
            entropy_window: Window for entropy calculation
            
        Returns:
            Entropy-weighted aggregated series
        """
        # Calculate entropy using Shannon entropy approximation
        def calculate_entropy(x):
            # Use histogram to approximate probability distribution
            hist, _ = np.histogram(x, bins=10, density=True)
            # Filter out zeros to avoid log(0)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))
        
        # Calculate rolling entropy
        entropy = data.rolling(window=entropy_window).apply(
            calculate_entropy, raw=True
        )
        
        # Normalize entropy to 0-1 range
        max_entropy = entropy.rolling(window=entropy_window).max()
        normalized_entropy = entropy / max_entropy.replace(0, 1)
        
        # Higher entropy (more randomness) = larger window
        # Lower entropy (more structure) = smaller window
        adaptive_window = (
            (normalized_entropy * 0.8 + 0.2) * window
        ).fillna(window).astype(int)
        
        # Enforce minimum and maximum window size
        adaptive_window = adaptive_window.clip(window // 2, window * 2)
        
        # Apply adaptive windows
        result = pd.Series(index=data.index, dtype='float64')
        
        for i in range(len(data)):
            if i < window // 2:
                result.iloc[i] = data.iloc[:i+1].mean()
            else:
                window_size = min(adaptive_window.iloc[i], i+1)
                result.iloc[i] = data.iloc[i-window_size+1:i+1].mean()
                
        return result
        
    def perform_aggregation(
        self,
        data: Union[pd.Series, pd.DataFrame],
        aggregation_type: str,
        params: Dict = None
    ) -> AggregationResult:
        """
        Perform aggregation operation with timing and memory tracking.
        
        Args:
            data: Input data to aggregate
            aggregation_type: Type of aggregation to perform
            params: Parameters for the aggregation
            
        Returns:
            AggregationResult with data and metadata
        """
        import time
        
        if params is None:
            params = {}
            
        # Start timing
        start_time = time.time()
        
        # Get input shape
        input_shape = data.shape
        
        # Select aggregation function
        aggregation_func = getattr(self, aggregation_type, None)
        
        if aggregation_func is None:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")
            
        # Perform aggregation
        result = aggregation_func(data, **params)
        
        # Calculate performance metrics
        computation_time = time.time() - start_time
        memory_usage = calculate_memory_usage(result)
        
        # Create and return result object
        return AggregationResult(
            data=result,
            computation_time=computation_time,
            memory_usage=memory_usage,
            aggregation_type=aggregation_type,
            parameters=params,
            input_shape=input_shape,
            output_shape=result.shape
        )


# Optimizer function to be used with numba
@jit(nopython=True, cache=True)
def _optimize_rolling_window(values, window, func_type='mean'):
    """
    Optimized rolling window calculation using numba.
    
    Args:
        values: Input array
        window: Window size
        func_type: Type of function to apply
        
    Returns:
        Array of results from rolling calculation
    """
    n = len(values)
    result = np.empty(n, dtype=np.float64)
    result[:window-1] = np.nan
    
    if func_type == 'mean':
        # Optimized moving average
        window_sum = np.sum(values[:window-1])
        
        for i in range(window-1, n):
            window_sum = window_sum + values[i] - (values[i-window] if i >= window else 0)
            result[i] = window_sum / min(i+1, window)
            
    elif func_type == 'median':
        # Median requires sorting for each window
        for i in range(window-1, n):
            result[i] = np.median(values[i-window+1:i+1])
            
    elif func_type == 'std':
        # Standard deviation
        for i in range(window-1, n):
            window_values = values[i-window+1:i+1]
            result[i] = np.std(window_values)
            
    return result


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
    data = pd.Series(np.cumsum(np.random.normal(0, 1, 1000)), index=dates)
    
    # Initialize aggregator
    aggregator = TimeseriesAggregator(use_gpu=False)
    
    # Try some aggregations
    hull_ma = aggregator.hull_moving_average(data, period=20)
    vwap = aggregator.volume_weighted_average(
        data, 
        pd.Series(np.random.lognormal(0, 1, 1000), index=dates),
        window=20
    )
    kama = aggregator.kaufman_adaptive_moving_average(data, period=20)
    
    logger.info("Hull MA first 5 values: %s", hull_ma.head())
    logger.info("VWAP first 5 values: %s", vwap.head())
    logger.info("KAMA first 5 values: %s", kama.head())

