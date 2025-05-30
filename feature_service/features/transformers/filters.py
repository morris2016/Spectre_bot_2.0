

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Filters Module for Feature Service

This module implements various filtering techniques for preprocessing time series
data before feature computation, strategy development, or model training.
Advanced filtering methods help reduce noise while preserving important signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Callable, Any
from enum import Enum
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import pywt
import statsmodels.api as sm

from common.logger import get_logger
from common.utils import timeit, rolling_apply
from common.exceptions import (
    DataTransformationError, InsufficientDataError, 
    FeatureCalculationError
)
from feature_service.transformers.base import BaseTransformer

logger = get_logger(__name__)


class FilterMethod(Enum):
    """Enum defining the available filtering methods."""
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    KALMAN = "kalman"
    SAVITZKY_GOLAY = "savitzky_golay"
    BUTTERWORTH = "butterworth"
    WAVELET = "wavelet"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    HODRICK_PRESCOTT = "hodrick_prescott"
    ADAPTIVE = "adaptive"
    REGIME_AWARE = "regime_aware"
    HAMPEL = "hampel"
    BILATERAL = "bilateral"
    WIENER = "wiener"
    ENSEMBLE = "ensemble"


class BaseFilter(BaseTransformer):
    """Base class for all filter transformers."""
    
    def __init__(
        self, 
        preserve_original: bool = False,
        return_residuals: bool = False,
        persist_state: bool = True,
        dtype: np.dtype = np.float32
    ):
        """
        Initialize the base filter.
        
        Args:
            preserve_original: Whether to preserve the original data
            return_residuals: Whether to return residuals instead of filtered data
            persist_state: Whether to maintain state between calls
            dtype: Data type for numeric operations
        """
        super().__init__(persist_state=persist_state)
        self.preserve_original = preserve_original
        self.return_residuals = return_residuals
        self.dtype = dtype
        self._state = {}
    
    def _validate_data(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> None:
        """
        Validate that the data is suitable for filtering.
        
        Args:
            data: Input data to validate
            
        Raises:
            InsufficientDataError: If data doesn't meet minimum requirements
            DataTransformationError: If data has unexpected properties
        """
        # Check for minimum data length requirements
        # Will be implemented by subclasses
        pass
    
    def _handle_nans(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Handle NaN values in the data.
        
        Args:
            data: Input data with potential NaN values
            
        Returns:
            Data with NaN values handled appropriately
        """
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data.fillna(method='ffill').fillna(method='bfill')
        else:
            # For numpy arrays
            mask = np.isnan(data)
            if not np.any(mask):
                return data
                
            result = np.copy(data)
            # Forward fill
            indices = np.where(~mask)[0]
            if len(indices) == 0:
                return data  # All NaN, can't fill
                
            first_valid = indices[0]
            for i in range(len(result)):
                if i < first_valid:
                    result[i] = result[first_valid]
                elif mask[i]:
                    result[i] = result[i-1]
            return result
    
    def reset(self) -> None:
        """Reset the internal state of the filter."""
        self._state = {}


class MovingAverageFilter(BaseFilter):
    """
    Moving Average Filter Transformer.
    
    Smooths data using simple moving average with various window types.
    """
    
    def __init__(
        self, 
        window_size: int = 20,
        window_type: str = 'simple',
        center: bool = False,
        min_periods: Optional[int] = None,
        preserve_original: bool = False,
        return_residuals: bool = False,
        persist_state: bool = True
    ):
        """
        Initialize the Moving Average filter.
        
        Args:
            window_size: Size of the moving average window
            window_type: Type of window ('simple', 'triangular', 'weighted', etc.)
            center: Whether to center the window
            min_periods: Minimum number of observations to calculate the MA
            preserve_original: Whether to preserve the original data
            return_residuals: Whether to return residuals instead of filtered data
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            preserve_original=preserve_original,
            return_residuals=return_residuals,
            persist_state=persist_state
        )
        self.window_size = window_size
        self.window_type = window_type
        self.center = center
        self.min_periods = min_periods or window_size
    
    def _validate_data(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> None:
        """Validate input data for moving average filtering."""
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if len(data) < self.min_periods:
                raise InsufficientDataError(
                    f"Data length {len(data)} is less than min_periods {self.min_periods}"
                )
        else:
            if data.shape[0] < self.min_periods:
                raise InsufficientDataError(
                    f"Data length {data.shape[0]} is less than min_periods {self.min_periods}"
                )
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply Moving Average filtering to the data.
        
        Args:
            data: Input data to filter
            
        Returns:
            Filtered data
        """
        self._validate_data(data)
        data = self._handle_nans(data)
        
        # Apply the appropriate moving average based on window_type
        if isinstance(data, pd.Series):
            filtered = self._filter_series(data)
        elif isinstance(data, pd.DataFrame):
            filtered = self._filter_dataframe(data)
        else:
            filtered = self._filter_array(data)
        
        # Return residuals if requested
        if self.return_residuals:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                residuals = data - filtered
                return residuals
            else:
                return data - filtered
        else:
            return filtered
    
    def _filter_series(self, data: pd.Series) -> pd.Series:
        """Apply moving average filter to a pandas Series."""
        if self.window_type == 'simple':
            # Simple moving average
            filtered = data.rolling(
                window=self.window_size, 
                min_periods=self.min_periods, 
                center=self.center
            ).mean()
        elif self.window_type == 'weighted':
            # Weighted moving average
            weights = np.arange(1, self.window_size + 1)
            filtered = data.rolling(
                window=self.window_size, 
                min_periods=self.min_periods, 
                center=self.center
            ).apply(lambda x: np.average(x, weights=weights[-len(x):]), raw=True)
        elif self.window_type == 'triangular':
            # Triangular moving average
            triangle = np.concatenate([
                np.arange(1, self.window_size // 2 + 1),
                np.arange((self.window_size + 1) // 2, 0, -1)
            ])
            triangle = triangle / triangle.sum()
            filtered = data.rolling(
                window=self.window_size, 
                min_periods=self.min_periods, 
                center=self.center
            ).apply(lambda x: np.sum(x * triangle[-len(x):]) / np.sum(triangle[-len(x):]), raw=True)
        elif self.window_type == 'exponential':
            # Exponential moving average
            alpha = 2 / (self.window_size + 1)
            filtered = data.ewm(alpha=alpha, min_periods=self.min_periods).mean()
        else:
            raise ValueError(f"Unsupported window type: {self.window_type}")
        
        # Fill NaN values at the beginning
        if not self.center:
            filtered = filtered.fillna(method='bfill')
            
        return filtered
    
    def _filter_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply moving average filter to a pandas DataFrame."""
        result = pd.DataFrame(index=data.index, columns=data.columns)
        
        for col in data.columns:
            result[col] = self._filter_series(data[col])
            
        return result
    
    def _filter_array(self, data: np.ndarray) -> np.ndarray:
        """Apply moving average filter to a numpy array."""
        # For numpy arrays, use convolution for efficiency
        filtered = np.zeros_like(data, dtype=self.dtype)
        
        if self.window_type == 'simple':
            # Simple moving average - uniform weights
            weights = np.ones(self.window_size) / self.window_size
        elif self.window_type == 'weighted':
            # Weighted moving average - linear weights
            weights = np.arange(1, self.window_size + 1)
            weights = weights / weights.sum()
        elif self.window_type == 'triangular':
            # Triangular moving average
            triangle = np.concatenate([
                np.arange(1, self.window_size // 2 + 1),
                np.arange((self.window_size + 1) // 2, 0, -1)
            ])
            weights = triangle / triangle.sum()
        elif self.window_type == 'exponential':
            # Exponential moving average
            alpha = 2 / (self.window_size + 1)
            weights = np.zeros(self.window_size)
            weights[0] = alpha
            for i in range(1, self.window_size):
                weights[i] = alpha * (1 - alpha) ** i
            weights = weights / weights.sum()  # Normalize
        else:
            raise ValueError(f"Unsupported window type: {self.window_type}")
        
        # Apply convolution
        if self.center:
            # Center the window
            filtered = np.convolve(data, weights, mode='same')
        else:
            # Use 'valid' mode and pad
            valid_result = np.convolve(data, weights, mode='valid')
            filtered = np.zeros_like(data)
            filtered[self.window_size-1:] = valid_result
            # Fill initial values
            for i in range(self.window_size-1):
                # Use available data points
                valid_weights = weights[-(i+1):]
                valid_weights = valid_weights / valid_weights.sum()  # Renormalize
                filtered[i] = np.sum(data[:i+1] * valid_weights)
        
        return filtered


class ExponentialFilter(BaseFilter):
    """
    Exponential Filter Transformer.
    
    Implements exponential smoothing (single, double, triple) for time series data.
    """
    
    def __init__(
        self, 
        alpha: float = 0.3,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        seasonal_periods: Optional[int] = None,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        preserve_original: bool = False,
        return_residuals: bool = False,
        persist_state: bool = True
    ):
        """
        Initialize the Exponential filter.
        
        Args:
            alpha: Smoothing factor for level (0 < alpha < 1)
            beta: Smoothing factor for trend (0 < beta < 1)
            gamma: Smoothing factor for seasonality (0 < gamma < 1)
            seasonal_periods: Number of periods in a seasonal cycle
            trend: Type of trend component ('add', 'mul', None)
            seasonal: Type of seasonal component ('add', 'mul', None)
            preserve_original: Whether to preserve the original data
            return_residuals: Whether to return residuals instead of filtered data
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            preserve_original=preserve_original,
            return_residuals=return_residuals,
            persist_state=persist_state
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        
        # Determine the type of exponential smoothing
        if beta is None and gamma is None:
            self.method = 'simple'  # Simple Exponential Smoothing
        elif gamma is None:
            self.method = 'double'  # Double Exponential Smoothing (Holt's method)
        else:
            self.method = 'triple'  # Triple Exponential Smoothing (Holt-Winters)
            
        # State for tracking
        self._state = {
            "level": None,
            "trend": None,
            "seasonal": None,
            "last_values": None
        }
        
    def _validate_data(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> None:
        """Validate input data for exponential filtering."""
        # For Holt-Winters, need at least one complete seasonal cycle
        if self.method == 'triple' and self.seasonal_periods is not None:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                if len(data) < self.seasonal_periods:
                    raise InsufficientDataError(
                        f"Data length {len(data)} is less than seasonal_periods {self.seasonal_periods}"
                    )
            else:
                if data.shape[0] < self.seasonal_periods:
                    raise InsufficientDataError(
                        f"Data length {data.shape[0]} is less than seasonal_periods {self.seasonal_periods}"
                    )
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply Exponential filtering to the data.
        
        Args:
            data: Input data to filter
            
        Returns:
            Filtered data
        """
        self._validate_data(data)
        data = self._handle_nans(data)
        
        # Apply the appropriate exponential smoothing method
        if isinstance(data, pd.Series):
            filtered = self._filter_series(data)
        elif isinstance(data, pd.DataFrame):
            filtered = self._filter_dataframe(data)
        else:
            filtered = self._filter_array(data)
        
        # Return residuals if requested
        if self.return_residuals:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                residuals = data - filtered
                return residuals
            else:
                return data - filtered
        else:
            return filtered
    
    def _simple_exponential_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Apply simple exponential smoothing to a numpy array."""
        n = len(data)
        result = np.zeros(n, dtype=self.dtype)
        
        # Initialize level
        level = data[0]
        result[0] = level
        
        # Apply SES formula: y_hat[t] = alpha * y[t] + (1 - alpha) * y_hat[t-1]
        for t in range(1, n):
            level = self.alpha * data[t] + (1 - self.alpha) * level
            result[t] = level
        
        # Save state
        self._state["level"] = level
        self._state["last_values"] = data[-1]
        
        return result
    
    def _holt_linear(self, data: np.ndarray) -> np.ndarray:
        """Apply Holt's linear (double exponential) smoothing to a numpy array."""
        n = len(data)
        result = np.zeros(n, dtype=self.dtype)
        
        # Initialize level and trend
        level = data[0]
        trend = data[1] - data[0] if n > 1 else 0
        result[0] = level
        
        # Apply Holt's method formulas
        for t in range(1, n):
            prev_level = level
            level = self.alpha * data[t] + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
            result[t] = level
        
        # Save state
        self._state["level"] = level
        self._state["trend"] = trend
        self._state["last_values"] = data[-1]
        
        return result
    
    def _holt_winters(self, data: np.ndarray) -> np.ndarray:
        """Apply Holt-Winters (triple exponential) smoothing to a numpy array."""
        n = len(data)
        result = np.zeros(n, dtype=self.dtype)
        s = self.seasonal_periods
        
        # Initialize level, trend, and seasonal components
        if "level" not in self._state or self._state["level"] is None:
            # Initialize from data
            level = data[0]
            
            if self.trend == 'add':
                trend = (data[s] - data[0]) / s
            elif self.trend == 'mul':
                trend = (data[s] / data[0]) ** (1/s)
            else:
                trend = 0
                
            seasonal = np.zeros(s)
            if self.seasonal == 'add':
                for i in range(s):
                    if i < n:
                        seasonal[i] = data[i] - level
            elif self.seasonal == 'mul':
                for i in range(s):
                    if i < n:
                        seasonal[i] = data[i] / level
            
            # Normalize seasonal components
            if self.seasonal == 'add':
                seasonal = seasonal - np.mean(seasonal)
            elif self.seasonal == 'mul':
                seasonal = seasonal / np.mean(seasonal)
        else:
            # Use saved state
            level = self._state["level"]
            trend = self._state["trend"]
            seasonal = self._state["seasonal"]
        
        # Apply Holt-Winters formulas
        for t in range(n):
            # Adjust formulas based on additive or multiplicative components
            if t < s and seasonal is not None:
                # Use initial seasonal values
                season_idx = t
            else:
                # Use calculated seasonal values
                season_idx = t % s
                
            if t == 0:
                result[t] = level
                continue
                
            prev_level = level
            
            if self.seasonal == 'add':
                level = self.alpha * (data[t] - seasonal[season_idx]) + (1 - self.alpha) * (level + trend)
            elif self.seasonal == 'mul':
                level = self.alpha * (data[t] / seasonal[season_idx]) + (1 - self.alpha) * (level + trend)
            else:
                level = self.alpha * data[t] + (1 - self.alpha) * (level + trend)
                
            if self.trend == 'add':
                trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
            elif self.trend == 'mul':
                trend = self.beta * (level / prev_level) + (1 - self.beta) * trend
                
            if self.seasonal == 'add':
                seasonal[season_idx] = self.gamma * (data[t] - level) + (1 - self.gamma) * seasonal[season_idx]
                result[t] = level + seasonal[season_idx]
            elif self.seasonal == 'mul':
                seasonal[season_idx] = self.gamma * (data[t] / level) + (1 - self.gamma) * seasonal[season_idx]
                result[t] = level * seasonal[season_idx]
            else:
                result[t] = level
        
        # Save state
        self._state["level"] = level
        self._state["trend"] = trend
        self._state["seasonal"] = seasonal
        self._state["last_values"] = data[-1]
        
        return result
    
    def _filter_series(self, data: pd.Series) -> pd.Series:
        """Apply exponential filter to a pandas Series."""
        values = data.values
        
        if self.method == 'simple':
            filtered_values = self._simple_exponential_smoothing(values)
        elif self.method == 'double':
            filtered_values = self._holt_linear(values)
        elif self.method == 'triple':
            filtered_values = self._holt_winters(values)
        
        return pd.Series(filtered_values, index=data.index)
    
    def _filter_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply exponential filter to a pandas DataFrame."""
        result = pd.DataFrame(index=data.index, columns=data.columns)
        
        for col in data.columns:
            # Save current state
            temp_state = self._state.copy()
            
            # Apply filter to this column
            result[col] = self._filter_series(data[col])
            
            # Reset state for next column
            if col != data.columns[-1]:
                self._state = temp_state
            
        return result
    
    def _filter_array(self, data: np.ndarray) -> np.ndarray:
        """Apply exponential filter to a numpy array."""
        if self.method == 'simple':
            return self._simple_exponential_smoothing(data)
        elif self.method == 'double':
            return self._holt_linear(data)
        elif self.method == 'triple':
            return self._holt_winters(data)


class KalmanFilter(BaseFilter):
    """
    Kalman Filter Transformer.
    
    Implements a Kalman filter for optimal state estimation in noisy time series.
    """
    
    def __init__(
        self, 
        transition_matrices: Optional[np.ndarray] = None,
        observation_matrices: Optional[np.ndarray] = None,
        transition_covariance: Optional[np.ndarray] = None,
        observation_covariance: Optional[np.ndarray] = None,
        initial_state_mean: Optional[np.ndarray] = None,
        initial_state_covariance: Optional[np.ndarray] = None,
        em_iterations: int = 5,
        auto_init: bool = True,
        preserve_original: bool = False,
        return_residuals: bool = False,
        persist_state: bool = True
    ):
        """
        Initialize the Kalman filter.
        
        Args:
            transition_matrices: State transition matrix
            observation_matrices: Observation matrix
            transition_covariance: Transition covariance matrix
            observation_covariance: Observation covariance matrix
            initial_state_mean: Initial state mean
            initial_state_covariance: Initial state covariance
            em_iterations: Number of EM iterations for parameter estimation
            auto_init: Whether to automatically initialize parameters
            preserve_original: Whether to preserve the original data
            return_residuals: Whether to return residuals instead of filtered data
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            preserve_original=preserve_original,
            return_residuals=return_residuals,
            persist_state=persist_state
        )
        
        # Store Kalman filter parameters
        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.em_iterations = em_iterations
        self.auto_init = auto_init
        
        # Default values for univariate case if not specified
        if self.auto_init and transition_matrices is None:
            self.n_dim_state = 2  # State dimension (position, velocity)
            self.n_dim_obs = 1  # Observation dimension
            
            # Default matrices for a simple dynamic system
            self.transition_matrices = np.array([[1, 1], [0, 1]])  # Constant velocity model
            self.observation_matrices = np.array([[1, 0]])  # Observe only position
            self.transition_covariance = np.eye(2) * 0.01  # Small process noise
            self.observation_covariance = np.array([[1.0]])  # Observation noise
            self.initial_state_mean = np.zeros(2)  # Start at zero
            self.initial_state_covariance = np.eye(2)  # High initial uncertainty
        elif not self.auto_init:
            # Parameters will be estimated from data
            pass
        
        # State for tracking
        self._state = {
            "filtered_state_means": None,
            "filtered_state_covariances": None,
            "kf_params": None
        }
    
    def _validate_data(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> None:
        """Validate input data for Kalman filtering."""
        # Kalman filter needs enough data to estimate parameters
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if len(data) < 10:  # Arbitrary minimum for stable estimation
                raise InsufficientDataError(
                    f"Data length {len(data)} is too small for Kalman filter parameter estimation"
                )
        else:
            if data.shape[0] < 10:
                raise InsufficientDataError(
                    f"Data length {data.shape[0]} is too small for Kalman filter parameter estimation"
                )
    
    def _init_from_data(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize Kalman filter parameters from data."""
        n = len(data)
        
        # Simple differencing for velocity estimation
        positions = data.reshape(-1, 1)
        velocities = np.diff(positions, axis=0, prepend=positions[0].reshape(1, -1))
        
        # Initial state is first observation
        initial_state_mean = np.array([positions[0, 0], velocities[0, 0]])
        
        # Initial covariance based on data variance
        var_pos = np.var(positions)
        var_vel = np.var(velocities)
        initial_state_covariance = np.diag([var_pos, var_vel])
        
        # Transition matrix for constant velocity model
        transition_matrices = np.array([[1, 1], [0, 1]])
        
        # Observation matrix to observe position
        observation_matrices = np.array([[1, 0]])
        
        # Covariances estimated from data
        transition_covariance = np.diag([var_pos * 0.01, var_vel * 0.01])
        observation_covariance = np.array([[var_pos * 0.1]])
        
        return (
            transition_matrices,
            observation_matrices,
            transition_covariance,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance
        )
    
    def _custom_kalman_filter(
        self,
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom implementation of Kalman filter without external libraries.
        
        Args:
            data: Input data (observations)
            
        Returns:
            Tuple of filtered state means and covariances
        """
        n = len(data)
        dim_state = self.transition_matrices.shape[0]
        
        # Initialize storage for filtered states and covariances
        filtered_state_means = np.zeros((n, dim_state))
        filtered_state_covariances = np.zeros((n, dim_state, dim_state))
        
        # Initialize with initial state and covariance
        x_pred = self.initial_state_mean.copy()
        P_pred = self.initial_state_covariance.copy()
        
        for t in range(n):
            # Reshape observation for matrix operations
            y = np.array([data[t]])
            
            # Update step (correct predictions with measurement)
            K = P_pred @ self.observation_matrices.T @ np.linalg.inv(
                self.observation_matrices @ P_pred @ self.observation_matrices.T + 
                self.observation_covariance
            )
            x_filtered = x_pred + K @ (y - self.observation_matrices @ x_pred)
            P_filtered = (np.eye(dim_state) - K @ self.observation_matrices) @ P_pred
            
            # Store filtered estimates
            filtered_state_means[t] = x_filtered
            filtered_state_covariances[t] = P_filtered
            
            # Predict step (project state ahead)
            x_pred = self.transition_matrices @ x_filtered
            P_pred = (
                self.transition_matrices @ P_filtered @ self.transition_matrices.T + 
                self.transition_covariance
            )
        
        return filtered_state_means, filtered_state_covariances
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply Kalman filtering to the data.
        
        Args:
            data: Input data to filter
            
        Returns:
            Filtered data
        """
        self._validate_data(data)
        data = self._handle_nans(data)
        
        # Apply Kalman filter
        if isinstance(data, pd.Series):
            filtered = self._filter_series(data)
        elif isinstance(data, pd.DataFrame):
            filtered = self._filter_dataframe(data)
        else:
            filtered = self._filter_array(data)
        
        # Return residuals if requested
        if self.return_residuals:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                residuals = data - filtered
                return residuals
            else:
                return data - filtered
        else:
            return filtered
    
    def _filter_array(self, data: np.ndarray) -> np.ndarray:
        """Apply Kalman filter to a numpy array."""
        # Initialize parameters if needed
        if self.auto_init and (self.transition_matrices is None or self.observation_matrices is None):
            (
                self.transition_matrices,
                self.observation_matrices,
                self.transition_covariance,
                self.observation_covariance,
                self.initial_state_mean,
                self.initial_state_covariance
            ) = self._init_from_data(data)
        
        # Run Kalman filter
        filtered_state_means, filtered_state_covariances = self._custom_kalman_filter(data)
        
        # Extract filtered observations (first dimension of state)
        filtered_values = filtered_state_means[:, 0]
        
        # Save state
        self._state["filtered_state_means"] = filtered_state_means
        self._state["filtered_state_covariances"] = filtered_state_covariances
        self._state["kf_params"] = {
            "transition_matrices": self.transition_matrices,
            "observation_matrices": self.observation_matrices,
            "transition_covariance": self.transition_covariance,
            "observation_covariance": self.observation_covariance,
            "initial_state_mean": filtered_state_means[-1],  # Use last state as next initial
            "initial_state_covariance": filtered_state_covariances[-1]
        }
        
        return filtered_values
    
    def _filter_series(self, data: pd.Series) -> pd.Series:
        """Apply Kalman filter to a pandas Series."""
        values = data.values
        filtered_values = self._filter_array(values)
        return pd.Series(filtered_values, index=data.index)
    
    def _filter_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Kalman filter to a pandas DataFrame."""
        result = pd.DataFrame(index=data.index, columns=data.columns)
        
        for col in data.columns:
            # Save current state
            temp_state = self._state.copy()
            
            # Apply filter to this column
            result[col] = self._filter_series(data[col])
            
            # Reset state for next column
            if col != data.columns[-1]:
                self._state = temp_state
            
        return result


class SavitzkyGolayFilter(BaseFilter):
    """
    Savitzky-Golay Filter Transformer.
    
    Implements Savitzky-Golay smoothing by fitting local polynomials.
    """
    
    def __init__(
        self, 
        window_length: int = 11,
        polyorder: int = 3,
        deriv: int = 0,
        delta: float = 1.0,
        mode: str = 'interp',
        preserve_original: bool = False,
        return_residuals: bool = False,
        persist_state: bool = True
    ):
        """
        Initialize the Savitzky-Golay filter.
        
        Args:
            window_length: Length of the filter window (must be odd)
            polyorder: Order of the polynomial to fit
            deriv: Order of the derivative to compute
            delta: Spacing between the samples
            mode: How to handle edges ('mirror', 'constant', 'nearest', 'interp')
            preserve_original: Whether to preserve the original data
            return_residuals: Whether to return residuals instead of filtered data
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            preserve_original=preserve_original,
            return_residuals=return_residuals,
            persist_state=persist_state
        )
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
            logger.warning(f"Window length must be odd. Adjusted to {window_length}")
            
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.mode = mode
        
        # Validate parameters
        if polyorder >= window_length:
            raise ValueError(f"polyorder ({polyorder}) must be less than window_length ({window_length})")
    
    def _validate_data(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> None:
        """Validate input data for Savitzky-Golay filtering."""
        # Need enough data points for the window
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if len(data) < self.window_length:
                raise InsufficientDataError(
                    f"Data length {len(data)} is less than window_length {self.window_length}"
                )
        else:
            if data.shape[0] < self.window_length:
                raise InsufficientDataError(
                    f"Data length {data.shape[0]} is less than window_length {self.window_length}"
                )
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply Savitzky-Golay filtering to the data.
        
        Args:
            data: Input data to filter
            
        Returns:
            Filtered data
        """
        self._validate_data(data)
        data = self._handle_nans(data)
        
        # Apply Savitzky-Golay filter
        if isinstance(data, pd.Series):
            filtered = self._filter_series(data)
        elif isinstance(data, pd.DataFrame):
            filtered = self._filter_dataframe(data)
        else:
            filtered = self._filter_array(data)
        
        # Return residuals if requested
        if self.return_residuals:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                residuals = data - filtered
                return residuals
            else:
                return data - filtered
        else:
            return filtered
    
    def _filter_array(self, data: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter to a numpy array."""
        # Use SciPy's implementation
        filtered = signal.savgol_filter(
            data,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            mode=self.mode
        )
        
        return filtered
    
    def _filter_series(self, data: pd.Series) -> pd.Series:
        """Apply Savitzky-Golay filter to a pandas Series."""
        values = data.values
        filtered_values = self._filter_array(values)
        return pd.Series(filtered_values, index=data.index)
    
    def _filter_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Savitzky-Golay filter to a pandas DataFrame."""
        result = pd.DataFrame(index=data.index, columns=data.columns)
        
        for col in data.columns:
            result[col] = self._filter_series(data[col])
            
        return result


class ButterworthFilter(BaseFilter):
    """
    Butterworth Filter Transformer.
    
    Implements Butterworth low-pass, high-pass, band-pass and band-stop filters.
    """
    
    def __init__(
        self, 
        cutoff: Union[float, Tuple[float, float]],
        fs: float = 1.0,
        order: int = 4,
        btype: str = 'low',
        output: str = 'sos',
        preserve_original: bool = False,
        return_residuals: bool = False,
        persist_state: bool = True
    ):
        """
        Initialize the Butterworth filter.
        
        Args:
            cutoff: Cutoff frequency (or frequencies for band filters)
            fs: Sampling frequency
            order: Filter order
            btype: Filter type ('low', 'high', 'band', 'bandstop')
            output: Filter output format ('ba', 'sos')
            preserve_original: Whether to preserve the original data
            return_residuals: Whether to return residuals instead of filtered data
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            preserve_original=preserve_original,
            return_residuals=return_residuals,
            persist_state=persist_state
        )
        
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.btype = btype
        self.output = output
        
        # Validate parameters
        if btype in ['band', 'bandstop'] and not isinstance(cutoff, (list, tuple)):
            raise ValueError(f"For {btype} filter, cutoff must be a tuple or list of two frequencies")
            
        # Create filter coefficients
        if output == 'sos':
            self.sos = signal.butter(
                order, 
                cutoff, 
                btype=btype, 
                fs=fs, 
                output=output
            )
        else:  # 'ba'
            self.b, self.a = signal.butter(
                order, 
                cutoff, 
                btype=btype, 
                fs=fs, 
                output=output
            )
            
        # Filter state
        self.zi = None
    
    def _validate_data(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> None:
        """Validate input data for Butterworth filtering."""
        # Need enough data points for the filter
        min_samples = 2 * self.order + 1
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if len(data) < min_samples:
                raise InsufficientDataError(
                    f"Data length {len(data)} is less than minimum required samples {min_samples}"
                )
        else:
            if data.shape[0] < min_samples:
                raise InsufficientDataError(
                    f"Data length {data.shape[0]} is less than minimum required samples {min_samples}"
                )
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply Butterworth filtering to the data.
        
        Args:
            data: Input data to filter
            
        Returns:
            Filtered data
        """
        self._validate_data(data)
        data = self._handle_nans(data)
        
        # Apply Butterworth filter
        if isinstance(data, pd.Series):
            filtered = self._filter_series(data)
        elif isinstance(data, pd.DataFrame):
            filtered = self._filter_dataframe(data)
        else:
            filtered = self._filter_array(data)
        
        # Return residuals if requested
        if self.return_residuals:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                residuals = data - filtered
                return residuals
            else:
                return data - filtered
        else:
            return filtered
    
    def _filter_array(self, data: np.ndarray) -> np.ndarray:
        """Apply Butterworth filter to a numpy array."""
        # Apply the filter with initial conditions
        if self.output == 'sos':
            if self.zi is None:
                zi = signal.sosfilt_zi(self.sos)
                zi = np.tile(zi, (data.shape[0] if data.ndim > 1 else 1, 1, 1)).T
                self.zi = zi
                
            filtered, self.zi = signal.sosfilt(self.sos, data, zi=self.zi)
        else:  # 'ba'
            if self.zi is None:
                zi = signal.lfilter_zi(self.b, self.a)
                zi = zi * data[0]
                self.zi = zi
                
            filtered, self.zi = signal.lfilter(self.b, self.a, data, zi=self.zi)
            
        return filtered
    
    def _filter_series(self, data: pd.Series) -> pd.Series:
        """Apply Butterworth filter to a pandas Series."""
        values = data.values
        filtered_values = self._filter_array(values)
        return pd.Series(filtered_values, index=data.index)
    
    def _filter_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply Butterworth filter to a pandas DataFrame."""
        result = pd.DataFrame(index=data.index, columns=data.columns)
        
        for col in data.columns:
            # Save current filter state
            temp_zi = self.zi
            
            # Apply filter to this column
            result[col] = self._filter_series(data[col])
            
            # Reset filter state for next column
            if col != data.columns[-1]:
                self.zi = temp_zi
            
        return result


class WaveletFilter(BaseFilter):
    """
    Wavelet Filter Transformer.
    
    Implements wavelet-based denoising for time series data.
    """
    
    def __init__(
        self, 
        wavelet: str = 'db4',
        mode: str = 'symmetric',
        level: Optional[int] = None,
        threshold: Optional[float] = None,
        threshold_mode: str = 'soft',
        preserve_original: bool = False,
        return_residuals: bool = False,
        persist_state: bool = True
    ):
        """
        Initialize the Wavelet filter.
        
        Args:
            wavelet: Wavelet type
            mode: Signal extension mode
            level: Decomposition level (None for maximum)
            threshold: Threshold for coefficient shrinkage (None for automatic)
            threshold_mode: Thresholding mode ('soft', 'hard')
            preserve_original: Whether to preserve the original data
            return_residuals: Whether to return residuals instead of filtered data
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            preserve_original=preserve_original,
            return_residuals=return_residuals,
            persist_state=persist_state
        )
        
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        
        # Validate parameters
        try:
            pywt.Wavelet(wavelet)
        except Exception as e:
            raise ValueError(f"Invalid wavelet '{wavelet}': {e}")
    
    def _validate_data(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> None:
        """Validate input data for wavelet filtering."""
        # Need enough data points for wavelet decomposition
        wavelet_obj = pywt.Wavelet(self.wavelet)
        min_samples = 2 * wavelet_obj.dec_len
        
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if len(data) < min_samples:
                raise InsufficientDataError(
                    f"Data length {len(data)} is less than minimum required samples {min_samples}"
                )
        else:
            if data.shape[0] < min_samples:
                raise InsufficientDataError(
                    f"Data length {data.shape[0]} is less than minimum required samples {min_samples}"
                )
    
    def _denoise_signal(self, data: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising to a signal."""
        # Determine maximum decomposition level if not specified
        if self.level is None:
            self.level = pywt.dwt_max_level(len(data), pywt.Wavelet(self.wavelet).dec_len)
            
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(data, self.wavelet, mode=self.mode, level=self.level)
        
        # Determine threshold if not specified
        if self.threshold is None:
            # Universal threshold based on noise estimate from finest detail coefficients
            detail_coeffs = coeffs[-1]
            sigma = np.median(np.abs(detail_coeffs)) / 0.6745  # Estimate of noise level
            self.threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        # Apply thresholding to detail coefficients
        thresholded_coeffs = [coeffs[0]]  # Approximation coefficients
        for i in range(1, len(coeffs)):
            if self.threshold_mode == 'soft':
                # Soft thresholding (shrink coefficients)
                thresholded = pywt.threshold(coeffs[i], self.threshold, mode='soft')
            else:
                # Hard thresholding (set small coefficients to zero)
                thresholded = pywt.threshold(coeffs[i], self.threshold, mode='hard')
            thresholded_coeffs.append(thresholded)
        
        # Reconstruct signal from thresholded coefficients
        denoised = pywt.waverec(thresholded_coeffs, self.wavelet, mode=self.mode)
        
        # Ensure the output has the same length as input
        if len(denoised) > len(data):
            denoised = denoised[:len(data)]
            
        return denoised
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply wavelet filtering to the data.
        
        Args:
            data: Input data to filter
            
        Returns:
            Filtered data
        """
        self._validate_data(data)
        data = self._handle_nans(data)
        
        # Apply wavelet filter
        if isinstance(data, pd.Series):
            filtered = self._filter_series(data)
        elif isinstance(data, pd.DataFrame):
            filtered = self._filter_dataframe(data)
        else:
            filtered = self._filter_array(data)
        
        # Return residuals if requested
        if self.return_residuals:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                residuals = data - filtered
                return residuals
            else:
                return data - filtered
        else:
            return filtered
    
    def _filter_array(self, data: np.ndarray) -> np.ndarray:
        """Apply wavelet filter to a numpy array."""
        return self._denoise_signal(data)
    
    def _filter_series(self, data: pd.Series) -> pd.Series:
        """Apply wavelet filter to a pandas Series."""
        values = data.values
        filtered_values = self._filter_array(values)
        return pd.Series(filtered_values, index=data.index)
    
    def _filter_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply wavelet filter to a pandas DataFrame."""
        result = pd.DataFrame(index=data.index, columns=data.columns)
        
        for col in data.columns:
            result[col] = self._filter_series(data[col])
            
        return result


class HodrickPrescottFilter(BaseFilter):
    """
    Hodrick-Prescott Filter Transformer.
    
    Implements the Hodrick-Prescott filter for trend-cycle decomposition.
    """
    
    def __init__(
        self, 
        lambda_param: float = 1600,
        return_trend: bool = True,
        preserve_original: bool = False,
        return_residuals: bool = False,
        persist_state: bool = True
    ):
        """
        Initialize the Hodrick-Prescott filter.
        
        Args:
            lambda_param: Smoothing parameter
            return_trend: Whether to return trend (True) or cycle (False)
            preserve_original: Whether to preserve the original data
            return_residuals: Whether to return residuals instead of filtered data
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            preserve_original=preserve_original,
            return_residuals=return_residuals,
            persist_state=persist_state
        )
        
        self.lambda_param = lambda_param
        self.return_trend = return_trend
    
    def _validate_data(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> None:
        """Validate input data for HP filtering."""
        # HP filter works with any data length, but should have reasonable size
        min_samples = 8  # Arbitrary minimum for meaningful decomposition
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if len(data) < min_samples:
                raise InsufficientDataError(
                    f"Data length {len(data)} is less than recommended minimum {min_samples}"
                )
        else:
            if data.shape[0] < min_samples:
                raise InsufficientDataError(
                    f"Data length {data.shape[0]} is less than recommended minimum {min_samples}"
                )
    
    def _hp_filter(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Hodrick-Prescott filter to decompose time series."""
        # Use statsmodels implementation
        cycle, trend = sm.tsa.filters.hpfilter(data, lamb=self.lambda_param)
        return trend, cycle
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply Hodrick-Prescott filtering to the data.
        
        Args:
            data: Input data to filter
            
        Returns:
            Filtered data (trend or cycle component)
        """
        self._validate_data(data)
        data = self._handle_nans(data)
        
        # Apply HP filter
        if isinstance(data, pd.Series):
            filtered = self._filter_series(data)
        elif isinstance(data, pd.DataFrame):
            filtered = self._filter_dataframe(data)
        else:
            filtered = self._filter_array(data)
        
        # Return residuals if requested
        if self.return_residuals:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                residuals = data - filtered
                return residuals
            else:
                return data - filtered
        else:
            return filtered
    
    def _filter_array(self, data: np.ndarray) -> np.ndarray:
        """Apply HP filter to a numpy array."""
        trend, cycle = self._hp_filter(data)
        return trend if self.return_trend else cycle
    
    def _filter_series(self, data: pd.Series) -> pd.Series:
        """Apply HP filter to a pandas Series."""
        values = data.values
        filtered_values = self._filter_array(values)
        return pd.Series(filtered_values, index=data.index)
    
    def _filter_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply HP filter to a pandas DataFrame."""
        result = pd.DataFrame(index=data.index, columns=data.columns)
        
        for col in data.columns:
            result[col] = self._filter_series(data[col])
            
        return result


class AdaptiveFilter(BaseFilter):
    """
    Adaptive Filter Transformer.
    
    Dynamically selects the best filtering method based on data characteristics.
    """
    
    def __init__(
        self, 
        methods: Optional[List[FilterMethod]] = None,
        auto_select: bool = True,
        preserve_original: bool = False,
        return_residuals: bool = False,
        persist_state: bool = True
    ):
        """
        Initialize the Adaptive filter.
        
        Args:
            methods: List of filtering methods to consider
            auto_select: Whether to automatically select the best method
            preserve_original: Whether to preserve the original data
            return_residuals: Whether to return residuals instead of filtered data
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            preserve_original=preserve_original,
            return_residuals=return_residuals,
            persist_state=persist_state
        )
        
        self.methods = methods or [
            FilterMethod.MOVING_AVERAGE,
            FilterMethod.EXPONENTIAL,
            FilterMethod.SAVITZKY_GOLAY,
            FilterMethod.BUTTERWORTH
        ]
        self.auto_select = auto_select
        
        # Dictionary to store filter instances
        self._filters = {}
        self._selected_method = None
        
        # Initialize filter instances
        for method in self.methods:
            if method == FilterMethod.MOVING_AVERAGE:
                self._filters[method] = MovingAverageFilter(
                    preserve_original=preserve_original,
                    return_residuals=return_residuals,
                    persist_state=persist_state
                )
            elif method == FilterMethod.EXPONENTIAL:
                self._filters[method] = ExponentialFilter(
                    preserve_original=preserve_original,
                    return_residuals=return_residuals,
                    persist_state=persist_state
                )
            elif method == FilterMethod.SAVITZKY_GOLAY:
                self._filters[method] = SavitzkyGolayFilter(
                    preserve_original=preserve_original,
                    return_residuals=return_residuals,
                    persist_state=persist_state
                )
            elif method == FilterMethod.BUTTERWORTH:
                self._filters[method] = ButterworthFilter(
                    cutoff=0.1,  # Default cutoff
                    preserve_original=preserve_original,
                    return_residuals=return_residuals,
                    persist_state=persist_state
                )
    
    def _select_method(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> FilterMethod:
        """
        Select the best filtering method based on data characteristics.
        
        Args:
            data: Input data to analyze
            
        Returns:
            The selected filtering method
        """
        # Convert data to numpy for analysis
        if isinstance(data, pd.Series):
            values = data.values
        elif isinstance(data, pd.DataFrame):
            # For dataframes, we'll analyze the first column for simplicity
            values = data.iloc[:, 0].values
        else:
            values = data
            
        # Remove NaNs for analysis
        values = values[~np.isnan(values)]
        
        # Calculate statistics
        n = len(values)
        diff = np.diff(values)
        acf = np.correlate(diff, diff, mode='full')[n-1:] / (np.var(diff) * np.arange(n, 0, -1))
        
        # Check for noise characteristics
        noise_level = np.mean(np.abs(diff))
        autocorr = acf[1]  # First-lag autocorrelation
        
        logger.debug(f"Noise level: {noise_level}, Autocorrelation: {autocorr}")
        
        # Selection logic based on data characteristics
        if abs(autocorr) > 0.7:
            # Strong autocorrelation - likely smooth signal with some noise
            if noise_level > 0.1 * np.std(values):
                # Higher noise - use Savitzky-Golay for polynomial fitting
                return FilterMethod.SAVITZKY_GOLAY
            else:
                # Lower noise - use Exponential for smoother transitions
                return FilterMethod.EXPONENTIAL
        else:
            # Weaker autocorrelation - more random or complex signal
            if noise_level > 0.2 * np.std(values):
                # Higher noise - use Butterworth for frequency-based filtering
                return FilterMethod.BUTTERWORTH
            else:
                # Lower noise - use Moving Average for simplicity
                return FilterMethod.MOVING_AVERAGE
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply adaptive filtering to the data.
        
        Args:
            data: Input data to filter
            
        Returns:
            Filtered data
        """
        data = self._handle_nans(data)
        
        # Select the best method if auto_select is enabled
        if self.auto_select:
            method = self._select_method(data)
            self._selected_method = method
            logger.debug(f"Selected filtering method: {method.value}")
        elif self._selected_method is None:
            # Default to first method if not auto-selecting
            method = self.methods[0]
            self._selected_method = method
        else:
            # Use previously selected method
            method = self._selected_method
        
        # Apply the selected filter
        filter_obj = self._filters[method]
        return filter_obj.transform(data)
    
    def reset(self) -> None:
        """Reset the internal state of all filters."""
        super().reset()
        for filter_obj in self._filters.values():
            filter_obj.reset()
        self._selected_method = None


class FilterFactory:
    """Factory class for creating filter instances based on method names."""
    
    @staticmethod
    def create_filter(
        method: Union[str, FilterMethod],
        **kwargs
    ) -> BaseFilter:
        """
        Create a filter instance based on the specified method.
        
        Args:
            method: Filtering method name or enum value
            **kwargs: Additional arguments for the filter
            
        Returns:
            Instantiated filter object
            
        Raises:
            ValueError: If the method is not supported
        """
        # Convert string to enum if necessary
        if isinstance(method, str):
            try:
                method = FilterMethod(method)
            except ValueError:
                raise ValueError(f"Unknown filtering method: {method}")
        
        # Create the appropriate filter
        if method == FilterMethod.MOVING_AVERAGE:
            return MovingAverageFilter(**kwargs)
        elif method == FilterMethod.EXPONENTIAL:
            return ExponentialFilter(**kwargs)
        elif method == FilterMethod.KALMAN:
            return KalmanFilter(**kwargs)
        elif method == FilterMethod.SAVITZKY_GOLAY:
            return SavitzkyGolayFilter(**kwargs)
        elif method == FilterMethod.BUTTERWORTH:
            return ButterworthFilter(**kwargs)
        elif method == FilterMethod.WAVELET:
            return WaveletFilter(**kwargs)
        elif method == FilterMethod.HODRICK_PRESCOTT:
            return HodrickPrescottFilter(**kwargs)
        elif method == FilterMethod.ADAPTIVE:
            return AdaptiveFilter(**kwargs)
        else:
            raise ValueError(f"Filter not implemented for method: {method}")

