

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Normalizers Module for Feature Service

This module implements various normalization techniques for transforming 
features before they're used in strategies or machine learning models.
Advanced normalization methods help maintain feature quality across 
different market conditions and timeframes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Callable
from enum import Enum
import scipy.stats as stats
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    QuantileTransformer, PowerTransformer
)

from common.logger import get_logger
from common.utils import timeit, rolling_apply
from common.exceptions import (
    DataTransformationError, InsufficientDataError, 
    FeatureCalculationError
)
from feature_service.transformers.base import BaseTransformer

logger = get_logger(__name__)


class NormalizationMethod(Enum):
    """Enum defining the available normalization methods."""
    Z_SCORE = "z_score"
    MIN_MAX = "min_max"
    ROBUST = "robust"
    QUANTILE = "quantile"
    DECIMAL_SCALING = "decimal_scaling"
    LOG = "log"
    SQRT = "sqrt"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    POWER = "power"
    MAX_ABS = "max_abs"
    PERCENTILE = "percentile"
    RANK = "rank"
    RELATIVE_STRENGTH = "relative_strength"
    ADAPTIVE = "adaptive"
    ROBUST_SIGMOID = "robust_sigmoid"
    WINSORIZE = "winsorize"
    REGIME_AWARE = "regime_aware"


class BaseNormalizer(BaseTransformer):
    """Base class for all normalizer transformers."""
    
    def __init__(
        self, 
        window_size: Optional[int] = None,
        clip_outliers: bool = False,
        outlier_threshold: float = 3.0,
        persist_state: bool = True,
        dtype: np.dtype = np.float32
    ):
        """
        Initialize the base normalizer.
        
        Args:
            window_size: Size of the rolling window for normalization
            clip_outliers: Whether to clip outliers before normalization
            outlier_threshold: Z-score threshold for outlier clipping
            persist_state: Whether to maintain state between calls
            dtype: Data type for numeric operations
        """
        super().__init__(persist_state=persist_state)
        self.window_size = window_size
        self.clip_outliers = clip_outliers
        self.outlier_threshold = outlier_threshold
        self.dtype = dtype
        self._state = {}
        self._fitted = False
    
    def _preprocess(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Preprocess the data before normalization.
        
        Args:
            data: Input data for preprocessing
            
        Returns:
            Preprocessed data
        """
        if self.clip_outliers:
            return self._clip_outliers(data)
        return data
    
    def _clip_outliers(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Clip outliers in the data based on z-score threshold.
        
        Args:
            data: Input data for outlier clipping
            
        Returns:
            Data with outliers clipped
        """
        if isinstance(data, (pd.Series, pd.DataFrame)):
            mean = data.mean()
            std = data.std()
            lower_bound = mean - self.outlier_threshold * std
            upper_bound = mean + self.outlier_threshold * std
            return data.clip(lower=lower_bound, upper=upper_bound)
        else:
            mean = np.nanmean(data)
            std = np.nanstd(data)
            lower_bound = mean - self.outlier_threshold * std
            upper_bound = mean + self.outlier_threshold * std
            return np.clip(data, lower_bound, upper_bound)
    
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
    
    def _validate_data(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> None:
        """
        Validate that the data is suitable for normalization.
        
        Args:
            data: Input data to validate
            
        Raises:
            InsufficientDataError: If data doesn't meet minimum requirements
            DataTransformationError: If data has unexpected properties
        """
        if self.window_size is not None:
            if isinstance(data, (pd.Series, pd.DataFrame)):
                if len(data) < self.window_size:
                    raise InsufficientDataError(
                        f"Data length {len(data)} is less than window size {self.window_size}"
                    )
            else:
                if data.shape[0] < self.window_size:
                    raise InsufficientDataError(
                        f"Data length {data.shape[0]} is less than window size {self.window_size}"
                    )
                    
        # Check for non-finite values
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if data.isna().all().all():
                raise DataTransformationError("All values in data are NaN")
        else:
            if not np.any(np.isfinite(data)):
                raise DataTransformationError("No finite values in data")
    
    def _get_window_data(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        index: int
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Get the window of data for the given index.
        
        Args:
            data: Full dataset
            index: Current index position
            
        Returns:
            Window of data for the given index
        """
        if self.window_size is None:
            return data
            
        start_idx = max(0, index - self.window_size + 1)
        
        if isinstance(data, pd.Series):
            return data.iloc[start_idx:index+1]
        elif isinstance(data, pd.DataFrame):
            return data.iloc[start_idx:index+1]
        else:
            return data[start_idx:index+1]
    
    def reset(self) -> None:
        """Reset the internal state of the normalizer."""
        self._state = {}
        self._fitted = False


class ZScoreNormalizer(BaseNormalizer):
    """
    Z-Score (Standard) Normalization Transformer.
    
    Standardizes data by removing the mean and scaling to unit variance.
    Can operate in global or rolling window modes.
    """
    
    def __init__(
        self, 
        window_size: Optional[int] = None,
        center: bool = True,
        scale: bool = True,
        clip_outliers: bool = False,
        outlier_threshold: float = 3.0,
        epsilon: float = 1e-8,
        persist_state: bool = True
    ):
        """
        Initialize the Z-Score normalizer.
        
        Args:
            window_size: Size of the rolling window for normalization
            center: Whether to center the data (subtract mean)
            scale: Whether to scale the data (divide by std)
            clip_outliers: Whether to clip outliers before normalization
            outlier_threshold: Z-score threshold for outlier clipping
            epsilon: Small value to avoid division by zero
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            window_size=window_size,
            clip_outliers=clip_outliers,
            outlier_threshold=outlier_threshold,
            persist_state=persist_state
        )
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self._state = {"mean": None, "std": None}
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply Z-Score normalization to the data.
        
        Args:
            data: Input data to normalize
            
        Returns:
            Normalized data
        """
        self._validate_data(data)
        data = self._preprocess(data)
        data = self._handle_nans(data)
        
        if self.window_size is None:
            # Global normalization
            if isinstance(data, (pd.Series, pd.DataFrame)):
                mean = data.mean() if self.center else 0
                std = data.std() if self.scale else 1
                self._state["mean"] = mean
                self._state["std"] = std
                normalized = (data - mean) / (std + self.epsilon)
            else:
                mean = np.nanmean(data) if self.center else 0
                std = np.nanstd(data) if self.scale else 1
                self._state["mean"] = mean
                self._state["std"] = std
                normalized = (data - mean) / (std + self.epsilon)
            
            self._fitted = True
            return normalized
        else:
            # Rolling window normalization
            if isinstance(data, pd.Series):
                return self._rolling_normalize_series(data)
            elif isinstance(data, pd.DataFrame):
                return self._rolling_normalize_dataframe(data)
            else:
                return self._rolling_normalize_array(data)
    
    def _rolling_normalize_series(self, data: pd.Series) -> pd.Series:
        """Apply rolling normalization to a pandas Series."""
        result = pd.Series(index=data.index, dtype=self.dtype)
        
        for i in range(len(data)):
            window = self._get_window_data(data, i)
            mean = window.mean() if self.center else 0
            std = window.std() if self.scale else 1
            result.iloc[i] = (data.iloc[i] - mean) / (std + self.epsilon)
            
            # Update state with most recent values
            if i == len(data) - 1:
                self._state["mean"] = mean
                self._state["std"] = std
                
        self._fitted = True
        return result
    
    def _rolling_normalize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling normalization to a pandas DataFrame."""
        result = pd.DataFrame(index=data.index, columns=data.columns, dtype=self.dtype)
        
        for col in data.columns:
            result[col] = self._rolling_normalize_series(data[col])
            
        return result
    
    def _rolling_normalize_array(self, data: np.ndarray) -> np.ndarray:
        """Apply rolling normalization to a numpy array."""
        result = np.zeros_like(data, dtype=self.dtype)
        
        for i in range(data.shape[0]):
            window = self._get_window_data(data, i)
            mean = np.nanmean(window) if self.center else 0
            std = np.nanstd(window) if self.scale else 1
            result[i] = (data[i] - mean) / (std + self.epsilon)
            
            # Update state with most recent values
            if i == data.shape[0] - 1:
                self._state["mean"] = mean
                self._state["std"] = std
                
        self._fitted = True
        return result
    
    def inverse_transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Inverse the Z-Score normalization.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self._fitted:
            raise ValueError("Normalizer has not been fitted yet")
            
        if isinstance(data, pd.Series):
            return data * (self._state["std"] + self.epsilon) + self._state["mean"]
        elif isinstance(data, pd.DataFrame):
            result = pd.DataFrame(index=data.index, columns=data.columns)
            for col in data.columns:
                result[col] = data[col] * (self._state["std"] + self.epsilon) + self._state["mean"]
            return result
        else:
            return data * (self._state["std"] + self.epsilon) + self._state["mean"]


class MinMaxNormalizer(BaseNormalizer):
    """
    Min-Max Normalization Transformer.
    
    Scales features to a specified range (default [0, 1]).
    Can operate in global or rolling window modes.
    """
    
    def __init__(
        self, 
        window_size: Optional[int] = None,
        feature_range: Tuple[float, float] = (0, 1),
        clip_outliers: bool = False,
        outlier_threshold: float = 3.0,
        epsilon: float = 1e-8,
        persist_state: bool = True
    ):
        """
        Initialize the Min-Max normalizer.
        
        Args:
            window_size: Size of the rolling window for normalization
            feature_range: Desired range of transformed data
            clip_outliers: Whether to clip outliers before normalization
            outlier_threshold: Z-score threshold for outlier clipping
            epsilon: Small value to avoid division by zero
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            window_size=window_size,
            clip_outliers=clip_outliers,
            outlier_threshold=outlier_threshold,
            persist_state=persist_state
        )
        self.feature_range = feature_range
        self.epsilon = epsilon
        self._state = {"min": None, "max": None}
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply Min-Max normalization to the data.
        
        Args:
            data: Input data to normalize
            
        Returns:
            Normalized data
        """
        self._validate_data(data)
        data = self._preprocess(data)
        data = self._handle_nans(data)
        
        min_val, max_val = self.feature_range
        
        if self.window_size is None:
            # Global normalization
            if isinstance(data, (pd.Series, pd.DataFrame)):
                data_min = data.min()
                data_max = data.max()
                self._state["min"] = data_min
                self._state["max"] = data_max
                data_range = data_max - data_min
                normalized = min_val + (max_val - min_val) * (data - data_min) / (data_range + self.epsilon)
            else:
                data_min = np.nanmin(data)
                data_max = np.nanmax(data)
                self._state["min"] = data_min
                self._state["max"] = data_max
                data_range = data_max - data_min
                normalized = min_val + (max_val - min_val) * (data - data_min) / (data_range + self.epsilon)
            
            self._fitted = True
            return normalized
        else:
            # Rolling window normalization
            if isinstance(data, pd.Series):
                return self._rolling_normalize_series(data)
            elif isinstance(data, pd.DataFrame):
                return self._rolling_normalize_dataframe(data)
            else:
                return self._rolling_normalize_array(data)
    
    def _rolling_normalize_series(self, data: pd.Series) -> pd.Series:
        """Apply rolling min-max normalization to a pandas Series."""
        result = pd.Series(index=data.index, dtype=self.dtype)
        min_val, max_val = self.feature_range
        
        for i in range(len(data)):
            window = self._get_window_data(data, i)
            data_min = window.min()
            data_max = window.max()
            data_range = data_max - data_min
            result.iloc[i] = min_val + (max_val - min_val) * (data.iloc[i] - data_min) / (data_range + self.epsilon)
            
            # Update state with most recent values
            if i == len(data) - 1:
                self._state["min"] = data_min
                self._state["max"] = data_max
                
        self._fitted = True
        return result
    
    def _rolling_normalize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling min-max normalization to a pandas DataFrame."""
        result = pd.DataFrame(index=data.index, columns=data.columns, dtype=self.dtype)
        
        for col in data.columns:
            result[col] = self._rolling_normalize_series(data[col])
            
        return result
    
    def _rolling_normalize_array(self, data: np.ndarray) -> np.ndarray:
        """Apply rolling min-max normalization to a numpy array."""
        result = np.zeros_like(data, dtype=self.dtype)
        min_val, max_val = self.feature_range
        
        for i in range(data.shape[0]):
            window = self._get_window_data(data, i)
            data_min = np.nanmin(window)
            data_max = np.nanmax(window)
            data_range = data_max - data_min
            result[i] = min_val + (max_val - min_val) * (data[i] - data_min) / (data_range + self.epsilon)
            
            # Update state with most recent values
            if i == data.shape[0] - 1:
                self._state["min"] = data_min
                self._state["max"] = data_max
                
        self._fitted = True
        return result
    
    def inverse_transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Inverse the Min-Max normalization.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self._fitted:
            raise ValueError("Normalizer has not been fitted yet")
            
        min_val, max_val = self.feature_range
        data_min, data_max = self._state["min"], self._state["max"]
        data_range = data_max - data_min
        
        if isinstance(data, pd.Series):
            return (data - min_val) * (data_range + self.epsilon) / (max_val - min_val) + data_min
        elif isinstance(data, pd.DataFrame):
            result = pd.DataFrame(index=data.index, columns=data.columns)
            for col in data.columns:
                result[col] = (data[col] - min_val) * (data_range + self.epsilon) / (max_val - min_val) + data_min
            return result
        else:
            return (data - min_val) * (data_range + self.epsilon) / (max_val - min_val) + data_min


class RobustNormalizer(BaseNormalizer):
    """
    Robust Normalization Transformer.
    
    Scales features using statistics that are robust to outliers.
    Uses median and interquartile range instead of mean and standard deviation.
    """
    
    def __init__(
        self, 
        window_size: Optional[int] = None,
        center: bool = True,
        scale: bool = True,
        quantile_range: Tuple[float, float] = (0.25, 0.75),
        epsilon: float = 1e-8,
        persist_state: bool = True
    ):
        """
        Initialize the Robust normalizer.
        
        Args:
            window_size: Size of the rolling window for normalization
            center: Whether to center the data (subtract median)
            scale: Whether to scale the data (divide by IQR)
            quantile_range: Quantile range for calculating IQR
            epsilon: Small value to avoid division by zero
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            window_size=window_size,
            persist_state=persist_state
        )
        self.center = center
        self.scale = scale
        self.quantile_range = quantile_range
        self.epsilon = epsilon
        self._state = {"median": None, "iqr": None}
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply Robust normalization to the data.
        
        Args:
            data: Input data to normalize
            
        Returns:
            Normalized data
        """
        self._validate_data(data)
        data = self._preprocess(data)
        data = self._handle_nans(data)
        
        q_low, q_high = self.quantile_range
        
        if self.window_size is None:
            # Global normalization
            if isinstance(data, (pd.Series, pd.DataFrame)):
                median = data.median() if self.center else 0
                q1 = data.quantile(q_low)
                q3 = data.quantile(q_high)
                iqr = q3 - q1 if self.scale else 1
                self._state["median"] = median
                self._state["iqr"] = iqr
                normalized = (data - median) / (iqr + self.epsilon)
            else:
                median = np.nanmedian(data) if self.center else 0
                q1 = np.nanquantile(data, q_low)
                q3 = np.nanquantile(data, q_high)
                iqr = q3 - q1 if self.scale else 1
                self._state["median"] = median
                self._state["iqr"] = iqr
                normalized = (data - median) / (iqr + self.epsilon)
            
            self._fitted = True
            return normalized
        else:
            # Rolling window normalization
            if isinstance(data, pd.Series):
                return self._rolling_normalize_series(data)
            elif isinstance(data, pd.DataFrame):
                return self._rolling_normalize_dataframe(data)
            else:
                return self._rolling_normalize_array(data)
    
    def _rolling_normalize_series(self, data: pd.Series) -> pd.Series:
        """Apply rolling robust normalization to a pandas Series."""
        result = pd.Series(index=data.index, dtype=self.dtype)
        q_low, q_high = self.quantile_range
        
        for i in range(len(data)):
            window = self._get_window_data(data, i)
            median = window.median() if self.center else 0
            q1 = window.quantile(q_low)
            q3 = window.quantile(q_high)
            iqr = q3 - q1 if self.scale else 1
            result.iloc[i] = (data.iloc[i] - median) / (iqr + self.epsilon)
            
            # Update state with most recent values
            if i == len(data) - 1:
                self._state["median"] = median
                self._state["iqr"] = iqr
                
        self._fitted = True
        return result
    
    def _rolling_normalize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling robust normalization to a pandas DataFrame."""
        result = pd.DataFrame(index=data.index, columns=data.columns, dtype=self.dtype)
        
        for col in data.columns:
            result[col] = self._rolling_normalize_series(data[col])
            
        return result
    
    def _rolling_normalize_array(self, data: np.ndarray) -> np.ndarray:
        """Apply rolling robust normalization to a numpy array."""
        result = np.zeros_like(data, dtype=self.dtype)
        q_low, q_high = self.quantile_range
        
        for i in range(data.shape[0]):
            window = self._get_window_data(data, i)
            median = np.nanmedian(window) if self.center else 0
            q1 = np.nanquantile(window, q_low)
            q3 = np.nanquantile(window, q_high)
            iqr = q3 - q1 if self.scale else 1
            result[i] = (data[i] - median) / (iqr + self.epsilon)
            
            # Update state with most recent values
            if i == data.shape[0] - 1:
                self._state["median"] = median
                self._state["iqr"] = iqr
                
        self._fitted = True
        return result
    
    def inverse_transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Inverse the Robust normalization.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self._fitted:
            raise ValueError("Normalizer has not been fitted yet")
            
        median, iqr = self._state["median"], self._state["iqr"]
        
        if isinstance(data, pd.Series):
            return data * (iqr + self.epsilon) + median
        elif isinstance(data, pd.DataFrame):
            result = pd.DataFrame(index=data.index, columns=data.columns)
            for col in data.columns:
                result[col] = data[col] * (iqr + self.epsilon) + median
            return result
        else:
            return data * (iqr + self.epsilon) + median


class TanhNormalizer(BaseNormalizer):
    """
    Tanh Normalization Transformer.
    
    Maps the input to a range of (-1, 1) using the hyperbolic tangent function.
    Good for handling data with extreme outliers.
    """
    
    def __init__(
        self, 
        window_size: Optional[int] = None,
        center: bool = True,
        scaling_factor: float = 2.0,
        persist_state: bool = True
    ):
        """
        Initialize the Tanh normalizer.
        
        Args:
            window_size: Size of the rolling window for normalization
            center: Whether to center the data before applying tanh
            scaling_factor: Controls the steepness of the tanh function
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            window_size=window_size,
            persist_state=persist_state
        )
        self.center = center
        self.scaling_factor = scaling_factor
        self._state = {"mean": None, "std": None}
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply Tanh normalization to the data.
        
        Args:
            data: Input data to normalize
            
        Returns:
            Normalized data
        """
        self._validate_data(data)
        data = self._preprocess(data)
        data = self._handle_nans(data)
        
        if self.window_size is None:
            # Global normalization
            if isinstance(data, (pd.Series, pd.DataFrame)):
                mean = data.mean() if self.center else 0
                std = data.std()
                self._state["mean"] = mean
                self._state["std"] = std
                normalized = np.tanh((data - mean) / (std * self.scaling_factor))
            else:
                mean = np.nanmean(data) if self.center else 0
                std = np.nanstd(data)
                self._state["mean"] = mean
                self._state["std"] = std
                normalized = np.tanh((data - mean) / (std * self.scaling_factor))
            
            self._fitted = True
            return normalized
        else:
            # Rolling window normalization
            if isinstance(data, pd.Series):
                return self._rolling_normalize_series(data)
            elif isinstance(data, pd.DataFrame):
                return self._rolling_normalize_dataframe(data)
            else:
                return self._rolling_normalize_array(data)
    
    def _rolling_normalize_series(self, data: pd.Series) -> pd.Series:
        """Apply rolling tanh normalization to a pandas Series."""
        result = pd.Series(index=data.index, dtype=self.dtype)
        
        for i in range(len(data)):
            window = self._get_window_data(data, i)
            mean = window.mean() if self.center else 0
            std = window.std()
            result.iloc[i] = np.tanh((data.iloc[i] - mean) / (std * self.scaling_factor))
            
            # Update state with most recent values
            if i == len(data) - 1:
                self._state["mean"] = mean
                self._state["std"] = std
                
        self._fitted = True
        return result
    
    def _rolling_normalize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling tanh normalization to a pandas DataFrame."""
        result = pd.DataFrame(index=data.index, columns=data.columns, dtype=self.dtype)
        
        for col in data.columns:
            result[col] = self._rolling_normalize_series(data[col])
            
        return result
    
    def _rolling_normalize_array(self, data: np.ndarray) -> np.ndarray:
        """Apply rolling tanh normalization to a numpy array."""
        result = np.zeros_like(data, dtype=self.dtype)
        
        for i in range(data.shape[0]):
            window = self._get_window_data(data, i)
            mean = np.nanmean(window) if self.center else 0
            std = np.nanstd(window)
            result[i] = np.tanh((data[i] - mean) / (std * self.scaling_factor))
            
            # Update state with most recent values
            if i == data.shape[0] - 1:
                self._state["mean"] = mean
                self._state["std"] = std
                
        self._fitted = True
        return result
    
    def inverse_transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Inverse the Tanh normalization.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self._fitted:
            raise ValueError("Normalizer has not been fitted yet")
            
        mean, std = self._state["mean"], self._state["std"]
        
        # Using arctanh (inverse of tanh)
        if isinstance(data, pd.Series):
            return np.arctanh(data) * (std * self.scaling_factor) + mean
        elif isinstance(data, pd.DataFrame):
            result = pd.DataFrame(index=data.index, columns=data.columns)
            for col in data.columns:
                result[col] = np.arctanh(data[col]) * (std * self.scaling_factor) + mean
            return result
        else:
            return np.arctanh(data) * (std * self.scaling_factor) + mean


class AdaptiveNormalizer(BaseNormalizer):
    """
    Adaptive Normalization Transformer.
    
    Dynamically selects the best normalization method based on data characteristics.
    Combines multiple normalization techniques for optimal results.
    """
    
    def __init__(
        self, 
        window_size: Optional[int] = None,
        methods: Optional[List[NormalizationMethod]] = None,
        kurtosis_threshold: float = 3.0,
        skew_threshold: float = 0.5,
        regime_aware: bool = True,
        persist_state: bool = True
    ):
        """
        Initialize the Adaptive normalizer.
        
        Args:
            window_size: Size of the rolling window for normalization
            methods: List of normalization methods to consider
            kurtosis_threshold: Threshold for determining heavy-tailed distributions
            skew_threshold: Threshold for determining skewed distributions
            regime_aware: Whether to adapt normalization based on market regime
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            window_size=window_size,
            persist_state=persist_state
        )
        
        self.methods = methods or [
            NormalizationMethod.Z_SCORE,
            NormalizationMethod.ROBUST,
            NormalizationMethod.TANH,
            NormalizationMethod.MIN_MAX
        ]
        self.kurtosis_threshold = kurtosis_threshold
        self.skew_threshold = skew_threshold
        self.regime_aware = regime_aware
        self._normalizers = {}
        self._selected_method = None
        self._state = {"method": None, "params": {}}
        
        # Initialize all available normalizers
        for method in self.methods:
            if method == NormalizationMethod.Z_SCORE:
                self._normalizers[method] = ZScoreNormalizer(
                    window_size=window_size, persist_state=persist_state
                )
            elif method == NormalizationMethod.MIN_MAX:
                self._normalizers[method] = MinMaxNormalizer(
                    window_size=window_size, persist_state=persist_state
                )
            elif method == NormalizationMethod.ROBUST:
                self._normalizers[method] = RobustNormalizer(
                    window_size=window_size, persist_state=persist_state
                )
            elif method == NormalizationMethod.TANH:
                self._normalizers[method] = TanhNormalizer(
                    window_size=window_size, persist_state=persist_state
                )
    
    def _select_method(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> NormalizationMethod:
        """
        Select the best normalization method based on data characteristics.
        
        Args:
            data: Input data to analyze
            
        Returns:
            The selected normalization method
        """
        # Convert data to numpy for statistics calculation
        if isinstance(data, pd.Series):
            values = data.values
        elif isinstance(data, pd.DataFrame):
            # For dataframes, we'll analyze the first column for simplicity
            values = data.iloc[:, 0].values
        else:
            values = data
            
        # Remove NaNs for statistical calculations
        values = values[~np.isnan(values)]
        
        # Calculate statistics
        try:
            skewness = stats.skew(values)
            kurtosis = stats.kurtosis(values)
            
            logger.debug(f"Data skewness: {skewness}, kurtosis: {kurtosis}")
            
            # Logic to select appropriate method
            if abs(kurtosis) > self.kurtosis_threshold:
                # Heavy-tailed distribution, use robust methods
                if abs(skewness) > self.skew_threshold:
                    # Also skewed, use tanh for better handling
                    return NormalizationMethod.TANH
                else:
                    # Symmetric but heavy-tailed
                    return NormalizationMethod.ROBUST
            elif abs(skewness) > self.skew_threshold:
                # Skewed but not heavy-tailed
                return NormalizationMethod.ROBUST
            else:
                # Well-behaved data
                return NormalizationMethod.Z_SCORE
                
        except Exception as e:
            logger.warning(f"Error calculating statistics for method selection: {e}")
            # Default to Z-score in case of errors
            return NormalizationMethod.Z_SCORE
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply adaptive normalization to the data.
        
        Args:
            data: Input data to normalize
            
        Returns:
            Normalized data
        """
        self._validate_data(data)
        data = self._preprocess(data)
        data = self._handle_nans(data)
        
        # Select the best method for this data
        method = self._select_method(data)
        self._selected_method = method
        self._state["method"] = method
        
        # Use the selected normalizer
        normalizer = self._normalizers[method]
        normalized = normalizer.transform(data)
        
        # Store the normalizer's state
        self._state["params"] = normalizer._state
        
        self._fitted = True
        return normalized
    
    def inverse_transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Inverse the adaptive normalization.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self._fitted:
            raise ValueError("Normalizer has not been fitted yet")
            
        method = self._state["method"]
        normalizer = self._normalizers[method]
        
        # Set the normalizer's state to match our saved state
        normalizer._state = self._state["params"]
        normalizer._fitted = True
        
        return normalizer.inverse_transform(data)
    
    def reset(self) -> None:
        """Reset the internal state of the normalizer."""
        super().reset()
        for normalizer in self._normalizers.values():
            normalizer.reset()
        self._selected_method = None


class RelativeStrengthNormalizer(BaseNormalizer):
    """
    Relative Strength Normalization Transformer.
    
    Normalizes data based on its relative performance compared to a reference.
    Useful for cross-asset analysis and relative strength indicators.
    """
    
    def __init__(
        self, 
        window_size: Optional[int] = None,
        reference_data: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        use_percentage: bool = True,
        use_log: bool = False,
        base_normalizer: Optional[BaseNormalizer] = None,
        persist_state: bool = True
    ):
        """
        Initialize the Relative Strength normalizer.
        
        Args:
            window_size: Size of the rolling window for normalization
            reference_data: Reference data for relative comparison
            use_percentage: Whether to use percentage change for comparison
            use_log: Whether to use log returns for comparison
            base_normalizer: Additional normalizer to apply after relative calculation
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            window_size=window_size,
            persist_state=persist_state
        )
        self.reference_data = reference_data
        self.use_percentage = use_percentage
        self.use_log = use_log
        self.base_normalizer = base_normalizer or ZScoreNormalizer(
            window_size=window_size, persist_state=persist_state
        )
        self._state = {"reference": None, "base_normalizer_state": None}
    
    def set_reference(
        self, 
        reference_data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> None:
        """
        Set the reference data for relative comparison.
        
        Args:
            reference_data: Reference data for relative comparison
        """
        self.reference_data = reference_data
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        reference_data: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply relative strength normalization to the data.
        
        Args:
            data: Input data to normalize
            reference_data: Optional reference data to override the instance reference
            
        Returns:
            Normalized data
        """
        self._validate_data(data)
        data = self._preprocess(data)
        data = self._handle_nans(data)
        
        # Use provided reference or instance reference
        ref_data = reference_data if reference_data is not None else self.reference_data
        
        if ref_data is None:
            raise ValueError("Reference data must be provided either at initialization or when calling transform")
        
        # Ensure reference data has the same format
        ref_data = self._preprocess(ref_data)
        ref_data = self._handle_nans(ref_data)
        
        # Calculate relative strength
        if self.use_percentage:
            if self.use_log:
                if isinstance(data, pd.Series) and isinstance(ref_data, pd.Series):
                    relative = np.log(data / data.shift(1)) - np.log(ref_data / ref_data.shift(1))
                    relative.iloc[0] = 0  # First value is NaN, set to 0
                elif isinstance(data, pd.DataFrame) and isinstance(ref_data, pd.DataFrame):
                    relative = np.log(data / data.shift(1)) - np.log(ref_data / ref_data.shift(1))
                    relative.iloc[0, :] = 0  # First values are NaN, set to 0
                else:
                    # For numpy arrays
                    data_returns = np.log(data[1:] / data[:-1])
                    ref_returns = np.log(ref_data[1:] / ref_data[:-1])
                    relative = np.zeros_like(data)
                    relative[1:] = data_returns - ref_returns
            else:
                if isinstance(data, pd.Series) and isinstance(ref_data, pd.Series):
                    relative = (data / data.shift(1) - 1) - (ref_data / ref_data.shift(1) - 1)
                    relative.iloc[0] = 0  # First value is NaN, set to 0
                elif isinstance(data, pd.DataFrame) and isinstance(ref_data, pd.DataFrame):
                    relative = (data / data.shift(1) - 1) - (ref_data / ref_data.shift(1) - 1)
                    relative.iloc[0, :] = 0  # First values are NaN, set to 0
                else:
                    # For numpy arrays
                    data_returns = data[1:] / data[:-1] - 1
                    ref_returns = ref_data[1:] / ref_data[:-1] - 1
                    relative = np.zeros_like(data)
                    relative[1:] = data_returns - ref_returns
        else:
            # Direct ratio
            if isinstance(data, (pd.Series, pd.DataFrame)) and isinstance(ref_data, (pd.Series, pd.DataFrame)):
                relative = data / ref_data
            else:
                relative = data / ref_data
        
        # Store the reference for inverse transform
        self._state["reference"] = ref_data
        
        # Apply base normalizer if specified
        if self.base_normalizer is not None:
            normalized = self.base_normalizer.transform(relative)
            self._state["base_normalizer_state"] = self.base_normalizer._state
            return normalized
        else:
            self._fitted = True
            return relative
    
    def inverse_transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Inverse the relative strength normalization.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self._fitted:
            raise ValueError("Normalizer has not been fitted yet")
            
        ref_data = self._state["reference"]
        
        # First undo the base normalizer if applied
        if self.base_normalizer is not None:
            self.base_normalizer._state = self._state["base_normalizer_state"]
            self.base_normalizer._fitted = True
            relative = self.base_normalizer.inverse_transform(data)
        else:
            relative = data
        
        # Then undo the relative transformation
        if self.use_percentage:
            if self.use_log:
                if isinstance(relative, pd.Series) and isinstance(ref_data, pd.Series):
                    # Start with base 1 and accumulate log returns
                    result = pd.Series(index=relative.index, data=1.0)
                    for i in range(1, len(relative)):
                        ref_return = np.log(ref_data.iloc[i] / ref_data.iloc[i-1])
                        result.iloc[i] = result.iloc[i-1] * np.exp(relative.iloc[i] + ref_return)
                    return result * ref_data.iloc[0]  # Scale to match original starting point
                elif isinstance(relative, pd.DataFrame) and isinstance(ref_data, pd.DataFrame):
                    # For dataframes, process each column
                    result = pd.DataFrame(index=relative.index, columns=relative.columns, data=1.0)
                    for col in relative.columns:
                        for i in range(1, len(relative)):
                            ref_return = np.log(ref_data[col].iloc[i] / ref_data[col].iloc[i-1])
                            result[col].iloc[i] = result[col].iloc[i-1] * np.exp(relative[col].iloc[i] + ref_return)
                        result[col] = result[col] * ref_data[col].iloc[0]  # Scale to match original
                    return result
                else:
                    # For numpy arrays - more complex reconstruction
                    result = np.ones_like(relative)
                    for i in range(1, len(relative)):
                        ref_return = np.log(ref_data[i] / ref_data[i-1])
                        result[i] = result[i-1] * np.exp(relative[i] + ref_return)
                    return result * ref_data[0]  # Scale to match original
            else:
                if isinstance(relative, pd.Series) and isinstance(ref_data, pd.Series):
                    # Start with base 1 and accumulate percentage returns
                    result = pd.Series(index=relative.index, data=1.0)
                    for i in range(1, len(relative)):
                        ref_return = ref_data.iloc[i] / ref_data.iloc[i-1] - 1
                        result.iloc[i] = result.iloc[i-1] * (1 + relative.iloc[i] + ref_return)
                    return result * ref_data.iloc[0]  # Scale to match original
                elif isinstance(relative, pd.DataFrame) and isinstance(ref_data, pd.DataFrame):
                    # For dataframes, process each column
                    result = pd.DataFrame(index=relative.index, columns=relative.columns, data=1.0)
                    for col in relative.columns:
                        for i in range(1, len(relative)):
                            ref_return = ref_data[col].iloc[i] / ref_data[col].iloc[i-1] - 1
                            result[col].iloc[i] = result[col].iloc[i-1] * (1 + relative[col].iloc[i] + ref_return)
                        result[col] = result[col] * ref_data[col].iloc[0]  # Scale to match original
                    return result
                else:
                    # For numpy arrays
                    result = np.ones_like(relative)
                    for i in range(1, len(relative)):
                        ref_return = ref_data[i] / ref_data[i-1] - 1
                        result[i] = result[i-1] * (1 + relative[i] + ref_return)
                    return result * ref_data[0]  # Scale to match original
        else:
            # Direct ratio inversion
            return relative * ref_data


class RegimeAwareNormalizer(BaseNormalizer):
    """
    Regime-Aware Normalization Transformer.
    
    Applies different normalization techniques based on detected market regime.
    Optimizes transformation for different volatility environments.
    """
    
    def __init__(
        self, 
        window_size: Optional[int] = None,
        volatility_window: int = 20,
        trend_window: int = 50,
        regime_thresholds: Optional[Dict[str, float]] = None,
        persist_state: bool = True
    ):
        """
        Initialize the Regime-Aware normalizer.
        
        Args:
            window_size: Size of the rolling window for normalization
            volatility_window: Window for calculating volatility
            trend_window: Window for calculating trend strength
            regime_thresholds: Thresholds for regime classification
            persist_state: Whether to maintain state between calls
        """
        super().__init__(
            window_size=window_size,
            persist_state=persist_state
        )
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.regime_thresholds = regime_thresholds or {
            "high_vol": 1.5,  # Volatility threshold as multiple of average
            "trend_strength": 0.6  # Correlation threshold for trend identification
        }
        
        # Initialize regime-specific normalizers
        self.normalizers = {
            "high_vol_trend": TanhNormalizer(window_size=window_size, persist_state=persist_state),
            "high_vol_range": RobustNormalizer(window_size=window_size, persist_state=persist_state),
            "low_vol_trend": ZScoreNormalizer(window_size=window_size, persist_state=persist_state),
            "low_vol_range": MinMaxNormalizer(window_size=window_size, persist_state=persist_state)
        }
        
        self._state = {"regime": None, "normalizer_state": None}
        self._current_regime = None
    
    def _detect_regime(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> str:
        """
        Detect the current market regime based on data characteristics.
        
        Args:
            data: Input data to analyze
            
        Returns:
            The detected regime as a string
        """
        # Extract data for analysis
        if isinstance(data, pd.DataFrame):
            # Use the first column for regime detection
            values = data.iloc[:, 0].values
        elif isinstance(data, pd.Series):
            values = data.values
        else:
            values = data
            
        if len(values) < max(self.volatility_window, self.trend_window):
            # Not enough data, default to low volatility trending regime
            return "low_vol_trend"
        
        # Calculate volatility
        vol_window = min(self.volatility_window, len(values))
        returns = np.diff(values[-vol_window:]) / values[-vol_window:-1]
        current_vol = np.std(returns)
        historical_vol = np.std(np.diff(values) / values[:-1])
        
        # Determine if we're in high volatility regime
        high_vol = current_vol > (historical_vol * self.regime_thresholds["high_vol"])
        
        # Calculate trend strength
        trend_window = min(self.trend_window, len(values))
        x = np.arange(trend_window)
        y = values[-trend_window:]
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Determine if we're in trending or ranging regime
        trending = abs(correlation) > self.regime_thresholds["trend_strength"]
        
        # Determine regime
        if high_vol and trending:
            return "high_vol_trend"
        elif high_vol and not trending:
            return "high_vol_range"
        elif not high_vol and trending:
            return "low_vol_trend"
        else:  # low_vol and not trending
            return "low_vol_range"
    
    @timeit
    def transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Apply regime-aware normalization to the data.
        
        Args:
            data: Input data to normalize
            
        Returns:
            Normalized data
        """
        self._validate_data(data)
        data = self._preprocess(data)
        data = self._handle_nans(data)
        
        # Detect the current market regime
        regime = self._detect_regime(data)
        self._current_regime = regime
        logger.debug(f"Detected market regime: {regime}")
        
        # Get the appropriate normalizer for this regime
        normalizer = self.normalizers[regime]
        
        # Apply normalization
        normalized = normalizer.transform(data)
        
        # Store regime and normalizer state
        self._state["regime"] = regime
        self._state["normalizer_state"] = normalizer._state
        
        self._fitted = True
        return normalized
    
    def inverse_transform(
        self, 
        data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
        """
        Inverse the regime-aware normalization.
        
        Args:
            data: Normalized data to inverse transform
            
        Returns:
            Original scale data
        """
        if not self._fitted:
            raise ValueError("Normalizer has not been fitted yet")
            
        regime = self._state["regime"]
        normalizer = self.normalizers[regime]
        
        # Set normalizer state
        normalizer._state = self._state["normalizer_state"]
        normalizer._fitted = True
        
        # Apply inverse transform
        return normalizer.inverse_transform(data)
    
    def reset(self) -> None:
        """Reset the internal state of the normalizer."""
        super().reset()
        for normalizer in self.normalizers.values():
            normalizer.reset()
        self._current_regime = None


class NormalizerFactory:
    """Factory class for creating normalizer instances based on method names."""
    
    @staticmethod
    def create_normalizer(
        method: Union[str, NormalizationMethod],
        **kwargs
    ) -> BaseNormalizer:
        """
        Create a normalizer instance based on the specified method.
        
        Args:
            method: Normalization method name or enum value
            **kwargs: Additional arguments for the normalizer
            
        Returns:
            Instantiated normalizer object
            
        Raises:
            ValueError: If the method is not supported
        """
        # Convert string to enum if necessary
        if isinstance(method, str):
            try:
                method = NormalizationMethod(method)
            except ValueError:
                raise ValueError(f"Unknown normalization method: {method}")
        
        # Create the appropriate normalizer
        if method == NormalizationMethod.Z_SCORE:
            return ZScoreNormalizer(**kwargs)
        elif method == NormalizationMethod.MIN_MAX:
            return MinMaxNormalizer(**kwargs)
        elif method == NormalizationMethod.ROBUST:
            return RobustNormalizer(**kwargs)
        elif method == NormalizationMethod.TANH:
            return TanhNormalizer(**kwargs)
        elif method == NormalizationMethod.ADAPTIVE:
            return AdaptiveNormalizer(**kwargs)
        elif method == NormalizationMethod.RELATIVE_STRENGTH:
            return RelativeStrengthNormalizer(**kwargs)
        elif method == NormalizationMethod.REGIME_AWARE:
            return RegimeAwareNormalizer(**kwargs)
        else:
            raise ValueError(f"Normalizer not implemented for method: {method}")

