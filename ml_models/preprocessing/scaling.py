#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
ML Models - Scaling Module

This module provides advanced scaling techniques for preprocessing financial data.
It includes adaptive scalers that can handle streaming data and financial-specific
transformations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

# Import module registry
from . import register_preprocessor

# Initialize logger
logger = logging.getLogger('quantumspectre.ml_models.preprocessing.scaling')

class AdaptiveWindowScaler:
    """
    Base class for adaptive window-based scalers that can update
    with streaming data without full retraining.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize the adaptive window scaler.
        
        Args:
            window_size: Number of recent observations to use for scaling parameters
        """
        self.window_size = window_size
        self.history = []
        self.scaler = None
        self._initialize_scaler()
    
    def _initialize_scaler(self):
        """Initialize the underlying scaler - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_scaler")
    
    def partial_fit(self, X):
        """
        Update the scaler with new data.
        
        Args:
            X: New data to incorporate into scaling parameters
        """
        # Add new data to history
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        for row in X:
            self.history.append(row)
        
        # Keep only the most recent window_size elements
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        # Retrain scaler on the window
        history_array = np.array(self.history)
        self.scaler.fit(history_array)
        
        return self
    
    def transform(self, X):
        """
        Transform the data using current scaling parameters.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        was_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        original_index = X.index if was_pandas else None
        original_columns = X.columns if isinstance(X, pd.DataFrame) else None
        
        if was_pandas:
            X = X.values
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        transformed = self.scaler.transform(X)
        
        # Convert back to original format if needed
        if was_pandas:
            if original_columns is not None:
                transformed = pd.DataFrame(transformed, index=original_index, columns=original_columns)
            else:
                transformed = pd.Series(transformed.ravel(), index=original_index)
        
        return transformed
    
    def fit_transform(self, X):
        """
        Fit the scaler to the data and transform it.
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Transformed data
        """
        self.partial_fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Reverse the transformation.
        
        Args:
            X: Transformed data
            
        Returns:
            Original scale data
        """
        was_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        original_index = X.index if was_pandas else None
        original_columns = X.columns if isinstance(X, pd.DataFrame) else None
        
        if was_pandas:
            X = X.values
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        original = self.scaler.inverse_transform(X)
        
        # Convert back to original format if needed
        if was_pandas:
            if original_columns is not None:
                original = pd.DataFrame(original, index=original_index, columns=original_columns)
            else:
                original = pd.Series(original.ravel(), index=original_index)
        
        return original

class AdaptiveMinMaxScaler(AdaptiveWindowScaler):
    """Adaptive Min-Max scaler with sliding window approach."""
    
    def __init__(self, window_size: int = 1000, feature_range: Tuple[float, float] = (0, 1)):
        """
        Initialize the adaptive min-max scaler.
        
        Args:
            window_size: Number of recent observations to use for scaling
            feature_range: Range to scale features to
        """
        self.feature_range = feature_range
        super().__init__(window_size)
    
    def _initialize_scaler(self):
        """Initialize the min-max scaler."""
        self.scaler = MinMaxScaler(feature_range=self.feature_range)

class AdaptiveStandardScaler(AdaptiveWindowScaler):
    """Adaptive Standard scaler with sliding window approach."""
    
    def __init__(self, window_size: int = 1000, with_mean: bool = True, with_std: bool = True):
        """
        Initialize the adaptive standard scaler.
        
        Args:
            window_size: Number of recent observations to use for scaling
            with_mean: Whether to center the data
            with_std: Whether to scale the data to unit variance
        """
        self.with_mean = with_mean
        self.with_std = with_std
        super().__init__(window_size)
    
    def _initialize_scaler(self):
        """Initialize the standard scaler."""
        self.scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)

class AdaptiveRobustScaler(AdaptiveWindowScaler):
    """Adaptive Robust scaler with sliding window approach."""
    
    def __init__(self, window_size: int = 1000, quantile_range: Tuple[float, float] = (25.0, 75.0), 
                 with_centering: bool = True, with_scaling: bool = True):
        """
        Initialize the adaptive robust scaler.
        
        Args:
            window_size: Number of recent observations to use for scaling
            quantile_range: Quantile range for computing IQR
            with_centering: Whether to center the data
            with_scaling: Whether to scale the data to IQR
        """
        self.quantile_range = quantile_range
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        super().__init__(window_size)
    
    def _initialize_scaler(self):
        """Initialize the robust scaler."""
        self.scaler = RobustScaler(quantile_range=self.quantile_range, 
                                  with_centering=self.with_centering,
                                  with_scaling=self.with_scaling)

class LogScaler:
    """
    Scaler that applies logarithmic transformation to data.
    Useful for financial data with exponential properties like volume.
    """
    
    def __init__(self, base: float = 10, add_constant: float = 1.0):
        """
        Initialize the log scaler.
        
        Args:
            base: Logarithm base (default: 10)
            add_constant: Constant to add before taking log to handle zeros (default: 1.0)
        """
        self.base = base
        self.add_constant = add_constant
    
    def fit(self, X, y=None):
        """
        Fit method (no-op for log transformation).
        
        Args:
            X: Input data
            y: Target data (unused)
            
        Returns:
            self
        """
        return self
    
    def transform(self, X):
        """
        Apply logarithmic transformation.
        
        Args:
            X: Data to transform
            
        Returns:
            Log-transformed data
        """
        was_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        original_index = X.index if was_pandas else None
        original_columns = X.columns if isinstance(X, pd.DataFrame) else None
        
        if was_pandas:
            X = X.values
            
        X_transformed = np.log(X + self.add_constant) / np.log(self.base)
        
        # Convert back to original format
        if was_pandas:
            if original_columns is not None:
                X_transformed = pd.DataFrame(X_transformed, index=original_index, columns=original_columns)
            else:
                X_transformed = pd.Series(X_transformed.ravel(), index=original_index)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform (equivalent to just transform for log scaler).
        
        Args:
            X: Data to transform
            y: Target data (unused)
            
        Returns:
            Log-transformed data
        """
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Reverse the logarithmic transformation.
        
        Args:
            X: Transformed data
            
        Returns:
            Original scale data
        """
        was_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        original_index = X.index if was_pandas else None
        original_columns = X.columns if isinstance(X, pd.DataFrame) else None
        
        if was_pandas:
            X = X.values
            
        X_original = (self.base ** X) - self.add_constant
        
        # Convert back to original format
        if was_pandas:
            if original_columns is not None:
                X_original = pd.DataFrame(X_original, index=original_index, columns=original_columns)
            else:
                X_original = pd.Series(X_original.ravel(), index=original_index)
        
        return X_original

class FinancialDifferenceScaler:
    """
    Scaler that transforms financial time series by taking differences or returns.
    Useful for making time series stationary.
    """
    
    def __init__(self, method: str = 'returns', periods: int = 1):
        """
        Initialize the financial difference scaler.
        
        Args:
            method: Transformation method ('returns', 'log_returns', 'diff')
            periods: Number of periods to use in the transformation
        """
        if method not in ['returns', 'log_returns', 'diff']:
            raise ValueError(f"Unknown method '{method}'. Use 'returns', 'log_returns', or 'diff'.")
        
        self.method = method
        self.periods = periods
        self.last_values = None
    
    def fit(self, X, y=None):
        """
        Fit method (stores the last values for inverse transform).
        
        Args:
            X: Input data
            y: Target data (unused)
            
        Returns:
            self
        """
        # For inverse transform, we need to store the original values
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            self.last_values = X.iloc[:self.periods].copy()
        else:
            self.last_values = np.copy(X[:self.periods])
        
        return self
    
    def transform(self, X):
        """
        Apply financial transformation.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            if self.method == 'returns':
                transformed = X.pct_change(periods=self.periods)
            elif self.method == 'log_returns':
                transformed = np.log(X / X.shift(self.periods))
            else:  # diff
                transformed = X.diff(periods=self.periods)
                
            # First n values will be NaN, fill with zeros
            transformed.iloc[:self.periods] = 0
        else:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
                
            if self.method == 'returns':
                transformed = np.zeros_like(X)
                transformed[self.periods:] = (X[self.periods:] - X[:-self.periods]) / X[:-self.periods]
            elif self.method == 'log_returns':
                transformed = np.zeros_like(X)
                transformed[self.periods:] = np.log(X[self.periods:] / X[:-self.periods])
            else:  # diff
                transformed = np.zeros_like(X)
                transformed[self.periods:] = X[self.periods:] - X[:-self.periods]
        
        return transformed
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform the data.
        
        Args:
            X: Data to transform
            y: Target data (unused)
            
        Returns:
            Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_diff):
        """
        Reverse the transformation to get original values.
        This requires the initial values used during fit.
        
        Args:
            X_diff: Transformed differences/returns
            
        Returns:
            Reconstructed original time series
        """
        if self.last_values is None:
            raise ValueError("Inverse transform requires the scaler to be fitted first.")
        
        is_pandas = isinstance(X_diff, pd.DataFrame) or isinstance(X_diff, pd.Series)
        
        # Create a copy of the transformed data to avoid modifying the original
        if is_pandas:
            X_reconstructed = X_diff.copy()
            
            # Start with the initial values
            for i in range(self.periods):
                X_reconstructed.iloc[i] = self.last_values.iloc[i]
            
            # Reconstruct the time series
            for i in range(self.periods, len(X_reconstructed)):
                if self.method == 'returns':
                    X_reconstructed.iloc[i] = X_reconstructed.iloc[i-self.periods] * (1 + X_diff.iloc[i])
                elif self.method == 'log_returns':
                    X_reconstructed.iloc[i] = X_reconstructed.iloc[i-self.periods] * np.exp(X_diff.iloc[i])
                else:  # diff
                    X_reconstructed.iloc[i] = X_reconstructed.iloc[i-self.periods] + X_diff.iloc[i]
        else:
            X_reconstructed = np.zeros_like(X_diff)
            
            # Start with the initial values
            X_reconstructed[:self.periods] = self.last_values
            
            # Reconstruct the time series
            for i in range(self.periods, len(X_reconstructed)):
                if self.method == 'returns':
                    X_reconstructed[i] = X_reconstructed[i-self.periods] * (1 + X_diff[i])
                elif self.method == 'log_returns':
                    X_reconstructed[i] = X_reconstructed[i-self.periods] * np.exp(X_diff[i])
                else:  # diff
                    X_reconstructed[i] = X_reconstructed[i-self.periods] + X_diff[i]
        
        return X_reconstructed

class ZScoreOutlierScaler:
    """
    Scaler that handles outliers by capping values at specified Z-score.
    Useful for financial data with extreme values.
    """
    
    def __init__(self, threshold: float = 3.0, window_size: int = 1000):
        """
        Initialize the Z-score outlier scaler.
        
        Args:
            threshold: Z-score threshold for capping (default: 3.0)
            window_size: Window size for calculating mean and std
        """
        self.threshold = threshold
        self.window_size = window_size
        self.mean = None
        self.std = None
        self.history = []
    
    def partial_fit(self, X):
        """
        Update the scaler parameters with new data.
        
        Args:
            X: New data
            
        Returns:
            self
        """
        # Add new data to history
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        for row in X:
            self.history.append(row)
        
        # Keep only the most recent window_size elements
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        # Update mean and std
        history_array = np.array(self.history)
        self.mean = np.mean(history_array, axis=0)
        self.std = np.std(history_array, axis=0)
        
        return self
    
    def fit(self, X, y=None):
        """
        Fit the scaler to the data.
        
        Args:
            X: Input data
            y: Target data (unused)
            
        Returns:
            self
        """
        return self.partial_fit(X)
    
    def transform(self, X):
        """
        Apply Z-score capping to the data.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data with outliers capped
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform")
        
        was_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        original_index = X.index if was_pandas else None
        original_columns = X.columns if isinstance(X, pd.DataFrame) else None
        
        if was_pandas:
            X = X.values
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Calculate Z-scores
        z_scores = (X - self.mean) / self.std
        
        # Cap outliers
        X_capped = np.copy(X)
        mask_high = z_scores > self.threshold
        mask_low = z_scores < -self.threshold
        
        X_capped[mask_high] = self.mean + self.threshold * self.std
        X_capped[mask_low] = self.mean - self.threshold * self.std
        
        # Convert back to original format
        if was_pandas:
            if original_columns is not None:
                X_capped = pd.DataFrame(X_capped, index=original_index, columns=original_columns)
            else:
                X_capped = pd.Series(X_capped.ravel(), index=original_index)
        
        return X_capped
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform the data.
        
        Args:
            X: Data to transform
            y: Target data (unused)
            
        Returns:
            Transformed data with outliers capped
        """
        self.fit(X)
        return self.transform(X)

class PriceRelativeScaler:
    """
    Scaler that normalizes price data relative to a reference price.
    Useful for comparing multiple assets or creating relative strength indicators.
    """
    
    def __init__(self, reference_type: str = 'first', window_size: Optional[int] = None):
        """
        Initialize the price relative scaler.
        
        Args:
            reference_type: Type of reference price ('first', 'last', 'mean', 'min', 'max')
            window_size: Optional window size for rolling references
        """
        valid_types = ['first', 'last', 'mean', 'min', 'max']
        if reference_type not in valid_types:
            raise ValueError(f"Unknown reference_type '{reference_type}'. "
                            f"Use one of {valid_types}")
        
        self.reference_type = reference_type
        self.window_size = window_size
        self.reference_values = None
    
    def fit(self, X, y=None):
        """
        Calculate reference values from the data.
        
        Args:
            X: Input price data
            y: Target data (unused)
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            if self.window_size is None:
                if self.reference_type == 'first':
                    self.reference_values = X.iloc[0].values
                elif self.reference_type == 'last':
                    self.reference_values = X.iloc[-1].values
                elif self.reference_type == 'mean':
                    self.reference_values = X.mean().values
                elif self.reference_type == 'min':
                    self.reference_values = X.min().values
                elif self.reference_type == 'max':
                    self.reference_values = X.max().values
            else:
                # With window_size, reference values will be calculated dynamically during transform
                pass
        elif isinstance(X, pd.Series):
            if self.window_size is None:
                if self.reference_type == 'first':
                    self.reference_values = X.iloc[0]
                elif self.reference_type == 'last':
                    self.reference_values = X.iloc[-1]
                elif self.reference_type == 'mean':
                    self.reference_values = X.mean()
                elif self.reference_type == 'min':
                    self.reference_values = X.min()
                elif self.reference_type == 'max':
                    self.reference_values = X.max()
            else:
                # With window_size, reference values will be calculated dynamically
                pass
        else:  # numpy array
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
                
            if self.window_size is None:
                if self.reference_type == 'first':
                    self.reference_values = X[0, :]
                elif self.reference_type == 'last':
                    self.reference_values = X[-1, :]
                elif self.reference_type == 'mean':
                    self.reference_values = np.mean(X, axis=0)
                elif self.reference_type == 'min':
                    self.reference_values = np.min(X, axis=0)
                elif self.reference_type == 'max':
                    self.reference_values = np.max(X, axis=0)
            else:
                # With window_size, reference values will be calculated dynamically
                pass
        
        return self
    
    def transform(self, X):
        """
        Transform prices to relative values.
        
        Args:
            X: Price data to transform
            
        Returns:
            Relative price data
        """
        if self.window_size is None and self.reference_values is None:
            raise ValueError("Scaler must be fitted before transform when window_size is None")
        
        if isinstance(X, pd.DataFrame):
            if self.window_size is None:
                # Use global reference values
                relative_prices = X.divide(self.reference_values, axis=1)
            else:
                # Calculate rolling reference values
                if self.reference_type == 'mean':
                    reference = X.rolling(window=self.window_size).mean().shift(1)
                elif self.reference_type == 'min':
                    reference = X.rolling(window=self.window_size).min().shift(1)
                elif self.reference_type == 'max':
                    reference = X.rolling(window=self.window_size).max().shift(1)
                elif self.reference_type == 'first':
                    def first(x): return x.iloc[0] if len(x) > 0 else np.nan
                    reference = X.rolling(window=self.window_size).apply(first).shift(1)
                else:  # 'last'
                    def last(x): return x.iloc[-1] if len(x) > 0 else np.nan
                    reference = X.rolling(window=self.window_size).apply(last).shift(1)
                
                # Divide by reference values
                relative_prices = X.divide(reference)
                
                # Fill NaNs in the beginning with 1.0
                relative_prices.fillna(1.0, inplace=True)
                
        elif isinstance(X, pd.Series):
            if self.window_size is None:
                # Use global reference value
                relative_prices = X / self.reference_values
            else:
                # Calculate rolling reference values
                if self.reference_type == 'mean':
                    reference = X.rolling(window=self.window_size).mean().shift(1)
                elif self.reference_type == 'min':
                    reference = X.rolling(window=self.window_size).min().shift(1)
                elif self.reference_type == 'max':
                    reference = X.rolling(window=self.window_size).max().shift(1)
                elif self.reference_type == 'first':
                    def first(x): return x.iloc[0] if len(x) > 0 else np.nan
                    reference = X.rolling(window=self.window_size).apply(first).shift(1)
                else:  # 'last'
                    def last(x): return x.iloc[-1] if len(x) > 0 else np.nan
                    reference = X.rolling(window=self.window_size).apply(last).shift(1)
                
                # Divide by reference values
                relative_prices = X / reference
                
                # Fill NaNs in the beginning with 1.0
                relative_prices.fillna(1.0, inplace=True)
        else:  # numpy array
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
                
            if self.window_size is None:
                # Use global reference values
                relative_prices = X / self.reference_values
            else:
                # Calculate rolling references - more complex with numpy arrays
                n_samples, n_features = X.shape
                relative_prices = np.ones_like(X)
                
                for i in range(self.window_size, n_samples):
                    window = X[i-self.window_size:i, :]
                    
                    if self.reference_type == 'mean':
                        reference = np.mean(window, axis=0)
                    elif self.reference_type == 'min':
                        reference = np.min(window, axis=0)
                    elif self.reference_type == 'max':
                        reference = np.max(window, axis=0)
                    elif self.reference_type == 'first':
                        reference = window[0, :]
                    else:  # 'last'
                        reference = window[-1, :]
                    
                    relative_prices[i, :] = X[i, :] / reference
        
        return relative_prices
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform the data.
        
        Args:
            X: Price data to transform
            y: Target data (unused)
            
        Returns:
            Relative price data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_relative):
        """
        Convert relative prices back to original prices.
        Only works for global references (window_size=None).
        
        Args:
            X_relative: Relative price data
            
        Returns:
            Original price data
        """
        if self.window_size is not None:
            raise ValueError("Inverse transform not supported with rolling window references")
            
        if self.reference_values is None:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        if isinstance(X_relative, pd.DataFrame):
            original_prices = X_relative.multiply(self.reference_values, axis=1)
        elif isinstance(X_relative, pd.Series):
            original_prices = X_relative * self.reference_values
        else:  # numpy array
            if len(X_relative.shape) == 1:
                X_relative = X_relative.reshape(-1, 1)
            original_prices = X_relative * self.reference_values
        
        return original_prices

class CompositeScaler:
    """
    A composite scaler that applies different scalers to different feature groups.
    Useful for mixed financial data where different scaling is appropriate
    for different types of data.
    """
    
    def __init__(self, feature_groups: Dict[str, List[str]], scalers: Dict[str, Any]):
        """
        Initialize the composite scaler.
        
        Args:
            feature_groups: Dictionary mapping group names to lists of feature names
            scalers: Dictionary mapping group names to scaler instances
        """
        self.feature_groups = feature_groups
        self.scalers = scalers
        self.all_features = []
        
        # Collect all feature names
        for group, features in feature_groups.items():
            if group not in scalers:
                logger.warning(f"No scaler specified for feature group '{group}'")
            self.all_features.extend(features)
    
    def fit(self, X, y=None):
        """
        Fit each scaler to its corresponding feature group.
        
        Args:
            X: Input data (DataFrame)
            y: Target data (unused)
            
        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CompositeScaler requires DataFrame input")
            
        # Verify all features exist in the DataFrame
        missing_features = set(self.all_features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features {missing_features} not found in input DataFrame")
        
        # Fit each scaler to its group of features
        for group, features in self.feature_groups.items():
            if group in self.scalers:
                self.scalers[group].fit(X[features])
        
        return self
    
    def transform(self, X):
        """
        Transform each feature group with its corresponding scaler.
        
        Args:
            X: Input data (DataFrame)
            
        Returns:
            Transformed DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CompositeScaler requires DataFrame input")
            
        # Create a copy of the input DataFrame
        X_transformed = X.copy()
        
        # Transform each feature group
        for group, features in self.feature_groups.items():
            if group in self.scalers:
                X_transformed[features] = self.scalers[group].transform(X[features])
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform the data.
        
        Args:
            X: Input data (DataFrame)
            y: Target data (unused)
            
        Returns:
            Transformed DataFrame
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Inverse transform each feature group.
        
        Args:
            X: Transformed data (DataFrame)
            
        Returns:
            Original scale DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("CompositeScaler requires DataFrame input")
            
        # Create a copy of the input DataFrame
        X_original = X.copy()
        
        # Inverse transform each feature group
        for group, features in self.feature_groups.items():
            if group in self.scalers:
                try:
                    X_original[features] = self.scalers[group].inverse_transform(X[features])
                except AttributeError:
                    logger.warning(f"Scaler for group '{group}' does not support inverse_transform")
        
        return X_original

# Factory function to create scalers based on config
def create_scaler(config: Dict[str, Any]) -> Any:
    """
    Create a scaler instance based on configuration.
    
    Args:
        config: Configuration dictionary with method and parameters
        
    Returns:
        Configured scaler instance
        
    Raises:
        ValueError: If method is unknown
    """
    method = config.get('method', 'min_max')
    
    if method == 'min_max':
        feature_range = config.get('feature_range', (0, 1))
        window = config.get('window', None)
        
        if window is not None:
            return AdaptiveMinMaxScaler(window_size=window, feature_range=feature_range)
        else:
            return MinMaxScaler(feature_range=feature_range)
            
    elif method == 'standard':
        with_mean = config.get('with_mean', True)
        with_std = config.get('with_std', True)
        window = config.get('window', None)
        
        if window is not None:
            return AdaptiveStandardScaler(window_size=window, with_mean=with_mean, with_std=with_std)
        else:
            return StandardScaler(with_mean=with_mean, with_std=with_std)
            
    elif method == 'robust':
        quantile_range = config.get('quantile_range', (25, 75))
        with_centering = config.get('with_centering', True)
        with_scaling = config.get('with_scaling', True)
        window = config.get('window', None)
        
        if window is not None:
            return AdaptiveRobustScaler(window_size=window, quantile_range=quantile_range,
                                      with_centering=with_centering, with_scaling=with_scaling)
        else:
            return RobustScaler(quantile_range=quantile_range,
                              with_centering=with_centering, with_scaling=with_scaling)
            
    elif method == 'quantile':
        n_quantiles = config.get('n_quantiles', 1000)
        output_distribution = config.get('output_distribution', 'uniform')
        
        return QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)
        
    elif method == 'power':
        method_pow = config.get('method_pow', 'yeo-johnson')
        standardize = config.get('standardize', True)
        
        return PowerTransformer(method=method_pow, standardize=standardize)
        
    elif method == 'log':
        base = config.get('base', 10)
        add_constant = config.get('add_constant', 1)
        
        return LogScaler(base=base, add_constant=add_constant)
        
    elif method == 'diff':
        diff_method = config.get('diff_method', 'returns')
        periods = config.get('periods', 1)
        
        return FinancialDifferenceScaler(method=diff_method, periods=periods)
        
    elif method == 'z_score_cap':
        threshold = config.get('threshold', 3.0)
        window = config.get('window', 1000)
        
        return ZScoreOutlierScaler(threshold=threshold, window_size=window)
        
    elif method == 'relative':
        reference_type = config.get('reference_type', 'first')
        window = config.get('window', None)
        
        return PriceRelativeScaler(reference_type=reference_type, window_size=window)
        
    elif method == 'composite':
        feature_groups = config.get('feature_groups', {})
        scaler_configs = config.get('scaler_configs', {})
        
        scalers = {}
        for group, scaler_config in scaler_configs.items():
            scalers[group] = create_scaler(scaler_config)
            
        return CompositeScaler(feature_groups=feature_groups, scalers=scalers)
        
    else:
        raise ValueError(f"Unknown scaling method: {method}")

# Register all scalers with the preprocessing registry
register_preprocessor('min_max_scaler', AdaptiveMinMaxScaler)
register_preprocessor('standard_scaler', AdaptiveStandardScaler)
register_preprocessor('robust_scaler', AdaptiveRobustScaler)
register_preprocessor('log_scaler', LogScaler)
register_preprocessor('diff_scaler', FinancialDifferenceScaler)
register_preprocessor('z_score_cap_scaler', ZScoreOutlierScaler)
register_preprocessor('relative_scaler', PriceRelativeScaler)
register_preprocessor('composite_scaler', CompositeScaler)
register_preprocessor('create_scaler', create_scaler)


def get_scaler(name: str, **kwargs) -> Any:
    """Retrieve a scaler from the registry by name."""
    scaler_cls = PREPROCESSOR_REGISTRY.get(name)
    if scaler_cls is None:
        raise ValueError(f"Unknown scaler: {name}")
    if isinstance(scaler_cls, type):
        return scaler_cls(**kwargs)
    return scaler_cls(**kwargs)
