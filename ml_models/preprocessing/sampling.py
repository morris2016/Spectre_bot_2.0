#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
ML Models - Preprocessing - Sampling

This module provides advanced sampling techniques for machine learning, including
class balancing, time-based sampling, and specialized methods for financial time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from collections import Counter
from sklearn.utils import resample, shuffle
import warnings

from common.logger import get_logger
from common.utils import is_time_series, create_window_samples
from common.exceptions import SamplingError

logger = get_logger(__name__)


class SamplingManager:
    """
    Manages various sampling techniques for ML model training,
    with specialized methods for financial time series data.
    """
    
    def __init__(self):
        """Initialize the SamplingManager."""
        self._sampling_methods = {
            'random': self._random_sample,
            'stratified': self._stratified_sample,
            'undersampling': self._undersample,
            'oversampling': self._oversample,
            'smote': self._smote_sample,
            'adasyn': self._adasyn_sample,
            'time_series': self._time_series_sample,
            'walk_forward': self._walk_forward_sample,
            'purged': self._purged_sample,
            'weight_of_evidence': self._weight_of_evidence_sample,
            'importance': self._importance_sample,
            'bootstrap': self._bootstrap_sample,
            'temporal_bootstrap': self._temporal_bootstrap_sample,
            'price_change': self._price_change_sample
        }
        logger.info("SamplingManager initialized")
        
    def sample(self, df: pd.DataFrame, 
               method: str = 'random', 
               target_col: Optional[str] = None,
               **kwargs) -> pd.DataFrame:
        """
        Sample the data according to the specified method.
        
        Args:
            df: Input DataFrame to sample
            method: Sampling method to use
            target_col: Target column name for methods that require it
            **kwargs: Additional arguments specific to the sampling method
            
        Returns:
            Sampled DataFrame
        """
        if method not in self._sampling_methods:
            valid_methods = list(self._sampling_methods.keys())
            raise SamplingError(f"Unknown sampling method: {method}. Valid methods: {valid_methods}")
            
        # Methods that require target column
        target_required = ['stratified', 'undersampling', 'oversampling', 'smote', 'adasyn', 
                         'weight_of_evidence']
                         
        if method in target_required and target_col is None:
            raise SamplingError(f"Method '{method}' requires a target_col parameter")
            
        try:
            return self._sampling_methods[method](df, target_col, **kwargs)
        except Exception as e:
            logger.error(f"Error during {method} sampling: {str(e)}")
            raise SamplingError(f"Failed to apply {method} sampling") from e
            
    def create_train_test_split(self, 
                               df: pd.DataFrame, 
                               test_size: float = 0.2,
                               split_method: str = 'random',
                               target_col: Optional[str] = None,
                               **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split using the appropriate method for the data type.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data to use for testing
            split_method: Method to use for splitting
            target_col: Target column name (required for stratified split)
            **kwargs: Additional arguments specific to the split method
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if split_method == 'random':
            return self._random_split(df, test_size)
        elif split_method == 'stratified':
            if target_col is None:
                raise SamplingError("target_col is required for stratified split")
            return self._stratified_split(df, test_size, target_col)
        elif split_method == 'time_series':
            return self._time_series_split(df, test_size, **kwargs)
        elif split_method == 'purged':
            if target_col is None:
                raise SamplingError("target_col is required for purged split")
            return self._purged_split(df, test_size, target_col, **kwargs)
        elif split_method == 'trading_windows':
            return self._trading_windows_split(df, test_size, **kwargs)
        elif split_method == 'regime_based':
            regime_col = kwargs.get('regime_col')
            if regime_col is None:
                raise SamplingError("regime_col is required for regime_based split")
            return self._regime_based_split(df, test_size, regime_col, **kwargs)
        else:
            raise SamplingError(f"Unknown split method: {split_method}")
            
    def create_cross_validation(self,
                               df: pd.DataFrame,
                               n_splits: int = 5,
                               cv_method: str = 'time_series',
                               target_col: Optional[str] = None,
                               **kwargs) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create cross-validation folds using the appropriate method.
        
        Args:
            df: Input DataFrame
            n_splits: Number of CV folds
            cv_method: Method to use for CV
            target_col: Target column name (required for some methods)
            **kwargs: Additional arguments specific to the CV method
            
        Returns:
            List of (train_df, test_df) tuples for each fold
        """
        if cv_method == 'k_fold':
            return self._k_fold_cv(df, n_splits, **kwargs)
        elif cv_method == 'stratified_k_fold':
            if target_col is None:
                raise SamplingError("target_col is required for stratified_k_fold")
            return self._stratified_k_fold_cv(df, n_splits, target_col, **kwargs)
        elif cv_method == 'time_series':
            return self._time_series_cv(df, n_splits, **kwargs)
        elif cv_method == 'walk_forward':
            return self._walk_forward_cv(df, n_splits, **kwargs)
        elif cv_method == 'purged':
            return self._purged_cv(df, n_splits, **kwargs)
        elif cv_method == 'trading_windows':
            return self._trading_windows_cv(df, n_splits, **kwargs)
        elif cv_method == 'embargo':
            return self._embargo_cv(df, n_splits, **kwargs)
        elif cv_method == 'regime_based':
            regime_col = kwargs.get('regime_col')
            if regime_col is None:
                raise SamplingError("regime_col is required for regime_based CV")
            return self._regime_based_cv(df, n_splits, regime_col, **kwargs)
        else:
            raise SamplingError(f"Unknown CV method: {cv_method}")
            
    # --- Sampling methods ---
    
    def _random_sample(self, df: pd.DataFrame, target_col: Optional[str] = None, 
                     sample_size: Optional[int] = None, 
                     replace: bool = False, 
                     random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Random sampling from the DataFrame.
        
        Args:
            df: Input DataFrame
            target_col: Not used in this method
            sample_size: Size of the sample to draw (defaults to size of df)
            replace: Whether to sample with replacement
            random_state: Random seed for reproducibility
            
        Returns:
            Randomly sampled DataFrame
        """
        if sample_size is None:
            sample_size = len(df)
            
        if sample_size > len(df) and not replace:
            logger.warning(f"Sample size {sample_size} > dataframe size {len(df)}. Setting replace=True.")
            replace = True
            
        return df.sample(n=sample_size, replace=replace, random_state=random_state)
        
    def _stratified_sample(self, df: pd.DataFrame, target_col: str, 
                         sample_size: Optional[int] = None,
                         replace: bool = False,
                         random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Stratified sampling to maintain class distribution.
        
        Args:
            df: Input DataFrame
            target_col: Target column for stratification
            sample_size: Size of the sample to draw
            replace: Whether to sample with replacement
            random_state: Random seed for reproducibility
            
        Returns:
            Stratified sampled DataFrame
        """
        if target_col not in df.columns:
            raise SamplingError(f"Target column '{target_col}' not found in DataFrame")
            
        target_counts = df[target_col].value_counts(normalize=True)
        
        if sample_size is None:
            sample_size = len(df)
            
        # Calculate samples per class
        class_samples = {
            cls: int(np.ceil(prop * sample_size))
            for cls, prop in target_counts.items()
        }
        
        # Adjust to ensure we get exactly sample_size samples
        total_adjusted = sum(class_samples.values())
        if total_adjusted > sample_size:
            # Remove from largest classes
            for cls in sorted(class_samples, key=class_samples.get, reverse=True):
                if total_adjusted <= sample_size:
                    break
                class_samples[cls] -= 1
                total_adjusted -= 1
                
        # Sample from each class
        sampled_dfs = []
        for cls, size in class_samples.items():
            cls_df = df[df[target_col] == cls]
            
            # If replacement=False but we need more samples than available
            if size > len(cls_df) and not replace:
                logger.warning(f"Class {cls}: sample size {size} > available rows {len(cls_df)}. Using all rows.")
                sampled_dfs.append(cls_df)
            else:
                sampled_dfs.append(
                    cls_df.sample(n=size, replace=replace, random_state=random_state)
                )
                
        # Combine and shuffle
        result = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state)
        return result
        
    def _undersample(self, df: pd.DataFrame, target_col: str,
                   strategy: str = 'majority',
                   sampling_strategy: Optional[Dict] = None,
                   random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Undersampling to balance classes.
        
        Args:
            df: Input DataFrame
            target_col: Target column for classes
            strategy: Undersampling strategy ('majority', 'not_minority', 'all')
            sampling_strategy: Dict with {class_label: desired_count}
            random_state: Random seed for reproducibility
            
        Returns:
            Undersampled DataFrame with balanced classes
        """
        if target_col not in df.columns:
            raise SamplingError(f"Target column '{target_col}' not found in DataFrame")
            
        # Get class counts
        class_counts = df[target_col].value_counts()
        min_class_count = class_counts.min()
        
        # Determine sample sizes for each class
        if sampling_strategy is not None:
            class_samples = sampling_strategy
        else:
            if strategy == 'majority':
                # Only undersample the majority class
                majority_class = class_counts.idxmax()
                class_samples = {cls: count if cls != majority_class else min_class_count 
                              for cls, count in class_counts.items()}
            elif strategy == 'not_minority':
                # Undersample all but the minority class
                minority_class = class_counts.idxmin()
                class_samples = {cls: min_class_count if cls != minority_class else count 
                              for cls, count in class_counts.items()}
            elif strategy == 'all':
                # Undersample all classes to the minimum count
                class_samples = {cls: min_class_count for cls in class_counts.index}
            else:
                raise SamplingError(f"Unknown undersampling strategy: {strategy}")
                
        # Sample from each class
        sampled_dfs = []
        for cls, size in class_samples.items():
            cls_df = df[df[target_col] == cls]
            
            if size >= len(cls_df):
                sampled_dfs.append(cls_df)
            else:
                sampled_dfs.append(
                    cls_df.sample(n=size, random_state=random_state)
                )
                
        # Combine and shuffle
        result = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state)
        logger.info(f"Undersampled from {len(df)} to {len(result)} rows")
        return result
        
    def _oversample(self, df: pd.DataFrame, target_col: str,
                  strategy: str = 'minority',
                  sampling_strategy: Optional[Dict] = None,
                  random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Oversampling to balance classes.
        
        Args:
            df: Input DataFrame
            target_col: Target column for classes
            strategy: Oversampling strategy ('minority', 'not_majority', 'all')
            sampling_strategy: Dict with {class_label: desired_count}
            random_state: Random seed for reproducibility
            
        Returns:
            Oversampled DataFrame with balanced classes
        """
        if target_col not in df.columns:
            raise SamplingError(f"Target column '{target_col}' not found in DataFrame")
            
        # Get class counts
        class_counts = df[target_col].value_counts()
        max_class_count = class_counts.max()
        
        # Determine sample sizes for each class
        if sampling_strategy is not None:
            class_samples = sampling_strategy
        else:
            if strategy == 'minority':
                # Only oversample the minority class
                minority_class = class_counts.idxmin()
                class_samples = {cls: count if cls != minority_class else max_class_count 
                              for cls, count in class_counts.items()}
            elif strategy == 'not_majority':
                # Oversample all but the majority class
                majority_class = class_counts.idxmax()
                class_samples = {cls: max_class_count if cls != majority_class else count 
                              for cls, count in class_counts.items()}
            elif strategy == 'all':
                # Oversample all classes to the maximum count
                class_samples = {cls: max_class_count for cls in class_counts.index}
            else:
                raise SamplingError(f"Unknown oversampling strategy: {strategy}")
                
        # Sample from each class
        sampled_dfs = []
        for cls, size in class_samples.items():
            cls_df = df[df[target_col] == cls]
            
            if size <= len(cls_df):
                sampled_dfs.append(cls_df)
            else:
                # Oversample with replacement
                sampled_dfs.append(
                    resample(cls_df, n_samples=size, replace=True, random_state=random_state)
                )
                
        # Combine and shuffle
        result = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state)
        logger.info(f"Oversampled from {len(df)} to {len(result)} rows")
        return result
        
    def _smote_sample(self, df: pd.DataFrame, target_col: str,
                    k_neighbors: int = 5,
                    sampling_strategy: Union[str, Dict] = 'auto',
                    random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Synthetic Minority Over-sampling Technique (SMOTE).
        
        Args:
            df: Input DataFrame
            target_col: Target column for classes
            k_neighbors: Number of nearest neighbors to use
            sampling_strategy: Strategy for sampling
            random_state: Random seed for reproducibility
            
        Returns:
            Resampled DataFrame using SMOTE
        """
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise SamplingError("SMOTE requires the imbalanced-learn package. Install with: pip install imbalanced-learn")
            
        if target_col not in df.columns:
            raise SamplingError(f"Target column '{target_col}' not found in DataFrame")
            
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Apply SMOTE
        smote = SMOTE(sampling_strategy=sampling_strategy, 
                      k_neighbors=k_neighbors, 
                      random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Convert back to DataFrame
        result = pd.DataFrame(X_resampled, columns=X.columns)
        result[target_col] = y_resampled
        
        logger.info(f"SMOTE resampled from {len(df)} to {len(result)} rows")
        return result
        
    def _adasyn_sample(self, df: pd.DataFrame, target_col: str,
                     n_neighbors: int = 5,
                     sampling_strategy: Union[str, Dict] = 'auto',
                     random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Adaptive Synthetic Sampling Approach (ADASYN).
        
        Args:
            df: Input DataFrame
            target_col: Target column for classes
            n_neighbors: Number of nearest neighbors to use
            sampling_strategy: Strategy for sampling
            random_state: Random seed for reproducibility
            
        Returns:
            Resampled DataFrame using ADASYN
        """
        try:
            from imblearn.over_sampling import ADASYN
        except ImportError:
            raise SamplingError("ADASYN requires the imbalanced-learn package. Install with: pip install imbalanced-learn")
            
        if target_col not in df.columns:
            raise SamplingError(f"Target column '{target_col}' not found in DataFrame")
            
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Apply ADASYN
        adasyn = ADASYN(sampling_strategy=sampling_strategy, 
                       n_neighbors=n_neighbors, 
                       random_state=random_state)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        # Convert back to DataFrame
        result = pd.DataFrame(X_resampled, columns=X.columns)
        result[target_col] = y_resampled
        
        logger.info(f"ADASYN resampled from {len(df)} to {len(result)} rows")
        return result
        
    def _time_series_sample(self, df: pd.DataFrame, target_col: Optional[str] = None,
                          window_size: int = 20,
                          step_size: int = 1,
                          min_samples: int = 100,
                          max_samples: Optional[int] = None,
                          timestamp_col: Optional[str] = None,
                          **kwargs) -> pd.DataFrame:
        """
        Time series sampling with sliding windows.
        
        Args:
            df: Input DataFrame
            target_col: Not directly used, but retained for API consistency
            window_size: Number of consecutive timestamps to include in each sample
            step_size: Number of steps between consecutive windows
            min_samples: Minimum number of samples to return
            max_samples: Maximum number of samples to return
            timestamp_col: Column to use for time ordering (uses index if None)
            
        Returns:
            Sampled time series data
        """
        result_dfs = []
        
        # Sort by timestamp if specified
        if timestamp_col is not None and timestamp_col in df.columns:
            sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            # Assume index is already time-ordered
            sorted_df = df.reset_index(drop=True)
            
        # Create sliding windows
        for start_idx in range(0, len(sorted_df) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_df = sorted_df.iloc[start_idx:end_idx].copy()
            result_dfs.append(window_df)
            
            if max_samples is not None and len(result_dfs) >= max_samples:
                break
                
        # If we don't have enough samples, reduce step size
        if len(result_dfs) < min_samples:
            logger.warning(f"Not enough samples with window_size={window_size}, step_size={step_size}. Reducing step size.")
            # Recursively call with smaller step size
            return self._time_series_sample(
                df=df, target_col=target_col,
                window_size=window_size,
                step_size=max(1, step_size // 2),
                min_samples=min_samples,
                max_samples=max_samples,
                timestamp_col=timestamp_col,
                **kwargs
            )
            
        # Combine all windows
        result = pd.concat(result_dfs)
        logger.info(f"Time series sampling created {len(result_dfs)} windows with {len(result)} total rows")
        return result
        
    def _walk_forward_sample(self, df: pd.DataFrame, target_col: Optional[str] = None,
                           train_window: int = 252,  # Approximately 1 year of trading days
                           test_window: int = 20,    # Approximately 1 month of trading days
                           step_size: int = 20,      # Step forward by 1 month
                           min_samples: int = 3,     # Minimum number of train/test pairs
                           timestamp_col: Optional[str] = None,
                           **kwargs) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Walk-forward sampling for time series cross-validation.
        
        Args:
            df: Input DataFrame
            target_col: Not directly used, but retained for API consistency
            train_window: Number of periods in training window
            test_window: Number of periods in testing window
            step_size: Number of periods to step forward between samples
            min_samples: Minimum number of train/test pairs to return
            timestamp_col: Column to use for time ordering (uses index if None)
            
        Returns:
            List of (train_df, test_df) pairs
        """
        # Sort by timestamp if specified
        if timestamp_col is not None and timestamp_col in df.columns:
            sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            # Assume index is already time-ordered
            sorted_df = df.reset_index(drop=True)
            
        # Ensure we have enough data
        if len(sorted_df) < train_window + test_window:
            raise SamplingError(f"Not enough data for walk-forward sampling. Need at least {train_window + test_window} rows")
            
        # Create walk-forward windows
        result_pairs = []
        
        for start_idx in range(0, len(sorted_df) - train_window - test_window + 1, step_size):
            train_end_idx = start_idx + train_window
            test_end_idx = train_end_idx + test_window
            
            train_df = sorted_df.iloc[start_idx:train_end_idx].copy()
            test_df = sorted_df.iloc[train_end_idx:test_end_idx].copy()
            
            result_pairs.append((train_df, test_df))
            
        if len(result_pairs) < min_samples:
            logger.warning(f"Not enough samples with current parameters. Reducing step size.")
            # Recursively call with smaller step size
            return self._walk_forward_sample(
                df=df, target_col=target_col,
                train_window=train_window,
                test_window=test_window,
                step_size=max(1, step_size // 2),
                min_samples=min_samples,
                timestamp_col=timestamp_col,
                **kwargs
            )
            
        logger.info(f"Walk-forward sampling created {len(result_pairs)} train/test pairs")
        return result_pairs
        
    def _purged_sample(self, df: pd.DataFrame, target_col: Optional[str] = None,
                     embargo_size: int = 5,
                     min_gap: int = 0,
                     timestamp_col: Optional[str] = None,
                     sample_frac: float = 0.7,
                     random_state: Optional[int] = None,
                     **kwargs) -> pd.DataFrame:
        """
        Purged sampling to avoid look-ahead bias in financial time series.
        
        Args:
            df: Input DataFrame
            target_col: Not directly used, but retained for API consistency
            embargo_size: Number of periods to embargo after each sampled point
            min_gap: Minimum gap between sampled points
            timestamp_col: Column to use for time ordering (uses index if None)
            sample_frac: Fraction of data to sample
            random_state: Random seed for reproducibility
            
        Returns:
            Purged sampled DataFrame
        """
        # Sort by timestamp if specified
        if timestamp_col is not None and timestamp_col in df.columns:
            sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            # Assume index is already time-ordered
            sorted_df = df.reset_index(drop=True)
            
        # Calculate number of points to sample
        n_samples = int(len(sorted_df) * sample_frac)
        
        # Initialize random number generator
        rng = np.random.RandomState(random_state)
        
        # Sample with purging
        valid_indices = set(range(len(sorted_df)))
        selected_indices = []
        
        while len(selected_indices) < n_samples and valid_indices:
            # Randomly select from remaining valid indices
            idx = rng.choice(list(valid_indices))
            selected_indices.append(idx)
            
            # Remove the selected index and embargo period from valid indices
            valid_indices.remove(idx)
            
            # Apply embargo - remove following indices
            for i in range(idx + 1, min(idx + embargo_size + 1, len(sorted_df))):
                if i in valid_indices:
                    valid_indices.remove(i)
                    
            # Apply minimum gap - remove surrounding indices
            for i in range(max(0, idx - min_gap), min(idx + min_gap + 1, len(sorted_df))):
                if i in valid_indices:
                    valid_indices.remove(i)
                    
        # Sort selected indices to maintain time order
        selected_indices.sort()
        
        # Return the selected rows
        result = sorted_df.iloc[selected_indices].copy()
        logger.info(f"Purged sampling selected {len(result)} rows from {len(df)} total")
        return result
        
    def _weight_of_evidence_sample(self, df: pd.DataFrame, target_col: str,
                                 weight_col: Optional[str] = None,
                                 positive_class: Any = 1,
                                 sample_size: Optional[int] = None,
                                 random_state: Optional[int] = None,
                                 **kwargs) -> pd.DataFrame:
        """
        Sample based on weight of evidence or custom weights.
        
        Args:
            df: Input DataFrame
            target_col: Target column for classification
            weight_col: Column containing weights (created if None)
            positive_class: Value representing the positive class
            sample_size: Size of sample to return
            random_state: Random seed for reproducibility
            
        Returns:
            Weighted sampled DataFrame
        """
        if target_col not in df.columns:
            raise SamplingError(f"Target column '{target_col}' not found in DataFrame")
            
        # Create weight column if not provided
        if weight_col is None:
            weight_col = '_sample_weight'
            
            # Calculate Information Value (IV) per row
            # For binary classification: w = |p(y|x) - p(y)|
            positive_prob = df[target_col].mean()  # Overall positive rate
            
            # Create weights based on difference from base rate
            df[weight_col] = np.abs(
                df[target_col].apply(lambda x: 1 if x == positive_class else 0) - positive_prob
            )
            
        elif weight_col not in df.columns:
            raise SamplingError(f"Weight column '{weight_col}' not found in DataFrame")
            
        # Normalize weights to probabilities
        weights = df[weight_col].values
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            logger.warning("All weights are zero. Using uniform weights.")
            weights = np.ones(len(df)) / len(df)
            
        # Determine sample size
        if sample_size is None:
            sample_size = len(df)
            
        # Sample based on weights
        rng = np.random.RandomState(random_state)
        sampled_indices = rng.choice(
            np.arange(len(df)), 
            size=sample_size, 
            replace=True, 
            p=weights
        )
        
        # Get result DataFrame
        result = df.iloc[sampled_indices].copy()
        if weight_col == '_sample_weight':
            result = result.drop(columns=[weight_col])
            
        logger.info(f"Weight-of-evidence sampling selected {len(result)} rows")
        return result
        
    def _importance_sample(self, df: pd.DataFrame, target_col: Optional[str] = None,
                         importance_fn: Callable[[pd.DataFrame], np.ndarray] = None,
                         importance_col: Optional[str] = None,
                         sample_size: Optional[int] = None,
                         replace: bool = True,
                         random_state: Optional[int] = None,
                         **kwargs) -> pd.DataFrame:
        """
        Importance sampling based on a custom importance function.
        
        Args:
            df: Input DataFrame
            target_col: Not directly used, but retained for API consistency
            importance_fn: Function that returns importance weights for each row
            importance_col: Column containing importance weights
            sample_size: Size of sample to return
            replace: Whether to sample with replacement
            random_state: Random seed for reproducibility
            
        Returns:
            Importance-sampled DataFrame
        """
        # Get importance weights
        if importance_col is not None and importance_col in df.columns:
            weights = df[importance_col].values
        elif importance_fn is not None:
            weights = importance_fn(df)
        else:
            raise SamplingError("Either importance_col or importance_fn must be provided")
            
        # Ensure weights are positive
        if np.any(weights < 0):
            logger.warning("Negative importance weights found. Taking absolute values.")
            weights = np.abs(weights)
            
        # Normalize weights to probabilities
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            logger.warning("All importance weights are zero. Using uniform weights.")
            weights = np.ones(len(df)) / len(df)
            
        # Determine sample size
        if sample_size is None:
            sample_size = len(df)
            
        # Sample based on weights
        rng = np.random.RandomState(random_state)
        sampled_indices = rng.choice(
            np.arange(len(df)), 
            size=sample_size, 
            replace=replace, 
            p=weights
        )
        
        # Get result DataFrame
        result = df.iloc[sampled_indices].copy()
        logger.info(f"Importance sampling selected {len(result)} rows")
        return result
        
    def _bootstrap_sample(self, df: pd.DataFrame, target_col: Optional[str] = None,
                        n_bootstrap: int = 100,
                        sample_frac: float = 1.0,
                        random_state: Optional[int] = None,
                        **kwargs) -> List[pd.DataFrame]:
        """
        Create bootstrap samples for uncertainty estimation.
        
        Args:
            df: Input DataFrame
            target_col: Not directly used, but retained for API consistency
            n_bootstrap: Number of bootstrap samples to create
            sample_frac: Fraction of data to sample in each bootstrap
            random_state: Random seed for reproducibility
            
        Returns:
            List of bootstrap DataFrames
        """
        bootstrap_samples = []
        sample_size = int(len(df) * sample_frac)
        
        for i in range(n_bootstrap):
            # Set seed for reproducibility, but different for each bootstrap
            if random_state is not None:
                seed = random_state + i
            else:
                seed = None
                
            # Sample with replacement
            bootstrap_df = df.sample(n=sample_size, replace=True, random_state=seed)
            bootstrap_samples.append(bootstrap_df)
            
        logger.info(f"Created {n_bootstrap} bootstrap samples with {sample_size} rows each")
        return bootstrap_samples
        
    def _temporal_bootstrap_sample(self, df: pd.DataFrame, target_col: Optional[str] = None,
                                 block_size: int = 20,
                                 n_bootstrap: int = 100,
                                 sample_frac: float = 1.0,
                                 timestamp_col: Optional[str] = None,
                                 random_state: Optional[int] = None,
                                 **kwargs) -> List[pd.DataFrame]:
        """
        Create temporal block bootstrap samples for time series data.
        
        Args:
            df: Input DataFrame
            target_col: Not directly used, but retained for API consistency
            block_size: Size of contiguous blocks to sample
            n_bootstrap: Number of bootstrap samples to create
            sample_frac: Fraction of data to sample in each bootstrap
            timestamp_col: Column to use for time ordering
            random_state: Random seed for reproducibility
            
        Returns:
            List of temporal bootstrap DataFrames
        """
        # Sort by timestamp if specified
        if timestamp_col is not None and timestamp_col in df.columns:
            sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            # Assume index is already time-ordered
            sorted_df = df.reset_index(drop=True)
            
        # Calculate number of blocks needed
        n_blocks = int(np.ceil(len(sorted_df) * sample_frac / block_size))
        
        # Get all possible block starting indices
        max_start_idx = len(sorted_df) - block_size + 1
        if max_start_idx <= 0:
            raise SamplingError(f"Block size {block_size} is larger than dataframe ({len(sorted_df)} rows)")
            
        rng = np.random.RandomState(random_state)
        
        bootstrap_samples = []
        for i in range(n_bootstrap):
            # Select random blocks
            block_starts = rng.choice(max_start_idx, size=n_blocks, replace=True)
            
            # Collect all blocks
            bootstrap_blocks = []
            for start_idx in block_starts:
                end_idx = start_idx + block_size
                block = sorted_df.iloc[start_idx:end_idx].copy()
                bootstrap_blocks.append(block)
                
            # Combine blocks
            bootstrap_df = pd.concat(bootstrap_blocks)
            bootstrap_samples.append(bootstrap_df)
            
        logger.info(f"Created {n_bootstrap} temporal bootstrap samples with {n_blocks} blocks each")
        return bootstrap_samples
        
    def _price_change_sample(self, df: pd.DataFrame, target_col: Optional[str] = None,
                           price_col: str = 'close',
                           threshold: float = 0.01,  # 1% change
                           direction: str = 'both',
                           timestamp_col: Optional[str] = None,
                           **kwargs) -> pd.DataFrame:
        """
        Sample based on significant price changes for event-driven analysis.
        
        Args:
            df: Input DataFrame
            target_col: Not directly used, but retained for API consistency
            price_col: Column containing price data
            threshold: Minimum price change to consider significant
            direction: Direction of price change ('up', 'down', or 'both')
            timestamp_col: Column to use for time ordering
            
        Returns:
            DataFrame with samples around significant price changes
        """
        if price_col not in df.columns:
            raise SamplingError(f"Price column '{price_col}' not found in DataFrame")
            
        # Sort by timestamp if specified
        if timestamp_col is not None and timestamp_col in df.columns:
            sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            # Assume index is already time-ordered
            sorted_df = df.reset_index(drop=True)
            
        # Calculate price changes
        price_changes = sorted_df[price_col].pct_change()
        
        # Find significant changes based on direction
        if direction.lower() == 'up':
            significant_indices = price_changes >= threshold
        elif direction.lower() == 'down':
            significant_indices = price_changes <= -threshold
        else:  # 'both'
            significant_indices = np.abs(price_changes) >= threshold
            
        # Get rows with significant changes
        result = sorted_df[significant_indices].copy()
        
        if len(result) == 0:
            logger.warning(f"No significant price changes found with threshold={threshold}")
            
        logger.info(f"Price change sampling selected {len(result)} events out of {len(df)} rows")
        return result
        
    # --- Split methods ---
    
    def _random_split(self, df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Basic random train/test split."""
        test_size_int = int(len(df) * test_size)
        shuffled = df.sample(frac=1)
        train = shuffled.iloc[test_size_int:].copy()
        test = shuffled.iloc[:test_size_int].copy()
        return train, test
        
    def _stratified_split(self, df: pd.DataFrame, test_size: float, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified split that preserves class distribution."""
        from sklearn.model_selection import train_test_split
        
        # Split preserving target distribution
        train, test = train_test_split(
            df, test_size=test_size, stratify=df[target_col], shuffle=True
        )
        
        return train, test
        
    def _time_series_split(self, df: pd.DataFrame, test_size: float, timestamp_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based split using the most recent data for testing."""
        # Sort by timestamp if specified
        if timestamp_col is not None and timestamp_col in df.columns:
            sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            # Assume index is already time-ordered
            sorted_df = df.reset_index(drop=True)
            
        # Calculate split point
        split_idx = int(len(sorted_df) * (1 - test_size))
        
        # Split into train and test
        train = sorted_df.iloc[:split_idx].copy()
        test = sorted_df.iloc[split_idx:].copy()
        
        return train, test
        
    def _purged_split(self, df: pd.DataFrame, test_size: float, target_col: str, 
                    embargo_size: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Purged split that avoids data leakage between train and test sets."""
        # First create a time-based split
        train, test = self._time_series_split(df, test_size)
        
        # Apply embargo - remove last N rows from training
        if embargo_size > 0 and len(train) > embargo_size:
            train = train.iloc[:-embargo_size].copy()
            
        return train, test
        
    def _trading_windows_split(self, df: pd.DataFrame, test_size: float, 
                             n_windows: int = 1,
                             timestamp_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into multiple trading windows for more robust evaluation."""
        # Sort by timestamp if specified
        if timestamp_col is not None and timestamp_col in df.columns:
            sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            # Assume index is already time-ordered
            sorted_df = df.reset_index(drop=True)
            
        # Calculate window size
        window_size = len(sorted_df) // (n_windows + 1)
        
        train_dfs = []
        test_dfs = []
        
        for i in range(n_windows):
            start_idx = i * window_size
            train_end_idx = start_idx + int(window_size * (1 - test_size))
            test_end_idx = start_idx + window_size
            
            window_train = sorted_df.iloc[start_idx:train_end_idx].copy()
            window_test = sorted_df.iloc[train_end_idx:test_end_idx].copy()
            
            train_dfs.append(window_train)
            test_dfs.append(window_test)
            
        # Combine all windows
        train = pd.concat(train_dfs)
        test = pd.concat(test_dfs)
        
        return train, test
        
    def _regime_based_split(self, df: pd.DataFrame, test_size: float, regime_col: str,
                          hold_out_regime: Optional[Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split based on market regimes, optionally holding out a specific regime."""
        if regime_col not in df.columns:
            raise SamplingError(f"Regime column '{regime_col}' not found in DataFrame")
            
        # If specific regime to hold out
        if hold_out_regime is not None:
            train = df[df[regime_col] != hold_out_regime].copy()
            test = df[df[regime_col] == hold_out_regime].copy()
            
            # If test set is too small or large, adjust
            test_frac = len(test) / len(df)
            if test_frac < 0.1 or test_frac > 0.5:
                logger.warning(f"Hold-out regime creates imbalanced split ({test_frac:.1%}). Using stratified split instead.")
                # Fall back to stratified split based on regime
                return self._stratified_split(df, test_size, regime_col)
                
            return train, test
            
        # Otherwise, do a stratified split based on regime
        return self._stratified_split(df, test_size, regime_col)
        
    # --- Cross-validation methods ---
    
    def _k_fold_cv(self, df: pd.DataFrame, n_splits: int, shuffle: bool = True,
                 random_state: Optional[int] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Standard k-fold cross-validation."""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        folds = []
        
        for train_idx, test_idx in kf.split(df):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            folds.append((train_df, test_df))
            
        return folds
        
    def _stratified_k_fold_cv(self, df: pd.DataFrame, n_splits: int, target_col: str,
                            shuffle: bool = True, 
                            random_state: Optional[int] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Stratified k-fold cross-validation that preserves class distribution."""
        from sklearn.model_selection import StratifiedKFold
        
        if target_col not in df.columns:
            raise SamplingError(f"Target column '{target_col}' not found in DataFrame")
            
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        folds = []
        
        for train_idx, test_idx in skf.split(df, df[target_col]):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            folds.append((train_df, test_df))
            
        return folds
        
    def _time_series_cv(self, df: pd.DataFrame, n_splits: int, 
                      timestamp_col: Optional[str] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Time series cross-validation with expanding window."""
        from sklearn.model_selection import TimeSeriesSplit
        
        # Sort by timestamp if specified
        if timestamp_col is not None and timestamp_col in df.columns:
            sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            # Assume index is already time-ordered
            sorted_df = df.reset_index(drop=True)
            
        tscv = TimeSeriesSplit(n_splits=n_splits)
        folds = []
        
        for train_idx, test_idx in tscv.split(sorted_df):
            train_df = sorted_df.iloc[train_idx].copy()
            test_df = sorted_df.iloc[test_idx].copy()
            folds.append((train_df, test_df))
            
        return folds
        
    def _walk_forward_cv(self, df: pd.DataFrame, n_splits: int,
                       train_window: int = 252,
                       test_window: int = 20,
                       step_size: Optional[int] = None,
                       timestamp_col: Optional[str] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Walk-forward cross-validation with fixed-size windows."""
        # Calculate step size if not provided
        if step_size is None:
            # Estimate step size to get approximately n_splits
            total_steps = len(df) - train_window - test_window
            step_size = max(1, total_steps // n_splits)
            
        # Get walk-forward samples
        samples = self._walk_forward_sample(
            df=df, 
            train_window=train_window,
            test_window=test_window,
            step_size=step_size,
            min_samples=n_splits,
            timestamp_col=timestamp_col
        )
        
        # Limit to n_splits
        return samples[:n_splits]
        
    def _purged_cv(self, df: pd.DataFrame, n_splits: int,
                 embargo_size: int = 5,
                 timestamp_col: Optional[str] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Cross-validation with purging and embargo to prevent leakage."""
        # First get time series CV splits
        cv_splits = self._time_series_cv(df, n_splits, timestamp_col)
        
        # Apply purging and embargo to each split
        purged_splits = []
        for train_df, test_df in cv_splits:
            # Get the end date of training and start date of testing
            if timestamp_col is not None and timestamp_col in df.columns:
                train_end = train_df[timestamp_col].max()
                test_start = test_df[timestamp_col].min()
                
                # Remove any training data that overlaps with test period
                purged_train = train_df[train_df[timestamp_col] < test_start].copy()
                
                # Apply embargo - remove training samples close to test
                if embargo_size > 0:
                    # This requires timestamps to be sortable
                    try:
                        # Sort training dates
                        sorted_dates = sorted(purged_train[timestamp_col].unique())
                        # Find embargo cutoff date
                        if len(sorted_dates) > embargo_size:
                            cutoff_date = sorted_dates[-embargo_size]
                            purged_train = purged_train[purged_train[timestamp_col] <= cutoff_date].copy()
                    except Exception as e:
                        logger.warning(f"Could not apply embargo using timestamps: {str(e)}")
            else:
                # Without timestamps, use indices
                test_start_idx = test_df.index.min()
                
                # Remove training data with indices after test start
                purged_train = train_df[train_df.index < test_start_idx].copy()
                
                # Apply embargo by removing last N rows
                if embargo_size > 0 and len(purged_train) > embargo_size:
                    purged_train = purged_train.iloc[:-embargo_size].copy()
                    
            purged_splits.append((purged_train, test_df))
            
        return purged_splits
        
    def _trading_windows_cv(self, df: pd.DataFrame, n_splits: int,
                          window_size: Optional[int] = None,
                          test_ratio: float = 0.2,
                          timestamp_col: Optional[str] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Cross-validation using multiple trading windows."""
        # Sort by timestamp if specified
        if timestamp_col is not None and timestamp_col in df.columns:
            sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            # Assume index is already time-ordered
            sorted_df = df.reset_index(drop=True)
            
        # Determine window size if not provided
        if window_size is None:
            window_size = len(sorted_df) // n_splits
            
        folds = []
        for i in range(n_splits):
            start_idx = i * window_size
            train_end_idx = start_idx + int(window_size * (1 - test_ratio))
            test_end_idx = start_idx + window_size
            
            # Handle boundary case in last fold
            if test_end_idx > len(sorted_df):
                test_end_idx = len(sorted_df)
                train_end_idx = int(test_end_idx * (1 - test_ratio))
                
            window_train = sorted_df.iloc[start_idx:train_end_idx].copy()
            window_test = sorted_df.iloc[train_end_idx:test_end_idx].copy()
            
            # Only add if we have data in both train and test
            if len(window_train) > 0 and len(window_test) > 0:
                folds.append((window_train, window_test))
                
        # If we have fewer than requested splits
        if len(folds) < n_splits:
            logger.warning(f"Could only create {len(folds)} folds instead of requested {n_splits}")
            
        return folds
        
    def _embargo_cv(self, df: pd.DataFrame, n_splits: int,
                  embargo_size: int = 5,
                  timestamp_col: Optional[str] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Cross-validation with embargo periods between folds."""
        # Sort by timestamp if specified
        if timestamp_col is not None and timestamp_col in df.columns:
            sorted_df = df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            # Assume index is already time-ordered
            sorted_df = df.reset_index(drop=True)
            
        # Calculate sizes
        total_size = len(sorted_df)
        fold_size = total_size // n_splits
        
        folds = []
        for i in range(n_splits):
            start_fold = i * fold_size
            end_fold = (i + 1) * fold_size
            
            # Adjust last fold to include remainder
            if i == n_splits - 1:
                end_fold = total_size
                
            # Create train/test split within this fold
            split_idx = start_fold + (end_fold - start_fold) // 2
            
            train_df = sorted_df.iloc[start_fold:split_idx].copy()
            test_df = sorted_df.iloc[split_idx:end_fold].copy()
            
            # Apply embargo between train and test
            if embargo_size > 0 and len(train_df) > embargo_size:
                train_df = train_df.iloc[:-embargo_size].copy()
                
            # Add to folds if both train and test have data
            if len(train_df) > 0 and len(test_df) > 0:
                folds.append((train_df, test_df))
                
        return folds
        
    def _regime_based_cv(self, df: pd.DataFrame, n_splits: int, regime_col: str,
                       test_ratio: float = 0.2,
                       random_state: Optional[int] = None) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Cross-validation splits based on different market regimes."""
        if regime_col not in df.columns:
            raise SamplingError(f"Regime column '{regime_col}' not found in DataFrame")
            
        # Get unique regimes
        regimes = df[regime_col].unique()
        
        # If we have at least n_splits regimes, use leave-one-regime-out CV
        if len(regimes) >= n_splits:
            folds = []
            for i, regime in enumerate(regimes[:n_splits]):
                train = df[df[regime_col] != regime].copy()
                test = df[df[regime_col] == regime].copy()
                folds.append((train, test))
                
            return folds
            
        # Otherwise, do stratified K-fold based on regime
        return self._stratified_k_fold_cv(
            df=df, 
            n_splits=n_splits, 
            target_col=regime_col, 
            shuffle=True, 
            random_state=random_state
        )


# Factory function
def get_sampler(**kwargs) -> SamplingManager:
    """Create and return a SamplingManager with optional configuration."""
    return SamplingManager()


def balance_dataset(X: pd.DataFrame, y: pd.Series, method: str = 'oversampling', **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """Balance feature and target data using the chosen sampling method."""
    df = X.copy()
    df['_target'] = y.values
    sampler = get_sampler()
    balanced = sampler.sample(df, method=method, target_col='_target', **kwargs)
    y_bal = balanced.pop('_target')
    return balanced.reset_index(drop=True), y_bal.reset_index(drop=True)

