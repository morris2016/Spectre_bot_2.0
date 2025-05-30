#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
ML Models - Preprocessing - Encoding

This module provides advanced data encoding techniques for machine learning models,
including categorical variable encoding, cyclical feature encoding, and complex transformations
optimized for financial time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional, Callable
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
import category_encoders as ce
from scipy import stats
import warnings

from common.logger import get_logger
from common.utils import is_categorical, is_cyclical, is_ordinal
from common.exceptions import EncodingError
from ml_models.preprocessing.scaling import MinMaxScaler

logger = get_logger(__name__)


class EncodingManager:
    """
    Manages the creation, application, and persistence of various encoding techniques
    for categorical, cyclical, and special financial features.
    """
    
    def __init__(self):
        """Initialize the EncodingManager with empty encoder registry."""
        self.encoders = {}
        self.feature_types = {}
        self.inverse_enabled = {}
        self._initialized = False
        logger.info("EncodingManager initialized")
        
    def fit(self, df: pd.DataFrame, feature_types: Optional[Dict[str, str]] = None) -> 'EncodingManager':
        """
        Analyze the dataframe and fit appropriate encoders for each column based on type.
        
        Args:
            df: Input DataFrame to analyze and fit encoders for
            feature_types: Optional dictionary mapping column names to feature types
                           ('categorical', 'cyclical', 'ordinal', etc.)
                           
        Returns:
            self: The fitted EncodingManager instance
        """
        if feature_types is None:
            feature_types = {}
            
        # Auto-detect feature types if not specified
        for col in df.columns:
            if col in feature_types:
                continue
                
            if is_categorical(df[col]):
                feature_types[col] = 'categorical'
            elif is_cyclical(df[col], col):
                feature_types[col] = 'cyclical'
            elif is_ordinal(df[col]):
                feature_types[col] = 'ordinal'
                
        # Create encoders for each detected type
        for col, ftype in feature_types.items():
            if col not in df.columns:
                logger.warning(f"Column {col} specified in feature_types but not found in DataFrame")
                continue
                
            try:
                if ftype == 'categorical':
                    self._fit_categorical_encoder(df, col)
                elif ftype == 'cyclical':
                    self._fit_cyclical_encoder(df, col)
                elif ftype == 'ordinal':
                    self._fit_ordinal_encoder(df, col)
                elif ftype == 'target':
                    self._fit_target_encoder(df, col)
                elif ftype == 'binary':
                    self._fit_binary_encoder(df, col)
                elif ftype == 'hash':
                    self._fit_hash_encoder(df, col)
                elif ftype == 'count':
                    self._fit_count_encoder(df, col)
                elif ftype == 'woe':
                    self._fit_woe_encoder(df, col)
                elif ftype == 'time':
                    self._fit_time_encoder(df, col)
            except Exception as e:
                logger.error(f"Error fitting encoder for column {col} of type {ftype}: {str(e)}")
                raise EncodingError(f"Failed to fit encoder for {col}") from e
                
        self.feature_types = feature_types
        self._initialized = True
        logger.info(f"EncodingManager fitted with {len(self.encoders)} encoders")
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted encoders to transform the input DataFrame.
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            Transformed DataFrame with encoded features
        """
        if not self._initialized:
            raise EncodingError("EncodingManager must be fitted before transform")
            
        result_df = df.copy()
        
        for col, ftype in self.feature_types.items():
            if col not in df.columns:
                logger.warning(f"Column {col} not found in input DataFrame during transform")
                continue
                
            encoder_key = f"{ftype}_{col}"
            if encoder_key not in self.encoders:
                logger.warning(f"No encoder found for {col} of type {ftype}")
                continue
                
            try:
                encoder = self.encoders[encoder_key]
                
                if ftype == 'categorical':
                    result_df = self._transform_categorical(result_df, col, encoder)
                elif ftype == 'cyclical':
                    result_df = self._transform_cyclical(result_df, col, encoder)
                elif ftype == 'ordinal':
                    result_df = self._transform_ordinal(result_df, col, encoder)
                elif ftype == 'target':
                    result_df = self._transform_target(result_df, col, encoder)
                elif ftype == 'binary':
                    result_df = self._transform_binary(result_df, col, encoder)
                elif ftype == 'hash':
                    result_df = self._transform_hash(result_df, col, encoder)
                elif ftype == 'count':
                    result_df = self._transform_count(result_df, col, encoder)
                elif ftype == 'woe':
                    result_df = self._transform_woe(result_df, col, encoder)
                elif ftype == 'time':
                    result_df = self._transform_time(result_df, col, encoder)
            except Exception as e:
                logger.error(f"Error transforming column {col} of type {ftype}: {str(e)}")
                raise EncodingError(f"Failed to transform {col}") from e
                
        return result_df
        
    def fit_transform(self, df: pd.DataFrame, feature_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Fit encoders to the data and then transform it.
        
        Args:
            df: Input DataFrame to fit and transform
            feature_types: Optional dictionary mapping column names to feature types
            
        Returns:
            Transformed DataFrame with encoded features
        """
        return self.fit(df, feature_types).transform(df)
        
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the encoding transformations where possible.
        
        Args:
            df: Encoded DataFrame to reverse transform
            
        Returns:
            DataFrame with original feature representations where possible
        """
        if not self._initialized:
            raise EncodingError("EncodingManager must be fitted before inverse_transform")
            
        result_df = df.copy()
        
        for col, ftype in self.feature_types.items():
            encoder_key = f"{ftype}_{col}"
            if encoder_key not in self.encoders or not self.inverse_enabled.get(encoder_key, False):
                continue
                
            try:
                encoder = self.encoders[encoder_key]
                
                if ftype == 'categorical':
                    result_df = self._inverse_transform_categorical(result_df, col, encoder)
                elif ftype == 'cyclical':
                    result_df = self._inverse_transform_cyclical(result_df, col, encoder)
                elif ftype == 'ordinal':
                    result_df = self._inverse_transform_ordinal(result_df, col, encoder)
                elif ftype == 'time':
                    result_df = self._inverse_transform_time(result_df, col, encoder)
            except Exception as e:
                logger.warning(f"Could not inverse transform {col}: {str(e)}")
                # Continue with other columns even if one fails
                
        return result_df
        
    def save(self, filepath: str) -> None:
        """
        Save the fitted encoders to a file.
        
        Args:
            filepath: Path to save the encoders
        """
        import joblib
        joblib.dump({
            'encoders': self.encoders,
            'feature_types': self.feature_types,
            'inverse_enabled': self.inverse_enabled,
            '_initialized': self._initialized
        }, filepath)
        logger.info(f"EncodingManager saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> 'EncodingManager':
        """
        Load encoders from a file.
        
        Args:
            filepath: Path to load the encoders from
            
        Returns:
            Loaded EncodingManager instance
        """
        import joblib
        data = joblib.load(filepath)
        manager = cls()
        manager.encoders = data['encoders']
        manager.feature_types = data['feature_types']
        manager.inverse_enabled = data['inverse_enabled']
        manager._initialized = data['_initialized']
        logger.info(f"EncodingManager loaded from {filepath}")
        return manager
        
    # Private methods for specific encoder types
    
    def _fit_categorical_encoder(self, df: pd.DataFrame, col: str) -> None:
        """Fit a categorical encoder for the given column."""
        n_unique = df[col].nunique()
        
        # Choose encoding strategy based on cardinality
        if n_unique <= 2:
            encoder = LabelEncoder().fit(df[col].astype(str))
            self.encoders[f"categorical_{col}"] = encoder
            self.inverse_enabled[f"categorical_{col}"] = True
        elif n_unique <= 15:  # Low cardinality
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(df[[col]])
            self.encoders[f"categorical_{col}"] = encoder
            self.inverse_enabled[f"categorical_{col}"] = True
        else:  # High cardinality
            encoder = ce.BinaryEncoder(cols=[col])
            encoder.fit(df[[col]])
            self.encoders[f"categorical_{col}"] = encoder
            self.inverse_enabled[f"categorical_{col}"] = False
            
    def _fit_cyclical_encoder(self, df: pd.DataFrame, col: str) -> None:
        """Fit a cyclical encoder for the given column."""
        # For cyclical features, we need to know the cycle length
        if col.lower() in ['hour', 'hour_of_day']:
            max_value = 24
        elif col.lower() in ['day', 'day_of_week']:
            max_value = 7
        elif col.lower() in ['month']:
            max_value = 12
        elif col.lower() in ['day_of_month']:
            max_value = 31
        else:
            max_value = df[col].max()
            
        scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
        scaler.fit(df[[col]])
        
        self.encoders[f"cyclical_{col}"] = {
            'scaler': scaler,
            'max_value': max_value
        }
        self.inverse_enabled[f"cyclical_{col}"] = True
        
    def _fit_ordinal_encoder(self, df: pd.DataFrame, col: str) -> None:
        """Fit an ordinal encoder for the given column."""
        encoder = OrdinalEncoder()
        encoder.fit(df[[col]])
        self.encoders[f"ordinal_{col}"] = encoder
        self.inverse_enabled[f"ordinal_{col}"] = True
        
    def _fit_target_encoder(self, df: pd.DataFrame, col: str) -> None:
        """Fit a target encoder for the given column."""
        encoder = ce.TargetEncoder(cols=[col])
        # Note: This requires the target column, will be applied during transform
        self.encoders[f"target_{col}"] = encoder
        self.inverse_enabled[f"target_{col}"] = False
        
    def _fit_binary_encoder(self, df: pd.DataFrame, col: str) -> None:
        """Fit a binary encoder for the given column."""
        encoder = ce.BinaryEncoder(cols=[col])
        encoder.fit(df[[col]])
        self.encoders[f"binary_{col}"] = encoder
        self.inverse_enabled[f"binary_{col}"] = False
        
    def _fit_hash_encoder(self, df: pd.DataFrame, col: str) -> None:
        """Fit a hash encoder for high cardinality categorical features."""
        encoder = ce.HashingEncoder(cols=[col])
        encoder.fit(df[[col]])
        self.encoders[f"hash_{col}"] = encoder
        self.inverse_enabled[f"hash_{col}"] = False
        
    def _fit_count_encoder(self, df: pd.DataFrame, col: str) -> None:
        """Fit a count encoder to replace categories with their counts."""
        value_counts = df[col].value_counts().to_dict()
        self.encoders[f"count_{col}"] = value_counts
        self.inverse_enabled[f"count_{col}"] = False
        
    def _fit_woe_encoder(self, df: pd.DataFrame, col: str) -> None:
        """Fit a Weight of Evidence encoder."""
        encoder = ce.WOEEncoder(cols=[col])
        # Note: This requires the target column, will be applied during transform
        self.encoders[f"woe_{col}"] = encoder
        self.inverse_enabled[f"woe_{col}"] = False
        
    def _fit_time_encoder(self, df: pd.DataFrame, col: str) -> None:
        """Fit a time feature encoder for datetime columns."""
        # For datetime columns, we extract multiple features
        # Store column format if we can detect it
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.encoders[f"time_{col}"] = {'type': 'datetime64'}
            else:
                # Try to infer format
                test_date = pd.to_datetime(df[col].iloc[0])
                self.encoders[f"time_{col}"] = {'type': 'string'}
        except:
            logger.warning(f"Could not determine datetime format for {col}")
            self.encoders[f"time_{col}"] = {'type': 'unknown'}
            
        self.inverse_enabled[f"time_{col}"] = False
        
    # Transform methods
    
    def _transform_categorical(self, df: pd.DataFrame, col: str, encoder: Any) -> pd.DataFrame:
        """Transform categorical column using the fitted encoder."""
        result_df = df.copy()
        
        if isinstance(encoder, LabelEncoder):
            # Fill NaN values with a placeholder
            result_df[col] = result_df[col].fillna('NaN')
            result_df[col] = encoder.transform(result_df[col].astype(str))
            
        elif isinstance(encoder, OneHotEncoder):
            encoded = encoder.transform(result_df[[col]])
            # Get feature names
            feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=result_df.index)
            
            # Drop original column and add encoded columns
            result_df = pd.concat([result_df.drop(columns=[col]), encoded_df], axis=1)
            
        elif isinstance(encoder, ce.BinaryEncoder):
            encoded_df = encoder.transform(result_df[[col]])
            # Drop original column and join encoded result
            result_df = result_df.drop(columns=[col])
            result_df = pd.concat([result_df, encoded_df], axis=1)
            
        return result_df
        
    def _transform_cyclical(self, df: pd.DataFrame, col: str, encoder: Dict) -> pd.DataFrame:
        """Transform cyclical column to sine and cosine components."""
        result_df = df.copy()
        
        scaler = encoder['scaler']
        
        # Scale values to [0, 2π]
        scaled_values = scaler.transform(result_df[[col]]).flatten()
        
        # Create sine and cosine features
        result_df[f"{col}_sin"] = np.sin(scaled_values)
        result_df[f"{col}_cos"] = np.cos(scaled_values)
        
        # Drop original column
        result_df = result_df.drop(columns=[col])
        
        return result_df
        
    def _transform_ordinal(self, df: pd.DataFrame, col: str, encoder: Any) -> pd.DataFrame:
        """Transform ordinal column using the fitted encoder."""
        result_df = df.copy()
        
        # Handle missing values
        result_df[col] = result_df[col].fillna(result_df[col].mode()[0])
        
        # Apply ordinal encoding
        result_df[col] = encoder.transform(result_df[[col]]).flatten()
        
        return result_df
        
    def _transform_target(self, df: pd.DataFrame, col: str, encoder: Any) -> pd.DataFrame:
        """Transform using target encoder."""
        # Note: This would need the target column, so we apply during training
        result_df = df.copy()
        
        # For inference, we handle the column with mean encoding from training
        if hasattr(encoder, 'mapping'):
            mapping = encoder.mapping
            col_mapping = mapping.get(col, {})
            if col_mapping:
                global_mean = col_mapping.get('_global_mean', df[col].mean())
                result_df[col] = result_df[col].map(col_mapping).fillna(global_mean)
        
        return result_df
        
    def _transform_binary(self, df: pd.DataFrame, col: str, encoder: Any) -> pd.DataFrame:
        """Transform using binary encoder."""
        result_df = df.copy()
        
        encoded_df = encoder.transform(result_df[[col]])
        # Drop original column and join encoded result
        result_df = result_df.drop(columns=[col])
        result_df = pd.concat([result_df, encoded_df], axis=1)
        
        return result_df
        
    def _transform_hash(self, df: pd.DataFrame, col: str, encoder: Any) -> pd.DataFrame:
        """Transform using hash encoder for high cardinality features."""
        result_df = df.copy()
        
        encoded_df = encoder.transform(result_df[[col]])
        # Drop original column and join encoded result
        result_df = result_df.drop(columns=[col])
        result_df = pd.concat([result_df, encoded_df], axis=1)
        
        return result_df
        
    def _transform_count(self, df: pd.DataFrame, col: str, encoder: Dict) -> pd.DataFrame:
        """Transform using frequency counts."""
        result_df = df.copy()
        
        # Map values to their counts
        result_df[col] = result_df[col].map(encoder).fillna(1)  # Default to 1 for unseen values
        
        return result_df
        
    def _transform_woe(self, df: pd.DataFrame, col: str, encoder: Any) -> pd.DataFrame:
        """Transform using Weight of Evidence encoder."""
        # Similar to target encoding, this requires the target during training
        result_df = df.copy()
        
        # For inference, apply pre-calculated WOE values
        if hasattr(encoder, 'mapping'):
            mapping = encoder.mapping
            col_mapping = mapping.get(col, {})
            if col_mapping:
                default_woe = col_mapping.get('_global_woe', 0)
                result_df[col] = result_df[col].map(col_mapping).fillna(default_woe)
        
        return result_df
        
    def _transform_time(self, df: pd.DataFrame, col: str, encoder: Dict) -> pd.DataFrame:
        """Transform datetime column into multiple time-based features."""
        result_df = df.copy()
        
        # Convert to datetime if needed
        try:
            if encoder['type'] != 'datetime64':
                result_df[col] = pd.to_datetime(result_df[col])
        except:
            logger.warning(f"Could not convert {col} to datetime, skipping time encoding")
            return result_df
            
        # Extract time-based features
        result_df[f"{col}_year"] = result_df[col].dt.year
        result_df[f"{col}_month"] = result_df[col].dt.month
        result_df[f"{col}_day"] = result_df[col].dt.day
        result_df[f"{col}_dayofweek"] = result_df[col].dt.dayofweek
        result_df[f"{col}_hour"] = result_df[col].dt.hour
        
        # Optional: add quarter, week of year, etc.
        result_df[f"{col}_quarter"] = result_df[col].dt.quarter
        result_df[f"{col}_weekofyear"] = result_df[col].dt.isocalendar().week
        
        # Create cyclical encodings for day, month, hour
        result_df[f"{col}_hour_sin"] = np.sin(2 * np.pi * result_df[f"{col}_hour"] / 24)
        result_df[f"{col}_hour_cos"] = np.cos(2 * np.pi * result_df[f"{col}_hour"] / 24)
        
        result_df[f"{col}_month_sin"] = np.sin(2 * np.pi * result_df[f"{col}_month"] / 12)
        result_df[f"{col}_month_cos"] = np.cos(2 * np.pi * result_df[f"{col}_month"] / 12)
        
        result_df[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * result_df[f"{col}_dayofweek"] / 7)
        result_df[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * result_df[f"{col}_dayofweek"] / 7)
        
        # Drop original column
        result_df = result_df.drop(columns=[col])
        
        return result_df
        
    # Inverse transform methods
    
    def _inverse_transform_categorical(self, df: pd.DataFrame, col: str, encoder: Any) -> pd.DataFrame:
        """Inverse transform categorical data where possible."""
        result_df = df.copy()
        
        if isinstance(encoder, LabelEncoder):
            result_df[col] = encoder.inverse_transform(result_df[col].astype(int))
        elif isinstance(encoder, OneHotEncoder):
            # Get all columns related to this encoding
            encoded_cols = [c for c in df.columns if c.startswith(f"{col}_")]
            if encoded_cols:
                # Extract the encoded values
                encoded_values = result_df[encoded_cols].values
                # Inverse transform
                original_values = encoder.inverse_transform(encoded_values)
                # Replace with original column
                result_df[col] = original_values
                # Drop encoded columns
                result_df = result_df.drop(columns=encoded_cols)
                
        return result_df
        
    def _inverse_transform_cyclical(self, df: pd.DataFrame, col: str, encoder: Dict) -> pd.DataFrame:
        """Inverse transform cyclical features where possible."""
        result_df = df.copy()
        
        # Check if sine and cosine columns exist
        sin_col = f"{col}_sin"
        cos_col = f"{col}_cos"
        
        if sin_col in df.columns and cos_col in df.columns:
            # Convert back from sine and cosine
            angle = np.arctan2(result_df[sin_col], result_df[cos_col])
            # Adjust negative angles
            angle[angle < 0] += 2 * np.pi
            
            # Scale back to original range
            max_value = encoder['max_value']
            result_df[col] = angle * max_value / (2 * np.pi)
            
            # Round to int if the original was likely an int
            if max_value in [7, 12, 24, 31, 60]:
                result_df[col] = np.round(result_df[col]).astype(int)
                
            # Drop sine and cosine columns
            result_df = result_df.drop(columns=[sin_col, cos_col])
            
        return result_df
        
    def _inverse_transform_ordinal(self, df: pd.DataFrame, col: str, encoder: Any) -> pd.DataFrame:
        """Inverse transform ordinal data where possible."""
        result_df = df.copy()
        
        # Reshape for inverse transform
        original_values = encoder.inverse_transform(
            result_df[col].values.reshape(-1, 1)
        )
        
        # Replace with original values
        result_df[col] = original_values.flatten()
        
        return result_df
        
    def _inverse_transform_time(self, df: pd.DataFrame, col: str, encoder: Dict) -> pd.DataFrame:
        """Attempt to reconstruct datetime from components."""
        result_df = df.copy()
        
        # Check for required columns
        year_col = f"{col}_year"
        month_col = f"{col}_month"
        day_col = f"{col}_day"
        hour_col = f"{col}_hour"
        
        required_cols = [c for c in [year_col, month_col, day_col] if c in df.columns]
        
        if len(required_cols) >= 3:
            # Create datetime from components
            try:
                dates = pd.to_datetime({
                    'year': result_df[year_col],
                    'month': result_df[month_col],
                    'day': result_df[day_col],
                    'hour': result_df[hour_col] if hour_col in df.columns else 0
                })
                
                # Add original column back
                result_df[col] = dates
                
                # Get all time-derived columns
                time_cols = [c for c in df.columns if c.startswith(f"{col}_")]
                result_df = result_df.drop(columns=time_cols)
            except Exception as e:
                logger.warning(f"Error recreating datetime for {col}: {str(e)}")
                
        return result_df


class AdvancedFinancialEncoder:
    """
    Specialized encoder for financial data that implements domain-specific
    transformations beyond standard categorical/numerical encoding.
    """
    
    def __init__(self):
        """Initialize the AdvancedFinancialEncoder."""
        self.encoders = {}
        self.feature_configs = {}
        self._initialized = False
        logger.info("AdvancedFinancialEncoder initialized")
        
    def fit(self, df: pd.DataFrame, feature_configs: Dict[str, Dict[str, Any]]) -> 'AdvancedFinancialEncoder':
        """
        Fit financial encoders based on configuration.
        
        Args:
            df: Input DataFrame
            feature_configs: Dictionary of column names to configuration settings
                Example: {'price_change': {'type': 'momentum', 'lookback': [1, 5, 10]}}
                
        Returns:
            self: The fitted encoder
        """
        self.feature_configs = feature_configs
        
        for col, config in feature_configs.items():
            encode_type = config.get('type', 'identity')
            
            try:
                if encode_type == 'momentum':
                    self._fit_momentum_encoder(df, col, config)
                elif encode_type == 'volatility':
                    self._fit_volatility_encoder(df, col, config)
                elif encode_type == 'rank':
                    self._fit_rank_encoder(df, col, config)
                elif encode_type == 'returns':
                    self._fit_returns_encoder(df, col, config)
                elif encode_type == 'crosssectional':
                    self._fit_crosssectional_encoder(df, col, config)
                elif encode_type == 'zscore':
                    self._fit_zscore_encoder(df, col, config)
            except Exception as e:
                logger.error(f"Error fitting financial encoder for {col}: {str(e)}")
                raise EncodingError(f"Failed to fit financial encoder for {col}") from e
                
        self._initialized = True
        logger.info(f"AdvancedFinancialEncoder fitted with {len(self.encoders)} encoders")
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply financial encoders to transform the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame with financial features
        """
        if not self._initialized:
            raise EncodingError("AdvancedFinancialEncoder must be fitted before transform")
            
        result_df = df.copy()
        
        for col, config in self.feature_configs.items():
            if col not in df.columns:
                logger.warning(f"Column {col} not found in input DataFrame")
                continue
                
            encode_type = config.get('type', 'identity')
            encoder_key = f"{encode_type}_{col}"
            
            if encoder_key not in self.encoders:
                logger.warning(f"No encoder found for {col} of type {encode_type}")
                continue
                
            try:
                if encode_type == 'momentum':
                    result_df = self._transform_momentum(result_df, col, config)
                elif encode_type == 'volatility':
                    result_df = self._transform_volatility(result_df, col, config)
                elif encode_type == 'rank':
                    result_df = self._transform_rank(result_df, col, config)
                elif encode_type == 'returns':
                    result_df = self._transform_returns(result_df, col, config)
                elif encode_type == 'crosssectional':
                    result_df = self._transform_crosssectional(result_df, col, config)
                elif encode_type == 'zscore':
                    result_df = self._transform_zscore(result_df, col, config)
            except Exception as e:
                logger.error(f"Error transforming {col} with {encode_type}: {str(e)}")
                raise EncodingError(f"Failed to transform {col}") from e
                
        return result_df
        
    def fit_transform(self, df: pd.DataFrame, feature_configs: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: Input DataFrame
            feature_configs: Feature configuration dictionary
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(df, feature_configs).transform(df)
        
    def save(self, filepath: str) -> None:
        """
        Save the fitted encoders to a file.
        
        Args:
            filepath: Path to save the encoders
        """
        import joblib
        joblib.dump({
            'encoders': self.encoders,
            'feature_configs': self.feature_configs,
            '_initialized': self._initialized
        }, filepath)
        logger.info(f"AdvancedFinancialEncoder saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> 'AdvancedFinancialEncoder':
        """
        Load encoders from a file.
        
        Args:
            filepath: Path to load the encoders from
            
        Returns:
            Loaded encoder instance
        """
        import joblib
        data = joblib.load(filepath)
        encoder = cls()
        encoder.encoders = data['encoders']
        encoder.feature_configs = data['feature_configs']
        encoder._initialized = data['_initialized']
        logger.info(f"AdvancedFinancialEncoder loaded from {filepath}")
        return encoder
        
    # Private fitting methods
    
    def _fit_momentum_encoder(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> None:
        """Fit encoder for momentum indicators (price changes over time)."""
        lookbacks = config.get('lookback', [1, 5, 10, 20])
        self.encoders[f"momentum_{col}"] = {
            'lookbacks': lookbacks
        }
        
    def _fit_volatility_encoder(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> None:
        """Fit encoder for volatility indicators."""
        windows = config.get('windows', [5, 10, 20, 30])
        self.encoders[f"volatility_{col}"] = {
            'windows': windows
        }
        
    def _fit_rank_encoder(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> None:
        """Fit encoder for percentile rank transformation."""
        window = config.get('window', 20)
        self.encoders[f"rank_{col}"] = {
            'window': window
        }
        
    def _fit_returns_encoder(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> None:
        """Fit encoder for returns calculation."""
        periods = config.get('periods', [1, 5, 10, 20])
        log_returns = config.get('log_returns', True)
        self.encoders[f"returns_{col}"] = {
            'periods': periods,
            'log_returns': log_returns
        }
        
    def _fit_crosssectional_encoder(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> None:
        """Fit encoder for cross-sectional normalization."""
        group_col = config.get('group_col')
        if not group_col:
            raise EncodingError("group_col is required for cross-sectional encoding")
            
        self.encoders[f"crosssectional_{col}"] = {
            'group_col': group_col
        }
        
    def _fit_zscore_encoder(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> None:
        """Fit encoder for z-score normalization."""
        window = config.get('window', 20)
        outlier_threshold = config.get('outlier_threshold', 3.0)
        self.encoders[f"zscore_{col}"] = {
            'window': window,
            'outlier_threshold': outlier_threshold,
            'mean': df[col].mean(),
            'std': df[col].std()
        }
        
    # Transform methods
    
    def _transform_momentum(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Transform data using momentum indicators."""
        result_df = df.copy()
        lookbacks = self.encoders[f"momentum_{col}"]['lookbacks']
        
        for lb in lookbacks:
            result_df[f"{col}_mom_{lb}"] = result_df[col].pct_change(lb)
            
        return result_df
        
    def _transform_volatility(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Transform data using volatility indicators."""
        result_df = df.copy()
        windows = self.encoders[f"volatility_{col}"]['windows']
        
        for w in windows:
            # Standard deviation over window
            result_df[f"{col}_vol_{w}"] = result_df[col].rolling(window=w).std()
            
            # Range-based volatility (high-low range)
            if f"{col}_high" in result_df.columns and f"{col}_low" in result_df.columns:
                result_df[f"{col}_range_{w}"] = (
                    (result_df[f"{col}_high"] - result_df[f"{col}_low"]) / 
                    result_df[col]
                ).rolling(window=w).mean()
                
            # ARCH-like volatility (squared returns)
            returns = result_df[col].pct_change()
            result_df[f"{col}_arch_{w}"] = returns.pow(2).rolling(window=w).mean().pow(0.5)
            
        return result_df
        
    def _transform_rank(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Transform data using percentile rank."""
        result_df = df.copy()
        window = self.encoders[f"rank_{col}"]['window']
        
        # Compute rolling rank transformation
        result_df[f"{col}_rank"] = result_df[col].rolling(
            window=window, min_periods=1
        ).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0)
        
        return result_df
        
    def _transform_returns(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Transform data using returns calculation."""
        result_df = df.copy()
        periods = self.encoders[f"returns_{col}"]['periods']
        log_returns = self.encoders[f"returns_{col}"]['log_returns']
        
        for p in periods:
            if log_returns:
                # Log returns: log(price_t / price_t-1)
                result_df[f"{col}_return_{p}"] = np.log(
                    result_df[col] / result_df[col].shift(p)
                )
            else:
                # Simple returns: (price_t / price_t-1) - 1
                result_df[f"{col}_return_{p}"] = (
                    result_df[col] / result_df[col].shift(p) - 1
                )
                
        return result_df
        
    def _transform_crosssectional(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Transform data using cross-sectional normalization."""
        result_df = df.copy()
        group_col = self.encoders[f"crosssectional_{col}"]['group_col']
        
        if group_col not in result_df.columns:
            logger.warning(f"Group column {group_col} not found in DataFrame")
            return result_df
            
        # Group by the specified column and apply z-score normalization
        result_df[f"{col}_cs_norm"] = result_df.groupby(group_col)[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        
        # Also add percentile rank within group
        result_df[f"{col}_cs_rank"] = result_df.groupby(group_col)[col].transform(
            lambda x: x.rank(pct=True)
        )
        
        return result_df
        
    def _transform_zscore(self, df: pd.DataFrame, col: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Transform data using z-score normalization."""
        result_df = df.copy()
        window = self.encoders[f"zscore_{col}"]['window']
        outlier_threshold = self.encoders[f"zscore_{col}"]['outlier_threshold']
        
        # Compute rolling z-score
        roll_mean = result_df[col].rolling(window=window, min_periods=1).mean()
        roll_std = result_df[col].rolling(window=window, min_periods=1).std()
        
        # Handle case where std is 0
        roll_std = roll_std.replace(0, 1)
        
        result_df[f"{col}_zscore"] = (result_df[col] - roll_mean) / roll_std
        
        # Cap outliers if specified
        if outlier_threshold:
            result_df[f"{col}_zscore"] = result_df[f"{col}_zscore"].clip(
                lower=-outlier_threshold, upper=outlier_threshold
            )
            
        return result_df


def get_encoder_for_task(task: str, **kwargs) -> Union[EncodingManager, AdvancedFinancialEncoder]:
    """
    Factory function to create an appropriate encoder based on the task.
    
    Args:
        task: The encoding task ('general', 'financial', etc.)
        **kwargs: Additional arguments to pass to the encoder
        
    Returns:
        Appropriate encoder instance
    """
    if task.lower() == 'financial':
        return AdvancedFinancialEncoder()
    else:
        return EncodingManager()


def encode_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str = "onehot",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    """Simple wrapper to encode categorical features."""
    if method != "onehot":
        raise ValueError("Only onehot encoding is supported")

    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoder.fit(pd.concat([train_df, val_df]))
    train_enc = pd.DataFrame(
        encoder.transform(train_df),
        index=train_df.index,
        columns=encoder.get_feature_names_out(train_df.columns),
    )
    val_enc = pd.DataFrame(
        encoder.transform(val_df),
        index=val_df.index,
        columns=encoder.get_feature_names_out(val_df.columns),
    )
    test_enc = pd.DataFrame(
        encoder.transform(test_df),
        index=test_df.index,
        columns=encoder.get_feature_names_out(test_df.columns),
    )
    return train_enc, val_enc, test_enc, encoder

