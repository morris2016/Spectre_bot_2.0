#!/usr/bin/env python3
from __future__ import annotations
"""
QuantumSpectre Elite Trading System
Time Series Forecasting Models

This module provides advanced time series forecasting models specifically optimized
for financial market prediction, including ARIMA, Prophet, LSTM, GRU, and TCN models.
Each model is designed for high-performance prediction with GPU acceleration where
applicable, and includes specialized features for capturing market dynamics.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import joblib
from functools import partial

# Statistical models
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (
        Dense, LSTM, GRU, Dropout, BatchNormalization, Input,
        Bidirectional, Conv1D, MaxPooling1D, Flatten,
        TimeDistributed, Attention, MultiHeadAttention,
        LayerNormalization, Add, Activation
    )
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.regularizers import l1_l2
    TF_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    tf = None  # type: ignore
    Sequential = Model = load_model = None  # type: ignore
    Dense = LSTM = GRU = Dropout = BatchNormalization = Input = None  # type: ignore
    Bidirectional = Conv1D = MaxPooling1D = Flatten = None  # type: ignore
    TimeDistributed = Attention = MultiHeadAttention = None  # type: ignore
    LayerNormalization = Add = Activation = None  # type: ignore
    Adam = RMSprop = None  # type: ignore
    EarlyStopping = ModelCheckpoint = ReduceLROnPlateau = None  # type: ignore
    l1_l2 = None  # type: ignore
    TF_AVAILABLE = False

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Scikit-learn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Project imports
from common.logger import get_logger
from common.utils import timeit, gpu_info, save_to_file, load_from_file
from common.exceptions import ModelTrainingError, ModelPredictionError, ModelNotFoundError
from ml_models.models.base import BaseModel, ModelConfig

logger = get_logger(__name__)


@dataclass
class TimeSeriesConfig(ModelConfig):
    """Configuration for time series models with specialized parameters."""
    sequence_length: int = 60  # Default lookback window
    forecast_horizon: int = 10  # Default prediction horizon
    n_features: int = 1  # Number of features in input data
    hidden_units: List[int] = None  # Hidden layer units for neural networks
    dropout_rate: float = 0.2  # Dropout rate for regularization
    learning_rate: float = 0.001  # Learning rate for optimization
    batch_size: int = 64  # Batch size for training
    epochs: int = 100  # Maximum training epochs
    early_stopping_patience: int = 10  # Patience for early stopping
    validation_split: float = 0.2  # Portion of data for validation
    seasonal_periods: List[int] = None  # Seasonality periods for statistical models
    seasonality_mode: str = 'multiplicative'  # Seasonality mode (multiplicative/additive)
    
    def __post_init__(self):
        """Initialize default values for list parameters."""
        super().__post_init__()
        if self.hidden_units is None:
            self.hidden_units = [128, 64, 32]
        if self.seasonal_periods is None:
            self.seasonal_periods = [24, 168]  # Daily and weekly seasonality


class ARIMAModel(BaseModel):
    """
    Advanced ARIMA model implementation for time series forecasting with
    automatic parameter selection and seasonality handling.
    """
    
    def __init__(self, config: TimeSeriesConfig, name: str = "arima", **kwargs: Any):
        """
        Initialize the ARIMA model with the given configuration.
        
        Args:
            config: Configuration parameters for the model
            name: Unique identifier for this model instance
        """
        super().__init__(config, name=name, **kwargs)
        self.model = None
        self.scaler = StandardScaler()
        self.config = config
        self.best_order = None
        self.best_seasonal_order = None
        self.is_fitted = False
        self.train_info = {}
    
    @timeit
    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> Dict[str, Any]:
        """
        Fit the ARIMA model to the time series data with automatic order selection.
        
        Args:
            X: Time series data (if y is None, X is assumed to be the target series)
            y: Target series if X contains exogenous variables
            
        Returns:
            Dict containing training results and metrics
        """
        try:
            # Use X as the target if y is not provided
            target = y if y is not None else X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X
            
            # Ensure target is a pandas Series
            if not isinstance(target, pd.Series):
                if isinstance(target, pd.DataFrame):
                    target = target.iloc[:, 0]
                else:
                    target = pd.Series(target)
            
            # Scale the data
            target_scaled = pd.Series(
                self.scaler.fit_transform(target.values.reshape(-1, 1)).flatten(),
                index=target.index
            )
            
            # Auto-select optimal parameters if not specified
            if not hasattr(self, 'best_order') or self.best_order is None:
                self.best_order, self.best_seasonal_order = self._auto_select_parameters(target_scaled)
                logger.info(f"Selected ARIMA order: {self.best_order}, seasonal order: {self.best_seasonal_order}")
            
            # Get exogenous variables if provided
            exog = None
            if y is not None:
                # X contains exogenous variables
                exog = X
            
            # Fit the SARIMAX model
            self.model = SARIMAX(
                target_scaled,
                exog=exog,
                order=self.best_order,
                seasonal_order=self.best_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            results = self.model.fit(disp=False)
            self.model = results
            self.is_fitted = True
            
            # Calculate in-sample metrics
            predictions = self.model.predict()
            predictions = self.scaler.inverse_transform(predictions.values.reshape(-1, 1)).flatten()
            target_orig = target.values
            
            mse = mean_squared_error(target_orig, predictions)
            mae = mean_absolute_error(target_orig, predictions)
            r2 = r2_score(target_orig, predictions)
            
            self.train_info = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'order': self.best_order,
                'seasonal_order': self.best_seasonal_order,
                'aic': self.model.aic,
                'bic': self.model.bic
            }
            
            logger.info(f"ARIMA model fitted successfully: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
            return self.train_info
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise ModelTrainingError(f"Failed to train ARIMA model: {str(e)}")
    
    def _auto_select_parameters(self, series: pd.Series) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Automatically select optimal ARIMA parameters based on AIC.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (order, seasonal_order) with the best parameters
        """
        best_aic = float('inf')
        best_order = None
        best_seasonal_order = None
        
        # Define parameter search space
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        
        # Use default seasonal periods from config
        seasonal_period = self.config.seasonal_periods[0] if self.config.seasonal_periods else 0
        
        # Grid search for best parameters (simplified for efficiency)
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    # Try non-seasonal model first
                    try:
                        model = SARIMAX(
                            series,
                            order=(p, d, q),
                            seasonal_order=(0, 0, 0, 0),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        results = model.fit(disp=False)
                        aic = results.aic
                        
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            best_seasonal_order = (0, 0, 0, 0)
                            
                        # Only try seasonal if we have enough data
                        if seasonal_period > 0 and len(series) >= 3 * seasonal_period:
                            # Try with simple seasonality
                            model = SARIMAX(
                                series,
                                order=(p, d, q),
                                seasonal_order=(1, 1, 1, seasonal_period),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            results = model.fit(disp=False)
                            aic = results.aic
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                                best_seasonal_order = (1, 1, 1, seasonal_period)
                    
                    except Exception as e:
                        # Skip failed models
                        continue
        
        # If no model converged, use simple defaults
        if best_order is None:
            logger.warning("Could not find optimal ARIMA parameters, using defaults")
            best_order = (1, 1, 1)
            best_seasonal_order = (0, 0, 0, 0)
        
        return best_order, best_seasonal_order
    
    @timeit
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Generate predictions from the ARIMA model.
        
        Args:
            X: Input data (can be exogenous variables or steps to forecast)
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Model has not been fitted yet.")
        
        try:
            # Determine forecast horizon
            steps = kwargs.get('steps', self.config.forecast_horizon)
            
            # Handle different prediction modes
            if isinstance(X, int):
                # X is number of steps to forecast
                steps = X
                predictions = self.model.forecast(steps=steps)
            else:
                # X contains exogenous variables or is None
                exog = X if X is not None and len(X) > 0 else None
                if exog is not None:
                    predictions = self.model.forecast(steps=steps, exog=exog)
                else:
                    predictions = self.model.forecast(steps=steps)
            
            # Convert predictions back to original scale
            predictions = self.scaler.inverse_transform(
                predictions.values.reshape(-1, 1)
            ).flatten()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with ARIMA model: {str(e)}")
            raise ModelPredictionError(f"Failed to generate predictions: {str(e)}")
    
    def save(self, directory: str) -> str:
        """
        Save the ARIMA model to disk.
        
        Args:
            directory: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Cannot save unfitted model")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save model components
        model_path = os.path.join(directory, f"{self.name}_arima.pkl")
        scaler_path = os.path.join(directory, f"{self.name}_scaler.pkl")
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        # Save the fitted model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save the scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save configuration and metadata
        metadata = {
            'best_order': self.best_order,
            'best_seasonal_order': self.best_seasonal_order,
            'train_info': self.train_info,
            'config': self.config.__dict__,
            'is_fitted': self.is_fitted
        }
        save_to_file(metadata, config_path)
        
        logger.info(f"ARIMA model saved to {directory}")
        return directory
    
    def load(self, directory: str) -> 'ARIMAModel':
        """
        Load the ARIMA model from disk.
        
        Args:
            directory: Directory containing the saved model
            
        Returns:
            Loaded model instance
        """
        model_path = os.path.join(directory, f"{self.name}_arima.pkl")
        scaler_path = os.path.join(directory, f"{self.name}_scaler.pkl")
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"Model file not found at {model_path}")
        
        # Load the fitted model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load configuration and metadata
        metadata = load_from_file(config_path)
        self.best_order = metadata['best_order']
        self.best_seasonal_order = metadata['best_seasonal_order']
        self.train_info = metadata['train_info']
        self.is_fitted = metadata['is_fitted']
        
        # Update config if available
        if 'config' in metadata:
            self.config = TimeSeriesConfig(**metadata['config'])
        
        logger.info(f"ARIMA model loaded from {directory}")
        return self


class ProphetModel(BaseModel):
    """
    Facebook Prophet model implementation for time series forecasting with
    multiple seasonality patterns, changepoint detection, and uncertainty estimation.
    """
    
    def __init__(self, config: TimeSeriesConfig, name: str = "prophet", **kwargs: Any):
        """
        Initialize the Prophet model with the given configuration.
        
        Args:
            config: Configuration parameters for the model
            name: Unique identifier for this model instance
        """
        super().__init__(config, name=name, **kwargs)
        self.model = None
        self.config = config
        self.is_fitted = False
        self.train_info = {}
        self.scaler = MinMaxScaler()  # Prophet works better with unscaled data, but we keep this for consistency
    
    @timeit
    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> Dict[str, Any]:
        """
        Fit the Prophet model to the time series data.
        
        Args:
            X: Time series data (if y is None, X is assumed to be the target series)
            y: Target series if X contains exogenous variables
            
        Returns:
            Dict containing training results and metrics
        """
        try:
            # Use X as the target if y is not provided
            target = y if y is not None else X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X
            
            # Ensure target is a pandas Series with datetime index
            if not isinstance(target, pd.Series):
                if isinstance(target, pd.DataFrame):
                    target = target.iloc[:, 0]
                else:
                    target = pd.Series(target)
            
            # Prophet requires data in a specific format
            # Create a dataframe with 'ds' (dates) and 'y' (target values)
            df = pd.DataFrame({
                'ds': target.index if isinstance(target.index, pd.DatetimeIndex) else pd.date_range(
                    start='1970-01-01', periods=len(target), freq='D'
                ),
                'y': target.values
            })
            
            # Handle NaN values
            df = df.dropna()
            
            # Scale the data if needed (Prophet generally doesn't require scaling)
            if kwargs.get('scale_data', False):
                df['y'] = self.scaler.fit_transform(df[['y']])
            
            # Configure Prophet model with parameters from the config
            self.model = Prophet(
                seasonality_mode=self.config.seasonality_mode,
                changepoint_prior_scale=kwargs.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=kwargs.get('seasonality_prior_scale', 10.0),
                yearly_seasonality=kwargs.get('yearly_seasonality', 'auto'),
                weekly_seasonality=kwargs.get('weekly_seasonality', 'auto'),
                daily_seasonality=kwargs.get('daily_seasonality', 'auto')
            )
            
            # Add custom seasonalities if specified
            if self.config.seasonal_periods:
                for period in self.config.seasonal_periods:
                    # Add appropriate seasonalities based on period length
                    if period == 24:  # Hourly seasonality
                        self.model.add_seasonality(
                            name='hourly', period=1/24, fourier_order=5
                        )
                    elif period == 168:  # Weekly seasonality (if data is hourly)
                        self.model.add_seasonality(
                            name='weekly', period=7, fourier_order=3
                        )
                    elif period == 12:  # Monthly seasonality
                        self.model.add_seasonality(
                            name='monthly', period=30.5, fourier_order=5
                        )
            
            # Add additional regressors if provided
            if y is not None and isinstance(X, pd.DataFrame):
                for column in X.columns:
                    # Add each column as a regressor
                    self.model.add_regressor(column)
                
                # Combine regressors with the base dataframe
                for column in X.columns:
                    df[column] = X[column].values
            
            # Fit the model
            self.model.fit(df)
            self.is_fitted = True
            
            # Calculate in-sample metrics
            future = self.model.make_future_dataframe(
                periods=0,  # No future periods for in-sample evaluation
                freq='D'    # Use appropriate frequency
            )
            
            # Add regressors to future dataframe if needed
            if y is not None and isinstance(X, pd.DataFrame):
                for column in X.columns:
                    future[column] = X[column].values
            
            forecast = self.model.predict(future)
            
            # Extract predictions and calculate metrics
            predictions = forecast['yhat'].values
            
            # Inverse transform if scaled
            if kwargs.get('scale_data', False):
                predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                y_true = target.values
            else:
                y_true = df['y'].values
            
            # Calculate metrics
            mse = mean_squared_error(y_true, predictions[:len(y_true)])
            mae = mean_absolute_error(y_true, predictions[:len(y_true)])
            r2 = r2_score(y_true, predictions[:len(y_true)])
            
            self.train_info = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'parameters': {
                    'seasonality_mode': self.config.seasonality_mode,
                    'changepoint_prior_scale': kwargs.get('changepoint_prior_scale', 0.05),
                    'seasonality_prior_scale': kwargs.get('seasonality_prior_scale', 10.0)
                }
            }
            
            logger.info(f"Prophet model fitted successfully: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
            return self.train_info
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {str(e)}")
            raise ModelTrainingError(f"Failed to train Prophet model: {str(e)}")
    
    @timeit
    def predict(self, X: Union[pd.DataFrame, int], **kwargs) -> np.ndarray:
        """
        Generate predictions from the Prophet model.
        
        Args:
            X: Input data (can be exogenous variables, dataframe with 'ds' column, or steps to forecast)
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Model has not been fitted yet.")
        
        try:
            # Determine forecast horizon and frequency
            if isinstance(X, int):
                # X is the number of steps to forecast
                steps = X
                freq = kwargs.get('freq', 'D')  # Default to daily frequency
                future = self.model.make_future_dataframe(periods=steps, freq=freq)
            elif isinstance(X, pd.DataFrame) and 'ds' in X.columns:
                # X already contains the dates to forecast
                future = X
            else:
                # Default case: use config forecast horizon
                steps = kwargs.get('steps', self.config.forecast_horizon)
                freq = kwargs.get('freq', 'D')
                future = self.model.make_future_dataframe(periods=steps, freq=freq)
            
            # Add external regressors if provided
            if isinstance(X, pd.DataFrame) and len(X.columns) > 0:
                for column in X.columns:
                    if column != 'ds' and column in self.model.extra_regressors:
                        future[column] = X[column].values
            
            # Make predictions
            forecast = self.model.predict(future)
            
            # Determine which part of the forecast to return
            if kwargs.get('return_all_columns', False):
                # Return the full forecast dataframe
                return forecast
            elif kwargs.get('return_components', False):
                # Return the forecast components
                return self.model.predict_components(future)
            else:
                # Return only the point predictions for the requested period
                if isinstance(X, int) or (kwargs.get('steps', 0) > 0):
                    # Return only future predictions
                    predictions = forecast['yhat'].values[-steps:]
                else:
                    # Return all predictions
                    predictions = forecast['yhat'].values
                
                # Inverse transform if scaled
                if kwargs.get('scale_data', False):
                    predictions = self.scaler.inverse_transform(
                        predictions.reshape(-1, 1)
                    ).flatten()
                
                return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with Prophet model: {str(e)}")
            raise ModelPredictionError(f"Failed to generate predictions: {str(e)}")
    
    def save(self, directory: str) -> str:
        """
        Save the Prophet model to disk.
        
        Args:
            directory: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Cannot save unfitted model")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save model components
        model_path = os.path.join(directory, f"{self.name}_prophet.json")
        scaler_path = os.path.join(directory, f"{self.name}_scaler.pkl")
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        # Save the fitted model
        with open(model_path, 'w') as f:
            json.dump(model_to_json(self.model), f)
        
        # Save the scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save configuration and metadata
        metadata = {
            'train_info': self.train_info,
            'config': self.config.__dict__,
            'is_fitted': self.is_fitted
        }
        save_to_file(metadata, config_path)
        
        logger.info(f"Prophet model saved to {directory}")
        return directory
    
    def load(self, directory: str) -> 'ProphetModel':
        """
        Load the Prophet model from disk.
        
        Args:
            directory: Directory containing the saved model
            
        Returns:
            Loaded model instance
        """
        model_path = os.path.join(directory, f"{self.name}_prophet.json")
        scaler_path = os.path.join(directory, f"{self.name}_scaler.pkl")
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"Model file not found at {model_path}")
        
        # Load the fitted model
        with open(model_path, 'r') as f:
            self.model = model_from_json(json.load(f))
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load configuration and metadata
        metadata = load_from_file(config_path)
        self.train_info = metadata['train_info']
        self.is_fitted = metadata['is_fitted']
        
        # Update config if available
        if 'config' in metadata:
            self.config = TimeSeriesConfig(**metadata['config'])
        
        logger.info(f"Prophet model loaded from {directory}")
        return self


class LSTMModel(BaseModel):
    """
    Advanced LSTM model implementation for time series forecasting with
    multiple layers, regularization, and GPU acceleration.
    """
    
    def __init__(self, config: TimeSeriesConfig, name: str = "lstm", **kwargs: Any):
        """
        Initialize the LSTM model with the given configuration.
        
        Args:
            config: Configuration parameters for the model
            name: Unique identifier for this model instance
        """
        super().__init__(config, name=name, **kwargs)
        self.model = None
        self.config = config
        self.is_fitted = False
        self.train_info = {}
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # LSTM works better with normalized data
        
        # Check GPU availability
        self.has_gpu = tf.config.list_physical_devices('GPU')
        if self.has_gpu:
            # Limit memory growth to avoid OOM errors
            for gpu in tf.config.list_physical_devices('GPU'):
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except:
                    logger.warning("Unable to set memory growth for GPU")
            
            logger.info(f"GPU available for training: {gpu_info()}")
        else:
            logger.info("No GPU available, using CPU for training")
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build the LSTM model architecture based on configuration.
        
        Returns:
            Compiled Keras model
        """
        # Set up the model architecture
        model = Sequential()
        
        # Input shape: (sequence_length, n_features)
        input_shape = (self.config.sequence_length, self.config.n_features)
        
        # First LSTM layer
        return_sequences = len(self.config.hidden_units) > 1
        model.add(LSTM(
            units=self.config.hidden_units[0],
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences=return_sequences,
            input_shape=input_shape,
            recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5),
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5),
            bias_regularizer=l1_l2(l1=1e-5, l2=1e-5)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.config.dropout_rate))
        
        # Hidden LSTM layers
        for i, units in enumerate(self.config.hidden_units[1:]):
            return_sequences = i < len(self.config.hidden_units) - 2
            model.add(LSTM(
                units=units,
                activation='tanh',
                recurrent_activation='sigmoid',
                return_sequences=return_sequences,
                recurrent_regularizer=l1_l2(l1=1e-5, l2=1e-5),
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5),
                bias_regularizer=l1_l2(l1=1e-5, l2=1e-5)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.config.dropout_rate))
        
        # Output layer
        model.add(Dense(
            units=self.config.forecast_horizon,
            activation='linear'
        ))
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences and target values for LSTM training.
        
        Args:
            data: Input time series data
            
        Returns:
            Tuple of (X, y) with input sequences and target values
        """
        X, y = [], []
        
        for i in range(len(data) - self.config.sequence_length - self.config.forecast_horizon + 1):
            # Extract sequence
            seq = data[i:(i + self.config.sequence_length)]
            # Extract target (next forecast_horizon values)
            target = data[i + self.config.sequence_length:
                         i + self.config.sequence_length + self.config.forecast_horizon]
            
            X.append(seq)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    @timeit
    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> Dict[str, Any]:
        """
        Fit the LSTM model to the time series data.
        
        Args:
            X: Time series data (if y is None, X is assumed to be the target series)
            y: Target series if X contains exogenous variables
            
        Returns:
            Dict containing training results and metrics
        """
        try:
            # Use X as the target if y is not provided
            if y is not None:
                # Ensure y is a numpy array
                y_values = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.array(y)
                
                # Prepare features
                features = X.values if isinstance(X, pd.DataFrame) else np.array(X)
                
                # Scale the data
                features_scaled = self.scaler.fit_transform(features)
                
                # Reshape data for LSTM [samples, timesteps, features]
                if len(features_scaled.shape) == 1:
                    features_scaled = features_scaled.reshape(-1, 1)
                
                # This is multivariate forecasting
                self.config.n_features = features_scaled.shape[1]
                data = features_scaled
                
            else:
                # This is univariate forecasting
                target = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X
                
                # Ensure target is a numpy array
                target_values = target.values if isinstance(target, (pd.Series, pd.DataFrame)) else np.array(target)
                
                # Reshape for scaling if needed
                if len(target_values.shape) == 1:
                    target_values = target_values.reshape(-1, 1)
                
                # Scale the data
                data = self.scaler.fit_transform(target_values)
                self.config.n_features = data.shape[1]
            
            # Prepare sequences for LSTM
            X_train, y_train = self._prepare_sequences(data)
            
            # Reshape for LSTM input [samples, timesteps, features]
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], self.config.n_features))
            
            # Build the model
            self.model = self._build_model()
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Add ModelCheckpoint if directory specified
            if 'checkpoint_dir' in kwargs:
                os.makedirs(kwargs['checkpoint_dir'], exist_ok=True)
                callbacks.append(
                    ModelCheckpoint(
                        filepath=os.path.join(kwargs['checkpoint_dir'], f"{self.name}_best.h5"),
                        monitor='val_loss',
                        save_best_only=True
                    )
                )
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=kwargs.get('verbose', 1),
                shuffle=True
            )
            
            self.is_fitted = True
            
            # Evaluate the model
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_mae = history.history['mae'][-1]
            val_mae = history.history['val_mae'][-1]
            
            self.train_info = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'n_epochs': len(history.history['loss']),
                'min_val_loss': min(history.history['val_loss']),
                'min_val_loss_epoch': np.argmin(history.history['val_loss']) + 1
            }
            
            logger.info(
                f"LSTM model fitted successfully: "
                f"Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"MAE={train_mae:.4f}, Val MAE={val_mae:.4f}"
            )
            return self.train_info
            
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {str(e)}")
            raise ModelTrainingError(f"Failed to train LSTM model: {str(e)}")
    
    @timeit
    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """
        Generate predictions from the LSTM model.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Model has not been fitted yet.")
        
        try:
            # Handle different input types
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X_values = X.values
            else:
                X_values = np.array(X)
            
            # Handle recursive multi-step forecasting
            if kwargs.get('recursive', False) and X_values.shape[0] < self.config.sequence_length:
                return self._recursive_forecast(X_values, **kwargs)
            
            # Scale the input data
            X_scaled = self.scaler.transform(
                X_values.reshape(-1, self.config.n_features)
            )
            
            # Prepare sequences
            if X_scaled.shape[0] >= self.config.sequence_length:
                # Create sequences
                sequences = []
                for i in range(len(X_scaled) - self.config.sequence_length + 1):
                    seq = X_scaled[i:i+self.config.sequence_length]
                    sequences.append(seq)
                
                X_pred = np.array(sequences)
                X_pred = X_pred.reshape((X_pred.shape[0], self.config.sequence_length, self.config.n_features))
                
                # Make predictions
                y_pred = self.model.predict(X_pred)
                
                # Return only the last prediction if specified
                if kwargs.get('last_only', False):
                    y_pred = y_pred[-1]
                
                # Inverse transform predictions
                if y_pred.ndim == 3:  # Multiple samples, multiple timesteps, multiple features
                    # Reshape to 2D for inverse transform
                    y_pred_reshaped = y_pred.reshape(-1, y_pred.shape[-1])
                    y_pred_inv = self.scaler.inverse_transform(y_pred_reshaped)
                    # Reshape back to original shape
                    y_pred = y_pred_inv.reshape(y_pred.shape)
                elif y_pred.ndim == 2:  # Multiple samples, multiple timesteps
                    # For each sample, inverse transform the timesteps
                    y_pred_inv = np.array([
                        self.scaler.inverse_transform(p.reshape(-1, 1)).flatten()
                        for p in y_pred
                    ])
                    y_pred = y_pred_inv
                else:  # Single sample, multiple timesteps
                    y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                
                return y_pred
            else:
                # Not enough data for a sequence
                logger.warning(
                    f"Input data length {X_scaled.shape[0]} is less than "
                    f"sequence_length {self.config.sequence_length}. "
                    f"Using recursive forecasting."
                )
                return self._recursive_forecast(X_values, **kwargs)
            
        except Exception as e:
            logger.error(f"Error making predictions with LSTM model: {str(e)}")
            raise ModelPredictionError(f"Failed to generate predictions: {str(e)}")
    
    def _recursive_forecast(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate forecasts recursively when input data is shorter than sequence_length.
        
        Args:
            X: Input data
            
        Returns:
            Predicted values
        """
        # Scale the input data
        X_scaled = self.scaler.transform(X.reshape(-1, self.config.n_features))
        
        # Determine forecast horizon
        horizon = kwargs.get('steps', self.config.forecast_horizon)
        
        # Pad the input sequence if needed
        if X_scaled.shape[0] < self.config.sequence_length:
            # Pad with zeros (or other padding strategy)
            padding_length = self.config.sequence_length - X_scaled.shape[0]
            padding = np.zeros((padding_length, self.config.n_features))
            X_padded = np.vstack([padding, X_scaled])
        else:
            # Take the last sequence_length points
            X_padded = X_scaled[-self.config.sequence_length:]
        
        # Initialize input sequence
        input_seq = X_padded.copy()
        
        # Generate forecasts one step at a time
        forecasts = []
        for i in range(horizon):
            # Reshape for LSTM input [1, sequence_length, n_features]
            X_pred = input_seq.reshape(1, self.config.sequence_length, self.config.n_features)
            
            # Predict next step
            y_pred = self.model.predict(X_pred, verbose=0)
            
            # Extract the first forecast point (model outputs forecast_horizon points)
            next_point = y_pred[0, 0]
            forecasts.append(next_point)
            
            # Update input sequence for next iteration
            input_seq = np.vstack([input_seq[1:], next_point.reshape(1, -1)])
        
        # Convert forecasts to array and inverse transform
        forecasts = np.array(forecasts).reshape(-1, self.config.n_features)
        forecasts_inv = self.scaler.inverse_transform(forecasts)
        
        return forecasts_inv.flatten() if self.config.n_features == 1 else forecasts_inv
    
    def save(self, directory: str) -> str:
        """
        Save the LSTM model to disk.
        
        Args:
            directory: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Cannot save unfitted model")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save model components
        model_path = os.path.join(directory, f"{self.name}_lstm.h5")
        scaler_path = os.path.join(directory, f"{self.name}_scaler.pkl")
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        # Save the fitted model
        self.model.save(model_path)
        
        # Save the scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save configuration and metadata
        metadata = {
            'train_info': self.train_info,
            'config': self.config.__dict__,
            'is_fitted': self.is_fitted
        }
        save_to_file(metadata, config_path)
        
        logger.info(f"LSTM model saved to {directory}")
        return directory
    
    def load(self, directory: str) -> 'LSTMModel':
        """
        Load the LSTM model from disk.
        
        Args:
            directory: Directory containing the saved model
            
        Returns:
            Loaded model instance
        """
        model_path = os.path.join(directory, f"{self.name}_lstm.h5")
        scaler_path = os.path.join(directory, f"{self.name}_scaler.pkl")
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"Model file not found at {model_path}")
        
        # Load the fitted model
        self.model = load_model(model_path)
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load configuration and metadata
        metadata = load_from_file(config_path)
        self.train_info = metadata['train_info']
        self.is_fitted = metadata['is_fitted']
        
        # Update config if available
        if 'config' in metadata:
            self.config = TimeSeriesConfig(**metadata['config'])
        
        logger.info(f"LSTM model loaded from {directory}")
        return self


class TCNModel(BaseModel):
    """
    Temporal Convolutional Network implementation for time series forecasting,
    which is excellent for capturing long-range dependencies in financial data.
    """
    
    def __init__(self, config: TimeSeriesConfig, name: str = "tcn", **kwargs: Any):
        """
        Initialize the TCN model with the given configuration.
        
        Args:
            config: Configuration parameters for the model
            name: Unique identifier for this model instance
        """
        super().__init__(config, name=name, **kwargs)
        self.model = None
        self.config = config
        self.is_fitted = False
        self.train_info = {}
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Special TCN parameters
        self.tcn_params = {
            'nb_filters': 64,
            'kernel_size': 3,
            'nb_stacks': 1,
            'dilations': [1, 2, 4, 8, 16, 32],
            'padding': 'causal',
            'use_skip_connections': True,
            'dropout_rate': config.dropout_rate,
            'return_sequences': False
        }
    
    def _residual_block(self, x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.0):
        """
        TCN residual block implementation.
        
        Args:
            x: Input tensor
            dilation_rate: Dilation rate for this block
            nb_filters: Number of convolutional filters
            kernel_size: Size of the convolutional kernel
            padding: Type of padding
            dropout_rate: Dropout rate
            
        Returns:
            Output tensor
        """
        # First dilated convolution
        prev_x = x
        x = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=dilation_rate, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # Second dilated convolution
        x = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=dilation_rate, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # Skip connection
        if prev_x.shape[-1] != nb_filters:
            prev_x = Conv1D(nb_filters, 1, padding='same')(prev_x)
        
        return Add()([prev_x, x])
    
    def _build_tcn_model(self) -> tf.keras.Model:
        """
        Build the TCN model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Extract parameters
        nb_filters = self.tcn_params['nb_filters']
        kernel_size = self.tcn_params['kernel_size']
        nb_stacks = self.tcn_params['nb_stacks']
        dilations = self.tcn_params['dilations']
        padding = self.tcn_params['padding']
        use_skip_connections = self.tcn_params['use_skip_connections']
        dropout_rate = self.tcn_params['dropout_rate']
        return_sequences = self.tcn_params['return_sequences']
        
        # Define input shape
        input_shape = (self.config.sequence_length, self.config.n_features)
        inputs = Input(shape=input_shape)
        
        x = inputs
        skip_connections = []
        
        # Build TCN architecture
        for s in range(nb_stacks):
            for d in dilations:
                x = self._residual_block(
                    x, d, nb_filters, kernel_size, padding, dropout_rate
                )
                if use_skip_connections:
                    skip_connections.append(x)
        
        # Add skip connections if enabled
        if use_skip_connections:
            if len(skip_connections) > 0:
                x = Add()(skip_connections)
        
        # Final processing
        if not return_sequences:
            x = Lambda(lambda tt: tt[:, -1, :])(x)
        
        # Output layer
        outputs = Dense(self.config.forecast_horizon)(x)
        
        # Build model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    @timeit
    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> Dict[str, Any]:
        """
        Fit the TCN model to the time series data.
        
        Args:
            X: Time series data (if y is None, X is assumed to be the target series)
            y: Target series if X contains exogenous variables
            
        Returns:
            Dict containing training results and metrics
        """
        try:
            # Prepare data similarly to LSTM model
            if y is not None:
                # Multivariate forecasting
                y_values = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.array(y)
                features = X.values if isinstance(X, pd.DataFrame) else np.array(X)
                features_scaled = self.scaler.fit_transform(features)
                self.config.n_features = features_scaled.shape[1]
                data = features_scaled
            else:
                # Univariate forecasting
                target = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X
                target_values = target.values if isinstance(target, (pd.Series, pd.DataFrame)) else np.array(target)
                
                # Reshape for scaling if needed
                if len(target_values.shape) == 1:
                    target_values = target_values.reshape(-1, 1)
                
                data = self.scaler.fit_transform(target_values)
                self.config.n_features = data.shape[1]
            
            # Prepare sequences for TCN
            X_train, y_train = [], []
            
            for i in range(len(data) - self.config.sequence_length - self.config.forecast_horizon + 1):
                # Extract sequence
                seq = data[i:(i + self.config.sequence_length)]
                # Extract target (next forecast_horizon values)
                target = data[i + self.config.sequence_length:
                             i + self.config.sequence_length + self.config.forecast_horizon]
                
                X_train.append(seq)
                y_train.append(target)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # For TCN, reshape the target if needed
            if self.config.n_features > 1:
                y_train = y_train.reshape(y_train.shape[0], -1)
            else:
                y_train = y_train.reshape(y_train.shape[0], self.config.forecast_horizon)
            
            # Apply TCN-specific parameters from kwargs
            for param in self.tcn_params:
                if param in kwargs:
                    self.tcn_params[param] = kwargs[param]
            
            # Build the TCN model
            self.model = self._build_tcn_model()
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Add ModelCheckpoint if directory specified
            if 'checkpoint_dir' in kwargs:
                os.makedirs(kwargs['checkpoint_dir'], exist_ok=True)
                callbacks.append(
                    ModelCheckpoint(
                        filepath=os.path.join(kwargs['checkpoint_dir'], f"{self.name}_best.h5"),
                        monitor='val_loss',
                        save_best_only=True
                    )
                )
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=kwargs.get('verbose', 1),
                shuffle=True
            )
            
            self.is_fitted = True
            
            # Evaluate the model
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_mae = history.history['mae'][-1]
            val_mae = history.history['val_mae'][-1]
            
            self.train_info = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'n_epochs': len(history.history['loss']),
                'min_val_loss': min(history.history['val_loss']),
                'min_val_loss_epoch': np.argmin(history.history['val_loss']) + 1,
                'tcn_params': self.tcn_params
            }
            
            logger.info(
                f"TCN model fitted successfully: "
                f"Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"MAE={train_mae:.4f}, Val MAE={val_mae:.4f}"
            )
            return self.train_info
            
        except Exception as e:
            logger.error(f"Error fitting TCN model: {str(e)}")
            raise ModelTrainingError(f"Failed to train TCN model: {str(e)}")
    
    @timeit
    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """
        Generate predictions from the TCN model.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Model has not been fitted yet.")
        
        try:
            # Handle prediction similar to LSTM
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X_values = X.values
            else:
                X_values = np.array(X)
            
            # Handle recursive multi-step forecasting
            if kwargs.get('recursive', False) and X_values.shape[0] < self.config.sequence_length:
                return self._recursive_forecast(X_values, **kwargs)
            
            # Scale the input data
            X_scaled = self.scaler.transform(
                X_values.reshape(-1, self.config.n_features)
            )
            
            # Prepare sequences
            if X_scaled.shape[0] >= self.config.sequence_length:
                # Create sequences
                sequences = []
                for i in range(len(X_scaled) - self.config.sequence_length + 1):
                    seq = X_scaled[i:i+self.config.sequence_length]
                    sequences.append(seq)
                
                X_pred = np.array(sequences)
                
                # Make predictions
                y_pred = self.model.predict(X_pred)
                
                # Return only the last prediction if specified
                if kwargs.get('last_only', False):
                    y_pred = y_pred[-1]
                
                # Reshape predictions for inverse transform
                if self.config.n_features > 1:
                    # For multivariate output, reshape to appropriate dimensions
                    pred_length = y_pred.shape[1] // self.config.n_features
                    y_pred_reshaped = y_pred.reshape(-1, self.config.n_features)
                    y_pred_inv = self.scaler.inverse_transform(y_pred_reshaped)
                    # Reshape back to original shape
                    y_pred = y_pred_inv.reshape(-1, pred_length, self.config.n_features)
                else:
                    # For univariate output, reshape for inverse transform
                    # Each row represents forecast_horizon steps
                    original_shape = y_pred.shape
                    n_samples = original_shape[0]
                    
                    # Create inverse transform of each prediction point
                    y_pred_inv = np.array([
                        self.scaler.inverse_transform(y.reshape(-1, 1)).flatten()
                        for y in y_pred
                    ])
                    
                    y_pred = y_pred_inv
                
                return y_pred
            else:
                # Not enough data for a sequence
                logger.warning(
                    f"Input data length {X_scaled.shape[0]} is less than "
                    f"sequence_length {self.config.sequence_length}. "
                    f"Using recursive forecasting."
                )
                return self._recursive_forecast(X_values, **kwargs)
            
        except Exception as e:
            logger.error(f"Error making predictions with TCN model: {str(e)}")
            raise ModelPredictionError(f"Failed to generate predictions: {str(e)}")
    
    def _recursive_forecast(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate forecasts recursively when input data is shorter than sequence_length.
        Implementation is similar to LSTM's recursive forecasting.
        
        Args:
            X: Input data
            
        Returns:
            Predicted values
        """
        # Implementation similar to LSTM's recursive forecast
        # Scale the input data
        X_scaled = self.scaler.transform(X.reshape(-1, self.config.n_features))
        
        # Determine forecast horizon
        horizon = kwargs.get('steps', self.config.forecast_horizon)
        
        # Pad the input sequence if needed
        if X_scaled.shape[0] < self.config.sequence_length:
            # Pad with zeros (or other padding strategy)
            padding_length = self.config.sequence_length - X_scaled.shape[0]
            padding = np.zeros((padding_length, self.config.n_features))
            X_padded = np.vstack([padding, X_scaled])
        else:
            # Take the last sequence_length points
            X_padded = X_scaled[-self.config.sequence_length:]
        
        # Initialize input sequence
        input_seq = X_padded.copy()
        
        # Generate forecasts one step at a time
        forecasts = []
        for i in range(horizon):
            # Reshape for TCN input [1, sequence_length, n_features]
            X_pred = input_seq.reshape(1, self.config.sequence_length, self.config.n_features)
            
            # Predict next step
            y_pred = self.model.predict(X_pred, verbose=0)
            
            # Extract the first forecast point (model outputs forecast_horizon points)
            next_point = y_pred[0, 0] if self.config.forecast_horizon > 1 else y_pred[0]
            forecasts.append(next_point)
            
            # Update input sequence for next iteration
            input_seq = np.vstack([input_seq[1:], next_point.reshape(1, -1) if hasattr(next_point, 'shape') else np.array([next_point]).reshape(1, -1)])
        
        # Convert forecasts to array and inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts_inv = self.scaler.inverse_transform(forecasts)
        
        return forecasts_inv.flatten()
    
    def save(self, directory: str) -> str:
        """
        Save the TCN model to disk.
        
        Args:
            directory: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Cannot save unfitted model")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save model components
        model_path = os.path.join(directory, f"{self.name}_tcn.h5")
        scaler_path = os.path.join(directory, f"{self.name}_scaler.pkl")
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        # Save the fitted model
        self.model.save(model_path)
        
        # Save the scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save configuration and metadata
        metadata = {
            'train_info': self.train_info,
            'config': self.config.__dict__,
            'tcn_params': self.tcn_params,
            'is_fitted': self.is_fitted
        }
        save_to_file(metadata, config_path)
        
        logger.info(f"TCN model saved to {directory}")
        return directory
    
    def load(self, directory: str) -> 'TCNModel':
        """
        Load the TCN model from disk.
        
        Args:
            directory: Directory containing the saved model
            
        Returns:
            Loaded model instance
        """
        model_path = os.path.join(directory, f"{self.name}_tcn.h5")
        scaler_path = os.path.join(directory, f"{self.name}_scaler.pkl")
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"Model file not found at {model_path}")
        
        # Load the fitted model
        self.model = load_model(model_path)
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load configuration and metadata
        metadata = load_from_file(config_path)
        self.train_info = metadata['train_info']
        self.is_fitted = metadata['is_fitted']
        
        # Update TCN params if available
        if 'tcn_params' in metadata:
            self.tcn_params = metadata['tcn_params']
        
        # Update config if available
        if 'config' in metadata:
            self.config = TimeSeriesConfig(**metadata['config'])
        
        logger.info(f"TCN model loaded from {directory}")
        return self


class AttentionModel(BaseModel):
    """
    Transformer-based model with self-attention for time series forecasting,
    effective at capturing long-range dependencies and multi-timeframe patterns.
    """
    
    def __init__(self, config: TimeSeriesConfig, name: str = "attention", **kwargs: Any):
        """
        Initialize the Attention model with the given configuration.
        
        Args:
            config: Configuration parameters for the model
            name: Unique identifier for this model instance
        """
        super().__init__(config, name=name, **kwargs)
        self.model = None
        self.config = config
        self.is_fitted = False
        self.train_info = {}
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Attention-specific parameters
        self.attention_params = {
            'num_heads': 4,
            'key_dim': 16,
            'ff_dim': 128,
            'num_transformer_blocks': 2,
            'dropout_rate': config.dropout_rate,
            'use_mask': True
        }
    
    def _transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """
        Create a transformer encoder block.
        
        Args:
            inputs: Input tensor
            head_size: Size of each attention head
            num_heads: Number of attention heads
            ff_dim: Hidden layer size in feed forward network
            dropout: Dropout rate
            
        Returns:
            Output tensor
        """
        # Multi-head attention
        attention_output = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        
        # Skip connection & layer normalization
        attention_output = Dropout(dropout)(attention_output)
        attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = Dense(ff_dim, activation='relu')(attention_output)
        ffn_output = Dense(inputs.shape[-1])(ffn_output)
        
        # Skip connection & layer normalization
        ffn_output = Dropout(dropout)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
    
    def _build_attention_model(self) -> tf.keras.Model:
        """
        Build the Transformer/Attention model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Extract parameters
        num_heads = self.attention_params['num_heads']
        key_dim = self.attention_params['key_dim']
        ff_dim = self.attention_params['ff_dim']
        num_transformer_blocks = self.attention_params['num_transformer_blocks']
        dropout_rate = self.attention_params['dropout_rate']
        
        # Define input shape
        input_shape = (self.config.sequence_length, self.config.n_features)
        inputs = Input(shape=input_shape)
        
        # Create positional encoding for time steps
        positions = tf.range(start=0, limit=self.config.sequence_length, delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=self.config.sequence_length, 
            output_dim=self.config.n_features
        )(positions)
        
        # Add positional encoding to inputs
        x = inputs + position_embedding
        
        # Transformer blocks
        for _ in range(num_transformer_blocks):
            x = self._transformer_encoder(
                x, key_dim, num_heads, ff_dim, dropout_rate
            )
        
        # Global average pooling for sequence reduction
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output layer
        outputs = Dense(self.config.forecast_horizon)(x)
        
        # Build model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    @timeit
    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> Dict[str, Any]:
        """
        Fit the Attention model to the time series data.
        
        Args:
            X: Time series data (if y is None, X is assumed to be the target series)
            y: Target series if X contains exogenous variables
            
        Returns:
            Dict containing training results and metrics
        """
        try:
            # Prepare data similarly to LSTM model
            if y is not None:
                # Multivariate forecasting
                y_values = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else np.array(y)
                features = X.values if isinstance(X, pd.DataFrame) else np.array(X)
                features_scaled = self.scaler.fit_transform(features)
                self.config.n_features = features_scaled.shape[1]
                data = features_scaled
            else:
                # Univariate forecasting
                target = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X
                target_values = target.values if isinstance(target, (pd.Series, pd.DataFrame)) else np.array(target)
                
                # Reshape for scaling if needed
                if len(target_values.shape) == 1:
                    target_values = target_values.reshape(-1, 1)
                
                data = self.scaler.fit_transform(target_values)
                self.config.n_features = data.shape[1]
            
            # Prepare sequences for Attention model
            X_train, y_train = [], []
            
            for i in range(len(data) - self.config.sequence_length - self.config.forecast_horizon + 1):
                # Extract sequence
                seq = data[i:(i + self.config.sequence_length)]
                # Extract target (next forecast_horizon values)
                target = data[i + self.config.sequence_length:
                             i + self.config.sequence_length + self.config.forecast_horizon]
                
                X_train.append(seq)
                y_train.append(target)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # For Attention, reshape the target if needed
            if self.config.n_features > 1:
                y_train = y_train.reshape(y_train.shape[0], -1)
            else:
                y_train = y_train.reshape(y_train.shape[0], self.config.forecast_horizon)
            
            # Apply Attention-specific parameters from kwargs
            for param in self.attention_params:
                if param in kwargs:
                    self.attention_params[param] = kwargs[param]
            
            # Build the Attention model
            self.model = self._build_attention_model()
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Add ModelCheckpoint if directory specified
            if 'checkpoint_dir' in kwargs:
                os.makedirs(kwargs['checkpoint_dir'], exist_ok=True)
                callbacks.append(
                    ModelCheckpoint(
                        filepath=os.path.join(kwargs['checkpoint_dir'], f"{self.name}_best.h5"),
                        monitor='val_loss',
                        save_best_only=True
                    )
                )
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=kwargs.get('verbose', 1),
                shuffle=True
            )
            
            self.is_fitted = True
            
            # Evaluate the model
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            train_mae = history.history['mae'][-1]
            val_mae = history.history['val_mae'][-1]
            
            self.train_info = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'n_epochs': len(history.history['loss']),
                'min_val_loss': min(history.history['val_loss']),
                'min_val_loss_epoch': np.argmin(history.history['val_loss']) + 1,
                'attention_params': self.attention_params
            }
            
            logger.info(
                f"Attention model fitted successfully: "
                f"Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"MAE={train_mae:.4f}, Val MAE={val_mae:.4f}"
            )
            return self.train_info
            
        except Exception as e:
            logger.error(f"Error fitting Attention model: {str(e)}")
            raise ModelTrainingError(f"Failed to train Attention model: {str(e)}")
    
    @timeit
    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """
        Generate predictions from the Attention model.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Model has not been fitted yet.")
        
        try:
            # Handle prediction similar to LSTM
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X_values = X.values
            else:
                X_values = np.array(X)
            
            # Scale the input data
            X_scaled = self.scaler.transform(
                X_values.reshape(-1, self.config.n_features)
            )
            
            # Prepare sequences
            if X_scaled.shape[0] >= self.config.sequence_length:
                # Create sequences
                sequences = []
                for i in range(len(X_scaled) - self.config.sequence_length + 1):
                    seq = X_scaled[i:i+self.config.sequence_length]
                    sequences.append(seq)
                
                X_pred = np.array(sequences)
                
                # Make predictions
                y_pred = self.model.predict(X_pred)
                
                # Return only the last prediction if specified
                if kwargs.get('last_only', False):
                    y_pred = y_pred[-1]
                
                # Reshape predictions for inverse transform
                if self.config.n_features > 1:
                    # For multivariate output, reshape to appropriate dimensions
                    pred_length = y_pred.shape[1] // self.config.n_features
                    y_pred_reshaped = y_pred.reshape(-1, self.config.n_features)
                    y_pred_inv = self.scaler.inverse_transform(y_pred_reshaped)
                    # Reshape back to original shape
                    y_pred = y_pred_inv.reshape(-1, pred_length, self.config.n_features)
                else:
                    # For univariate output, reshape for inverse transform
                    # Each row represents forecast_horizon steps
                    original_shape = y_pred.shape
                    
                    # Create inverse transform of each prediction point
                    y_pred_inv = np.array([
                        self.scaler.inverse_transform(y.reshape(-1, 1)).flatten()
                        for y in y_pred
                    ])
                    
                    y_pred = y_pred_inv
                
                return y_pred
            else:
                # Not enough data for a sequence
                logger.warning(
                    f"Input data length {X_scaled.shape[0]} is less than "
                    f"sequence_length {self.config.sequence_length}. "
                    f"Cannot make predictions without sufficient history."
                )
                raise ValueError("Insufficient data for prediction with Attention model")
            
        except Exception as e:
            logger.error(f"Error making predictions with Attention model: {str(e)}")
            raise ModelPredictionError(f"Failed to generate predictions: {str(e)}")
    
    def save(self, directory: str) -> str:
        """
        Save the Attention model to disk.
        
        Args:
            directory: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Cannot save unfitted model")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save model components
        model_path = os.path.join(directory, f"{self.name}_attention.h5")
        scaler_path = os.path.join(directory, f"{self.name}_scaler.pkl")
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        # Save the fitted model
        self.model.save(model_path)
        
        # Save the scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save configuration and metadata
        metadata = {
            'train_info': self.train_info,
            'config': self.config.__dict__,
            'attention_params': self.attention_params,
            'is_fitted': self.is_fitted
        }
        save_to_file(metadata, config_path)
        
        logger.info(f"Attention model saved to {directory}")
        return directory
    
    def load(self, directory: str) -> 'AttentionModel':
        """
        Load the Attention model from disk.
        
        Args:
            directory: Directory containing the saved model
            
        Returns:
            Loaded model instance
        """
        model_path = os.path.join(directory, f"{self.name}_attention.h5")
        scaler_path = os.path.join(directory, f"{self.name}_scaler.pkl")
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"Model file not found at {model_path}")
        
        # Load the fitted model
        self.model = load_model(model_path)
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load configuration and metadata
        metadata = load_from_file(config_path)
        self.train_info = metadata['train_info']
        self.is_fitted = metadata['is_fitted']
        
        # Update attention params if available
        if 'attention_params' in metadata:
            self.attention_params = metadata['attention_params']
        
        # Update config if available
        if 'config' in metadata:
            self.config = TimeSeriesConfig(**metadata['config'])
        
        logger.info(f"Attention model loaded from {directory}")
        return self


class EnsembleTimeSeriesModel(BaseModel):
    """
    Ensemble model that combines multiple time series models for improved prediction accuracy,
    providing robustness and uncertainty estimation.
    """
    
    def __init__(self, config: TimeSeriesConfig, name: str = "ensemble_ts", **kwargs: Any):
        """
        Initialize the Ensemble model with the given configuration.
        
        Args:
            config: Configuration parameters for the model
            name: Unique identifier for this model instance
        """
        super().__init__(config, name=name, **kwargs)
        self.models = {}
        self.weights = {}
        self.config = config
        self.is_fitted = False
        self.train_info = {}
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def add_model(self, model_type: str, weight: float = 1.0, **kwargs) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model_type: Type of model to add ('arima', 'prophet', 'lstm', 'tcn', 'attention')
            weight: Weight for this model in the ensemble
            **kwargs: Additional parameters for the model
        """
        model_name = f"{self.name}_{model_type}_{len(self.models)}"
        
        if model_type.lower() == 'arima':
            model = ARIMAModel(self.config, name=model_name)
        elif model_type.lower() == 'prophet':
            model = ProphetModel(self.config, name=model_name)
        elif model_type.lower() == 'lstm':
            model = LSTMModel(self.config, name=model_name)
        elif model_type.lower() == 'tcn':
            model = TCNModel(self.config, name=model_name)
        elif model_type.lower() == 'attention':
            model = AttentionModel(self.config, name=model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.models[model_name] = model
        self.weights[model_name] = weight
        
        logger.info(f"Added {model_type} model to ensemble with weight {weight}")
    
    @timeit
    def fit(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> Dict[str, Any]:
        """
        Fit all models in the ensemble to the time series data.
        
        Args:
            X: Time series data (if y is None, X is assumed to be the target series)
            y: Target series if X contains exogenous variables
            
        Returns:
            Dict containing training results and metrics
        """
        if not self.models:
            raise ValueError("No models in the ensemble. Use add_model() to add models.")
        
        try:
            # Prepare the data (scaling is done inside each model's fit method)
            results = {}
            
            # Train each model in the ensemble
            for model_name, model in self.models.items():
                logger.info(f"Training ensemble model: {model_name}")
                model_results = model.fit(X, y, **kwargs)
                results[model_name] = model_results
            
            # Update weights based on validation performance if requested
            if kwargs.get('auto_weight', False):
                self._update_weights_by_performance()
            
            self.is_fitted = True
            self.train_info = {
                'model_results': results,
                'weights': self.weights,
                'ensemble_size': len(self.models)
            }
            
            logger.info(f"Ensemble model fitted successfully with {len(self.models)} models")
            return self.train_info
            
        except Exception as e:
            logger.error(f"Error fitting ensemble model: {str(e)}")
            raise ModelTrainingError(f"Failed to train ensemble model: {str(e)}")
    
    def _update_weights_by_performance(self) -> None:
        """
        Update model weights based on validation performance.
        Better performing models get higher weights.
        """
        # Extract validation metrics (lower is better)
        val_metrics = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'train_info') and 'val_loss' in model.train_info:
                val_metrics[model_name] = model.train_info['val_loss']
            elif hasattr(model, 'train_info') and 'val_mae' in model.train_info:
                val_metrics[model_name] = model.train_info['val_mae']
            else:
                # Use MSE as fallback
                val_metrics[model_name] = model.train_info.get('mse', 1.0)
        
        # Calculate weights inversely proportional to validation metric
        # (lower metric = higher weight)
        total = sum(1.0 / (metric + 1e-10) for metric in val_metrics.values())
        for model_name, metric in val_metrics.items():
            self.weights[model_name] = (1.0 / (metric + 1e-10)) / total
        
        logger.info(f"Updated ensemble weights based on performance: {self.weights}")
    
    @timeit
    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """
        Generate predictions from the ensemble by combining predictions from all models.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Ensemble has not been fitted yet.")
        
        try:
            all_predictions = {}
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(X, **kwargs)
                    all_predictions[model_name] = pred
                except Exception as e:
                    logger.warning(f"Error in model {model_name} prediction: {str(e)}. Skipping this model.")
            
            if not all_predictions:
                raise ModelPredictionError("All ensemble models failed to generate predictions")
            
            # Normalize prediction shapes if needed
            self._normalize_prediction_shapes(all_predictions)
            
            # Weight and combine predictions
            weighted_preds = []
            total_weight = 0.0
            
            for model_name, pred in all_predictions.items():
                weight = self.weights[model_name]
                weighted_preds.append(pred * weight)
                total_weight += weight
            
            # Compute weighted average
            ensemble_pred = sum(weighted_preds) / total_weight
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making predictions with ensemble model: {str(e)}")
            raise ModelPredictionError(f"Failed to generate ensemble predictions: {str(e)}")
    
    def _normalize_prediction_shapes(self, predictions: Dict[str, np.ndarray]) -> None:
        """
        Ensure all predictions have the same shape by truncating or padding.
        
        Args:
            predictions: Dictionary of model predictions
        """
        # Find the minimum length across all predictions
        min_length = min(len(pred) for pred in predictions.values())
        
        # Truncate predictions to the minimum length
        for model_name in predictions:
            predictions[model_name] = predictions[model_name][:min_length]
    
    def save(self, directory: str) -> str:
        """
        Save the ensemble model to disk by saving each component model.
        
        Args:
            directory: Directory to save the models
            
        Returns:
            Path to the saved models
        """
        if not self.is_fitted:
            raise ModelNotFoundError("Cannot save unfitted ensemble")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save each model in a subdirectory
        model_paths = {}
        for model_name, model in self.models.items():
            model_dir = os.path.join(directory, model_name)
            os.makedirs(model_dir, exist_ok=True)
            model.save(model_dir)
            model_paths[model_name] = model_dir
        
        # Save ensemble configuration and metadata
        config_path = os.path.join(directory, f"{self.name}_config.json")
        metadata = {
            'train_info': self.train_info,
            'config': self.config.__dict__,
            'weights': self.weights,
            'model_paths': model_paths,
            'is_fitted': self.is_fitted
        }
        save_to_file(metadata, config_path)
        
        logger.info(f"Ensemble model saved to {directory}")
        return directory
    
    def load(self, directory: str) -> 'EnsembleTimeSeriesModel':
        """
        Load the ensemble model from disk.
        
        Args:
            directory: Directory containing the saved model
            
        Returns:
            Loaded ensemble model instance
        """
        config_path = os.path.join(directory, f"{self.name}_config.json")
        
        if not os.path.exists(config_path):
            raise ModelNotFoundError(f"Ensemble configuration not found at {config_path}")
        
        # Load configuration and metadata
        metadata = load_from_file(config_path)
        self.train_info = metadata['train_info']
        self.weights = metadata['weights']
        self.is_fitted = metadata['is_fitted']
        
        # Update config if available
        if 'config' in metadata:
            self.config = TimeSeriesConfig(**metadata['config'])
        
        # Load each model
        self.models = {}
        model_paths = metadata.get('model_paths', {})
        
        for model_name, model_dir in model_paths.items():
            # Determine model type from name
            model_type = model_name.split('_')[1].lower()
            
            if model_type == 'arima':
                model = ARIMAModel(self.config, name=model_name)
            elif model_type == 'prophet':
                model = ProphetModel(self.config, name=model_name)
            elif model_type == 'lstm':
                model = LSTMModel(self.config, name=model_name)
            elif model_type == 'tcn':
                model = TCNModel(self.config, name=model_name)
            elif model_type == 'attention':
                model = AttentionModel(self.config, name=model_name)
            else:
                logger.warning(f"Unknown model type: {model_type}, skipping")
                continue
            
            # Load the model
            model.load(model_dir)
            self.models[model_name] = model
        
        logger.info(f"Ensemble model loaded from {directory} with {len(self.models)} models")
        return self


def create_time_series_model(model_type: str, config: TimeSeriesConfig, **kwargs: Any) -> BaseModel:
    """Factory function to create a time series model by type."""
    model_type = model_type.lower()
    model_map = {
        "arima": ARIMAModel,
        "prophet": ProphetModel,
        "lstm": LSTMModel,
        "tcn": TCNModel,
        "attention": AttentionModel,
    }
    model_cls = model_map.get(model_type)
    if not model_cls:
        raise ModelNotFoundError(f"Unknown time series model: {model_type}")
    return model_cls(config, **kwargs)
