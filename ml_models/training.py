#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
ML Models - Training Service

This module handles training and retraining of machine learning models
with advanced optimization, hardware acceleration, and training pipelines.
"""

import os
import time
import json
import uuid
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import pickle
import joblib
import tempfile
import shutil
from pathlib import Path

# ML Libraries
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import (
        layers,
        models,
        optimizers,
        callbacks,
        regularizers,
    )
except Exception:  # pragma: no cover - optional dependency
    tf = None  # type: ignore
    layers = models = optimizers = callbacks = regularizers = None  # type: ignore
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    DataLoader = TensorDataset = None  # type: ignore
    TORCH_AVAILABLE = False
import sklearn
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from ml_models.hyperopt import HyperparameterOptimizer

# Internal imports
from config import Config
from common.logger import get_logger
from common.exceptions import (
    ModelTrainingError, DataValidationError, ResourceExhaustionError,
    HyperparameterOptimizationError, ModelPersistenceError, GPUNotAvailableError
)
from common.utils import create_timeframes, calculate_metrics, validate_data
from common.constants import ML_MODEL_TYPES, FEATURE_IMPORTANCE_METHODS
from common.metrics import MetricsCollector
from common.async_utils import run_in_thread_pool
from common.redis_client import RedisClient

from data_storage.market_data import MarketDataRepository
from feature_service.feature_extraction import FeatureExtractor

from ml_models.models.regression import create_regression_model
from ml_models.models.classification import create_classification_model
from ml_models.models.time_series import create_time_series_model
from ml_models.preprocessing.scaling import get_scaler
from ml_models.preprocessing.encoding import encode_features
from ml_models.preprocessing.sampling import balance_dataset
from ml_models.hardware.gpu import setup_gpu, get_gpu_memory_usage
from ml_models.feature_importance import calculate_feature_importance

# Configure logger
logger = get_logger("ml_models.training")

try:
    from ml_models.models.ensemble import create_ensemble_model
except Exception as e:  # pragma: no cover - optional dependency
    create_ensemble_model = None  # type: ignore
    logger.warning(f"Ensemble models not available: {e}")

try:
    from ml_models.models.deep_learning import create_deep_learning_model
except Exception as e:  # pragma: no cover - optional dependency
    create_deep_learning_model = None  # type: ignore
    logger.warning(f"Deep learning models not available: {e}")

class ModelTrainer:
    """
    Advanced model training pipeline with hardware acceleration, 
    hyperparameter optimization, and performance tracking.
    """
    
    def __init__(self, config: Config, metrics_collector: MetricsCollector = None):
        """
        Initialize the model trainer with configuration and dependencies.
        
        Args:
            config: Application configuration
            metrics_collector: Optional metrics collector for performance tracking
        """
        self.config = config
        self.metrics_collector = metrics_collector or MetricsCollector("ml_models", "trainer")
        self.market_data_repo = MarketDataRepository(config)
        features_list = config.get("features.enabled", ["sma", "ema", "rsi", "macd", "atr"])
        self.feature_extractor = FeatureExtractor(features_list)
        self.redis_client = RedisClient(config)
        self.use_gpu = config.get("ml_models.use_gpu", True)
        self.cuda_available = torch.cuda.is_available() if torch else False
        self.tf_gpu_available = (
            tf.config.list_physical_devices('GPU') if tf else []
        )
        
        # Configure GPU usage
        if self.use_gpu:
            self._setup_gpu()
        
        # Prepare model storage directory
        self.model_dir = Path(config.get("ml_models.storage_path", "./models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for data and preprocessors
        self._feature_cache = {}
        self._preprocessor_cache = {}
        
        # Semaphore to limit concurrent training processes
        self._training_semaphore = asyncio.Semaphore(
            config.get("ml_models.max_concurrent_training", 2)
        )
        
        # Stats for monitoring
        self.training_stats = {
            "total_training_time": 0,
            "models_trained": 0,
            "successful_trainings": 0,
            "failed_trainings": 0,
            "avg_training_time": 0,
            "gpu_utilization": 0,
        }
        
        logger.info(f"ModelTrainer initialized with GPU support: {self.use_gpu}")
        if self.use_gpu:
            if self.cuda_available:
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            if self.tf_gpu_available:
                logger.info(f"TensorFlow GPU available: {self.tf_gpu_available}")
    
    def _setup_gpu(self):
        """Configure GPU environment for optimal training performance."""
        try:
            # Setup GPU for TensorFlow
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set visible devices based on config
                visible_gpus = self.config.get("ml_models.visible_gpus", [0])
                if len(gpus) > 1 and visible_gpus:
                    tf.config.set_visible_devices([gpus[i] for i in visible_gpus], 'GPU')
            
            # Setup GPU for PyTorch
            if torch.cuda.is_available():
                # Set CUDA device
                cuda_device = self.config.get("ml_models.cuda_device", 0)
                torch.cuda.set_device(cuda_device)
                
                # Set PyTorch to deterministic mode if specified
                if self.config.get("ml_models.deterministic", True):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            
            logger.info("GPU setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up GPU: {str(e)}")
            self.use_gpu = False
    
    async def train_model(
        self,
        model_name: str,
        model_type: str,
        symbol: str,
        exchange: str = "binance",
        timeframe: str = "1h",
        training_period: str = "1y",
        features: List[str] = None,
        target: str = "future_price_change",
        target_horizon: str = "24h",
        labels: Optional[List] = None,
        hyperparams: Dict = None,
        optimization_metric: str = "f1_score",
        optimize_hyperparams: bool = True,
        save_model: bool = True,
        validation_size: float = 0.2,
        test_size: float = 0.1,
        walkforward: bool = True,
        resampling_method: str = None,
        ensemble_models: List[str] = None,
        feature_selection: bool = True,
        retrain_existing: bool = False,
        incremental: bool = False,
    ) -> Dict[str, Any]:
        """
        Train a machine learning model with advanced pipeline and optimization.
        
        Args:
            model_name: Unique name for the model
            model_type: Type of model (regression, classification, time_series, deep_learning)
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (default: 'binance')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            training_period: Period of data to use for training (e.g., '1y', '6m')
            features: List of features to use for model training
            target: Target variable for prediction
            target_horizon: Future prediction horizon (e.g., '24h')
            labels: For classification, the class labels
            hyperparams: Optional hyperparameters dictionary
            optimization_metric: Metric to optimize (e.g., 'f1_score', 'rmse')
            optimize_hyperparams: Whether to perform hyperparameter optimization
            save_model: Whether to save the trained model to disk
            validation_size: Fraction of data to use for validation
            test_size: Fraction of data to use for test evaluation
            walkforward: Use time series walk-forward validation
            resampling_method: Method for resampling imbalanced datasets
            ensemble_models: List of models to use in ensemble
            feature_selection: Whether to perform feature selection
            retrain_existing: Whether to retrain an existing model
            incremental: Whether to use incremental learning for existing model
            
        Returns:
            Dictionary containing trained model info and performance metrics
        """
        async with self._training_semaphore:
            start_time = time.time()
            model_id = f"{model_name}_{str(uuid.uuid4())[:8]}"

            try:
                logger.info(f"Starting training for model {model_name} ({model_type}) for {symbol} on {exchange}")

                if model_type in ['reinforcement_learning', 'rl']:
                    from intelligence.adaptive_learning.reinforcement import MarketEnvironment
                    market_data = await self.market_data_repo.get_ohlcv_data(
                        exchange, symbol, timeframe
                    )
                    features_df = self.feature_extractor.extract_features(market_data)
                    env = MarketEnvironment(market_data, features_df)
                    agent = await self.train_rl_model(env, hyperparams)
                    result = {
                        "model_id": model_id,
                        "model_name": model_name,
                        "model_type": model_type,
                        "agent": agent,
                    }
                    if save_model:
                        model_path = self.model_dir / f"{model_id}.pt"
                        agent.save_model(model_path)
                        result["model_path"] = str(model_path)
                    return result
                

                # Delegate to reinforcement learning pipeline when configured
                if self.config.get("ml_models.type") == "reinforcement":
                    from ml_models.rl.trainer import _train_rl_agent
                    return await _train_rl_agent(
                        self.config,
                        symbol,
                        exchange,
                        timeframe,
                        training_period,
                    )

                # Track GPU memory usage if available
                if self.use_gpu and (self.cuda_available or self.tf_gpu_available):
                    initial_gpu_memory = await run_in_thread_pool(get_gpu_memory_usage)
                
                # 1. Prepare and validate data
                X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_info = \
                    await self._prepare_training_data(
                        symbol, exchange, timeframe, training_period, features, 
                        target, target_horizon, labels, resampling_method
                    )
                
                if X_train.shape[0] == 0 or y_train.shape[0] == 0:
                    raise DataValidationError(f"Empty training dataset for {symbol}")
                
                # Log data shapes for debugging
                logger.info(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
                logger.info(f"Validation data shapes - X: {X_val.shape}, y: {y_val.shape}")
                logger.info(f"Test data shapes - X: {X_test.shape}, y: {y_test.shape}")
                
                # 2. Load or initialize hyperparameters
                final_hyperparams = await self._get_hyperparameters(
                    model_type, hyperparams, optimize_hyperparams, 
                    X_train, y_train, X_val, y_val, optimization_metric
                )
                
                # 3. Create and train the model
                model, training_history = await self._create_and_train_model(
                    model_type, X_train, y_train, X_val, y_val, 
                    final_hyperparams, model_name, incremental, ensemble_models
                )
                
                # 4. Evaluate model performance
                metrics = await self._evaluate_model(
                    model, model_type, X_test, y_test, optimization_metric
                )
                
                # 5. Calculate feature importance if applicable
                feature_importance = {}
                if feature_selection and hasattr(model, 'feature_importances_') or \
                   model_type in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
                    feature_importance = await run_in_thread_pool(
                        calculate_feature_importance, model, X_train.columns.tolist(), 
                        model_type, self.config.get("ml_models.feature_importance_method", "permutation")
                    )
                
                # 6. Save model if requested
                model_path = None
                if save_model:
                    model_path = await self._save_model(
                        model, model_id, model_name, model_type, preprocessing_info, 
                        final_hyperparams, metrics, feature_importance, symbol, 
                        exchange, timeframe, features
                    )
                
                # 7. Record training metrics
                training_duration = time.time() - start_time
                self._update_training_stats(training_duration, True)
                
                # Track GPU memory usage if available
                if self.use_gpu and (self.cuda_available or self.tf_gpu_available):
                    final_gpu_memory = await run_in_thread_pool(get_gpu_memory_usage)
                    gpu_memory_used = final_gpu_memory - initial_gpu_memory
                    logger.info(f"GPU memory used during training: {gpu_memory_used} MB")
                
                # 8. Compose and return results
                result = {
                    "model_id": model_id,
                    "model_name": model_name,
                    "model_type": model_type,
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "training_duration": training_duration,
                    "training_samples": X_train.shape[0],
                    "features_used": list(X_train.columns),
                    "metrics": metrics,
                    "feature_importance": feature_importance,
                    "hyperparameters": final_hyperparams,
                    "model_path": str(model_path) if model_path else None,
                    "training_timestamp": datetime.datetime.now().isoformat(),
                    "status": "success"
                }
                
                # Store training result in Redis for quick lookup
                await self.redis_client.set_async(
                    f"model:training:result:{model_id}", 
                    json.dumps(result),
                    ex=self.config.get("ml_models.training_result_ttl", 86400 * 7)  # 1 week default
                )
                
                logger.info(f"Successfully trained model {model_name} in {training_duration:.2f} seconds")
                return result
                
            except Exception as e:
                self._update_training_stats(time.time() - start_time, False)
                error_msg = f"Error training model {model_name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                
                # Record failure in Redis for tracking
                await self.redis_client.set_async(
                    f"model:training:error:{model_id}",
                    json.dumps({
                        "model_id": model_id,
                        "model_name": model_name,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.datetime.now().isoformat()
                    }),
                    ex=self.config.get("ml_models.error_log_ttl", 86400 * 7)  # 1 week default
                )
                
                raise ModelTrainingError(error_msg) from e
    
    async def _prepare_training_data(
        self, 
        symbol: str,
        exchange: str,
        timeframe: str,
        training_period: str,
        features: List[str],
        target: str,
        target_horizon: str,
        labels: Optional[List] = None,
        resampling_method: Optional[str] = None
    ) -> Tuple:
        """
        Prepare and preprocess data for model training.
        
        Returns:
            Tuple containing (X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_info)
        """
        # Generate cache key
        cache_key = f"{symbol}_{exchange}_{timeframe}_{training_period}_{target}_{target_horizon}"
        
        # Check if we have this data in cache
        if cache_key in self._feature_cache:
            logger.info(f"Using cached feature data for {cache_key}")
            feature_data = self._feature_cache[cache_key]
            # Add any missing features that might be requested
            if features:
                missing_features = [f for f in features if f not in feature_data.columns]
                if missing_features:
                    logger.info(f"Fetching {len(missing_features)} additional features for cached data")
                    # Get only the missing features and merge with cached data
                    additional_data = await self._fetch_feature_data(
                        symbol, exchange, timeframe, training_period, missing_features, target, target_horizon
                    )
                    feature_data = pd.concat([feature_data, additional_data[missing_features]], axis=1)
                    self._feature_cache[cache_key] = feature_data
        else:
            # Fetch all feature data from repositories
            feature_data = await self._fetch_feature_data(
                symbol, exchange, timeframe, training_period, features, target, target_horizon
            )
            # Cache the data for future use
            self._feature_cache[cache_key] = feature_data
            
        # Validate the data
        if feature_data.empty:
            raise DataValidationError(f"No data found for {symbol} on {exchange} with {timeframe} timeframe")
        
        # Handle missing values
        feature_data = await run_in_thread_pool(self._handle_missing_values, feature_data)
        
        # Select features if provided
        if features:
            available_features = [f for f in features if f in feature_data.columns]
            if len(available_features) < len(features):
                missing = set(features) - set(available_features)
                logger.warning(f"Some requested features not available: {missing}")
            X = feature_data[available_features]
        else:
            # Use all columns except the target
            X = feature_data.drop(columns=[target], errors='ignore')
        
        # Get the target variable
        if target in feature_data.columns:
            y = feature_data[target]
        else:
            raise DataValidationError(f"Target variable {target} not found in dataset")
            
        # For classification, convert target to categorical if needed
        if labels is not None:
            y = pd.cut(y, bins=[-float('inf')] + labels + [float('inf')], 
                      labels=range(len(labels) + 1))
        
        # Split data chronologically (time series aware)
        train_val_split_idx = int(len(X) * (1 - self.config.get("ml_models.validation_test_size", 0.3)))
        val_test_split_idx = int(len(X) * (1 - self.config.get("ml_models.test_size", 0.1)))
        
        X_train = X.iloc[:train_val_split_idx]
        y_train = y.iloc[:train_val_split_idx]
        
        X_val = X.iloc[train_val_split_idx:val_test_split_idx]
        y_val = y.iloc[train_val_split_idx:val_test_split_idx]
        
        X_test = X.iloc[val_test_split_idx:]
        y_test = y.iloc[val_test_split_idx:]
        
        # Apply preprocessing (scaling, encoding, etc.)
        X_train, X_val, X_test, preprocessing_info = await self._preprocess_features(
            X_train, X_val, X_test
        )
        
        # Handle imbalanced datasets if needed
        if resampling_method and labels is not None:
            X_train, y_train = await run_in_thread_pool(
                balance_dataset, X_train, y_train, method=resampling_method
            )
            
        return X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_info
    
    async def _fetch_feature_data(
        self, 
        symbol: str,
        exchange: str,
        timeframe: str,
        training_period: str,
        features: List[str],
        target: str,
        target_horizon: str
    ) -> pd.DataFrame:
        """
        Fetch historical data and calculate features for model training.
        """
        # Convert training period to actual timestamps
        end_time = datetime.datetime.now()
        
        if training_period.endswith('y'):
            years = int(training_period[:-1])
            start_time = end_time - datetime.timedelta(days=365 * years)
        elif training_period.endswith('m'):
            months = int(training_period[:-1])
            start_time = end_time - datetime.timedelta(days=30 * months)
        elif training_period.endswith('d'):
            days = int(training_period[:-1])
            start_time = end_time - datetime.timedelta(days=days)
        else:
            raise ValueError(f"Invalid training period format: {training_period}")
        
        # Fetch raw market data
        market_data = await self.market_data_repo.get_historical_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        if market_data.empty:
            raise DataValidationError(f"No historical data found for {symbol} on {exchange}")
        
        # Calculate all required features
        feature_data = await self.feature_extractor.calculate_features(
            market_data=market_data,
            features=features,
            symbol=symbol,
            timeframe=timeframe
        )
        
        # Calculate target variable based on horizon
        if target == 'future_price_change':
            # Convert target horizon to number of periods
            if target_horizon.endswith('h'):
                periods = int(target_horizon[:-1])
                if timeframe == '1h':
                    shift_periods = periods
                elif timeframe == '15m':
                    shift_periods = periods * 4
                elif timeframe == '5m':
                    shift_periods = periods * 12
                elif timeframe == '1m':
                    shift_periods = periods * 60
                elif timeframe == '4h':
                    shift_periods = periods // 4
                elif timeframe == '1d':
                    shift_periods = periods // 24
                else:
                    # Default to raw hours if timeframe is unknown
                    shift_periods = periods
            elif target_horizon.endswith('d'):
                days = int(target_horizon[:-1])
                if timeframe == '1d':
                    shift_periods = days
                elif timeframe == '4h':
                    shift_periods = days * 6
                elif timeframe == '1h':
                    shift_periods = days * 24
                elif timeframe == '15m':
                    shift_periods = days * 24 * 4
                elif timeframe == '5m':
                    shift_periods = days * 24 * 12
                elif timeframe == '1m':
                    shift_periods = days * 24 * 60
                else:
                    # Default to raw days if timeframe is unknown
                    shift_periods = days
            else:
                raise ValueError(f"Invalid target horizon format: {target_horizon}")
            
            # Calculate future price change as percentage
            future_close = feature_data['close'].shift(-shift_periods)
            feature_data['future_price_change'] = (future_close - feature_data['close']) / feature_data['close'] * 100
        
        # Drop rows with NaN in target (usually at the end due to future_price_change)
        feature_data = feature_data.dropna(subset=[target])
        
        return feature_data
    
    async def _preprocess_features(
        self, 
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame, 
        X_test: pd.DataFrame
    ) -> Tuple:
        """
        Apply preprocessing to features including scaling and encoding.
        
        Returns:
            Tuple of (X_train_processed, X_val_processed, X_test_processed, preprocessing_info)
        """
        # Create preprocessing pipeline based on column types
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        
        # Initialize preprocessing_info
        preprocessing_info = {
            'numeric_scaler': None,
            'categorical_encoders': {},
            'feature_names': X_train.columns.tolist()
        }
        
        # Process data in thread pool for performance
        if len(numeric_cols) > 0:
            # Get scaler method from config
            scaler_method = self.config.get("ml_models.scaler_method", "standard")
            scaler = await run_in_thread_pool(get_scaler, scaler_method)
            
            # Fit scaler on training data
            scaler.fit(X_train[numeric_cols])
            preprocessing_info['numeric_scaler'] = {
                'method': scaler_method,
                'scaler_object': scaler
            }
            
            # Transform all datasets
            X_train_numeric = pd.DataFrame(
                scaler.transform(X_train[numeric_cols]), 
                columns=numeric_cols,
                index=X_train.index
            )
            X_val_numeric = pd.DataFrame(
                scaler.transform(X_val[numeric_cols]), 
                columns=numeric_cols,
                index=X_val.index
            )
            X_test_numeric = pd.DataFrame(
                scaler.transform(X_test[numeric_cols]), 
                columns=numeric_cols,
                index=X_test.index
            )
        else:
            X_train_numeric = pd.DataFrame(index=X_train.index)
            X_val_numeric = pd.DataFrame(index=X_val.index)
            X_test_numeric = pd.DataFrame(index=X_test.index)
        
        # Process categorical features if any
        if len(categorical_cols) > 0:
            X_train_cat, X_val_cat, X_test_cat, cat_encoders = await run_in_thread_pool(
                encode_features, 
                X_train[categorical_cols], 
                X_val[categorical_cols], 
                X_test[categorical_cols],
                self.config.get("ml_models.encoding_method", "onehot")
            )
            preprocessing_info['categorical_encoders'] = cat_encoders
            
            # Combine numeric and categorical
            X_train_processed = pd.concat([X_train_numeric, X_train_cat], axis=1)
            X_val_processed = pd.concat([X_val_numeric, X_val_cat], axis=1)
            X_test_processed = pd.concat([X_test_numeric, X_test_cat], axis=1)
        else:
            X_train_processed = X_train_numeric
            X_val_processed = X_val_numeric
            X_test_processed = X_test_numeric
        
        return X_train_processed, X_val_processed, X_test_processed, preprocessing_info
    
    async def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Check for missing values
        null_counts = data.isnull().sum()
        if null_counts.sum() > 0:
            cols_with_nulls = null_counts[null_counts > 0].index.tolist()
            logger.info(f"Handling missing values in columns: {cols_with_nulls}")
            
            # Handle based on data type
            for col in cols_with_nulls:
                if data[col].dtype in ['int64', 'float64']:
                    # For numeric, use forward fill then backward fill
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                    
                    # If still has NaN, use median
                    if data[col].isnull().sum() > 0:
                        data[col] = data[col].fillna(data[col].median())
                else:
                    # For categorical, use most frequent value
                    most_frequent = data[col].mode()[0]
                    data[col] = data[col].fillna(most_frequent)
        
        return data
    
    async def _get_hyperparameters(
        self,
        model_type: str, 
        hyperparams: Dict, 
        optimize_hyperparams: bool,
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        optimization_metric: str
    ) -> Dict:
        """
        Get hyperparameters either from input or through optimization.
        
        Returns:
            Dictionary of optimized hyperparameters
        """
        if not optimize_hyperparams or hyperparams is not None:
            logger.info(f"Using provided hyperparameters for {model_type} model")
            return hyperparams or {}
        
        logger.info(f"Starting hyperparameter optimization for {model_type} model")

        # Set search space based on model type
        if model_type == 'random_forest':
            search_space = {
                'n_estimators': (50, 500),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None],
            }
        elif model_type == 'gradient_boosting':
            search_space = {
                'n_estimators': (50, 500),
                'learning_rate': (0.001, 0.1),
                'max_depth': (3, 10),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'subsample': (0.5, 1.0),
            }
        elif model_type == 'xgboost':
            search_space = {
                'n_estimators': (50, 500),
                'learning_rate': (0.001, 0.1),
                'max_depth': (3, 10),
                'min_child_weight': (1, 10),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
                'gamma': (0.0, 1.0),
            }
        elif model_type == 'lightgbm':
            search_space = {
                'n_estimators': (50, 500),
                'learning_rate': (0.001, 0.1),
                'max_depth': (3, 10),
                'num_leaves': (20, 100),
                'min_child_samples': (5, 50),
                'subsample': (0.5, 1.0),
                'colsample_bytree': (0.5, 1.0),
            }
        elif model_type in ['lstm', 'gru', 'cnn']:
            search_space = {
                'units': (32, 256),
                'layers': (1, 3),
                'dropout': (0.1, 0.5),
                'learning_rate': (0.0001, 0.01),
                'batch_size': [16, 32, 64, 128],
            }
        else:
            search_space = {
                'n_estimators': (50, 300),
                'max_depth': (3, 10),
                'learning_rate': (0.001, 0.1),
            }

        try:
            def objective(params):
                for param in [
                    'n_estimators',
                    'max_depth',
                    'min_samples_split',
                    'min_samples_leaf',
                    'layers',
                    'units',
                    'batch_size',
                    'num_leaves',
                    'min_child_samples',
                ]:
                    if param in params:
                        params[param] = int(params[param])

                model = self._create_model_instance(model_type, params)
                model.fit(X_train, y_train)

                if model_type in ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']:
                    y_pred = model.predict(X_val)
                else:
                    y_pred = model.predict(X_val).flatten()

                if len(np.unique(y_train)) <= 10:
                    if optimization_metric == 'accuracy':
                        score = accuracy_score(y_val, y_pred)
                    elif optimization_metric == 'f1_score':
                        score = f1_score(y_val, y_pred, average='weighted')
                    elif optimization_metric == 'precision':
                        score = precision_score(y_val, y_pred, average='weighted')
                    elif optimization_metric == 'recall':
                        score = recall_score(y_val, y_pred, average='weighted')
                    else:
                        score = f1_score(y_val, y_pred, average='weighted')
                else:
                    if optimization_metric == 'rmse':
                        score = mean_squared_error(y_val, y_pred, squared=False)
                    elif optimization_metric == 'mae':
                        score = mean_absolute_error(y_val, y_pred)
                    elif optimization_metric == 'r2':
                        score = r2_score(y_val, y_pred)
                    else:
                        score = mean_squared_error(y_val, y_pred, squared=False)

                return score

            optimizer = HyperparameterOptimizer(
                metrics=self.metrics_collector,
                direction='minimize' if optimization_metric in ['rmse', 'mae'] else 'maximize',
            )

            best_params = await run_in_thread_pool(
                optimizer.optimize,
                objective,
                search_space,
                self.config.get('ml_models.optimize_method', 'bayesian'),
                self.config.get('ml_models.optimize_trials', 50),
            )

            logger.info(f"Optimized hyperparameters for {model_type}: {best_params}")
            return best_params

        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {str(e)}", exc_info=True)
            raise HyperparameterOptimizationError(
                f"Failed to optimize hyperparameters: {str(e)}"
            ) from e
    
    def _create_model_instance(self, model_type: str, params: Dict) -> Any:
        """Create a model instance based on model type and parameters."""
        if model_type == 'random_forest':
            if len(np.unique(y_train)) <= 10:  # Classification
                return RandomForestClassifier(**params, random_state=42)
            else:  # Regression
                return RandomForestRegressor(**params, random_state=42)
        
        elif model_type == 'gradient_boosting':
            if len(np.unique(y_train)) <= 10:  # Classification
                return GradientBoostingClassifier(**params, random_state=42)
            else:  # Regression
                return GradientBoostingRegressor(**params, random_state=42)
        
        elif model_type == 'xgboost':
            if len(np.unique(y_train)) <= 10:  # Classification
                return xgb.XGBClassifier(**params, random_state=42)
            else:  # Regression
                return xgb.XGBRegressor(**params, random_state=42)
        
        elif model_type == 'lightgbm':
            if len(np.unique(y_train)) <= 10:  # Classification
                return lgb.LGBMClassifier(**params, random_state=42)
            else:  # Regression
                return lgb.LGBMRegressor(**params, random_state=42)
        
        elif model_type in ['lstm', 'gru', 'cnn']:
            if not create_deep_learning_model:
                raise ModelTrainingError('Deep learning support is not available')

            if len(np.unique(y_train)) <= 10:
                problem_type = 'classification'
            else:
                problem_type = 'regression'

            return create_deep_learning_model(
                model_type=model_type,
                input_shape=X_train.shape[1:],
                output_shape=1,
                problem_type=problem_type,
                hyperparams=params,
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    async def _create_and_train_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparams: Dict,
        model_name: str,
        incremental: bool = False,
        ensemble_models: List[str] = None
    ) -> Tuple[Any, Dict]:
        """
        Create and train the model with given hyperparameters.
        
        Returns:
            Tuple containing (trained_model, training_history)
        """
        # Determine problem type
        n_unique_targets = len(np.unique(y_train))
        problem_type = 'classification' if n_unique_targets <= 10 else 'regression'
        
        logger.info(f"Creating {problem_type} model of type {model_type}")
        
        # Prepare model based on type
        if model_type == 'regression':
            model = await run_in_thread_pool(
                create_regression_model,
                X_train.shape[1],
                hyperparams,
                self.config
            )
        
        elif model_type == 'classification':
            model = await run_in_thread_pool(
                create_classification_model,
                X_train.shape[1],
                n_unique_targets,
                hyperparams,
                self.config
            )
        
        elif model_type == 'time_series':
            model = await run_in_thread_pool(
                create_time_series_model,
                X_train.shape[1],
                hyperparams,
                self.config
            )
        
        elif model_type in ['lstm', 'gru', 'cnn', 'mlp', 'transformer']:
            if not create_deep_learning_model:
                raise ModelTrainingError('Deep learning support is not available')

            model = await run_in_thread_pool(
                create_deep_learning_model,
                model_type,
                X_train.shape[1:],
                1 if problem_type == 'regression' else n_unique_targets,
                problem_type,
                hyperparams,
                self.use_gpu,
            )
        
        elif model_type == 'ensemble':
            if not ensemble_models:
                raise ValueError("Ensemble models list must be provided for ensemble model type")

            if not create_ensemble_model:
                raise ModelTrainingError('Ensemble model support is not available')

            model = await run_in_thread_pool(
                create_ensemble_model,
                ensemble_models,
                X_train.shape[1:],
                problem_type,
                hyperparams,
                self.config
            )
        
        elif model_type == 'random_forest':
            if problem_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', None),
                    min_samples_split=hyperparams.get('min_samples_split', 2),
                    min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                    max_features=hyperparams.get('max_features', 'sqrt'),
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', None),
                    min_samples_split=hyperparams.get('min_samples_split', 2),
                    min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                    max_features=hyperparams.get('max_features', 'sqrt'),
                    random_state=42,
                    n_jobs=-1
                )
        
        elif model_type == 'gradient_boosting':
            if problem_type == 'classification':
                model = GradientBoostingClassifier(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    learning_rate=hyperparams.get('learning_rate', 0.1),
                    max_depth=hyperparams.get('max_depth', 3),
                    min_samples_split=hyperparams.get('min_samples_split', 2),
                    min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                    subsample=hyperparams.get('subsample', 1.0),
                    random_state=42
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    learning_rate=hyperparams.get('learning_rate', 0.1),
                    max_depth=hyperparams.get('max_depth', 3),
                    min_samples_split=hyperparams.get('min_samples_split', 2),
                    min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                    subsample=hyperparams.get('subsample', 1.0),
                    random_state=42
                )
        
        elif model_type == 'xgboost':
            # Convert parameters for XGBoost
            xgb_params = {
                'n_estimators': hyperparams.get('n_estimators', 100),
                'learning_rate': hyperparams.get('learning_rate', 0.1),
                'max_depth': hyperparams.get('max_depth', 3),
                'min_child_weight': hyperparams.get('min_child_weight', 1),
                'subsample': hyperparams.get('subsample', 1.0),
                'colsample_bytree': hyperparams.get('colsample_bytree', 1.0),
                'gamma': hyperparams.get('gamma', 0),
                'random_state': 42
            }
            
            if problem_type == 'classification':
                model = xgb.XGBClassifier(**xgb_params)
                if self.use_gpu and self.cuda_available:
                    model.set_params(tree_method='gpu_hist')
            else:
                model = xgb.XGBRegressor(**xgb_params)
                if self.use_gpu and self.cuda_available:
                    model.set_params(tree_method='gpu_hist')
        
        elif model_type == 'lightgbm':
            # Convert parameters for LightGBM
            lgb_params = {
                'n_estimators': hyperparams.get('n_estimators', 100),
                'learning_rate': hyperparams.get('learning_rate', 0.1),
                'max_depth': hyperparams.get('max_depth', 3),
                'num_leaves': hyperparams.get('num_leaves', 31),
                'min_child_samples': hyperparams.get('min_child_samples', 20),
                'subsample': hyperparams.get('subsample', 1.0),
                'colsample_bytree': hyperparams.get('colsample_bytree', 1.0),
                'random_state': 42
            }
            
            if problem_type == 'classification':
                model = lgb.LGBMClassifier(**lgb_params)
                if self.use_gpu and self.cuda_available:
                    model.set_params(device='gpu')
            else:
                model = lgb.LGBMRegressor(**lgb_params)
                if self.use_gpu and self.cuda_available:
                    model.set_params(device='gpu')
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        logger.info(f"Training {model_type} model with {len(X_train)} samples")
        
        history = {}
        try:
            # Special handling for deep learning models
            if model_type in ['lstm', 'gru', 'cnn', 'mlp', 'transformer']:
                # Convert data format if needed
                if isinstance(model, tf.keras.Model):
                    # For TensorFlow models
                    batch_size = hyperparams.get('batch_size', 32)
                    epochs = hyperparams.get('epochs', 100)
                    
                    # Create early stopping callback
                    early_stop = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=hyperparams.get('patience', 10),
                        restore_best_weights=True
                    )
                    
                    # Create ModelCheckpoint callback
                    checkpoint_path = os.path.join(
                        tempfile.gettempdir(), 
                        f"model_checkpoint_{model_name}_{int(time.time())}.h5"
                    )
                    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                        checkpoint_path,
                        monitor='val_loss',
                        save_best_only=True,
                        mode='min'
                    )
                    
                    # Add TensorBoard if configured
                    callbacks = [early_stop, checkpoint]
                    if self.config.get("ml_models.use_tensorboard", False):
                        log_dir = os.path.join(
                            self.config.get("ml_models.tensorboard_dir", "./logs/tensorboard"),
                            f"{model_name}_{int(time.time())}"
                        )
                        tensorboard = tf.keras.callbacks.TensorBoard(
                            log_dir=log_dir,
                            histogram_freq=1
                        )
                        callbacks.append(tensorboard)
                    
                    # Train model
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=self.config.get("ml_models.training_verbose", 1)
                    ).history
                    
                    # Load best model weights
                    if os.path.exists(checkpoint_path):
                        model.load_weights(checkpoint_path)
                        os.remove(checkpoint_path)
                
                elif isinstance(model, nn.Module):
                    # For PyTorch models
                    model, history = await run_in_thread_pool(
                        self._train_pytorch_model,
                        model, X_train, y_train, X_val, y_val, hyperparams, problem_type
                    )
            else:
                # For scikit-learn based models
                fit_params = {}
                
                # Add validation data for models that support it
                if model_type in ['xgboost', 'lightgbm']:
                    if problem_type == 'classification':
                        fit_params['eval_set'] = [(X_val, y_val)]
                        fit_params['early_stopping_rounds'] = hyperparams.get('early_stopping_rounds', 20)
                        fit_params['verbose'] = self.config.get("ml_models.training_verbose", 0)
                    else:
                        fit_params['eval_set'] = [(X_val, y_val)]
                        fit_params['early_stopping_rounds'] = hyperparams.get('early_stopping_rounds', 20)
                        fit_params['verbose'] = self.config.get("ml_models.training_verbose", 0)
                
                # Train model
                model.fit(X_train, y_train, **fit_params)
                
                # Get evaluation metrics for history
                if hasattr(model, 'evals_result_'):
                    history = model.evals_result_
                
            logger.info(f"Successfully trained {model_type} model")
            return model, history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}", exc_info=True)
            raise ModelTrainingError(f"Failed to train {model_type} model: {str(e)}") from e
    
    async def _train_pytorch_model(
        self, 
        model, 
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        hyperparams, 
        problem_type
    ):
        """Train a PyTorch model."""
        # Convert numpy arrays to torch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
        
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1)
        
        # Create dataloaders
        batch_size = hyperparams.get('batch_size', 32)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Move model to GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu else "cpu")
        model.to(device)
        
        # Define loss function and optimizer
        if problem_type == 'classification':
            criterion = nn.BCEWithLogitsLoss() if model.output_dim == 1 else nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=hyperparams.get('learning_rate', 0.001),
            weight_decay=hyperparams.get('weight_decay', 0.0)
        )
        
        # Learning rate scheduler if specified
        if hyperparams.get('use_lr_scheduler', False):
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                verbose=True
            )
        
        # Training loop
        epochs = hyperparams.get('epochs', 100)
        patience = hyperparams.get('patience', 10)
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        # Save best model state
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            
            # Update learning rate scheduler if used
            if hyperparams.get('use_lr_scheduler', False):
                scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, history
    
    async def _evaluate_model(
        self, 
        model, 
        model_type: str, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        optimization_metric: str
    ) -> Dict:
        """
        Evaluate model performance on test data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_type} model on {len(X_test)} test samples")
        
        try:
            # Determine problem type (classification or regression)
            n_unique_targets = len(np.unique(y_test))
            problem_type = 'classification' if n_unique_targets <= 10 else 'regression'
            
            # Make predictions based on model type
            if model_type in ['lstm', 'gru', 'cnn', 'mlp', 'transformer']:
                if isinstance(model, tf.keras.Model):
                    y_pred = model.predict(X_test)
                    # Convert probabilities to class labels for classification
                    if problem_type == 'classification' and y_pred.shape[1] > 1:
                        y_pred = np.argmax(y_pred, axis=1)
                    else:
                        y_pred = y_pred.flatten()
                elif isinstance(model, nn.Module):
                    # For PyTorch models
                    device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu else "cpu")
                    model.eval()
                    with torch.no_grad():
                        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
                        y_pred = model(X_test_tensor).cpu().numpy()
                        # Convert probabilities to class labels for classification
                        if problem_type == 'classification' and y_pred.shape[1] > 1:
                            y_pred = np.argmax(y_pred, axis=1)
                        else:
                            y_pred = y_pred.flatten()
            else:
                # For scikit-learn based models
                y_pred = model.predict(X_test)
            
            # Calculate metrics based on problem type
            metrics = {}
            if problem_type == 'classification':
                # Binary or Multi-class Classification metrics
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                
                # For multi-class, use weighted average
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
                metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
                
                # Add ROC-AUC if binary classification and model supports predict_proba
                if n_unique_targets == 2 and hasattr(model, 'predict_proba'):
                    try:
                        from sklearn.metrics import roc_auc_score
                        y_prob = model.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
                    except Exception as e:
                        logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
                
                # Calculate confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
                
                # Calculate class distribution
                from collections import Counter
                metrics['class_distribution'] = dict(Counter(y_test))
                
            else:
                # Regression metrics
                metrics['mse'] = mean_squared_error(y_test, y_pred)
                metrics['rmse'] = mean_squared_error(y_test, y_pred, squared=False)
                metrics['mae'] = mean_absolute_error(y_test, y_pred)
                metrics['r2'] = r2_score(y_test, y_pred)
                
                # Add more detailed regression metrics
                from sklearn.metrics import median_absolute_error, explained_variance_score
                metrics['median_absolute_error'] = median_absolute_error(y_test, y_pred)
                metrics['explained_variance'] = explained_variance_score(y_test, y_pred)
                
                # Calculate distribution stats
                metrics['target_min'] = float(y_test.min())
                metrics['target_max'] = float(y_test.max())
                metrics['target_mean'] = float(y_test.mean())
                metrics['prediction_min'] = float(np.min(y_pred))
                metrics['prediction_max'] = float(np.max(y_pred))
                metrics['prediction_mean'] = float(np.mean(y_pred))
            
            # Add common metrics
            metrics['test_samples'] = len(X_test)
            
            # Log evaluation results
            log_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
            logger.info(f"Model evaluation results: {log_metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}", exc_info=True)
            raise ModelTrainingError(f"Failed to evaluate model: {str(e)}") from e
    
    async def _save_model(
        self,
        model,
        model_id: str,
        model_name: str,
        model_type: str,
        preprocessing_info: Dict,
        hyperparams: Dict,
        metrics: Dict,
        feature_importance: Dict,
        symbol: str,
        exchange: str,
        timeframe: str,
        features: List[str]
    ) -> Path:
        """
        Save trained model and metadata to disk.
        
        Returns:
            Path to saved model directory
        """
        logger.info(f"Saving model {model_name} (id: {model_id})")
        
        try:
            # Create model directory
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = self.model_dir / f"{model_name}_{timestamp}_{model_id}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model based on its type
            if isinstance(model, tf.keras.Model):
                # For TensorFlow models
                model_path = model_dir / "model.h5"
                model.save(model_path)
            elif isinstance(model, nn.Module):
                # For PyTorch models
                model_path = model_dir / "model.pt"
                torch.save(model.state_dict(), model_path)
                # Also save model architecture
                with open(model_dir / "model_architecture.pkl", 'wb') as f:
                    pickle.dump(type(model), f)
            else:
                # For scikit-learn based models
                model_path = model_dir / "model.joblib"
                joblib.dump(model, model_path)
            
            # Save preprocessing information
            if preprocessing_info:
                if 'numeric_scaler' in preprocessing_info and preprocessing_info['numeric_scaler']:
                    scaler_path = model_dir / "scaler.joblib"
                    joblib.dump(preprocessing_info['numeric_scaler']['scaler_object'], scaler_path)
                    # Remove actual scaler object from metadata to avoid serialization issues
                    preprocessing_info['numeric_scaler'] = {
                        'method': preprocessing_info['numeric_scaler']['method'],
                        'path': 'scaler.joblib'
                    }
                
                # Save categorical encoders
                if 'categorical_encoders' in preprocessing_info and preprocessing_info['categorical_encoders']:
                    encoder_dir = model_dir / "encoders"
                    encoder_dir.mkdir(exist_ok=True)
                    
                    for col, encoder in preprocessing_info['categorical_encoders'].items():
                        if hasattr(encoder, 'save'):
                            encoder_path = encoder_dir / f"{col}_encoder.joblib"
                            joblib.dump(encoder, encoder_path)
                            # Update path in metadata
                            preprocessing_info['categorical_encoders'][col] = f"encoders/{col}_encoder.joblib"
            
            # Save model metadata
            metadata = {
                "model_id": model_id,
                "model_name": model_name,
                "model_type": model_type,
                "created_at": datetime.datetime.now().isoformat(),
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "features": features,
                "hyperparameters": hyperparams,
                "metrics": metrics,
                "feature_importance": feature_importance,
                "preprocessing_info": preprocessing_info,
                "model_path": model_path.name
            }
            
            # Save metadata as JSON
            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved successfully to {model_dir}")
            return model_dir
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            raise ModelPersistenceError(f"Failed to save model: {str(e)}") from e
    
    def _update_training_stats(self, training_duration: float, success: bool):
        """Update training statistics."""
        self.training_stats['total_training_time'] += training_duration
        self.training_stats['models_trained'] += 1
        
        if success:
            self.training_stats['successful_trainings'] += 1
        else:
            self.training_stats['failed_trainings'] += 1
        
        self.training_stats['avg_training_time'] = (
            self.training_stats['total_training_time'] / self.training_stats['models_trained']
        )
    
    async def get_training_stats(self) -> Dict:
        """Get current training statistics."""
        # Update GPU usage if available
        if self.use_gpu and (self.cuda_available or self.tf_gpu_available):
            try:
                self.training_stats['gpu_utilization'] = await run_in_thread_pool(get_gpu_memory_usage)
            except Exception as e:
                logger.warning(f"Could not get GPU utilization: {str(e)}")
        
        return self.training_stats
    
    async def clean_up(self):
        """Clean up resources and temporary files."""
        # Clear cache to free memory
        self._feature_cache.clear()
        self._preprocessor_cache.clear()
        
        # Clean up any temporary files
        tmp_dir = tempfile.gettempdir()
        pattern = "model_checkpoint_*.h5"
        
        for file in Path(tmp_dir).glob(pattern):
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temporary file {file}: {str(e)}")
        
        # Release GPU memory if using TensorFlow
        if self.use_gpu and self.tf_gpu_available:
            try:
                tf.keras.backend.clear_session()
            except Exception as e:
                logger.warning(f"Could not clear TensorFlow session: {str(e)}")

    async def train_rl_model(
        self,
        env,
        agent_config: dict | None = None,
        episodes: int = 100,
    ) -> Any:
        """Train a reinforcement learning agent."""
        from ml_models.rl import DQNAgent

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = DQNAgent(state_dim, action_dim, agent_config)

        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.update_model()
                state = next_state

        return agent
