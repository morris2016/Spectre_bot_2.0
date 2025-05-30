#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
ML Models - Prediction Service

This module provides high-performance prediction capabilities with 
hardware acceleration, model ensembling, and confidence estimation.
"""

import os
import time
import json
import uuid
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import asyncio
import pickle
import joblib
from collections import defaultdict

# ML Libraries
try:
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tf = None  # type: ignore
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Internal imports
from config import Config
from common.logger import get_logger
from common.exceptions import (
    ModelNotFoundError, PredictionError, DataValidationError,
    HardwareAccelerationError, ModelLoadingError
)
from common.utils import validate_data
from common.metrics import MetricsCollector
from common.async_utils import run_in_thread_pool
from common.redis_client import RedisClient

from data_storage.market_data import MarketDataRepository
from feature_service.feature_extraction import FeatureExtractor
from ml_models.hardware.gpu import setup_gpu, get_gpu_memory_usage

# Configure logger
logger = get_logger("ml_models.prediction")

class ModelPredictor:
    """
    High-performance prediction service with model ensembling,
    confidence estimation, and hardware acceleration.
    """
    
    def __init__(self, config: Config, metrics_collector: MetricsCollector = None):
        """
        Initialize the model predictor with configuration and dependencies.
        
        Args:
            config: Application configuration
            metrics_collector: Optional metrics collector for performance tracking
        """
        self.config = config
        self.metrics_collector = metrics_collector or MetricsCollector("ml_models.prediction")
        self.market_data_repo = MarketDataRepository(config)
        self.feature_extractor = FeatureExtractor(config)
        self.redis_client = RedisClient(config)
        
        # Configure GPU usage
        self.use_gpu = config.get("ml_models.use_gpu", True)
        self.cuda_available = torch.cuda.is_available() if torch is not None else False
        self.tf_gpu_available = tf.config.list_physical_devices('GPU') if tf is not None else []
        
        if self.use_gpu:
            self._setup_gpu()
        
        # Prepare model storage directory
        self.model_dir = Path(config.get("ml_models.storage_path", "./models"))
        
        # Model cache for frequently used models
        self._model_cache = {}
        self._metadata_cache = {}
        self._preprocessor_cache = {}
        
        # Cache size limit
        self.max_cache_size = config.get("ml_models.max_cache_size", 10)
        
        # Cache for feature data
        self._feature_cache = {}
        self._feature_cache_ttl = config.get("ml_models.feature_cache_ttl", 300)  # 5 minutes
        self._feature_cache_timestamps = {}
        
        # Tracking performance metrics
        self.prediction_stats = {
            "total_predictions": 0,
            "avg_prediction_time": 0,
            "total_prediction_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }
        
        logger.info(f"ModelPredictor initialized with GPU support: {self.use_gpu}")
        if self.use_gpu:
            if self.cuda_available:
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            if self.tf_gpu_available:
                logger.info(f"TensorFlow GPU available: {self.tf_gpu_available}")
    
    def _setup_gpu(self):
        """Configure GPU environment for optimal prediction performance."""
        try:
            # Setup GPU for TensorFlow
            if tf is not None:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set visible devices based on config
                    visible_gpus = self.config.get("ml_models.visible_gpus", [0])
                    if len(gpus) > 1 and visible_gpus:
                        tf.config.set_visible_devices([gpus[i] for i in visible_gpus], 'GPU')
            
            # Setup GPU for PyTorch
            if torch is not None and torch.cuda.is_available():
                # Set CUDA device
                cuda_device = self.config.get("ml_models.cuda_device", 0)
                torch.cuda.set_device(cuda_device)
            
            logger.info("GPU setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up GPU: {str(e)}")
            self.use_gpu = False
            raise HardwareAccelerationError(f"Failed to setup GPU: {str(e)}") from e
    
    async def predict(
        self,
        model_id: str = None,
        model_name: str = None,
        data: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        exchange: str = "binance",
        timeframe: str = "1h",
        num_candles: int = 100,
        features: List[str] = None,
        ensemble: bool = False,
        ensemble_method: str = "weighted_average",
        calculate_confidence: bool = True,
        use_cached_data: bool = True,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Make predictions using trained models.
        
        Args:
            model_id: Specific model ID to use for prediction
            model_name: Model name to use (latest version if multiple exist)
            data: Optional dataframe with pre-calculated features
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (default: 'binance')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            num_candles: Number of candles to fetch for feature calculation
            features: List of features to use for prediction
            ensemble: Whether to use model ensembling for prediction
            ensemble_method: Method for ensembling (average, weighted_average, voting)
            calculate_confidence: Whether to calculate prediction confidence
            use_cached_data: Whether to use cached feature data if available
            return_features: Whether to return feature data with prediction
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        start_time = time.time()
        
        try:
            # Track and update stats
            self.prediction_stats["total_predictions"] += 1
            
            # Step 1: Identify which model(s) to use
            if ensemble and not model_id and not model_name:
                # Use all available models for the symbol
                model_metadata_list = await self._get_models_for_symbol(symbol, exchange, timeframe)
                if not model_metadata_list:
                    raise ModelNotFoundError(f"No models found for {symbol} on {exchange}")
            elif ensemble and model_name:
                # Use all models with the given name
                model_metadata_list = await self._get_models_by_name(model_name)
                if not model_metadata_list:
                    raise ModelNotFoundError(f"No models found with name {model_name}")
            elif model_id:
                # Use specific model ID
                model_metadata = await self._get_model_by_id(model_id)
                if not model_metadata:
                    raise ModelNotFoundError(f"Model with ID {model_id} not found")
                model_metadata_list = [model_metadata]
            elif model_name:
                # Use latest model with the given name
                model_metadata = await self._get_latest_model_by_name(model_name)
                if not model_metadata:
                    raise ModelNotFoundError(f"Model with name {model_name} not found")
                model_metadata_list = [model_metadata]
            else:
                raise ValueError("Either model_id, model_name, or ensemble=True with symbol must be provided")
            
            # Step 2: Prepare feature data
            if data is not None:
                # Use provided data
                feature_data = data
            else:
                # Use cached data or fetch new data
                feature_data = await self._get_feature_data(
                    symbol, exchange, timeframe, num_candles, features, 
                    model_metadata_list, use_cached_data
                )
            
            # Check if we have enough data
            if feature_data.empty:
                raise DataValidationError(f"No valid feature data for prediction")
            
            # Step 3: Make predictions with each model
            predictions = []
            confidence_scores = []
            
            for model_metadata in model_metadata_list:
                # Load and prepare model
                model, preprocessor = await self._load_model(
                    model_metadata["model_id"], model_metadata["model_path"], 
                    model_metadata["model_type"]
                )
                
                # Prepare features for this specific model
                processed_features = await self._prepare_features_for_model(
                    feature_data, model_metadata, preprocessor
                )
                
                # Make prediction
                prediction_result, confidence = await self._make_single_prediction(
                    model, processed_features, model_metadata, calculate_confidence
                )
                
                # Store results with metadata
                predictions.append({
                    "model_id": model_metadata["model_id"],
                    "model_name": model_metadata["model_name"],
                    "prediction": prediction_result,
                    "confidence": confidence,
                    "model_weight": model_metadata.get("performance_weight", 1.0)
                })
                
                # Store confidence scores for calculation
                if confidence is not None:
                    confidence_scores.append(confidence)
            
            # Step 4: Combine predictions if ensembling
            if ensemble and len(predictions) > 1:
                ensemble_result = await self._combine_predictions(
                    predictions, ensemble_method, model_metadata_list[0].get("prediction_type", "regression")
                )
                final_prediction = ensemble_result["prediction"]
                final_confidence = ensemble_result["confidence"]
                methods_used = "ensemble:" + ensemble_method
            else:
                # Use single model prediction
                final_prediction = predictions[0]["prediction"]
                final_confidence = predictions[0]["confidence"]
                methods_used = f"single_model:{predictions[0]['model_id']}"
            
            # Calculate time taken
            prediction_time = time.time() - start_time
            self._update_prediction_stats(prediction_time)
            
            # Step 5: Prepare and return result
            timestamp = datetime.datetime.now().isoformat()
            
            result = {
                "prediction": final_prediction,
                "confidence": final_confidence,
                "timestamp": timestamp,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "prediction_time": prediction_time,
                "methods_used": methods_used,
            }
            
            # Add individual model predictions if ensembling
            if ensemble and len(predictions) > 1:
                result["individual_predictions"] = predictions
                result["ensemble_method"] = ensemble_method
            
            # Add feature data if requested
            if return_features:
                result["feature_data"] = feature_data.to_dict(orient="records")
            
            # Return different result formats based on prediction type
            prediction_type = model_metadata_list[0].get("prediction_type", "regression")
            if prediction_type == "classification":
                # For classification, return predicted class and probabilities
                if isinstance(final_prediction, list):
                    result["predicted_class"] = np.argmax(final_prediction)
                    result["class_probabilities"] = final_prediction
                else:
                    result["predicted_class"] = int(final_prediction)
            
            return result
            
        except Exception as e:
            self.prediction_stats["errors"] += 1
            error_msg = f"Error making prediction: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise PredictionError(error_msg) from e
    
    async def _get_models_for_symbol(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ) -> List[Dict]:
        """Get all available models for a specific symbol, exchange and timeframe."""
        models = []
        
        # Scan model directory for matching models
        for model_dir in self.model_dir.glob("*"):
            if not model_dir.is_dir():
                continue
            
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check if model matches the criteria
                if (metadata.get("symbol") == symbol and 
                    metadata.get("exchange") == exchange and
                    metadata.get("timeframe") == timeframe):
                    # Add full path to model
                    metadata["model_path"] = model_dir
                    models.append(metadata)
            except Exception as e:
                logger.warning(f"Error reading metadata from {metadata_path}: {str(e)}")
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Calculate performance weights for ensembling
        if models:
            await self._calculate_performance_weights(models)
        
        return models
    
    async def _get_models_by_name(self, model_name: str) -> List[Dict]:
        """Get all models with a specific name."""
        models = []
        
        # Scan model directory for matching models
        for model_dir in self.model_dir.glob(f"{model_name}_*"):
            if not model_dir.is_dir():
                continue
            
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check if model name matches
                if metadata.get("model_name") == model_name:
                    # Add full path to model
                    metadata["model_path"] = model_dir
                    models.append(metadata)
            except Exception as e:
                logger.warning(f"Error reading metadata from {metadata_path}: {str(e)}")
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Calculate performance weights for ensembling
        if models:
            await self._calculate_performance_weights(models)
        
        return models
    
    async def _get_model_by_id(self, model_id: str) -> Optional[Dict]:
        """Get model metadata by its ID."""
        # Check if model metadata is in cache
        if model_id in self._metadata_cache:
            return self._metadata_cache[model_id]
        
        # Scan model directory for matching model ID
        for model_dir in self.model_dir.glob("*"):
            if not model_dir.is_dir():
                continue
            
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check if model ID matches
                if metadata.get("model_id") == model_id:
                    # Add full path to model
                    metadata["model_path"] = model_dir
                    
                    # Update cache
                    self._metadata_cache[model_id] = metadata
                    
                    # Manage cache size
                    if len(self._metadata_cache) > self.max_cache_size:
                        # Remove oldest entry
                        oldest_key = next(iter(self._metadata_cache))
                        self._metadata_cache.pop(oldest_key)
                    
                    return metadata
            except Exception as e:
                logger.warning(f"Error reading metadata from {metadata_path}: {str(e)}")
        
        # Try to get from Redis if not found in filesystem
        try:
            redis_key = f"model:training:result:{model_id}"
            model_data = await self.redis_client.get_async(redis_key)
            if model_data:
                metadata = json.loads(model_data)
                metadata["model_path"] = Path(metadata.get("model_path", "")) if metadata.get("model_path") else None
                
                # Update cache
                self._metadata_cache[model_id] = metadata
                return metadata
        except Exception as e:
            logger.warning(f"Error fetching model metadata from Redis: {str(e)}")
        
        return None
    
    async def _get_latest_model_by_name(self, model_name: str) -> Optional[Dict]:
        """Get the latest model with a specific name."""
        models = await self._get_models_by_name(model_name)
        return models[0] if models else None
    
    async def _calculate_performance_weights(self, models: List[Dict]):
        """Calculate performance weights for ensembling based on model metrics."""
        # Initialize weights with default value
        for model in models:
            model["performance_weight"] = 1.0
        
        # Get metrics from all models to normalize
        metrics = {}
        for model in models:
            model_metrics = model.get("metrics", {})
            
            # Use appropriate metric based on model type
            if "f1_score" in model_metrics:
                # Classification model
                metrics[model["model_id"]] = model_metrics.get("f1_score", 0.5)
            elif "r2" in model_metrics:
                # Regression model
                metrics[model["model_id"]] = model_metrics.get("r2", 0.0)
            elif "rmse" in model_metrics:
                # Regression model (lower is better, so invert)
                rmse = model_metrics.get("rmse", 0.0)
                metrics[model["model_id"]] = 1.0 / (1.0 + rmse) if rmse > 0 else 0.5
        
        # Normalize weights
        if metrics:
            min_metric = min(metrics.values())
            max_metric = max(metrics.values())
            
            if max_metric > min_metric:
                # Scale between 0.5 and 2.0 to ensure all models contribute
                for model in models:
                    model_metric = metrics.get(model["model_id"], min_metric)
                    model["performance_weight"] = 0.5 + 1.5 * (model_metric - min_metric) / (max_metric - min_metric)
    
    async def _get_feature_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        num_candles: int,
        features: List[str],
        model_metadata_list: List[Dict],
        use_cached_data: bool
    ) -> pd.DataFrame:
        """Get feature data for prediction, either from cache or by fetching and calculating."""
        cache_key = f"{symbol}_{exchange}_{timeframe}_{num_candles}"
        
        # Check cache expiry
        if cache_key in self._feature_cache_timestamps:
            cache_age = time.time() - self._feature_cache_timestamps[cache_key]
            if cache_age > self._feature_cache_ttl:
                # Cache expired
                if cache_key in self._feature_cache:
                    del self._feature_cache[cache_key]
                if cache_key in self._feature_cache_timestamps:
                    del self._feature_cache_timestamps[cache_key]
        
        # Use cached data if available and requested
        if use_cached_data and cache_key in self._feature_cache:
            logger.debug(f"Using cached feature data for {cache_key}")
            self.prediction_stats["cache_hits"] += 1
            return self._feature_cache[cache_key]
        
        # Cache miss, need to fetch data
        self.prediction_stats["cache_misses"] += 1
        
        # Fetch market data
        end_time = datetime.datetime.now()
        
        # Calculate start time based on number of candles and timeframe
        time_multiplier = 1.5  # Fetch more candles than needed for calculations
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            start_time = end_time - datetime.timedelta(minutes=minutes * num_candles * time_multiplier)
        elif timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            start_time = end_time - datetime.timedelta(hours=hours * num_candles * time_multiplier)
        elif timeframe.endswith('d'):
            days = int(timeframe[:-1])
            start_time = end_time - datetime.timedelta(days=days * num_candles * time_multiplier)
        else:
            # Default to hours if unknown format
            start_time = end_time - datetime.timedelta(hours=num_candles * time_multiplier)
        
        # Fetch market data
        market_data = await self.market_data_repo.get_historical_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        if market_data.empty:
            raise DataValidationError(f"No historical data found for {symbol} on {exchange}")
        
        # Get list of required features from model metadata
        if not features:
            # Combine features from all models
            features = set()
            for metadata in model_metadata_list:
                model_features = metadata.get("preprocessing_info", {}).get("feature_names", [])
                features.update(model_features)
            
            features = list(features)
        
        # Calculate features
        feature_data = await self.feature_extractor.calculate_features(
            market_data=market_data,
            features=features,
            symbol=symbol,
            timeframe=timeframe
        )
        
        # Handle missing values
        feature_data = await run_in_thread_pool(self._handle_missing_values, feature_data)
        
        # Cache the computed features
        self._feature_cache[cache_key] = feature_data
        self._feature_cache_timestamps[cache_key] = time.time()
        
        # Manage cache size
        if len(self._feature_cache) > self.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self._feature_cache_timestamps, key=self._feature_cache_timestamps.get)
            if oldest_key in self._feature_cache:
                del self._feature_cache[oldest_key]
            if oldest_key in self._feature_cache_timestamps:
                del self._feature_cache_timestamps[oldest_key]
        
        return feature_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature data."""
        # Check for missing values
        null_counts = data.isnull().sum()
        if null_counts.sum() > 0:
            cols_with_nulls = null_counts[null_counts > 0].index.tolist()
            logger.debug(f"Handling missing values in columns: {cols_with_nulls}")
            
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
    
    async def _load_model(
        self, 
        model_id: str, 
        model_path: Path, 
        model_type: str
    ) -> Tuple[Any, Dict]:
        """
        Load model and preprocessor from storage.
        
        Returns:
            Tuple containing (model, preprocessor)
        """
        # Check if model is in cache
        if model_id in self._model_cache:
            logger.debug(f"Using cached model for {model_id}")
            return self._model_cache[model_id]
        
        logger.info(f"Loading model {model_id} of type {model_type}")
        
        try:
            # Determine model file path based on type
            if model_type in ['lstm', 'gru', 'cnn', 'mlp', 'transformer'] or 'deep_learning' in model_type:
                # Try TensorFlow first
                tf_model_path = model_path / "model.h5"
                if tf_model_path.exists():
                    model = tf.keras.models.load_model(tf_model_path)
                else:
                    # Try PyTorch
                    pt_model_path = model_path / "model.pt"
                    architecture_path = model_path / "model_architecture.pkl"
                    
                    if pt_model_path.exists() and architecture_path.exists():
                        # Load model architecture
                        with open(architecture_path, 'rb') as f:
                            model_class = pickle.load(f)
                        
                        # Initialize model instance
                        model = model_class()
                        
                        # Load state dict
                        state_dict = torch.load(
                            pt_model_path, 
                            map_location=torch.device('cuda') if self.use_gpu and self.cuda_available else torch.device('cpu')
                        )
                        model.load_state_dict(state_dict)
                        
                        # Set model to evaluation mode
                        model.eval()
                    else:
                        raise ModelLoadingError(f"No compatible model file found in {model_path}")
            else:
                # Scikit-learn based model
                sklearn_model_path = model_path / "model.joblib"
                if sklearn_model_path.exists():
                    model = joblib.load(sklearn_model_path)
                else:
                    raise ModelLoadingError(f"No model file found in {model_path}")
            
            # Load preprocessor
            preprocessor = {}
            metadata_path = model_path / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract preprocessing info
                preprocessing_info = metadata.get("preprocessing_info", {})
                
                # Load scaler if exists
                if "numeric_scaler" in preprocessing_info and preprocessing_info["numeric_scaler"]:
                    scaler_info = preprocessing_info["numeric_scaler"]
                    if "path" in scaler_info:
                        scaler_path = model_path / scaler_info["path"]
                        if scaler_path.exists():
                            scaler = joblib.load(scaler_path)
                            preprocessor["scaler"] = scaler
                            preprocessor["scaler_method"] = scaler_info.get("method", "standard")
                
                # Load encoders if exist
                if "categorical_encoders" in preprocessing_info and preprocessing_info["categorical_encoders"]:
                    encoders = {}
                    for col, encoder_path in preprocessing_info["categorical_encoders"].items():
                        if isinstance(encoder_path, str) and encoder_path.startswith("encoders/"):
                            full_path = model_path / encoder_path
                            if full_path.exists():
                                encoder = joblib.load(full_path)
                                encoders[col] = encoder
                    
                    if encoders:
                        preprocessor["encoders"] = encoders
                
                # Store feature names
                if "feature_names" in preprocessing_info:
                    preprocessor["feature_names"] = preprocessing_info["feature_names"]
            
            # Update model cache
            self._model_cache[model_id] = (model, preprocessor)
            
            # Manage cache size
            if len(self._model_cache) > self.max_cache_size:
                # Remove oldest entry (first key)
                oldest_key = next(iter(self._model_cache))
                self._model_cache.pop(oldest_key)
            
            return model, preprocessor
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}", exc_info=True)
            raise ModelLoadingError(f"Failed to load model {model_id}: {str(e)}") from e
    
    async def _prepare_features_for_model(
        self, 
        feature_data: pd.DataFrame, 
        model_metadata: Dict, 
        preprocessor: Dict
    ) -> pd.DataFrame:
        """
        Prepare feature data for a specific model using its preprocessing requirements.
        
        Returns:
            Processed feature data ready for prediction
        """
        # Get feature names for this model
        feature_names = preprocessor.get("feature_names", [])
        
        if not feature_names:
            # Fallback to metadata
            feature_names = model_metadata.get("features", [])
        
        # Check if we have all required features
        missing_features = [f for f in feature_names if f not in feature_data.columns]
        if missing_features:
            logger.warning(f"Missing features for model {model_metadata['model_id']}: {missing_features}")
            # Use available features only
            available_features = [f for f in feature_names if f in feature_data.columns]
            if not available_features:
                raise DataValidationError(f"No required features available for model {model_metadata['model_id']}")
            feature_names = available_features
        
        # Create copy of data with only required features
        X = feature_data[feature_names].copy()
        
        # Apply preprocessing
        
        # Handle numeric scaling
        if "scaler" in preprocessor:
            scaler = preprocessor["scaler"]
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
            
            if not numeric_cols.empty:
                # Get intersection of available numeric columns and those the scaler expects
                if hasattr(scaler, 'feature_names_in_'):
                    expected_cols = scaler.feature_names_in_
                    numeric_cols = [col for col in numeric_cols if col in expected_cols]
                
                # Apply scaling
                if numeric_cols:
                    X[numeric_cols] = scaler.transform(X[numeric_cols])
        
        # Handle categorical encoding
        if "encoders" in preprocessor:
            encoders = preprocessor["encoders"]
            for col, encoder in encoders.items():
                if col in X.columns:
                    # Get encoded column(s)
                    encoded = encoder.transform(X[[col]])
                    
                    # If encoder returns a dense array with multiple columns
                    if hasattr(encoded, 'shape') and len(encoded.shape) > 1 and encoded.shape[1] > 1:
                        # Create column names for each encoded feature
                        encoded_cols = [f"{col}_{i}" for i in range(encoded.shape[1])]
                        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=X.index)
                        
                        # Remove original column and add encoded columns
                        X = X.drop(columns=[col])
                        X = pd.concat([X, encoded_df], axis=1)
                    else:
                        # Single column output, replace original
                        X[col] = encoded
        
        # Select most recent row for prediction
        if len(X) > 1:
            X = X.iloc[-1:].reset_index(drop=True)
        
        return X
    
    async def _make_single_prediction(
        self, 
        model, 
        features: pd.DataFrame, 
        model_metadata: Dict,
        calculate_confidence: bool
    ) -> Tuple[Any, Optional[float]]:
        """
        Make prediction with a single model and calculate confidence if requested.
        
        Returns:
            Tuple containing (prediction_result, confidence_score)
        """
        model_type = model_metadata.get("model_type", "")
        prediction_type = model_metadata.get("prediction_type", "regression")
        
        try:
            # Make prediction based on model type
            if isinstance(model, tf.keras.Model):
                # For TensorFlow models
                prediction = model.predict(features.values)
                
                # Format prediction result
                if prediction_type == "classification":
                    if prediction.shape[1] > 1:
                        # Multi-class classification
                        prediction_result = prediction[0].tolist()
                    else:
                        # Binary classification
                        prediction_result = float(prediction[0][0])
                else:
                    # Regression
                    prediction_result = float(prediction[0][0])
            
            elif isinstance(model, nn.Module):
                # For PyTorch models
                device = torch.device("cuda:0" if self.use_gpu and self.cuda_available else "cpu")
                model.to(device)
                model.eval()
                
                with torch.no_grad():
                    features_tensor = torch.tensor(features.values, dtype=torch.float32).to(device)
                    prediction = model(features_tensor).cpu().numpy()
                
                # Format prediction result
                if prediction_type == "classification":
                    if prediction.shape[1] > 1:
                        # Multi-class classification
                        prediction_result = prediction[0].tolist()
                    else:
                        # Binary classification
                        prediction_result = float(prediction[0][0])
                else:
                    # Regression
                    prediction_result = float(prediction[0][0])
            
            else:
                # For scikit-learn based models
                
                # Check if model supports probability estimates
                if prediction_type == "classification" and hasattr(model, "predict_proba"):
                    # Get class probabilities
                    proba = model.predict_proba(features)
                    classes = model.classes_
                    
                    if len(classes) > 2:
                        # Multi-class
                        prediction_result = proba[0].tolist()
                    else:
                        # Binary classification
                        prediction_result = float(proba[0][1])  # Probability of positive class
                else:
                    # Use regular prediction
                    prediction = model.predict(features)
                    
                    if prediction.shape[0] == 1:
                        prediction_result = float(prediction[0])
                    else:
                        prediction_result = prediction.tolist()
            
            # Calculate confidence score if requested
            confidence = None
            if calculate_confidence:
                if prediction_type == "classification":
                    # For classification, use prediction probability
                    if isinstance(prediction_result, list):
                        # Multi-class, use max probability
                        confidence = max(prediction_result)
                    elif hasattr(model, "predict_proba"):
                        # Binary classification with probability
                        proba = model.predict_proba(features)
                        confidence = float(np.max(proba[0]))
                    else:
                        # Default confidence
                        confidence = 0.8
                else:
                    # For regression, use model-specific methods
                    confidence = await self._estimate_regression_confidence(
                        model, features, prediction_result, model_type
                    )
            
            return prediction_result, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction with model {model_metadata['model_id']}: {str(e)}", exc_info=True)
            raise PredictionError(f"Failed to make prediction: {str(e)}") from e
    
    async def _estimate_regression_confidence(
        self, 
        model, 
        features: pd.DataFrame, 
        prediction: float, 
        model_type: str
    ) -> float:
        """
        Estimate prediction confidence for regression models.
        
        Returns:
            Confidence score between 0 and 1
        """
        # Different methods depending on model type
        if hasattr(model, "predict_proba"):
            # Some models provide prediction intervals
            try:
                proba = model.predict_proba(features)
                return float(np.max(proba[0]))
            except:
                pass
        
        if model_type in ["random_forest", "gradient_boosting", "xgboost", "lightgbm"]:
            # For tree-based models, use prediction standard deviation
            if hasattr(model, "estimators_"):
                # Get predictions from all trees
                predictions = []
                for estimator in model.estimators_:
                    est_pred = estimator.predict(features)
                    predictions.append(float(est_pred[0]))
                
                # Calculate std dev and normalize to confidence
                if len(predictions) > 1:
                    std_dev = np.std(predictions)
                    mean_pred = np.mean(predictions)
                    cv = std_dev / mean_pred if mean_pred != 0 else 1.0
                    
                    # Convert coefficient of variation to confidence (lower CV = higher confidence)
                    return 1.0 / (1.0 + cv)
        
        # Default confidence based on model type
        default_confidences = {
            "random_forest": 0.8,
            "gradient_boosting": 0.75,
            "xgboost": 0.85,
            "lightgbm": 0.85,
            "lstm": 0.75,
            "gru": 0.75,
            "cnn": 0.7,
            "linear": 0.6
        }
        
        return default_confidences.get(model_type, 0.7)
    
    async def _combine_predictions(
        self, 
        predictions: List[Dict], 
        ensemble_method: str, 
        prediction_type: str
    ) -> Dict:
        """
        Combine predictions from multiple models using the specified method.
        
        Returns:
            Dict with combined prediction and confidence
        """
        if len(predictions) == 1:
            return {
                "prediction": predictions[0]["prediction"],
                "confidence": predictions[0]["confidence"]
            }
        
        # Extract predictions and weights
        pred_values = [p["prediction"] for p in predictions]
        weights = [p.get("model_weight", 1.0) for p in predictions]
        confidences = [p.get("confidence", 0.7) for p in predictions]
        
        # Normalize weights
        sum_weights = sum(weights)
        if sum_weights > 0:
            weights = [w / sum_weights for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        # Handle different prediction types
        if prediction_type == "classification":
            # Check if predictions are class probabilities (lists)
            if isinstance(pred_values[0], list):
                # Multi-class classification
                num_classes = len(pred_values[0])
                ensemble_pred = [0] * num_classes
                
                if ensemble_method == "weighted_average":
                    # Weighted average of probabilities
                    for i, pred in enumerate(pred_values):
                        for c in range(num_classes):
                            ensemble_pred[c] += pred[c] * weights[i]
                else:
                    # Simple average
                    for pred in pred_values:
                        for c in range(num_classes):
                            ensemble_pred[c] += pred[c]
                    
                    # Normalize
                    ensemble_pred = [p / len(pred_values) for p in ensemble_pred]
                
                # Normalize to ensure sum = 1
                sum_probs = sum(ensemble_pred)
                if sum_probs > 0:
                    ensemble_pred = [p / sum_probs for p in ensemble_pred]
                
                # Confidence is the max probability
                ensemble_conf = max(ensemble_pred)
                
            else:
                # Binary classification with probabilities
                if ensemble_method == "weighted_average":
                    ensemble_pred = sum(p * w for p, w in zip(pred_values, weights))
                else:
                    ensemble_pred = sum(pred_values) / len(pred_values)
                
                # For binary, confidence is how far from 0.5
                ensemble_conf = abs(ensemble_pred - 0.5) * 2
        else:
            # Regression
            if ensemble_method == "weighted_average":
                # Weighted average of predictions
                ensemble_pred = sum(p * w for p, w in zip(pred_values, weights))
                
                # Weight confidences by model weights
                ensemble_conf = sum(c * w for c, w in zip(confidences, weights))
            else:
                # Simple average
                ensemble_pred = sum(pred_values) / len(pred_values)
                
                # Average confidence
                ensemble_conf = sum(confidences) / len(confidences)
                
                # Adjust confidence based on prediction agreement
                std_dev = np.std(pred_values)
                mean_pred = np.mean(pred_values)
                cv = std_dev / abs(mean_pred) if mean_pred != 0 else 1.0
                
                # Higher agreement = higher confidence
                agreement_factor = 1.0 / (1.0 + cv)
                ensemble_conf = 0.7 * ensemble_conf + 0.3 * agreement_factor
        
        return {
            "prediction": ensemble_pred,
            "confidence": ensemble_conf
        }
    
    def _update_prediction_stats(self, prediction_time: float):
        """Update prediction statistics."""
        self.prediction_stats["total_prediction_time"] += prediction_time
        self.prediction_stats["avg_prediction_time"] = (
            self.prediction_stats["total_prediction_time"] / self.prediction_stats["total_predictions"]
        )
    
    async def get_prediction_stats(self) -> Dict:
        """Get current prediction statistics."""
        return self.prediction_stats
    
    async def clear_cache(self):
        """Clear all caches to free memory."""
        self._model_cache.clear()
        self._metadata_cache.clear()
        self._preprocessor_cache.clear()
        self._feature_cache.clear()
        self._feature_cache_timestamps.clear()
        
        # Clean up TensorFlow memory if used
        if self.tf_gpu_available:
            tf.keras.backend.clear_session()
        
        logger.info("All prediction caches cleared")
    
    async def clean_up(self):
        """Clean up resources before shutdown."""
        await self.clear_cache()

        # Additional cleanup if needed
        logger.info("ModelPredictor resources cleaned up")


def predict_momentum_score(data: pd.DataFrame) -> float:
    """Simple heuristic to compute a momentum score from price data."""
    if data.empty:
        return 0.0
    # Use percent change as a basic momentum measure
    momentum = data.pct_change().mean().mean()
    return float(momentum)


def predict_mean_reversion_probability(data: pd.DataFrame) -> float:
    """Return a naive probability of mean reversion based on momentum."""
    momentum = predict_momentum_score(data)
    return float(max(0.0, 1.0 - abs(momentum)))


