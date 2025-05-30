#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
ML Brain - Advanced Machine Learning Trading Strategy

This module implements sophisticated machine learning-based trading strategies including:
- Ensemble models for classification and regression
- Deep learning for price prediction and pattern recognition
- Reinforcement learning for adaptive trading
- Feature importance analysis and selection
- Model explainability for trading decisions
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from joblib import Parallel, delayed

# Import mixin for historical learning
from .historical_memory import HistoricalMemoryMixin

# Internal imports
from common.utils import calculate_sharpe_ratio, calculate_sortino_ratio
from common.constants import (
    DEFAULT_LOOKBACK_PERIODS, DEFAULT_CONFIDENCE_LEVELS, DEFAULT_MODEL_TYPES,
    DEFAULT_FEATURE_IMPORTANCE_THRESHOLD, ML_MODEL_VOTING_WEIGHTS,
    DEFAULT_PREDICTION_HORIZONS, DEFAULT_CONFIDENCE_SCALING_FACTOR,
    MARKET_REGIMES
)
from common.async_utils import run_in_executor
from feature_service.features.technical import calculate_technical_features
from feature_service.features.volatility import calculate_volatility_features
from feature_service.features.volume import calculate_volume_features
from feature_service.features.market_structure import calculate_structure_features
from data_storage.market_data import MarketDataRepository
from ml_models.models.classification import ClassificationModel
from ml_models.models.regression import RegressionModel
from ml_models.models.deep_learning import DeepLearningModel
from ml_models.models.ensemble import EnsembleModel
from ml_models.model_manager import ModelManager
from ml_models.feature_importance import FeatureImportanceAnalyzer
from strategy_brains.base_brain import BaseBrain


logger = logging.getLogger(__name__)


class MLBrain(HistoricalMemoryMixin, BaseBrain):
    """
    Advanced Machine Learning Trading Brain implementing various ML-based strategies
    with model selection, ensemble techniques, and continuous learning.
    """
    
    def __init__(
        self,
        asset_id: str,
        timeframe: str,
        parameters: Dict[str, Any] = None,
        name: str = "ml_brain",
        platform: str = "binance",
        market_data_repo: Optional[MarketDataRepository] = None,
        model_manager: Optional[ModelManager] = None,
    ):
        """
        Initialize the ML Brain with configuration parameters
        
        Args:
            asset_id: Trading asset identifier
            timeframe: Trading timeframe
            parameters: Configuration parameters for the strategy
            name: Brain name for identification
            platform: Trading platform (binance/deriv)
            market_data_repo: Repository for market data access
            model_manager: Manager for ML models
        """
        HistoricalMemoryMixin.__init__(
            self,
            short_window=(parameters or {}).get("short_memory", 50),
            long_window=(parameters or {}).get("long_memory", 500),
        )

        super().__init__(
            asset_id=asset_id,
            timeframe=timeframe,
            parameters=parameters or {},
            name=name,
            platform=platform,
            market_data_repo=market_data_repo,
        )
        
        # Strategy parameters with sensible defaults
        self.lookback_periods = self.parameters.get("lookback_periods", DEFAULT_LOOKBACK_PERIODS)
        self.confidence_level = self.parameters.get("confidence_level", DEFAULT_CONFIDENCE_LEVELS["ml"])
        self.model_types = self.parameters.get("model_types", DEFAULT_MODEL_TYPES)
        self.feature_importance_threshold = self.parameters.get(
            "feature_importance_threshold", DEFAULT_FEATURE_IMPORTANCE_THRESHOLD
        )
        self.prediction_horizons = self.parameters.get("prediction_horizons", DEFAULT_PREDICTION_HORIZONS)
        self.confidence_scaling = self.parameters.get("confidence_scaling", DEFAULT_CONFIDENCE_SCALING_FACTOR)
        self.voting_weights = self.parameters.get("voting_weights", ML_MODEL_VOTING_WEIGHTS)
        
        # Strategy state and components
        self.features_cache = {}
        self.last_signals = {}
        self.current_regime = None
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Model management
        self.model_manager = model_manager or ModelManager()
        self.models = {}
        self.feature_analyzer = FeatureImportanceAnalyzer()
        
        # Classification models for direction prediction
        self.classification_models = {}
        
        # Regression models for price target prediction
        self.regression_models = {}
        
        # Deep learning models for complex pattern recognition
        self.deep_learning_models = {}
        
        # Ensemble model for combining predictions
        self.ensemble_model = EnsembleModel()
        
        # Asset-specific optimization
        self._initialize_asset_specific_parameters()
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"ML Brain initialized for {asset_id} on {platform} with {timeframe} timeframe")
    
    def _initialize_asset_specific_parameters(self):
        """Initialize parameters specifically optimized for the current asset"""
        # For production, these would be loaded from a database of optimized parameters
        # We simulate asset-specific optimization here with asset-based variations
        asset_hash = hash(self.asset_id) % 100
        
        # Adjust prediction horizons based on asset volatility
        # This would normally come from a volatility analysis
        volatility_factor = 0.8 + (asset_hash % 20) / 100  # 0.8 to 1.0
        
        # Adjust prediction horizons
        self.prediction_horizons = [
            max(1, int(horizon * volatility_factor))
            for horizon in self.prediction_horizons
        ]
        
        # Adjust model weights based on asset characteristics
        # Example: More weight to ML for crypto, more weight to traditional for forex
        if "BTC" in self.asset_id or "ETH" in self.asset_id:
            # Crypto assets - more weight to deep learning
            self.voting_weights = {
                "classification": 0.25,
                "regression": 0.25,
                "deep_learning": 0.40,
                "ensemble": 0.10
            }
        elif "USD" in self.asset_id or "EUR" in self.asset_id:
            # Forex pairs - more weight to traditional models
            self.voting_weights = {
                "classification": 0.35,
                "regression": 0.35,
                "deep_learning": 0.20,
                "ensemble": 0.10
            }
        
        logger.debug(f"Asset-specific parameters initialized for {self.asset_id}")
    
    def _initialize_models(self):
        """Initialize ML models for the strategy"""
        model_suffix = f"{self.asset_id}_{self.timeframe}"
        
        # Create unique model IDs
        classification_id = f"clf_{model_suffix}"
        regression_id = f"reg_{model_suffix}"
        deep_learning_id = f"dl_{model_suffix}"
        ensemble_id = f"ens_{model_suffix}"
        
        # Initialize or load classification models
        self.classification_models = {
            "xgboost": self.model_manager.get_or_create_model(
                f"xgb_{classification_id}", 
                lambda: xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    scale_pos_weight=1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            ),
            "lightgbm": self.model_manager.get_or_create_model(
                f"lgb_{classification_id}",
                lambda: lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary',
                    random_state=42
                )
            ),
            "random_forest": self.model_manager.get_or_create_model(
                f"rf_{classification_id}",
                lambda: RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            )
        }
        
        # Initialize or load regression models
        self.regression_models = {
            "xgboost": self.model_manager.get_or_create_model(
                f"xgb_{regression_id}",
                lambda: xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='reg:squarederror',
                    random_state=42
                )
            ),
            "lightgbm": self.model_manager.get_or_create_model(
                f"lgb_{regression_id}",
                lambda: lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='regression',
                    random_state=42
                )
            ),
            "gradient_boosting": self.model_manager.get_or_create_model(
                f"gb_{regression_id}",
                lambda: GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            )
        }
        
        # Initialize or load deep learning models
        self.deep_learning_models = {
            "lstm": self.model_manager.get_or_create_model(
                f"lstm_{deep_learning_id}",
                lambda: DeepLearningModel(
                    model_type="lstm",
                    input_dim=None,  # Will be set during training
                    output_dim=1,
                    hidden_layers=[64, 32],
                    dropout_rate=0.2
                )
            ),
            "gru": self.model_manager.get_or_create_model(
                f"gru_{deep_learning_id}",
                lambda: DeepLearningModel(
                    model_type="gru",
                    input_dim=None,  # Will be set during training
                    output_dim=1,
                    hidden_layers=[64, 32],
                    dropout_rate=0.2
                )
            ),
            "cnn": self.model_manager.get_or_create_model(
                f"cnn_{deep_learning_id}",
                lambda: DeepLearningModel(
                    model_type="cnn",
                    input_dim=None,  # Will be set during training
                    output_dim=1,
                    hidden_layers=[64, 32],
                    dropout_rate=0.2
                )
            )
        }
        
        # Ensemble model to combine predictions
        self.ensemble_model = self.model_manager.get_or_create_model(
            ensemble_id,
            lambda: EnsembleModel(
                models=[
                    self.classification_models["xgboost"],
                    self.classification_models["lightgbm"],
                    self.classification_models["random_forest"],
                    self.regression_models["xgboost"],
                    self.regression_models["lightgbm"],
                    self.regression_models["gradient_boosting"],
                    self.deep_learning_models["lstm"],
                    self.deep_learning_models["gru"],
                    self.deep_learning_models["cnn"]
                ],
                weights=None  # Will be set during training based on performance
            )
        )
        
        logger.debug(f"ML models initialized for {self.asset_id}")
    
    async def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data using machine learning models
        
        Args:
            data: Market data in pandas DataFrame format
            
        Returns:
            dict: Analysis results including model predictions and signals
        """
        logger.debug(f"ML analysis started for {self.asset_id}")
        
        # Detect current market regime
        self.current_regime = await self._detect_market_regime(data)
        
        # Prepare features for model input
        features = await self._prepare_features(data)
        
        # Generate predictions from different models
        predictions = await self._generate_predictions(features)
        
        # Combine predictions using ensemble techniques
        ensemble_prediction = await self._ensemble_predictions(predictions)
        
        # Calculate confidence based on prediction consistency
        confidence = await self._calculate_confidence(predictions, ensemble_prediction)
        
        # Generate trading signals with entry/exit points
        signals = await self._generate_signals(ensemble_prediction, confidence, data)
        
        # Store last signals for reference
        self.last_signals = signals
        
        logger.debug(f"ML analysis completed for {self.asset_id}, signal: {signals['signal']}, confidence: {signals['confidence']:.4f}")
        
        return {
            "signal": signals["signal"],
            "confidence": signals["confidence"],
            "entry_price": signals["entry_price"],
            "stop_loss": signals["stop_loss"],
            "take_profit": signals["take_profit"],
            "regime": self.current_regime,
            "predictions": predictions,
            "ensemble": ensemble_prediction,
            "feature_importance": self.feature_importance,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Detect the current market regime using ML techniques
        
        Args:
            data: Market data
            
        Returns:
            str: Identified market regime
        """
        # Calculate key features for regime detection
        returns = np.log(data['close'] / data['close'].shift(1)).dropna()
        volatility = returns.rolling(20).std().iloc[-1]
        
        # Calculate trend strength using linear regression slope
        x = np.arange(len(data['close'][-20:]))
        y = data['close'][-20:].values
        slope, _, r_value, _, _ = np.polyfit(x, y, 1, full=True)[0:5]
        trend_strength = abs(r_value[0]) if isinstance(r_value, np.ndarray) else abs(r_value)
        
        # Calculate trading range
        high_low_range = (data['high'].iloc[-20:].max() - data['low'].iloc[-20:].min()) / data['close'].iloc[-1]
        
        # Use a simple ML classifier for regime detection
        # In a real system, this would be a trained model
        
        # Simple rule-based regime detection (would be ML-based in production)
        if volatility > 1.5 * returns.rolling(50).std().mean():
            if trend_strength > 0.6:
                regime = MARKET_REGIMES["TRENDING_VOLATILE"]
            else:
                regime = MARKET_REGIMES["CHOPPY_VOLATILE"]
        else:
            if trend_strength > 0.6:
                regime = MARKET_REGIMES["TRENDING_CALM"]
            else:
                if high_low_range < 0.03:  # Tight range
                    regime = MARKET_REGIMES["RANGING_TIGHT"]
                else:
                    regime = MARKET_REGIMES["RANGING_NORMAL"]
        
        logger.debug(f"Detected market regime: {regime} for {self.asset_id}")
        return regime
    
    async def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model input
        
        Args:
            data: Market data
            
        Returns:
            DataFrame: Engineered features
        """
        # Check if we have recent cached features
        cache_key = f"{len(data)}_{data['timestamp'].iloc[-1]}"
        if cache_key in self.features_cache:
            logger.debug(f"Using cached features for {self.asset_id}")
            return self.features_cache[cache_key]
        
        # Ensure we have enough data
        if len(data) < 50:
            logger.warning(f"Insufficient data for ML feature calculation: {len(data)} rows")
            return pd.DataFrame()
        
        # Run feature calculations in parallel using executor
        technical_features = await run_in_executor(
            calculate_technical_features, data, extended=True
        )
        
        volatility_features = await run_in_executor(
            calculate_volatility_features, data
        )
        
        volume_features = await run_in_executor(
            calculate_volume_features, data
        )
        
        structure_features = await run_in_executor(
            calculate_structure_features, data
        )
        
        # Combine all features
        all_features = pd.concat(
            [technical_features, volatility_features, volume_features, structure_features],
            axis=1
        )
        
        # Handle NaN values - forward fill, then backfill, then replace remaining with 0
        all_features = all_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Feature selection based on importance (if available)
        if self.feature_importance:
            important_features = [
                feature for feature, importance in self.feature_importance.items()
                if importance >= self.feature_importance_threshold
            ]
            
            if important_features:
                all_features = all_features[important_features]
        
        # Calculate target variables for different prediction horizons
        for horizon in self.prediction_horizons:
            # Classification target: price direction (up=1, down=0)
            all_features[f'target_direction_{horizon}'] = (
                data['close'].shift(-horizon) > data['close']
            ).astype(int)
            
            # Regression target: price change percentage
            all_features[f'target_return_{horizon}'] = (
                (data['close'].shift(-horizon) - data['close']) / data['close']
            )
        
        # Drop rows with NaN in target variables (due to future shift)
        all_features = all_features.dropna()
        
        # Cache features for reuse
        self.features_cache[cache_key] = all_features
        
        logger.debug(f"Prepared {all_features.shape[1]} features with {all_features.shape[0]} samples for {self.asset_id}")
        return all_features
    
    async def _generate_predictions(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate predictions from different ML models
        
        Args:
            features: Engineered features
            
        Returns:
            dict: Predictions from various models
        """
        if features.empty:
            logger.warning("Empty features dataframe, cannot generate predictions")
            return {
                "classification": {},
                "regression": {},
                "deep_learning": {},
                "direction": None,
                "price_change": None,
                "price_targets": {}
            }
        
        logger.debug(f"Generating predictions for {self.asset_id}")
        
        # Select latest data point for prediction
        prediction_features = features.drop(
            [col for col in features.columns if col.startswith('target_')],
            axis=1
        ).iloc[-1:].copy()
        
        # Classification predictions (direction)
        classification_predictions = {}
        for name, model in self.classification_models.items():
            try:
                for horizon in self.prediction_horizons:
                    target_col = f'target_direction_{horizon}'
                    
                    # Skip if model not trained
                    if not hasattr(model, 'trained') or not model.trained:
                        await self._train_model(
                            model, features.drop([target_col], axis=1), features[target_col], 'classification'
                        )
                    
                    # Get prediction and probability
                    pred = model.predict(prediction_features)
                    prob = model.predict_proba(prediction_features)
                    
                    # Store prediction and probability
                    if name not in classification_predictions:
                        classification_predictions[name] = {}
                    
                    classification_predictions[name][horizon] = {
                        'direction': int(pred[0]),
                        'probability': prob[0][1] if pred[0] == 1 else 1 - prob[0][1]
                    }
            except Exception as e:
                logger.error(f"Classification prediction failed for {name}: {str(e)}")
                classification_predictions[name] = {
                    horizon: {'direction': None, 'probability': 0.5} for horizon in self.prediction_horizons
                }
        
        # Regression predictions (price change)
        regression_predictions = {}
        for name, model in self.regression_models.items():
            try:
                for horizon in self.prediction_horizons:
                    target_col = f'target_return_{horizon}'
                    
                    # Skip if model not trained
                    if not hasattr(model, 'trained') or not model.trained:
                        await self._train_model(
                            model, features.drop([target_col], axis=1), features[target_col], 'regression'
                        )
                    
                    # Get prediction
                    pred = model.predict(prediction_features)
                    
                    # Store prediction
                    if name not in regression_predictions:
                        regression_predictions[name] = {}
                    
                    regression_predictions[name][horizon] = float(pred[0])
            except Exception as e:
                logger.error(f"Regression prediction failed for {name}: {str(e)}")
                regression_predictions[name] = {
                    horizon: 0.0 for horizon in self.prediction_horizons
                }
        
        # Deep learning predictions
        deep_learning_predictions = {}
        for name, model in self.deep_learning_models.items():
            try:
                for horizon in self.prediction_horizons:
                    target_col = f'target_return_{horizon}'
                    
                    # Skip if model not trained
                    if not hasattr(model, 'trained') or not model.trained:
                        # Prepare sequence data for deep learning
                        sequence_length = 10  # Look back 10 periods
                        X, y = self._prepare_sequence_data(
                            features.drop([col for col in features.columns if col.startswith('target_')], axis=1),
                            features[target_col],
                            sequence_length
                        )
                        
                        await self._train_model(model, X, y, 'deep_learning')
                    
                    # Prepare sequence data for prediction
                    X_pred = self._prepare_prediction_sequence(
                        features.drop([col for col in features.columns if col.startswith('target_')], axis=1),
                        sequence_length
                    )
                    
                    # Get prediction
                    pred = model.predict(X_pred)
                    
                    # Store prediction
                    if name not in deep_learning_predictions:
                        deep_learning_predictions[name] = {}
                    
                    deep_learning_predictions[name][horizon] = float(pred[0][0])
            except Exception as e:
                logger.error(f"Deep learning prediction failed for {name}: {str(e)}")
                deep_learning_predictions[name] = {
                    horizon: 0.0 for horizon in self.prediction_horizons
                }
        
        # Calculate consensus direction and price change
        direction_votes = {}
        price_changes = {}
        
        for horizon in self.prediction_horizons:
            # Count direction votes
            votes_up = sum(1 for name, preds in classification_predictions.items() 
                         if horizon in preds and preds[horizon]['direction'] == 1)
            votes_down = sum(1 for name, preds in classification_predictions.items() 
                           if horizon in preds and preds[horizon]['direction'] == 0)
            
            direction_votes[horizon] = 1 if votes_up > votes_down else 0
            
            # Average price change predictions
            reg_changes = [preds[horizon] for name, preds in regression_predictions.items() 
                          if horizon in preds and preds[horizon] is not None]
            
            dl_changes = [preds[horizon] for name, preds in deep_learning_predictions.items() 
                         if horizon in preds and preds[horizon] is not None]
            
            # Weight deep learning more in volatile regimes
            if self.current_regime in [MARKET_REGIMES["TRENDING_VOLATILE"], MARKET_REGIMES["CHOPPY_VOLATILE"]]:
                dl_weight = 0.6
                reg_weight = 0.4
            else:
                dl_weight = 0.4
                reg_weight = 0.6
            
            # Calculate weighted average if we have predictions
            if reg_changes and dl_changes:
                price_changes[horizon] = (sum(reg_changes) * reg_weight / len(reg_changes) + 
                                        sum(dl_changes) * dl_weight / len(dl_changes))
            elif reg_changes:
                price_changes[horizon] = sum(reg_changes) / len(reg_changes)
            elif dl_changes:
                price_changes[horizon] = sum(dl_changes) / len(dl_changes)
            else:
                price_changes[horizon] = 0.0
        
        # Calculate price targets
        current_price = features['close'].iloc[-1] if 'close' in features else 0
        price_targets = {
            horizon: current_price * (1 + change) 
            for horizon, change in price_changes.items()
        }
        
        logger.debug(f"Generated predictions for {self.asset_id} across {len(self.prediction_horizons)} horizons")
        
        return {
            "classification": classification_predictions,
            "regression": regression_predictions,
            "deep_learning": deep_learning_predictions,
            "direction": direction_votes,
            "price_change": price_changes,
            "price_targets": price_targets
        }
    
    async def _train_model(self, model, X, y, model_type):
        """
        Train a model asynchronously
        
        Args:
            model: Model to train
            X: Features
            y: Target
            model_type: Type of model (classification, regression, deep_learning)
        """
        try:
            logger.debug(f"Training {model_type} model for {self.asset_id}")
            
            # Train classifier or regressor
            if model_type in ['classification', 'regression']:
                # Use time series split for validation
                tscv = TimeSeriesSplit(n_splits=5)
                best_score = -np.inf
                best_params = None
                
                # Simple hyperparameter tuning
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1]
                }
                
                for n_estimators in param_grid['n_estimators']:
                    for max_depth in param_grid['max_depth']:
                        for learning_rate in param_grid['learning_rate']:
                            params = {
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'learning_rate': learning_rate
                            }
                            
                            # Set parameters if model supports it
                            if hasattr(model, 'set_params'):
                                model.set_params(**params)
                            
                            scores = []
                            for train_idx, test_idx in tscv.split(X):
                                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                                
                                model.fit(X_train, y_train)
                                if model_type == 'classification':
                                    score = model.score(X_test, y_test)
                                else:
                                    y_pred = model.predict(X_test)
                                    score = -np.mean((y_test - y_pred) ** 2)  # Negative MSE
                                
                                scores.append(score)
                            
                            avg_score = np.mean(scores)
                            if avg_score > best_score:
                                best_score = avg_score
                                best_params = params
                
                # Train final model with best params
                if best_params and hasattr(model, 'set_params'):
                    model.set_params(**best_params)
                
                model.fit(X, y)
                model.trained = True
                
                # Update feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self._update_feature_importance(model.feature_importances_, X.columns)
            
            # Train deep learning model
            elif model_type == 'deep_learning':
                # Basic training for sequence models
                model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
                model.trained = True
        
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
    
    def _prepare_sequence_data(self, features, target, sequence_length):
        """
        Prepare sequence data for deep learning models
        
        Args:
            features: Feature dataframe
            target: Target series
            sequence_length: Length of each sequence
            
        Returns:
            tuple: (X_sequences, y_targets)
        """
        # Convert to numpy for sequence creation
        feature_array = features.values
        target_array = target.values
        
        X_sequences = []
        y_targets = []
        
        # Create sequences
        for i in range(len(feature_array) - sequence_length):
            X_sequences.append(feature_array[i:i+sequence_length])
            y_targets.append(target_array[i+sequence_length])
        
        return np.array(X_sequences), np.array(y_targets)
    
    def _prepare_prediction_sequence(self, features, sequence_length):
        """
        Prepare sequence data for prediction
        
        Args:
            features: Feature dataframe
            sequence_length: Length of sequence
            
        Returns:
            array: Sequence for prediction
        """
        # Get the most recent sequence
        feature_array = features.values
        sequence = feature_array[-sequence_length:]
        
        # Reshape for model input (batch_size, sequence_length, features)
        return np.array([sequence])
    
    def _update_feature_importance(self, importances, feature_names):
        """
        Update feature importance dictionary
        
        Args:
            importances: Feature importance values
            feature_names: Feature names
        """
        # Create or update feature importance dictionary
        for i, feature in enumerate(feature_names):
            if feature in self.feature_importance:
                # Rolling average of importance
                self.feature_importance[feature] = (
                    self.feature_importance[feature] * 0.7 + importances[i] * 0.3
                )
            else:
                self.feature_importance[feature] = importances[i]
    
    async def _ensemble_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine predictions using ensemble techniques
        
        Args:
            predictions: Predictions from various models
            
        Returns:
            dict: Ensemble predictions
        """
        ensemble_results = {}
        
        # Check if we have predictions
        if not predictions["direction"]:
            logger.warning("No predictions available for ensemble")
            return {
                "direction": {horizon: None for horizon in self.prediction_horizons},
                "price_change": {horizon: 0.0 for horizon in self.prediction_horizons},
                "probability": {horizon: 0.5 for horizon in self.prediction_horizons}
            }
        
        # For each prediction horizon
        for horizon in self.prediction_horizons:
            # Direction prediction - weighted voting
            direction_probs = []
            
            # Get classification probabilities
            for name, model_preds in predictions["classification"].items():
                if horizon in model_preds and model_preds[horizon]["direction"] is not None:
                    # Higher weight for more confident predictions
                    confidence = model_preds[horizon]["probability"]
                    direction = model_preds[horizon]["direction"]
                    weight = confidence * self.voting_weights["classification"] / len(predictions["classification"])
                    
                    direction_probs.append((direction, weight))
            
            # Get implied direction from regression models
            for name, model_preds in predictions["regression"].items():
                if horizon in model_preds and model_preds[horizon] is not None:
                    # Direction implied by predicted return
                    price_change = model_preds[horizon]
                    implied_direction = 1 if price_change > 0 else 0
                    confidence = min(1.0, abs(price_change) * 10)  # Scale small changes
                    weight = confidence * self.voting_weights["regression"] / len(predictions["regression"])
                    
                    direction_probs.append((implied_direction, weight))
            
            # Get implied direction from deep learning models
            for name, model_preds in predictions["deep_learning"].items():
                if horizon in model_preds and model_preds[horizon] is not None:
                    # Direction implied by predicted return
                    price_change = model_preds[horizon]
                    implied_direction = 1 if price_change > 0 else 0
                    confidence = min(1.0, abs(price_change) * 10)  # Scale small changes
                    weight = confidence * self.voting_weights["deep_learning"] / len(predictions["deep_learning"])
                    
                    direction_probs.append((implied_direction, weight))
            
            # Calculate weighted direction probability
            if direction_probs:
                up_votes = sum(weight for direction, weight in direction_probs if direction == 1)
                down_votes = sum(weight for direction, weight in direction_probs if direction == 0)
                
                total_votes = up_votes + down_votes
                if total_votes > 0:
                    up_probability = up_votes / total_votes
                    ensemble_direction = 1 if up_probability > 0.5 else 0
                    direction_probability = up_probability if ensemble_direction == 1 else (1 - up_probability)
                else:
                    ensemble_direction = None
                    direction_probability = 0.5
            else:
                ensemble_direction = None
                direction_probability = 0.5
            
            # Price change prediction - weighted average
            weighted_changes = []
            
            # Regression model predictions
            for name, model_preds in predictions["regression"].items():
                if horizon in model_preds and model_preds[horizon] is not None:
                    price_change = model_preds[horizon]
                    weight = self.voting_weights["regression"] / len(predictions["regression"])
                    weighted_changes.append((price_change, weight))
            
            # Deep learning model predictions
            for name, model_preds in predictions["deep_learning"].items():
                if horizon in model_preds and model_preds[horizon] is not None:
                    price_change = model_preds[horizon]
                    weight = self.voting_weights["deep_learning"] / len(predictions["deep_learning"])
                    weighted_changes.append((price_change, weight))
            
            # Calculate weighted average price change
            if weighted_changes:
                ensemble_price_change = sum(change * weight for change, weight in weighted_changes) / \
                                      sum(weight for _, weight in weighted_changes)
            else:
                ensemble_price_change = 0.0
            
            # Store ensemble results for this horizon
            if "direction" not in ensemble_results:
                ensemble_results["direction"] = {}
                ensemble_results["price_change"] = {}
                ensemble_results["probability"] = {}
            
            ensemble_results["direction"][horizon] = ensemble_direction
            ensemble_results["price_change"][horizon] = ensemble_price_change
            ensemble_results["probability"][horizon] = direction_probability
        
        logger.debug(f"Ensembled predictions for {self.asset_id} across {len(self.prediction_horizons)} horizons")
        
        return ensemble_results
    
    async def _calculate_confidence(
        self, predictions: Dict[str, Any], ensemble: Dict[str, Any]
    ) -> Dict[int, float]:
        """
        Calculate confidence level for predictions
        
        Args:
            predictions: Predictions from various models
            ensemble: Ensemble predictions
            
        Returns:
            dict: Confidence levels by horizon
        """
        confidence_by_horizon = {}
        
        # For each prediction horizon
        for horizon in self.prediction_horizons:
            # Skip if no ensemble direction
            if horizon not in ensemble["direction"] or ensemble["direction"][horizon] is None:
                confidence_by_horizon[horizon] = 0.5
                continue
            
            ensemble_direction = ensemble["direction"][horizon]
            
            # Count models agreeing with ensemble
            agreeing_models = 0
            total_models = 0
            
            # Check classification models
            for name, model_preds in predictions["classification"].items():
                if horizon in model_preds and model_preds[horizon]["direction"] is not None:
                    total_models += 1
                    if model_preds[horizon]["direction"] == ensemble_direction:
                        agreeing_models += 1
            
            # Check regression models (implied direction)
            for name, model_preds in predictions["regression"].items():
                if horizon in model_preds and model_preds[horizon] is not None:
                    total_models += 1
                    implied_direction = 1 if model_preds[horizon] > 0 else 0
                    if implied_direction == ensemble_direction:
                        agreeing_models += 1
            
            # Check deep learning models (implied direction)
            for name, model_preds in predictions["deep_learning"].items():
                if horizon in model_preds and model_preds[horizon] is not None:
                    total_models += 1
                    implied_direction = 1 if model_preds[horizon] > 0 else 0
                    if implied_direction == ensemble_direction:
                        agreeing_models += 1
            
            # Calculate agreement ratio
            if total_models > 0:
                agreement_ratio = agreeing_models / total_models
            else:
                agreement_ratio = 0.5
            
            # Calculate confidence based on agreement and probability
            direction_prob = ensemble["probability"][horizon]
            
            # Confidence is a combination of agreement and probability
            confidence = (agreement_ratio * 0.5 + direction_prob * 0.5) * self.confidence_scaling
            
            # Adjust confidence based on market regime
            if self.current_regime in [MARKET_REGIMES["CHOPPY_VOLATILE"], MARKET_REGIMES["RANGING_TIGHT"]]:
                # Reduce confidence in difficult regimes
                confidence *= 0.8
            elif self.current_regime == MARKET_REGIMES["TRENDING_CALM"]:
                # Increase confidence in trending regimes
                confidence *= 1.2
            
            # Cap confidence at realistic value
            confidence_by_horizon[horizon] = min(0.95, max(0.1, confidence))
        
        logger.debug(f"Calculated confidence for {self.asset_id} across {len(self.prediction_horizons)} horizons")
        
        return confidence_by_horizon
    
    async def _generate_signals(
        self, ensemble: Dict[str, Any], confidence: Dict[int, float], data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate trading signals with entry/exit points
        
        Args:
            ensemble: Ensemble predictions
            confidence: Confidence levels by horizon
            data: Market data
            
        Returns:
            dict: Trading signals with metadata
        """
        # Default values
        current_price = data['close'].iloc[-1]
        signal = 0
        entry_price = current_price
        stop_loss = current_price * 0.97  # Default 3% stop loss
        take_profit = current_price * 1.05  # Default 5% take profit
        confidence_value = 0.5
        horizon_used = self.prediction_horizons[0]
        
        # Check if we have ensemble predictions
        if not ensemble["direction"]:
            logger.warning("No ensemble predictions available for signal generation")
            return {
                "signal": 0,
                "confidence": 0.5,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "horizon": horizon_used,
                "price_change": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Select most confident prediction horizon
        best_horizon = max(confidence.items(), key=lambda x: x[1])[0]
        best_confidence = confidence[best_horizon]
        
        # Only generate signal if confidence exceeds threshold
        if best_confidence >= self.confidence_level:
            ensemble_direction = ensemble["direction"][best_horizon]
            ensemble_price_change = ensemble["price_change"][best_horizon]
            
            if ensemble_direction is not None:
                # Generate buy/sell signal
                signal = 1 if ensemble_direction == 1 else -1
                
                # Set entry price (current price with a small buffer)
                entry_price = current_price * (1 + 0.001 * signal)  # 0.1% buffer
                
                # Calculate take profit based on predicted price change
                profit_target = abs(ensemble_price_change) * 1.5  # Add 50% to predicted change
                profit_target = max(0.01, min(0.10, profit_target))  # Cap between 1-10%
                
                # Calculate stop loss based on risk-reward ratio and recent volatility
                atr = data['high'].iloc[-14:].max() - data['low'].iloc[-14:].min()
                atr_pct = atr / current_price
                
                # Risk factor based on confidence (higher confidence = tighter stop)
                risk_factor = 1.2 - best_confidence * 0.4  # 0.8 to 1.2
                
                # Stop loss is 1-2 ATR based on confidence
                stop_loss_pct = min(0.05, max(0.01, atr_pct * risk_factor))
                
                # Set take profit and stop loss
                if signal == 1:  # Buy signal
                    take_profit = entry_price * (1 + profit_target)
                    stop_loss = entry_price * (1 - stop_loss_pct)
                else:  # Sell signal
                    take_profit = entry_price * (1 - profit_target)
                    stop_loss = entry_price * (1 + stop_loss_pct)
                
                confidence_value = best_confidence
                horizon_used = best_horizon
        
        logger.debug(f"Generated signal for {self.asset_id}: {signal} with confidence {confidence_value:.4f}")
        
        return {
            "signal": signal,
            "confidence": confidence_value,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "horizon": horizon_used,
            "price_change": ensemble["price_change"].get(horizon_used, 0.0),
            "timestamp": datetime.now().isoformat()
        }
    
    async def learn(self, performance_data: Dict[str, Any]):
        """
        Learn from trading performance and improve models
        
        Args:
            performance_data: Trading performance metrics
        """
        # Process learning in a background task to avoid blocking
        asyncio.create_task(self._process_learning(performance_data))
    
    async def _process_learning(self, performance_data: Dict[str, Any]):
        """Process learning from trading performance"""
        try:
            # Extract performance metrics
            trades = performance_data.get("trades", [])
            win_rate = performance_data.get("win_rate", 0)
            profit_factor = performance_data.get("profit_factor", 0)
            
            if not trades:
                return
            
            # Update performance metrics
            self.performance_metrics["win_rate"] = win_rate
            self.performance_metrics["profit_factor"] = profit_factor
            self.performance_metrics["trade_count"] = len(trades)
            
            # Prepare training data from actual trades
            train_features = []
            train_targets = []
            
            for trade in trades:
                feature_vector = trade.get("feature_vector", {})
                outcome = 1 if trade.get("profitable", False) else 0
                pnl = trade.get("pnl_percent", 0.0)

                # Record trade outcome for historical memory
                self.record_trade_result(pnl)
                
                if feature_vector:
                    train_features.append(feature_vector)
                    train_targets.append(outcome)
            
            # Only retrain if we have enough data
            if len(train_features) >= 10:
                # Convert to DataFrame and Series
                X = pd.DataFrame(train_features)
                y = pd.Series(train_targets)
                
                # Retrain classification models
                for name, model in self.classification_models.items():
                    try:
                        model.fit(X, y)
                        logger.info(f"Retrained {name} model with {len(X)} samples")
                    except Exception as e:
                        logger.error(f"Failed to retrain {name} model: {str(e)}")
                
                # Update feature importance
                for name, model in self.classification_models.items():
                    if hasattr(model, 'feature_importances_'):
                        self._update_feature_importance(model.feature_importances_, X.columns)
            
            # Adjust confidence threshold based on win rate
            if win_rate < 0.4:
                # Increase confidence threshold if win rate is low
                self.confidence_level = min(0.85, self.confidence_level + 0.05)
            elif win_rate > 0.6:
                # Decrease confidence threshold if win rate is high
                self.confidence_level = max(0.55, self.confidence_level - 0.03)
            
            logger.info(f"ML Brain learned from {len(trades)} trades, adjusted confidence threshold to {self.confidence_level:.2f}")

            # Mutate parameters based on historical success rates
            self.mutate_parameters()
        
        except Exception as e:
            logger.error(f"Learning process failed: {str(e)}")
    
    async def adapt(self, market_conditions: Dict[str, Any]):
        """
        Adapt strategy to changing market conditions
        
        Args:
            market_conditions: Current market conditions
        """
        # Extract market conditions
        regime = market_conditions.get("regime", self.current_regime)
        volatility = market_conditions.get("volatility", 1.0)
        
        # Adjust strategy parameters based on market conditions
        if regime != self.current_regime:
            self.current_regime = regime
            
            # Adjust model weights based on regime
            if regime == MARKET_REGIMES["TRENDING_VOLATILE"]:
                # In volatile trending markets, favor deep learning
                self.voting_weights = {
                    "classification": 0.25,
                    "regression": 0.25,
                    "deep_learning": 0.40,
                    "ensemble": 0.10
                }
            elif regime == MARKET_REGIMES["CHOPPY_VOLATILE"]:
                # In choppy volatile markets, favor ensemble approaches
                self.voting_weights = {
                    "classification": 0.20,
                    "regression": 0.20,
                    "deep_learning": 0.30,
                    "ensemble": 0.30
                }
            elif regime == MARKET_REGIMES["TRENDING_CALM"]:
                # In calm trending markets, balance all models
                self.voting_weights = {
                    "classification": 0.30,
                    "regression": 0.30,
                    "deep_learning": 0.30,
                    "ensemble": 0.10
                }
            elif regime == MARKET_REGIMES["RANGING_TIGHT"]:
                # In tight ranging markets, favor classification
                self.voting_weights = {
                    "classification": 0.40,
                    "regression": 0.25,
                    "deep_learning": 0.25,
                    "ensemble": 0.10
                }
        
        # Adjust confidence scaling based on volatility
        self.confidence_scaling = DEFAULT_CONFIDENCE_SCALING_FACTOR * (1.1 - volatility * 0.2)
        self.confidence_scaling = max(0.7, min(1.3, self.confidence_scaling))
        
        logger.debug(f"Adapted ML Brain to market conditions: regime={regime}, volatility={volatility:.2f}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state of the strategy brain
        
        Returns:
            dict: Current strategy state
        """
        return {
            "name": self.name,
            "asset_id": self.asset_id,
            "platform": self.platform,
            "timeframe": self.timeframe,
            "parameters": {
                "lookback_periods": self.lookback_periods,
                "confidence_level": self.confidence_level,
                "model_types": self.model_types,
                "feature_importance_threshold": self.feature_importance_threshold,
                "prediction_horizons": self.prediction_horizons,
                "confidence_scaling": self.confidence_scaling,
                "voting_weights": self.voting_weights
            },
            "performance_metrics": self.performance_metrics,
            "current_regime": self.current_regime,
            "last_signals": self.last_signals,
            "last_updated": datetime.now().isoformat()
        }
    
    def set_state(self, state: Dict[str, Any]):
        """
        Restore strategy state from saved state
        
        Args:
            state: Strategy state to restore
        """
        if state.get("asset_id") != self.asset_id:
            logger.warning(f"State asset ID mismatch: {state.get('asset_id')} vs {self.asset_id}")
            return
        
        parameters = state.get("parameters", {})
        self.lookback_periods = parameters.get("lookback_periods", self.lookback_periods)
        self.confidence_level = parameters.get("confidence_level", self.confidence_level)
        self.model_types = parameters.get("model_types", self.model_types)
        self.feature_importance_threshold = parameters.get("feature_importance_threshold", self.feature_importance_threshold)
        self.prediction_horizons = parameters.get("prediction_horizons", self.prediction_horizons)
        self.confidence_scaling = parameters.get("confidence_scaling", self.confidence_scaling)
        self.voting_weights = parameters.get("voting_weights", self.voting_weights)
        
        self.performance_metrics = state.get("performance_metrics", self.performance_metrics)
        self.current_regime = state.get("current_regime", self.current_regime)
        self.last_signals = state.get("last_signals", self.last_signals)
        
        logger.info(f"Restored ML Brain state for {self.asset_id}")
