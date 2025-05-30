#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Ensemble ML Models Module

This module implements advanced ensemble techniques optimized for financial prediction
with high win rates. It integrates multiple model types for robust predictions across
market conditions, adapting to regime changes and employing dynamic weighting based on
model performance.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
import logging
import joblib
import time
import os
from datetime import datetime, timedelta
import json

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    tf = None  # type: ignore
    TF_AVAILABLE = False

from ml_models.models.regression import AdvancedRegressionModel
from ml_models.models.classification import ClassificationModel
from ml_models.models.time_series import EnsembleTimeSeriesModel
from common.utils import calculate_sharpe_ratio
from common.logger import get_logger
from common.exceptions import ModelTrainingError, ModelPredictionError, InvalidParameterError
from common.constants import MODEL_SAVE_PATH, TRADING_CUTOFFS

logger = get_logger(__name__)

try:
    from ml_models.models.deep_learning import DeepLearningModel
except Exception as e:  # pragma: no cover - optional dependency
    DeepLearningModel = None  # type: ignore
    logger.warning(f"DeepLearningModel not available: {e}")

class EnsembleWeighter:
    """
    Dynamic weighting system for ensemble models based on recent performance,
    asset specifics, market regime, and volatility conditions.
    """
    def __init__(
        self, 
        base_weights: Optional[Dict[str, float]] = None,
        performance_metrics_weight: float = 0.5,
        recency_weight: float = 0.3,
        regime_adapt_weight: float = 0.2,
        recency_half_life: int = 14,
        min_weight: float = 0.05,
        max_weight: float = 0.6
    ):
        """
        Initialize the ensemble weighter with configurable parameters.
        
        Args:
            base_weights: Initial weights for each model
            performance_metrics_weight: Influence of performance metrics on weights
            recency_weight: Influence of recent performance vs historical
            regime_adapt_weight: Influence of market regime adaptation
            recency_half_life: Half-life in days for recency decay
            min_weight: Minimum weight for any model
            max_weight: Maximum weight for any model
        """
        self.base_weights = base_weights or {}
        self.performance_metrics_weight = performance_metrics_weight
        self.recency_weight = recency_weight
        self.regime_adapt_weight = regime_adapt_weight
        self.recency_half_life = recency_half_life
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.performance_history = {}
        self.current_weights = {}
        self.current_regime = None
        self.regime_performance = {}
        
    def update_performance(
        self, 
        model_id: str, 
        metrics: Dict[str, float], 
        timestamp: Optional[datetime] = None,
        regime_type: Optional[str] = None
    ):
        """
        Update performance metrics for a specific model.
        
        Args:
            model_id: Identifier for the model
            metrics: Dictionary of performance metrics
            timestamp: When the performance was recorded
            regime_type: Current market regime classification
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
            
        record = {
            "timestamp": timestamp,
            "metrics": metrics,
            "regime": regime_type
        }
        
        self.performance_history[model_id].append(record)
        
        # Update regime-specific performance
        if regime_type:
            self.current_regime = regime_type
            if regime_type not in self.regime_performance:
                self.regime_performance[regime_type] = {}
                
            if model_id not in self.regime_performance[regime_type]:
                self.regime_performance[regime_type][model_id] = []
                
            self.regime_performance[regime_type][model_id].append(record)
            
        # Recalculate weights after update
        self._recalculate_weights()
            
    def _calculate_recency_weight(self, timestamp: datetime) -> float:
        """
        Calculate time-decay weight based on recency.
        
        Args:
            timestamp: The time when performance was recorded
            
        Returns:
            Recency weight factor
        """
        now = datetime.now()
        days_ago = (now - timestamp).total_seconds() / (24 * 3600)
        decay_factor = 0.5 ** (days_ago / self.recency_half_life)
        return decay_factor
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall performance score from multiple metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Combined performance score
        """
        # Different weights for different types of metrics
        metric_weights = {
            "accuracy": 0.2,
            "precision": 0.15,
            "recall": 0.15,
            "f1": 0.2,
            "roc_auc": 0.3,
            "mse": -0.2,
            "mae": -0.15,
            "rmse": -0.2,
            "mape": -0.15,
            "r2": 0.3,
            "sharpe": 0.4,
            "sortino": 0.3,
            "calmar": 0.3,
            "win_rate": 0.4,
            "profit_factor": 0.3
        }
        
        score = 0.0
        metrics_used = 0
        
        for metric, value in metrics.items():
            if metric.lower() in metric_weights:
                weight = metric_weights[metric.lower()]
                score += value * weight
                metrics_used += abs(weight)
                
        if metrics_used > 0:
            score /= metrics_used
            
        return score
    
    def _calculate_regime_adaptation_score(self, model_id: str) -> float:
        """
        Calculate how well a model adapts to the current market regime.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Regime adaptation score
        """
        if not self.current_regime or self.current_regime not in self.regime_performance:
            return 1.0
            
        if model_id not in self.regime_performance[self.current_regime]:
            return 0.8  # Slight penalty for no regime data
            
        # Calculate average performance in current regime
        regime_records = self.regime_performance[self.current_regime][model_id]
        if not regime_records:
            return 0.8
            
        scores = [self._calculate_performance_score(r["metrics"]) for r in regime_records]
        return sum(scores) / len(scores)
    
    def _recalculate_weights(self):
        """Recalculate all model weights based on performance history."""
        raw_weights = {}
        
        # Calculate raw weights for each model
        for model_id, history in self.performance_history.items():
            base_weight = self.base_weights.get(model_id, 1.0)
            
            if not history:
                raw_weights[model_id] = base_weight
                continue
                
            # Calculate performance component
            performance_scores = []
            recency_weights = []
            
            for record in history:
                perf_score = self._calculate_performance_score(record["metrics"])
                recency_weight = self._calculate_recency_weight(record["timestamp"])
                
                performance_scores.append(perf_score)
                recency_weights.append(recency_weight)
                
            # Calculate weighted average with recency bias
            total_recency_weight = sum(recency_weights)
            if total_recency_weight > 0:
                weighted_performance = sum(s * w for s, w in zip(performance_scores, recency_weights)) / total_recency_weight
            else:
                weighted_performance = sum(performance_scores) / len(performance_scores)
                
            # Get regime adaptation score
            regime_score = self._calculate_regime_adaptation_score(model_id)
            
            # Combine all factors
            raw_weights[model_id] = (
                base_weight * (1 - self.performance_metrics_weight - self.recency_weight - self.regime_adapt_weight) +
                weighted_performance * self.performance_metrics_weight +
                (performance_scores[-1] if performance_scores else 1.0) * self.recency_weight +
                regime_score * self.regime_adapt_weight
            )
        
        # Normalize weights
        total_weight = sum(raw_weights.values())
        if total_weight > 0:
            for model_id in raw_weights:
                self.current_weights[model_id] = raw_weights[model_id] / total_weight
                
                # Apply min/max constraints
                self.current_weights[model_id] = max(self.min_weight, min(self.max_weight, self.current_weights[model_id]))
                
            # Re-normalize after applying constraints
            total_weight = sum(self.current_weights.values())
            for model_id in self.current_weights:
                self.current_weights[model_id] /= total_weight
        else:
            # Equal weights if no data
            model_count = len(raw_weights)
            for model_id in raw_weights:
                self.current_weights[model_id] = 1.0 / model_count
    
    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.current_weights
    
    def set_current_regime(self, regime_type: str):
        """Set the current market regime classification."""
        self.current_regime = regime_type
        self._recalculate_weights()


class DynamicEnsembleModel(BaseEstimator):
    """
    Advanced ensemble model with dynamic weighting and adaptive composition
    based on market conditions and performance monitoring.
    """
    def __init__(
        self,
        model_type: str = "classification",
        base_models: Optional[List[BaseEstimator]] = None,
        meta_model: Optional[BaseEstimator] = None,
        ensemble_method: str = "stacking",
        cv_folds: int = 5,
        use_probabilities: bool = True,
        weights: Optional[List[float]] = None,
        dynamic_weighting: bool = True,
        weighter_config: Optional[Dict[str, Any]] = None,
        custom_evaluation_metric: Optional[Callable] = None,
        prediction_threshold: float = 0.5,
        n_jobs: int = -1,
        random_state: int = 42,
        gpu_acceleration: bool = True,
        memory_efficient: bool = False,
        asset_specific: bool = True,
        asset_id: Optional[str] = None,
        market_regime_adaptive: bool = True,
        feature_subset_per_model: Optional[List[List[str]]] = None
    ):
        """
        Initialize the dynamic ensemble model.
        
        Args:
            model_type: Either "classification" or "regression"
            base_models: List of scikit-learn compatible base models
            meta_model: Meta-model for stacking (optional)
            ensemble_method: "voting", "stacking", "bagging", or "boosting"
            cv_folds: Number of cross-validation folds
            use_probabilities: Use probabilities instead of predictions for stacking
            weights: Initial static weights for base models
            dynamic_weighting: Whether to use dynamic weight adjustment
            weighter_config: Configuration for the ensemble weighter
            custom_evaluation_metric: Custom function for model evaluation
            prediction_threshold: Threshold for binary classification
            n_jobs: Number of parallel jobs
            random_state: Random seed
            gpu_acceleration: Whether to use GPU acceleration
            memory_efficient: Use memory-efficient algorithms
            asset_specific: Whether this model is specifically for one asset
            asset_id: Identifier for the specific asset
            market_regime_adaptive: Adapt to market regime changes
            feature_subset_per_model: Use different feature subsets for diversity
        """
        self.model_type = model_type
        self.base_models = base_models or []
        self.meta_model = meta_model
        self.ensemble_method = ensemble_method
        self.cv_folds = cv_folds
        self.use_probabilities = use_probabilities
        self.weights = weights
        self.dynamic_weighting = dynamic_weighting
        self.weighter_config = weighter_config or {}
        self.custom_evaluation_metric = custom_evaluation_metric
        self.prediction_threshold = prediction_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.gpu_acceleration = gpu_acceleration
        self.memory_efficient = memory_efficient
        self.asset_specific = asset_specific
        self.asset_id = asset_id
        self.market_regime_adaptive = market_regime_adaptive
        self.feature_subset_per_model = feature_subset_per_model
        
        # Internal attributes
        self.models_ = []
        self.model_names_ = []
        self.ensemble_model_ = None
        self.feature_names_ = None
        self.trained_ = False
        self.training_timestamp_ = None
        self.model_metrics_ = {}
        self.current_market_regime_ = None
        
        # Initialize weighter if using dynamic weighting
        if self.dynamic_weighting:
            base_weights = {f"model_{i}": (self.weights[i] if self.weights and i < len(self.weights) else 1.0) 
                           for i in range(len(self.base_models))}
            self.weighter_ = EnsembleWeighter(base_weights=base_weights, **self.weighter_config)
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate model parameters and configurations."""
        if self.model_type not in ["classification", "regression"]:
            raise InvalidParameterError(f"Invalid model_type: {self.model_type}. Must be 'classification' or 'regression'.")
            
        if self.ensemble_method not in ["voting", "stacking", "bagging", "boosting"]:
            raise InvalidParameterError(f"Invalid ensemble_method: {self.ensemble_method}. Must be 'voting', 'stacking', 'bagging', or 'boosting'.")
            
        if self.weights and len(self.weights) != len(self.base_models):
            raise InvalidParameterError(f"Length of weights ({len(self.weights)}) must match number of base models ({len(self.base_models)}).")
            
        if self.feature_subset_per_model and len(self.feature_subset_per_model) != len(self.base_models):
            raise InvalidParameterError(f"Length of feature_subset_per_model ({len(self.feature_subset_per_model)}) must match number of base models ({len(self.base_models)}).")
    
    def _prepare_ensemble_model(self):
        """Prepare the ensemble model based on the selected method."""
        if not self.base_models:
            raise ModelTrainingError("No base models provided for ensemble.")
            
        # Create named model tuples for scikit-learn ensemble methods
        self.models_ = []
        self.model_names_ = []
        
        for i, model in enumerate(self.base_models):
            model_name = f"model_{i}"
            self.models_.append((model_name, model))
            self.model_names_.append(model_name)
        
        # Determine current weights
        if self.dynamic_weighting and hasattr(self, 'weighter_'):
            current_weights = list(self.weighter_.get_weights().values())
            if len(current_weights) != len(self.models_):
                current_weights = None  # Fall back to uniform weights
        else:
            current_weights = self.weights
            
        # Create the appropriate ensemble model
        if self.model_type == "classification":
            if self.ensemble_method == "voting":
                self.ensemble_model_ = VotingClassifier(
                    estimators=self.models_,
                    voting='soft' if self.use_probabilities else 'hard',
                    weights=current_weights,
                    n_jobs=self.n_jobs
                )
            elif self.ensemble_method == "stacking":
                if self.meta_model is None:
                    # Default meta-model if none provided
                    self.meta_model = XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                self.ensemble_model_ = StackingClassifier(
                    estimators=self.models_,
                    final_estimator=self.meta_model,
                    cv=self.cv_folds,
                    stack_method='predict_proba' if self.use_probabilities else 'predict',
                    n_jobs=self.n_jobs
                )
            elif self.ensemble_method == "bagging":
                # For bagging, we'll create individual bagging models for each base model
                bagging_models = []
                for name, model in self.models_:
                    bagging_model = BaggingClassifier(
                        base_estimator=model,
                        n_estimators=10,
                        max_samples=0.8,
                        max_features=0.8,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state
                    )
                    bagging_models.append((f"bagging_{name}", bagging_model))
                
                # Then use voting to combine them
                self.ensemble_model_ = VotingClassifier(
                    estimators=bagging_models,
                    voting='soft',
                    weights=current_weights,
                    n_jobs=self.n_jobs
                )
            elif self.ensemble_method == "boosting":
                # For simplicity, use AdaBoost as default boosting method
                self.ensemble_model_ = AdaBoostClassifier(
                    base_estimator=self.base_models[0],
                    n_estimators=50,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
        else:  # regression
            if self.ensemble_method == "voting":
                self.ensemble_model_ = VotingRegressor(
                    estimators=self.models_,
                    weights=current_weights,
                    n_jobs=self.n_jobs
                )
            elif self.ensemble_method == "stacking":
                if self.meta_model is None:
                    # Default meta-model if none provided
                    self.meta_model = XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=self.random_state
                    )
                self.ensemble_model_ = StackingRegressor(
                    estimators=self.models_,
                    final_estimator=self.meta_model,
                    cv=self.cv_folds,
                    n_jobs=self.n_jobs
                )
            elif self.ensemble_method == "bagging":
                # For bagging, we'll create individual bagging models for each base model
                bagging_models = []
                for name, model in self.models_:
                    bagging_model = BaggingRegressor(
                        base_estimator=model,
                        n_estimators=10,
                        max_samples=0.8,
                        max_features=0.8,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=self.n_jobs,
                        random_state=self.random_state
                    )
                    bagging_models.append((f"bagging_{name}", bagging_model))
                
                # Then use voting to combine them
                self.ensemble_model_ = VotingRegressor(
                    estimators=bagging_models,
                    weights=current_weights,
                    n_jobs=self.n_jobs
                )
            elif self.ensemble_method == "boosting":
                # For simplicity, use AdaBoost as default boosting method
                self.ensemble_model_ = AdaBoostRegressor(
                    base_estimator=self.base_models[0],
                    n_estimators=50,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
    
    def fit(self, X, y, sample_weight=None, **kwargs):
        """
        Fit the ensemble model to the training data.
        
        Args:
            X: Training features
            y: Target values
            sample_weight: Sample weights for training
            **kwargs: Additional arguments passed to base models
        
        Returns:
            Self
        """
        start_time = time.time()
        logger.info(f"Starting training dynamic ensemble model with method: {self.ensemble_method}")
        
        try:
            # Save feature names if DataFrame
            if hasattr(X, 'columns'):
                self.feature_names_ = list(X.columns)
            
            # Create feature subsets if needed
            if self.feature_subset_per_model:
                X_subsets = []
                for feature_subset in self.feature_subset_per_model:
                    if hasattr(X, 'loc'):
                        X_subsets.append(X.loc[:, feature_subset])
                    else:
                        # For numpy arrays, create a mask
                        mask = [self.feature_names_.index(f) for f in feature_subset]
                        X_subsets.append(X[:, mask])
            
            # Handle market regime for adaptation
            if self.market_regime_adaptive and 'market_regime' in kwargs:
                self.current_market_regime_ = kwargs['market_regime']
                if self.dynamic_weighting:
                    self.weighter_.set_current_regime(self.current_market_regime_)
            
            # First train individual models if using feature subsets
            if self.feature_subset_per_model:
                for i, ((name, model), X_subset) in enumerate(zip(self.models_, X_subsets)):
                    logger.info(f"Training base model {name} with feature subset")
                    if hasattr(model, 'random_state'):
                        model.random_state = self.random_state
                    if hasattr(model, 'n_jobs'):
                        model.n_jobs = self.n_jobs
                    
                    model.fit(X_subset, y, sample_weight=sample_weight)
                    
                    # Evaluate and update weights if dynamic
                    if self.dynamic_weighting:
                        metrics = self._evaluate_model(model, X_subset, y)
                        self.model_metrics_[name] = metrics
                        self.weighter_.update_performance(
                            name, 
                            metrics, 
                            timestamp=datetime.now(),
                            regime_type=self.current_market_regime_
                        )
            
            # Prepare the ensemble model
            self._prepare_ensemble_model()
            
            # Fit the ensemble model
            if self.ensemble_method == "boosting":
                # Boosting uses a different approach
                self.ensemble_model_.fit(X, y, sample_weight=sample_weight)
            else:
                # Standard ensemble methods
                if sample_weight is not None:
                    logger.warning("Sample weights may not be directly supported by all ensemble methods.")
                
                self.ensemble_model_.fit(X, y)
            
            self.trained_ = True
            self.training_timestamp_ = datetime.now()
            
            training_time = time.time() - start_time
            logger.info(f"Ensemble model training completed in {training_time:.2f} seconds")
            
            return self
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {str(e)}")
            logger.error(traceback.format_exc())
            raise ModelTrainingError(f"Failed to train ensemble model: {str(e)}")
    
    def predict(self, X):
        """
        Generate predictions from the ensemble model.
        
        Args:
            X: Features to predict
        
        Returns:
            Model predictions
        """
        if not self.trained_:
            raise ModelPredictionError("Model has not been trained yet.")
            
        try:
            # Create feature subsets if needed
            if self.feature_subset_per_model and self.ensemble_method in ["voting", "stacking"]:
                # For these methods, we need to let the ensemble handle the predictions
                return self.ensemble_model_.predict(X)
            elif self.feature_subset_per_model:
                # For bagging or custom ensembling with feature subsets
                predictions = []
                for i, ((name, model), feature_subset) in enumerate(zip(self.models_, self.feature_subset_per_model)):
                    if hasattr(X, 'loc'):
                        X_subset = X.loc[:, feature_subset]
                    else:
                        # For numpy arrays, create a mask
                        mask = [self.feature_names_.index(f) for f in feature_subset]
                        X_subset = X[:, mask]
                    
                    model_pred = model.predict(X_subset)
                    predictions.append(model_pred)
                
                # Combine predictions using current weights
                if self.dynamic_weighting:
                    weights = list(self.weighter_.get_weights().values())
                else:
                    weights = self.weights if self.weights else [1/len(predictions)] * len(predictions)
                
                # Stack and weight predictions
                predictions = np.stack(predictions, axis=1)
                weighted_sum = np.sum(predictions * np.array(weights), axis=1)
                
                if self.model_type == "classification":
                    return (weighted_sum >= self.prediction_threshold).astype(int)
                else:
                    return weighted_sum
            else:
                # Regular prediction without feature subsets
                return self.ensemble_model_.predict(X)
                
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            logger.error(traceback.format_exc())
            raise ModelPredictionError(f"Failed to generate predictions: {str(e)}")
    
    def predict_proba(self, X):
        """
        Generate probability predictions for classification models.
        
        Args:
            X: Features to predict
        
        Returns:
            Class probabilities
        """
        if self.model_type != "classification":
            raise ModelPredictionError("predict_proba is only available for classification models")
            
        if not self.trained_:
            raise ModelPredictionError("Model has not been trained yet.")
            
        try:
            # Create feature subsets if needed
            if self.feature_subset_per_model and self.ensemble_method in ["voting", "stacking"]:
                # For these methods, we need to let the ensemble handle the predictions
                return self.ensemble_model_.predict_proba(X)
            elif self.feature_subset_per_model:
                # For bagging or custom ensembling with feature subsets
                probas = []
                for i, ((name, model), feature_subset) in enumerate(zip(self.models_, self.feature_subset_per_model)):
                    if hasattr(X, 'loc'):
                        X_subset = X.loc[:, feature_subset]
                    else:
                        # For numpy arrays, create a mask
                        mask = [self.feature_names_.index(f) for f in feature_subset]
                        X_subset = X[:, mask]
                    
                    if hasattr(model, 'predict_proba'):
                        model_proba = model.predict_proba(X_subset)
                        probas.append(model_proba)
                    else:
                        logger.warning(f"Model {name} doesn't support predict_proba")
                
                if not probas:
                    raise ModelPredictionError("None of the base models support predict_proba")
                
                # Combine probabilities using current weights
                if self.dynamic_weighting:
                    weights = list(self.weighter_.get_weights().values())
                else:
                    weights = self.weights if self.weights else [1/len(probas)] * len(probas)
                
                # Ensure all probabilities have the same shape
                n_classes = probas[0].shape[1]
                for i, p in enumerate(probas):
                    if p.shape[1] != n_classes:
                        raise ModelPredictionError(f"Inconsistent number of classes in base model probabilities")
                
                # Weight and combine probabilities
                weighted_probas = np.zeros_like(probas[0])
                for i, p in enumerate(probas):
                    weighted_probas += p * weights[i]
                
                # Normalize to ensure they sum to 1
                row_sums = weighted_probas.sum(axis=1)
                weighted_probas = weighted_probas / row_sums[:, np.newaxis]
                
                return weighted_probas
            else:
                # Regular prediction without feature subsets
                return self.ensemble_model_.predict_proba(X)
                
        except Exception as e:
            logger.error(f"Error generating probability predictions: {str(e)}")
            logger.error(traceback.format_exc())
            raise ModelPredictionError(f"Failed to generate probability predictions: {str(e)}")
    
    def _evaluate_model(self, model, X, y) -> Dict[str, float]:
        """
        Evaluate a single model's performance for dynamic weighting.
        
        Args:
            model: The model to evaluate
            X: Evaluation features
            y: True labels
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        try:
            # Get predictions
            y_pred = model.predict(X)
            
            if self.model_type == "classification":
                # Classification metrics
                metrics["accuracy"] = accuracy_score(y, y_pred)
                
                if len(np.unique(y)) == 2:  # Binary classification
                    metrics["precision"] = precision_score(y, y_pred, average='binary')
                    metrics["recall"] = recall_score(y, y_pred, average='binary')
                    metrics["f1"] = f1_score(y, y_pred, average='binary')
                    
                    # ROC AUC if predict_proba is available
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_proba = model.predict_proba(X)[:, 1]
                            metrics["roc_auc"] = roc_auc_score(y, y_proba)
                        except:
                            pass
            else:
                # Regression metrics
                metrics["mse"] = mean_squared_error(y, y_pred)
                metrics["rmse"] = np.sqrt(metrics["mse"])
                metrics["mae"] = mean_absolute_error(y, y_pred)
                metrics["r2"] = r2_score(y, y_pred)
                
                try:
                    metrics["mape"] = mean_absolute_percentage_error(y, y_pred)
                except:
                    pass
            
            # Add custom evaluation metric if provided
            if self.custom_evaluation_metric:
                metrics["custom"] = self.custom_evaluation_metric(y, y_pred)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error evaluating model for dynamic weighting: {str(e)}")
            return {"error": 0.0}  # Return default metrics on error
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Manually update model weights.
        
        Args:
            new_weights: Dictionary mapping model names to weights
        """
        if not self.dynamic_weighting:
            logger.warning("Manual weight updates have no effect when dynamic_weighting is False")
            return
            
        for name, weight in new_weights.items():
            if name in self.model_names_:
                # Update the dynamic weighter weights
                self.weighter_.base_weights[name] = weight
                
        # Recalculate weights
        self.weighter_._recalculate_weights()
        
        # Create a new ensemble model with updated weights
        if self.trained_:
            self._prepare_ensemble_model()
            logger.info("Ensemble model weights updated")
    
    def set_market_regime(self, regime_type: str):
        """
        Update the current market regime for adaptation.
        
        Args:
            regime_type: Current market regime classification
        """
        if not self.market_regime_adaptive:
            logger.warning("Market regime updates have no effect when market_regime_adaptive is False")
            return
            
        self.current_market_regime_ = regime_type
        
        if self.dynamic_weighting:
            self.weighter_.set_current_regime(regime_type)
            self._prepare_ensemble_model()
            logger.info(f"Market regime updated to {regime_type}, weights adjusted")
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get the current model weights."""
        if self.dynamic_weighting:
            return self.weighter_.get_weights()
        else:
            if self.weights:
                return {name: weight for name, weight in zip(self.model_names_, self.weights)}
            else:
                equal_weight = 1.0 / len(self.model_names_)
                return {name: equal_weight for name in self.model_names_}
    
    def save(self, path: Optional[str] = None):
        """
        Save the ensemble model to disk.
        
        Args:
            path: Path to save the model (optional)
        """
        if not self.trained_:
            raise ValueError("Cannot save untrained model")
            
        if path is None:
            # Use default path with asset ID if available
            asset_str = f"_{self.asset_id}" if self.asset_specific and self.asset_id else ""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_{self.model_type}_{self.ensemble_method}{asset_str}_{timestamp}.joblib"
            path = os.path.join(MODEL_SAVE_PATH, filename)
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        joblib.dump(self, path)
        logger.info(f"Ensemble model saved to {path}")
        
        # Save auxiliary information in a JSON file
        info_path = path.replace('.joblib', '_info.json')
        model_info = {
            "model_type": self.model_type,
            "ensemble_method": self.ensemble_method,
            "trained_timestamp": self.training_timestamp_.isoformat() if self.training_timestamp_ else None,
            "dynamic_weighting": self.dynamic_weighting,
            "asset_specific": self.asset_specific,
            "asset_id": self.asset_id,
            "feature_names": self.feature_names_,
            "model_names": self.model_names_,
            "current_weights": self.get_current_weights(),
            "current_market_regime": self.current_market_regime_,
            "model_metrics": self.model_metrics_
        }
        
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return path
    
    @classmethod
    def load(cls, path: str):
        """
        Load an ensemble model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(path)
            logger.info(f"Ensemble model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Error loading ensemble model: {str(e)}")
            raise ValueError(f"Failed to load model: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        info = {
            "model_type": self.model_type,
            "ensemble_method": self.ensemble_method,
            "trained": self.trained_,
            "trained_timestamp": self.training_timestamp_.isoformat() if self.training_timestamp_ else None,
            "base_model_count": len(self.base_models),
            "dynamic_weighting": self.dynamic_weighting,
            "current_weights": self.get_current_weights(),
            "asset_specific": self.asset_specific,
            "asset_id": self.asset_id,
            "market_regime_adaptive": self.market_regime_adaptive,
            "current_market_regime": self.current_market_regime_,
            "feature_count": len(self.feature_names_) if self.feature_names_ else None,
            "model_metrics": self.model_metrics_
        }
        return info


class AdvancedEnsembleFactory:
    """
    Factory class for creating sophisticated ensemble models based on specific needs.
    This factory produces highly optimized ensembles for different trading tasks.
    """
    @staticmethod
    def create_trend_direction_ensemble(
        asset_id: str,
        timeframe: str,
        model_type: str = "classification",
        dynamic_weighting: bool = True,
        gpu_acceleration: bool = True,
        market_regime_adaptive: bool = True
    ) -> DynamicEnsembleModel:
        """
        Create an ensemble specifically for trend direction prediction.
        
        Args:
            asset_id: The asset identifier
            timeframe: Trading timeframe (e.g., "1h", "4h", "1d")
            model_type: "classification" for direction, "regression" for price prediction
            dynamic_weighting: Whether to use dynamic model weighting
            gpu_acceleration: Use GPU acceleration if available
            market_regime_adaptive: Adapt to changing market regimes
            
        Returns:
            Configured ensemble model for trend direction prediction
        """
        logger.info(f"Creating trend direction ensemble for {asset_id} on {timeframe} timeframe")
        
        # Define base models
        if model_type == "classification":
            base_models = [
                RandomForestClassifier(
                    n_estimators=200, 
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    class_weight='balanced',
                    random_state=42
                ),
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_split=200,
                    min_samples_leaf=50,
                    subsample=0.8,
                    max_features=None,
                    random_state=42
                ),
                xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    use_label_encoder=False,
                    eval_metric='logloss',
                    tree_method='gpu_hist' if gpu_acceleration else 'hist',
                    random_state=42
                ),
                lgb.LGBMClassifier(
                    n_estimators=200,
                    num_leaves=31,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary',
                    class_weight='balanced',
                    random_state=42
                ),
                cb.CatBoostClassifier(
                    iterations=150,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3,
                    loss_function='Logloss',
                    random_seed=42,
                    task_type='GPU' if gpu_acceleration else 'CPU',
                    verbose=False
                )
            ]
            
            # Meta-model for stacking
            meta_model = lgb.LGBMClassifier(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary',
                random_state=42
            )
        else:  # regression
            base_models = [
                RandomForestRegressor(
                    n_estimators=200, 
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42
                ),
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_split=200,
                    min_samples_leaf=50,
                    subsample=0.8,
                    max_features=None,
                    random_state=42
                ),
                xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='reg:squarederror',
                    tree_method='gpu_hist' if gpu_acceleration else 'hist',
                    random_state=42
                ),
                lgb.LGBMRegressor(
                    n_estimators=200,
                    num_leaves=31,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='regression',
                    random_state=42
                ),
                cb.CatBoostRegressor(
                    iterations=150,
                    depth=6,
                    learning_rate=0.05,
                    l2_leaf_reg=3,
                    loss_function='RMSE',
                    random_seed=42,
                    task_type='GPU' if gpu_acceleration else 'CPU',
                    verbose=False
                )
            ]
            
            # Meta-model for stacking
            meta_model = lgb.LGBMRegressor(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='regression',
                random_state=42
            )
        
        # Configure the ensemble model
        ensemble = DynamicEnsembleModel(
            model_type=model_type,
            base_models=base_models,
            meta_model=meta_model,
            ensemble_method="stacking",
            cv_folds=5,
            use_probabilities=True,
            weights=None,  # Will be determined dynamically
            dynamic_weighting=dynamic_weighting,
            weighter_config={
                "performance_metrics_weight": 0.6,
                "recency_weight": 0.3,
                "regime_adapt_weight": 0.1,
                "recency_half_life": 14,
                "min_weight": 0.05,
                "max_weight": 0.4
            },
            prediction_threshold=0.5,
            n_jobs=-1,
            random_state=42,
            gpu_acceleration=gpu_acceleration,
            asset_specific=True,
            asset_id=asset_id,
            market_regime_adaptive=market_regime_adaptive,
            feature_subset_per_model=None  # Use all features for all models
        )
        
        return ensemble
    
    @staticmethod
    def create_entry_exit_ensemble(
        asset_id: str,
        timeframe: str,
        entry_type: str = "long",  # "long" or "short"
        dynamic_weighting: bool = True,
        gpu_acceleration: bool = True
    ) -> DynamicEnsembleModel:
        """
        Create an ensemble specifically for optimal entry/exit point detection.
        
        Args:
            asset_id: The asset identifier
            timeframe: Trading timeframe (e.g., "1h", "4h", "1d")
            entry_type: "long" for buy signals, "short" for sell signals
            dynamic_weighting: Whether to use dynamic model weighting
            gpu_acceleration: Use GPU acceleration if available
            
        Returns:
            Configured ensemble model for entry/exit point detection
        """
        logger.info(f"Creating {entry_type} entry/exit ensemble for {asset_id} on {timeframe} timeframe")
        
        # Define base models optimized for entry/exit detection
        base_models = [
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.6,
                objective='binary:logistic',
                scale_pos_weight=2.0 if entry_type == "long" else 0.5,  # Balance class weights
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='gpu_hist' if gpu_acceleration else 'hist',
                random_state=42
            ),
            lgb.LGBMClassifier(
                n_estimators=150,
                num_leaves=31,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.7,
                objective='binary',
                class_weight='balanced',
                random_state=42
            ),
            cb.CatBoostClassifier(
                iterations=120,
                depth=5,
                learning_rate=0.04,
                l2_leaf_reg=5,
                loss_function='Logloss',
                random_seed=42,
                task_type='GPU' if gpu_acceleration else 'CPU',
                verbose=False
            ),
            RandomForestClassifier(
                n_estimators=150, 
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                random_state=42
            )
        ]
        
        # Meta-model for stacking
        meta_model = xgb.XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=1.5 if entry_type == "long" else 0.7,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='gpu_hist' if gpu_acceleration else 'hist',
            random_state=42
        )
        
        # Configure the ensemble model
        ensemble = DynamicEnsembleModel(
            model_type="classification",
            base_models=base_models,
            meta_model=meta_model,
            ensemble_method="stacking",
            cv_folds=5,
            use_probabilities=True,
            weights=None,  # Will be determined dynamically
            dynamic_weighting=dynamic_weighting,
            weighter_config={
                "performance_metrics_weight": 0.5,
                "recency_weight": 0.4,  # Higher recency weight for entry/exit timing
                "regime_adapt_weight": 0.1,
                "recency_half_life": 7,  # Shorter half-life for more responsiveness
                "min_weight": 0.1,
                "max_weight": 0.5
            },
            # Higher threshold for more conservative entry signals
            prediction_threshold=0.65 if entry_type == "long" else 0.6,
            n_jobs=-1,
            random_state=42,
            gpu_acceleration=gpu_acceleration,
            asset_specific=True,
            asset_id=f"{asset_id}_{entry_type}",
            market_regime_adaptive=True
        )
        
        return ensemble
    
    @staticmethod
    def create_price_movement_ensemble(
        asset_id: str,
        timeframe: str,
        prediction_horizon: int,  # Number of periods ahead to predict
        gpu_acceleration: bool = True
    ) -> DynamicEnsembleModel:
        """
        Create an ensemble for predicting price movement magnitude.
        
        Args:
            asset_id: The asset identifier
            timeframe: Trading timeframe (e.g., "1h", "4h", "1d")
            prediction_horizon: How many periods ahead to predict
            gpu_acceleration: Use GPU acceleration if available
            
        Returns:
            Configured ensemble model for price movement prediction
        """
        logger.info(f"Creating price movement ensemble for {asset_id} on {timeframe} timeframe with {prediction_horizon} period horizon")
        
        # Define base models for regression task
        base_models = [
            xgb.XGBRegressor(
                n_estimators=150,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                tree_method='gpu_hist' if gpu_acceleration else 'hist',
                random_state=42
            ),
            lgb.LGBMRegressor(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.7,
                objective='regression',
                random_state=42
            ),
            cb.CatBoostRegressor(
                iterations=120,
                depth=6,
                learning_rate=0.03,
                l2_leaf_reg=3,
                loss_function='RMSE',
                random_seed=42,
                task_type='GPU' if gpu_acceleration else 'CPU',
                verbose=False
            ),
            # Neural network for capturing complex patterns
            DeepLearningModel(
                input_dim=50,  # Will be set during fitting
                output_dim=1,
                hidden_layers=[64, 32, 16],
                dropout_rate=0.3,
                learning_rate=0.001,
                batch_size=64,
                epochs=50,
                early_stopping=True,
                model_type="regression",
                gpu_acceleration=gpu_acceleration
            ),
            # Time series specific model
            EnsembleTimeSeriesModel(
                model_type="arimax",
                forecast_horizon=prediction_horizon,
                seasonal=True,
                seasonal_periods=24 if timeframe == "1h" else (5 if timeframe == "1d" else 7),
                exog_vars=True
            )
        ]
        
        # Meta-model for stacking
        meta_model = cb.CatBoostRegressor(
            iterations=80,
            depth=4,
            learning_rate=0.02,
            l2_leaf_reg=5,
            loss_function='RMSE',
            random_seed=42,
            task_type='GPU' if gpu_acceleration else 'CPU',
            verbose=False
        )
        
        # Custom evaluation metric: directional accuracy + magnitude error
        def custom_movement_metric(y_true, y_pred):
            # Directional accuracy (sign agreement)
            direction_match = np.sign(y_true) == np.sign(y_pred)
            directional_accuracy = np.mean(direction_match)
            
            # Magnitude error
            magnitude_error = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-6))
            
            # Combined score, weighted toward directional accuracy
            combined_score = (0.7 * directional_accuracy) - (0.3 * magnitude_error)
            return combined_score
        
        # Configure the ensemble model
        ensemble = DynamicEnsembleModel(
            model_type="regression",
            base_models=base_models,
            meta_model=meta_model,
            ensemble_method="stacking",
            cv_folds=5,
            use_probabilities=False,
            weights=None,  # Will be determined dynamically
            dynamic_weighting=True,
            weighter_config={
                "performance_metrics_weight": 0.5,
                "recency_weight": 0.3,
                "regime_adapt_weight": 0.2,
                "recency_half_life": 10,
                "min_weight": 0.05,
                "max_weight": 0.4
            },
            custom_evaluation_metric=custom_movement_metric,
            n_jobs=-1,
            random_state=42,
            gpu_acceleration=gpu_acceleration,
            asset_specific=True,
            asset_id=f"{asset_id}_h{prediction_horizon}",
            market_regime_adaptive=True
        )
        
        return ensemble
    
    @staticmethod
    def create_volatility_prediction_ensemble(
        asset_id: str,
        timeframe: str,
        prediction_horizon: int,
        gpu_acceleration: bool = True
    ) -> DynamicEnsembleModel:
        """
        Create an ensemble specifically for volatility prediction.
        
        Args:
            asset_id: The asset identifier
            timeframe: Trading timeframe (e.g., "1h", "4h", "1d")
            prediction_horizon: How many periods ahead to predict
            gpu_acceleration: Use GPU acceleration if available
            
        Returns:
            Configured ensemble model for volatility prediction
        """
        logger.info(f"Creating volatility prediction ensemble for {asset_id} on {timeframe} timeframe with {prediction_horizon} period horizon")
        
        # Define base models optimized for volatility prediction
        base_models = [
            # Linear models work well for volatility
            AdvancedRegressionModel(
                model_type="elastic_net",
                alpha=0.1,
                l1_ratio=0.5,
                max_iter=10000,
                random_state=42
            ),
            # Gradient boosting for non-linear relationships
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                loss='huber',
                random_state=42
            ),
            # XGBoost for handling different volatility regimes
            xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                tree_method='gpu_hist' if gpu_acceleration else 'hist',
                random_state=42
            ),
            # GARCH-like model for time-varying volatility
            EnsembleTimeSeriesModel(
                model_type="garch",
                forecast_horizon=prediction_horizon,
                p=1, q=1,  # GARCH(1,1) is often sufficient
                vol_model=True
            ),
            # Random Forest for robustness
            RandomForestRegressor(
                n_estimators=150, 
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                random_state=42
            )
        ]
        
        # Meta-model for stacking
        meta_model = lgb.LGBMRegressor(
            n_estimators=80,
            num_leaves=31,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='regression_l1',  # L1 loss is good for volatility prediction
            random_state=42
        )
        
        # Configure the ensemble model
        ensemble = DynamicEnsembleModel(
            model_type="regression",
            base_models=base_models,
            meta_model=meta_model,
            ensemble_method="stacking",
            cv_folds=5,
            use_probabilities=False,
            weights=None,  # Will be determined dynamically
            dynamic_weighting=True,
            weighter_config={
                "performance_metrics_weight": 0.5,
                "recency_weight": 0.4,  # Higher recency weight for volatility which changes quickly
                "regime_adapt_weight": 0.1,
                "recency_half_life": 5,  # Shorter half-life for more responsiveness to regime changes
                "min_weight": 0.05,
                "max_weight": 0.5
            },
            n_jobs=-1,
            random_state=42,
            gpu_acceleration=gpu_acceleration,
            asset_specific=True,
            asset_id=f"{asset_id}_vol_h{prediction_horizon}",
            market_regime_adaptive=True
        )
        
        return ensemble
    
    @staticmethod
    def create_regime_classification_ensemble(
        gpu_acceleration: bool = True
    ) -> DynamicEnsembleModel:
        """
        Create an ensemble for market regime classification.
        
        Args:
            gpu_acceleration: Use GPU acceleration if available
            
        Returns:
            Configured ensemble model for market regime classification
        """
        logger.info("Creating market regime classification ensemble")
        
        # Define base models for multi-class classification
        base_models = [
            RandomForestClassifier(
                n_estimators=200, 
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                random_state=42
            ),
            xgb.XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                use_label_encoder=False,
                tree_method='gpu_hist' if gpu_acceleration else 'hist',
                random_state=42
            ),
            lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multiclass',
                class_weight='balanced',
                random_state=42
            ),
            cb.CatBoostClassifier(
                iterations=150,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3,
                loss_function='MultiClass',
                random_seed=42,
                task_type='GPU' if gpu_acceleration else 'CPU',
                verbose=False
            )
        ]
        
        # Meta-model for stacking
        meta_model = lgb.LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multiclass',
            random_state=42
        )
        
        # Configure the ensemble model
        ensemble = DynamicEnsembleModel(
            model_type="classification",
            base_models=base_models,
            meta_model=meta_model,
            ensemble_method="stacking",
            cv_folds=5,
            use_probabilities=True,
            weights=None,  # Will be determined dynamically
            dynamic_weighting=True,
            weighter_config={
                "performance_metrics_weight": 0.6,
                "recency_weight": 0.3,
                "regime_adapt_weight": 0.1,
                "recency_half_life": 14,
                "min_weight": 0.1,
                "max_weight": 0.4
            },
            prediction_threshold=0.5,  # For multi-class, this is ignored
            n_jobs=-1,
            random_state=42,
            gpu_acceleration=gpu_acceleration,
            asset_specific=False,  # Regime classification is generally market-wide
            market_regime_adaptive=False  # We don't adapt to regimes since we're predicting them
        )
        
        return ensemble


# Make this module callable as a script for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ensemble models module")
    parser.add_argument("--asset", type=str, default="BTC/USDT", help="Asset to test")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe to test")
    parser.add_argument("--test", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("Testing ensemble models module")
        
        # Example: Create a trend direction ensemble
        trend_ensemble = AdvancedEnsembleFactory.create_trend_direction_ensemble(
            asset_id=args.asset,
            timeframe=args.timeframe,
            dynamic_weighting=True
        )
        
        logger.info(f"Created trend direction ensemble for {args.asset} on {args.timeframe}")
        logger.info(f"Model info: {trend_ensemble.get_model_info()}")
        
        # Example: Create an entry/exit point ensemble
        entry_ensemble = AdvancedEnsembleFactory.create_entry_exit_ensemble(
            asset_id=args.asset,
            timeframe=args.timeframe,
            entry_type="long"
        )
        
        logger.info(f"Created entry/exit ensemble for {args.asset} on {args.timeframe}")
        logger.info(f"Model info: {entry_ensemble.get_model_info()}")

        logger.info("Ensemble models module tests completed successfully")

