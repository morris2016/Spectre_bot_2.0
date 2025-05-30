#!/usr/bin/env python3
from __future__ import annotations
"""
QuantumSpectre Elite Trading System
Regression Models Implementation

This module implements various regression models used for price prediction, 
volatility estimation, and other continuous value forecasting tasks. The models
are optimized for financial time series data and include custom features for
handling the unique characteristics of market data.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
import joblib
import warnings

# ML libraries
import sklearn
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    SGDRegressor, BayesianRidge, HuberRegressor,
    RANSACRegressor, TheilSenRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.svm import SVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_score, TimeSeriesSplit, GridSearchCV, 
    RandomizedSearchCV
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    median_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer,
    FunctionTransformer
)

# Optional GPU accelerated libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from cuml.ensemble import RandomForestRegressor as cuRFR
    from cuml.linear_model import Ridge as cuRidge
    from cuml.linear_model import Lasso as cuLasso
    from cuml.svm import SVR as cuSVR
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Custom modules
from common.logger import get_logger
from common.exceptions import ModelError, TrainingError, InferenceError
from ml_models.models.base import BaseModel, ModelConfig

logger = get_logger(__name__)


class RegressionModelFactory:
    """
    Factory class to create and configure regression models based on configuration.
    """
    
    @staticmethod
    def create_model(config: ModelConfig) -> BaseRegressionModel:
        """
        Create a regression model instance based on configuration.
        
        Args:
            config: Model configuration parameters
            
        Returns:
            Instantiated regression model
        """
        model_type = config.model_type.lower()
        
        if model_type == "linear":
            return LinearRegressionModel(config)
        elif model_type == "ridge":
            return RidgeRegressionModel(config)
        elif model_type == "lasso":
            return LassoRegressionModel(config)
        elif model_type == "elastic_net":
            return ElasticNetRegressionModel(config)
        elif model_type == "svr":
            return SVRModel(config)
        elif model_type == "random_forest":
            return RandomForestRegressionModel(config)
        elif model_type == "gradient_boosting":
            return GradientBoostingRegressionModel(config)
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            return XGBoostRegressionModel(config)
        elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            return LightGBMRegressionModel(config)
        elif model_type == "catboost" and CATBOOST_AVAILABLE:
            return CatBoostRegressionModel(config)
        elif model_type == "neural_network":
            return NeuralNetworkRegressionModel(config)
        elif model_type == "ensemble":
            return EnsembleRegressionModel(config)
        elif model_type == "time_weighted":
            return TimeWeightedRegressionModel(config)
        elif model_type == "periodic":
            return PeriodicRegressionModel(config)
        elif model_type == "adaptive":
            return AdaptiveRegressionModel(config)
        elif model_type == "advanced":
            return AdvancedRegressionModel(config)
        elif model_type == "quantum":
            return QuantumInspiredRegressionModel(config)
        else:
            raise ValueError(f"Unknown regression model type: {model_type}")


def create_regression_model(config: ModelConfig) -> BaseRegressionModel:
    """Convenience wrapper to instantiate a regression model from config."""

    return RegressionModelFactory.create_model(config)


class BaseRegressionModel(BaseModel):
    """
    Base class for all regression models in the system.
    """
    
    def __init__(self, config: ModelConfig, name: str = "base_regression", **kwargs: Any):
        """
        Initialize regression model with configuration.
        
        Args:
            config: Model configuration parameters
        """
        super().__init__(config, name=name, **kwargs)
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.metrics = {}
        self.feature_importances_ = None
        self.trained = False
        self.cv_results = None
        self.setup_scalers()
        
    def setup_scalers(self):
        """Configure input and output scalers based on config."""
        scaler_type = self.config.get('scaler', 'standard')
        
        if scaler_type == 'standard':
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler_X = RobustScaler()
            self.scaler_y = RobustScaler()
        elif scaler_type == 'quantile':
            self.scaler_X = QuantileTransformer(output_distribution='normal')
            self.scaler_y = QuantileTransformer(output_distribution='normal')
        elif scaler_type == 'power':
            self.scaler_X = PowerTransformer(method='yeo-johnson')
            self.scaler_y = PowerTransformer(method='yeo-johnson')
        elif scaler_type == 'none':
            self.scaler_X = None
            self.scaler_y = None
        else:
            logger.warning(f"Unknown scaler type {scaler_type}, using StandardScaler")
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
    
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Scale input features if scaler is configured."""
        if self.scaler_X is not None:
            return self.scaler_X.transform(X)
        return X
    
    def _scale_target(self, y: np.ndarray) -> np.ndarray:
        """Scale target values if scaler is configured."""
        if self.scaler_y is not None:
            # Reshape for 1D arrays to ensure correct scaling
            y_reshaped = y.reshape(-1, 1) if len(y.shape) == 1 else y
            return self.scaler_y.transform(y_reshaped)
        return y
    
    def _inverse_scale_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform scaled predictions back to original scale."""
        if self.scaler_y is not None:
            # Reshape for 1D arrays to ensure correct scaling
            y_reshaped = y.reshape(-1, 1) if len(y.shape) == 1 else y
            return self.scaler_y.inverse_transform(y_reshaped)
        return y
    
    def preprocess(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess input data before training or inference.
        
        Args:
            X: Input features
            y: Target values (optional, for training)
            
        Returns:
            Tuple of preprocessed X and y (y can be None)
        """
        # Handle NaN values
        if np.isnan(X).any():
            logger.warning("Input contains NaN values. Applying NaN handling strategy.")
            nan_strategy = self.config.get('nan_strategy', 'mean')
            
            if nan_strategy == 'mean':
                col_means = np.nanmean(X, axis=0)
                indices = np.where(np.isnan(X))
                X[indices] = np.take(col_means, indices[1])
            elif nan_strategy == 'median':
                col_medians = np.nanmedian(X, axis=0)
                indices = np.where(np.isnan(X))
                X[indices] = np.take(col_medians, indices[1])
            elif nan_strategy == 'zero':
                X = np.nan_to_num(X)
            else:
                raise ValueError(f"Unknown NaN strategy: {nan_strategy}")
        
        # Handle outliers if configured
        outlier_strategy = self.config.get('outlier_strategy', None)
        if outlier_strategy:
            X = self._handle_outliers(X, outlier_strategy)
        
        # Apply any custom preprocessing specific to the model type
        X = self._custom_preprocess_features(X)
        
        # Scale features during training (fit) or inference (transform)
        if self.scaler_X is not None:
            if y is not None:  # Training mode
                self.scaler_X.fit(X)
            X = self.scaler_X.transform(X)
        
        # Scale target values during training
        if y is not None and self.scaler_y is not None:
            y_reshaped = y.reshape(-1, 1) if len(y.shape) == 1 else y
            self.scaler_y.fit(y_reshaped)
            y = self.scaler_y.transform(y_reshaped).ravel() if len(y.shape) == 1 else self.scaler_y.transform(y)
        
        return X, y
    
    def _handle_outliers(self, X: np.ndarray, strategy: str) -> np.ndarray:
        """Handle outliers in the data using the specified strategy."""
        if strategy == 'clip':
            # Clip values beyond 3 standard deviations
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            X = np.clip(X, means - 3 * stds, means + 3 * stds)
        elif strategy == 'winsorize':
            # Winsorize data to the 1st and 99th percentiles
            lower_bounds = np.percentile(X, 1, axis=0)
            upper_bounds = np.percentile(X, 99, axis=0)
            for i in range(X.shape[1]):
                X[:, i] = np.clip(X[:, i], lower_bounds[i], upper_bounds[i])
        elif strategy == 'iqr':
            # Remove values outside 1.5 * IQR
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            iqr = q3 - q1
            lower_bounds = q1 - 1.5 * iqr
            upper_bounds = q3 + 1.5 * iqr
            for i in range(X.shape[1]):
                X[:, i] = np.clip(X[:, i], lower_bounds[i], upper_bounds[i])
        else:
            logger.warning(f"Unknown outlier strategy: {strategy}")
        
        return X
    
    def _custom_preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """
        Apply custom preprocessing specific to the model.
        Override in subclasses if needed.
        """
        return X
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the regression model.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Preprocess the data
            X_processed, y_processed = self.preprocess(X, y)
            
            # Create and configure the model
            self._create_model()
            
            # Log training start
            logger.info(f"Training {self.__class__.__name__} with {X_processed.shape[0]} samples and {X_processed.shape[1]} features")
            
            # Perform cross-validation if configured
            if self.config.get('cross_validation', False):
                self._perform_cross_validation(X_processed, y_processed)
            
            # Perform hyperparameter tuning if configured
            if self.config.get('hyperparameter_tuning', False):
                self._tune_hyperparameters(X_processed, y_processed)
            
            # Train the final model
            start_time = datetime.now()
            self.model.fit(X_processed, y_processed)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate training metrics
            y_pred = self.model.predict(X_processed)
            if self.scaler_y is not None:
                y_pred_original = self._inverse_scale_target(y_pred).ravel() if len(y_pred.shape) == 1 else self._inverse_scale_target(y_pred)
                y_original = self._inverse_scale_target(y_processed.reshape(-1, 1)).ravel() if len(y_processed.shape) == 1 else self._inverse_scale_target(y_processed)
                self.metrics = self._calculate_metrics(y_original, y_pred_original)
            else:
                self.metrics = self._calculate_metrics(y_processed, y_pred)
            
            # Extract feature importances if available
            self._extract_feature_importances()
            
            # Update model metadata
            self.trained = True
            self.metadata = {
                'training_date': datetime.now().isoformat(),
                'samples': X.shape[0],
                'features': X.shape[1],
                'training_time': training_time,
                'model_type': self.__class__.__name__,
                'config': self.config.to_dict()
            }
            
            logger.info(f"Training completed. Metrics: {self.metrics}")
            
            # Store additional training artifacts if configured
            if self.config.get('store_artifacts', False):
                self._store_training_artifacts(X_processed, y_processed, y_pred)
            
            return {
                'metrics': self.metrics,
                'metadata': self.metadata
            }
        
        except Exception as e:
            error_msg = f"Error during model training: {str(e)}"
            logger.error(error_msg)
            raise TrainingError(error_msg) from e
    
    def _create_model(self):
        """
        Create and configure the actual ML model.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _create_model()")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression performance metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'median_ae': median_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
        
        # Calculate MAPE if no zeros in y_true
        if not np.any(y_true == 0):
            try:
                metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
            except:
                # Fallback calculation if sklearn version doesn't have MAPE
                metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Calculate directional accuracy (for time series)
        if len(y_true) > 1:
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            dir_match = np.sum(actual_direction == pred_direction)
            metrics['directional_accuracy'] = dir_match / len(actual_direction) if len(actual_direction) > 0 else 0
        
        return metrics
    
    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray):
        """Perform time series cross-validation."""
        cv_strategy = self.config.get('cv_strategy', 'time_series')
        n_splits = self.config.get('cv_splits', 5)
        
        if cv_strategy == 'time_series':
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            # Default k-fold CV
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.config.get('random_state', 42))
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=cv, 
            scoring=self.config.get('cv_scoring', 'neg_mean_squared_error')
        )
        
        self.cv_results = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        logger.info(f"Cross-validation results: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """Tune hyperparameters using grid or random search."""
        param_grid = self.config.get('hyperparameter_grid', {})
        if not param_grid:
            logger.warning("Hyperparameter tuning enabled but no parameter grid specified")
            return
        
        tuning_strategy = self.config.get('tuning_strategy', 'random')
        n_splits = self.config.get('cv_splits', 5)
        cv = TimeSeriesSplit(n_splits=n_splits)
        
        if tuning_strategy == 'grid':
            search = GridSearchCV(
                self.model, param_grid, 
                cv=cv,
                scoring=self.config.get('cv_scoring', 'neg_mean_squared_error'),
                n_jobs=self.config.get('n_jobs', -1),
                verbose=self.config.get('verbose', 1)
            )
        else:  # Random search
            n_iter = self.config.get('tuning_iterations', 20)
            search = RandomizedSearchCV(
                self.model, param_grid, 
                n_iter=n_iter,
                cv=cv,
                scoring=self.config.get('cv_scoring', 'neg_mean_squared_error'),
                n_jobs=self.config.get('n_jobs', -1),
                verbose=self.config.get('verbose', 1),
                random_state=self.config.get('random_state', 42)
            )
        
        logger.info(f"Starting hyperparameter tuning with {tuning_strategy} search")
        search.fit(X, y)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best score: {search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = search.best_estimator_
    
    def _extract_feature_importances(self):
        """Extract feature importances if the model supports it."""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models
            self.feature_importances_ = np.abs(self.model.coef_)
        else:
            self.feature_importances_ = None
    
    def _store_training_artifacts(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        """Store additional training artifacts for analysis."""
        # This could include residual plots, feature importance plots, etc.
        # Implement in subclasses if needed
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        if not self.trained or self.model is None:
            raise InferenceError("Model is not trained. Call train() first.")
        
        try:
            # Preprocess input features
            X_processed, _ = self.preprocess(X)
            
            # Generate predictions
            predictions = self.model.predict(X_processed)
            
            # Transform back to original scale if needed
            if self.scaler_y is not None:
                predictions = self._inverse_scale_target(predictions)
                # Ensure we return the right shape
                if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                    predictions = predictions.ravel()
            
            return predictions
        
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            logger.error(error_msg)
            raise InferenceError(error_msg) from e
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence intervals.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        # Default implementation for models that don't natively support uncertainty
        # Override in subclasses that support native uncertainty estimation
        predictions = self.predict(X)
        
        # Use a simple uncertainty estimation method based on model metrics
        # This is a fallback and should be improved in specific model implementations
        if 'rmse' in self.metrics:
            confidence = np.ones_like(predictions) * self.metrics['rmse']
        else:
            # If no RMSE available, use a percentage of the prediction value
            confidence = np.abs(predictions) * 0.1  # 10% of prediction value
        
        return predictions, confidence
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances if available.
        
        Returns:
            Feature importance values or None if not supported
        """
        return self.feature_importances_

    def export_model(self, path: str) -> str:
        """
        Export the trained model to a file.
        
        Args:
            path: Directory to save the model
            
        Returns:
            Path to the saved model file
        """
        if not self.trained:
            raise ModelError("Cannot export untrained model")
        
        try:
            os.makedirs(path, exist_ok=True)
            model_file = os.path.join(path, f"{self.config.model_id}.joblib")
            
            # Create a model bundle with all necessary components
            model_bundle = {
                'model': self.model,
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'config': self.config.to_dict(),
                'metadata': self.metadata,
                'metrics': self.metrics,
                'feature_importances': self.feature_importances_,
                'model_class': self.__class__.__name__
            }
            
            joblib.dump(model_bundle, model_file)
            logger.info(f"Model exported to {model_file}")
            
            return model_file
        
        except Exception as e:
            error_msg = f"Error exporting model: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    @classmethod
    def import_model(cls, path: str) -> 'BaseRegressionModel':
        """
        Import a trained model from a file.
        
        Args:
            path: Path to the model file
            
        Returns:
            Loaded model instance
        """
        try:
            model_bundle = joblib.load(path)
            
            # Get the correct model class
            model_class_name = model_bundle.get('model_class', cls.__name__)
            model_config = ModelConfig(**model_bundle.get('config', {}))
            
            # Find the correct class based on name
            if model_class_name == cls.__name__:
                model_instance = cls(model_config)
            else:
                # Try to find the model class in the current module
                import sys
                current_module = sys.modules[__name__]
                model_class = getattr(current_module, model_class_name, None)
                
                if model_class is None:
                    logger.warning(f"Model class {model_class_name} not found, using {cls.__name__}")
                    model_instance = cls(model_config)
                else:
                    model_instance = model_class(model_config)
            
            # Restore model state
            model_instance.model = model_bundle['model']
            model_instance.scaler_X = model_bundle.get('scaler_X')
            model_instance.scaler_y = model_bundle.get('scaler_y')
            model_instance.metadata = model_bundle.get('metadata', {})
            model_instance.metrics = model_bundle.get('metrics', {})
            model_instance.feature_importances_ = model_bundle.get('feature_importances')
            model_instance.trained = True
            
            logger.info(f"Model imported from {path}")
            
            return model_instance
        
        except Exception as e:
            error_msg = f"Error importing model: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e


class LinearRegressionModel(BaseRegressionModel):
    """Linear regression model implementation."""
    
    def _create_model(self):
        """Create and configure a linear regression model."""
        fit_intercept = self.config.get('fit_intercept', True)
        n_jobs = self.config.get('n_jobs', None)
        
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            n_jobs=n_jobs
        )


class RidgeRegressionModel(BaseRegressionModel):
    """Ridge regression model with L2 regularization."""
    
    def _create_model(self):
        """Create and configure a ridge regression model."""
        alpha = self.config.get('alpha', 1.0)
        fit_intercept = self.config.get('fit_intercept', True)
        solver = self.config.get('solver', 'auto')
        
        # Use GPU-accelerated version if available and configured
        if CUML_AVAILABLE and self.config.get('use_gpu', False):
            self.model = cuRidge(
                alpha=alpha,
                fit_intercept=fit_intercept
            )
        else:
            self.model = Ridge(
                alpha=alpha,
                fit_intercept=fit_intercept,
                solver=solver,
                random_state=self.config.get('random_state', 42)
            )


class LassoRegressionModel(BaseRegressionModel):
    """Lasso regression model with L1 regularization."""
    
    def _create_model(self):
        """Create and configure a lasso regression model."""
        alpha = self.config.get('alpha', 1.0)
        fit_intercept = self.config.get('fit_intercept', True)
        max_iter = self.config.get('max_iter', 1000)
        
        # Use GPU-accelerated version if available and configured
        if CUML_AVAILABLE and self.config.get('use_gpu', False):
            self.model = cuLasso(
                alpha=alpha,
                fit_intercept=fit_intercept
            )
        else:
            self.model = Lasso(
                alpha=alpha,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                random_state=self.config.get('random_state', 42)
            )


class ElasticNetRegressionModel(BaseRegressionModel):
    """ElasticNet regression model with combined L1 and L2 regularization."""
    
    def _create_model(self):
        """Create and configure an elastic net regression model."""
        alpha = self.config.get('alpha', 1.0)
        l1_ratio = self.config.get('l1_ratio', 0.5)
        fit_intercept = self.config.get('fit_intercept', True)
        max_iter = self.config.get('max_iter', 1000)
        
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            random_state=self.config.get('random_state', 42)
        )


class SVRModel(BaseRegressionModel):
    """Support Vector Regression model."""
    
    def _create_model(self):
        """Create and configure a support vector regression model."""
        kernel = self.config.get('kernel', 'rbf')
        C = self.config.get('C', 1.0)
        epsilon = self.config.get('epsilon', 0.1)
        gamma = self.config.get('gamma', 'scale')
        
        # Use GPU-accelerated version if available and configured
        if CUML_AVAILABLE and self.config.get('use_gpu', False):
            self.model = cuSVR(
                C=C,
                epsilon=epsilon,
                kernel=kernel,
                gamma=gamma
            )
        else:
            self.model = SVR(
                kernel=kernel,
                C=C,
                epsilon=epsilon,
                gamma=gamma
            )


class RandomForestRegressionModel(BaseRegressionModel):
    """Random Forest regression model."""
    
    def _create_model(self):
        """Create and configure a random forest regression model."""
        n_estimators = self.config.get('n_estimators', 100)
        max_depth = self.config.get('max_depth', None)
        min_samples_split = self.config.get('min_samples_split', 2)
        min_samples_leaf = self.config.get('min_samples_leaf', 1)
        
        # Use GPU-accelerated version if available and configured
        if CUML_AVAILABLE and self.config.get('use_gpu', False):
            self.model = cuRFR(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                n_jobs=self.config.get('n_jobs', -1),
                random_state=self.config.get('random_state', 42)
            )
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with confidence intervals using forest's tree variance.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        if not self.trained:
            raise InferenceError("Model is not trained")
        
        # Preprocess input features
        X_processed, _ = self.preprocess(X)
        
        # Get predictions from each tree
        predictions = []
        for tree in self.model.estimators_:
            if hasattr(tree, 'predict'):
                tree_pred = tree.predict(X_processed)
                predictions.append(tree_pred)
        
        # Convert to array
        predictions = np.array(predictions)
        
        # Mean prediction across all trees
        mean_prediction = np.mean(predictions, axis=0)
        
        # Standard deviation as a measure of uncertainty
        std_prediction = np.std(predictions, axis=0)
        
        # Transform back to original scale if needed
        if self.scaler_y is not None:
            mean_prediction_reshaped = mean_prediction.reshape(-1, 1)
            std_prediction_reshaped = std_prediction.reshape(-1, 1)
            
            mean_prediction = self._inverse_scale_target(mean_prediction_reshaped).ravel()
            
            # For std, we need to scale by the same factor but not shift
            # This is an approximation, as proper error propagation can be complex
            if isinstance(self.scaler_y, StandardScaler):
                std_prediction = std_prediction * self.scaler_y.scale_
            elif isinstance(self.scaler_y, MinMaxScaler):
                std_prediction = std_prediction * (self.scaler_y.data_max_ - self.scaler_y.data_min_)
            else:
                # For other scalers, we use a simple proportional approach
                # This is an approximation
                mean_scale = np.mean(mean_prediction / mean_prediction_reshaped.ravel())
                std_prediction = std_prediction * mean_scale
        
        return mean_prediction, std_prediction


class GradientBoostingRegressionModel(BaseRegressionModel):
    """Gradient Boosting regression model."""
    
    def _create_model(self):
        """Create and configure a gradient boosting regression model."""
        n_estimators = self.config.get('n_estimators', 100)
        learning_rate = self.config.get('learning_rate', 0.1)
        max_depth = self.config.get('max_depth', 3)
        subsample = self.config.get('subsample', 1.0)
        
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=self.config.get('random_state', 42)
        )


class XGBoostRegressionModel(BaseRegressionModel):
    """XGBoost regression model implementation."""
    
    def _create_model(self):
        """Create and configure an XGBoost regression model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with 'pip install xgboost'")
        
        n_estimators = self.config.get('n_estimators', 100)
        learning_rate = self.config.get('learning_rate', 0.1)
        max_depth = self.config.get('max_depth', 6)
        subsample = self.config.get('subsample', 1.0)
        colsample_bytree = self.config.get('colsample_bytree', 1.0)
        gamma = self.config.get('gamma', 0)
        reg_alpha = self.config.get('reg_alpha', 0)
        reg_lambda = self.config.get('reg_lambda', 1)
        
        # Configure GPU usage if available
        tree_method = 'gpu_hist' if self.config.get('use_gpu', False) else 'auto'
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method=tree_method,
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )


class LightGBMRegressionModel(BaseRegressionModel):
    """LightGBM regression model implementation."""
    
    def _create_model(self):
        """Create and configure a LightGBM regression model."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with 'pip install lightgbm'")
        
        n_estimators = self.config.get('n_estimators', 100)
        learning_rate = self.config.get('learning_rate', 0.1)
        max_depth = self.config.get('max_depth', -1)  # -1 means no limit
        num_leaves = self.config.get('num_leaves', 31)
        subsample = self.config.get('subsample', 1.0)
        colsample_bytree = self.config.get('colsample_bytree', 1.0)
        reg_alpha = self.config.get('reg_alpha', 0)
        reg_lambda = self.config.get('reg_lambda', 0)
        
        # Configure GPU usage if available
        device = 'gpu' if self.config.get('use_gpu', False) else 'cpu'
        
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            device=device,
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )


class CatBoostRegressionModel(BaseRegressionModel):
    """CatBoost regression model implementation."""
    
    def _create_model(self):
        """Create and configure a CatBoost regression model."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not available. Install with 'pip install catboost'")
        
        iterations = self.config.get('iterations', 100)
        learning_rate = self.config.get('learning_rate', 0.1)
        depth = self.config.get('depth', 6)
        l2_leaf_reg = self.config.get('l2_leaf_reg', 3.0)
        rsm = self.config.get('rsm', 1.0)  # Random subspace method
        
        # Configure GPU usage if available
        task_type = 'GPU' if self.config.get('use_gpu', False) else 'CPU'
        
        self.model = cb.CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            rsm=rsm,
            task_type=task_type,
            random_seed=self.config.get('random_state', 42),
            thread_count=self.config.get('n_jobs', -1),
            verbose=self.config.get('verbose', False)
        )


class NeuralNetworkRegressionModel(BaseRegressionModel):
    """Neural Network regression model implementation using SKLearn MLPRegressor."""
    
    def _create_model(self):
        """Create and configure a neural network regression model."""
        hidden_layer_sizes = self.config.get('hidden_layer_sizes', (100, 50))
        activation = self.config.get('activation', 'relu')
        solver = self.config.get('solver', 'adam')
        alpha = self.config.get('alpha', 0.0001)
        learning_rate = self.config.get('learning_rate', 'adaptive')
        max_iter = self.config.get('max_iter', 1000)
        early_stopping = self.config.get('early_stopping', True)
        
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=self.config.get('random_state', 42),
            verbose=self.config.get('verbose', False)
        )


class EnsembleRegressionModel(BaseRegressionModel):
    """Ensemble regression model combining multiple regression models."""
    
    def _create_model(self):
        """Create and configure an ensemble of regression models."""
        ensemble_type = self.config.get('ensemble_type', 'voting')
        estimators = []
        
        # Create base estimators
        for name, model_config in self.config.get('base_models', {}).items():
            model_type = model_config.get('model_type')
            model_params = model_config.get('params', {})
            
            if model_type == 'linear':
                estimator = LinearRegression(**model_params)
            elif model_type == 'ridge':
                estimator = Ridge(**model_params)
            elif model_type == 'lasso':
                estimator = Lasso(**model_params)
            elif model_type == 'svr':
                estimator = SVR(**model_params)
            elif model_type == 'random_forest':
                estimator = RandomForestRegressor(**model_params)
            elif model_type == 'gbm':
                estimator = GradientBoostingRegressor(**model_params)
            elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                estimator = xgb.XGBRegressor(**model_params)
            elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                estimator = lgb.LGBMRegressor(**model_params)
            elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                estimator = cb.CatBoostRegressor(**model_params)
            else:
                logger.warning(f"Unknown or unavailable model type: {model_type}")
                continue
            
            estimators.append((name, estimator))
        
        if not estimators:
            # If no valid estimators were configured, use defaults
            estimators = [
                ('linear', LinearRegression()),
                ('ridge', Ridge(alpha=0.5)),
                ('rf', RandomForestRegressor(n_estimators=50, random_state=42))
            ]
        
        if ensemble_type == 'voting':
            self.model = VotingRegressor(
                estimators=estimators,
                weights=self.config.get('weights', None),
                n_jobs=self.config.get('n_jobs', -1)
            )
        elif ensemble_type == 'stacking':
            # Configure meta-learner
            meta_model_config = self.config.get('meta_model', {})
            meta_model_type = meta_model_config.get('model_type', 'ridge')
            meta_model_params = meta_model_config.get('params', {})
            
            if meta_model_type == 'ridge':
                final_estimator = Ridge(**meta_model_params)
            elif meta_model_type == 'gbm':
                final_estimator = GradientBoostingRegressor(**meta_model_params)
            else:
                final_estimator = Ridge(alpha=0.5)  # Default
            
            self.model = StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=self.config.get('cv', 5),
                n_jobs=self.config.get('n_jobs', -1)
            )
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")


class TimeWeightedRegressionModel(BaseRegressionModel):
    """Regression model that gives more weight to recent data."""
    
    def _create_model(self):
        """Create and configure a time-weighted regression model."""
        base_model_type = self.config.get('base_model_type', 'random_forest')
        decay_factor = self.config.get('decay_factor', 0.95)  # Weight decay per time step
        
        # Configure the base model
        if base_model_type == 'linear':
            self.base_model = LinearRegression()
        elif base_model_type == 'ridge':
            alpha = self.config.get('alpha', 1.0)
            self.base_model = Ridge(alpha=alpha)
        elif base_model_type == 'random_forest':
            n_estimators = self.config.get('n_estimators', 100)
            self.base_model = RandomForestRegressor(
                n_estimators=n_estimators, 
                random_state=self.config.get('random_state', 42)
            )
        elif base_model_type == 'gbm':
            n_estimators = self.config.get('n_estimators', 100)
            self.base_model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                random_state=self.config.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported base model type: {base_model_type}")
        
        # Store decay factor for use during training
        self.decay_factor = decay_factor
        
        # Initialize the model (will be replaced during training)
        self.model = self.base_model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the time-weighted regression model.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Preprocess the data
            X_processed, y_processed = self.preprocess(X, y)
            
            # Calculate sample weights based on time
            # Assumes samples are ordered chronologically (newest last)
            n_samples = X_processed.shape[0]
            sample_weights = np.power(self.decay_factor, np.arange(n_samples - 1, -1, -1))
            
            # Normalize weights
            sample_weights = sample_weights / np.sum(sample_weights) * n_samples
            
            # Log training start
            logger.info(f"Training {self.__class__.__name__} with time-weighted samples")
            
            # Train the model with sample weights
            start_time = datetime.now()
            if hasattr(self.base_model, 'fit') and 'sample_weight' in self.base_model.fit.__code__.co_varnames:
                self.base_model.fit(X_processed, y_processed, sample_weight=sample_weights)
                self.model = self.base_model
            else:
                # Fall back to LinearRegression with analytical weights if model doesn't support sample_weight
                logger.warning(f"Base model doesn't support sample weights, using weighted LinearRegression")
                from sklearn.linear_model import LinearRegression
                self.model = LinearRegression()
                self.model.fit(X_processed, y_processed, sample_weight=sample_weights)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate training metrics
            y_pred = self.model.predict(X_processed)
            if self.scaler_y is not None:
                y_pred_original = self._inverse_scale_target(y_pred)
                y_original = self._inverse_scale_target(y_processed)
                self.metrics = self._calculate_metrics(y_original, y_pred_original)
            else:
                self.metrics = self._calculate_metrics(y_processed, y_pred)
            
            # Extract feature importances if available
            self._extract_feature_importances()
            
            # Update model metadata
            self.trained = True
            self.metadata = {
                'training_date': datetime.now().isoformat(),
                'samples': X.shape[0],
                'features': X.shape[1],
                'training_time': training_time,
                'model_type': self.__class__.__name__,
                'base_model_type': self.config.get('base_model_type'),
                'decay_factor': self.decay_factor,
                'config': self.config.to_dict()
            }
            
            logger.info(f"Time-weighted training completed. Metrics: {self.metrics}")
            
            return {
                'metrics': self.metrics,
                'metadata': self.metadata
            }
        
        except Exception as e:
            error_msg = f"Error during time-weighted model training: {str(e)}"
            logger.error(error_msg)
            raise TrainingError(error_msg) from e


class PeriodicRegressionModel(BaseRegressionModel):
    """Regression model that accounts for periodic patterns in time series data."""
    
    def _create_model(self):
        """Create and configure a periodic regression model."""
        base_model_type = self.config.get('base_model_type', 'random_forest')
        
        # Configure the base model
        if base_model_type == 'random_forest':
            n_estimators = self.config.get('n_estimators', 100)
            self.base_model = RandomForestRegressor(
                n_estimators=n_estimators, 
                random_state=self.config.get('random_state', 42)
            )
        elif base_model_type == 'gbm':
            n_estimators = self.config.get('n_estimators', 100)
            self.base_model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                random_state=self.config.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported base model type: {base_model_type}")
        
        # Store periodicity parameters
        self.period_daily = self.config.get('period_daily', True)
        self.period_weekly = self.config.get('period_weekly', True)
        self.period_monthly = self.config.get('period_monthly', False)
        
        # Initialize the model (will be a pipeline with feature creation)
        self.model = None
    
    def _create_periodic_features(self, X: np.ndarray) -> np.ndarray:
        """Create periodic features for time-series data."""
        # Check if timestamp column is specified
        timestamp_col = self.config.get('timestamp_column', None)
        if timestamp_col is None:
            logger.warning("No timestamp column specified for periodic features")
            return X
        
        # Get timestamp column index
        try:
            if isinstance(timestamp_col, str):
                # Find column by name if X is a DataFrame
                if hasattr(X, 'columns'):
                    timestamp_col_idx = list(X.columns).index(timestamp_col)
                else:
                    raise ValueError(f"Cannot find column by name in ndarray: {timestamp_col}")
            else:
                timestamp_col_idx = timestamp_col
        except (ValueError, IndexError) as e:
            logger.error(f"Error finding timestamp column: {str(e)}")
            return X
        
        # Extract timestamps
        try:
            if hasattr(X, 'values'):
                timestamps = X.values[:, timestamp_col_idx]
            else:
                timestamps = X[:, timestamp_col_idx]
                
            # Convert to datetime if not already
            if not np.issubdtype(timestamps.dtype, np.datetime64):
                timestamps = pd.to_datetime(timestamps)
        except Exception as e:
            logger.error(f"Error processing timestamps: {str(e)}")
            return X
        
        # Create new feature array
        X_new = X.copy() if hasattr(X, 'copy') else np.copy(X)
        
        # Extract periodic components
        if self.period_daily:
            # Hour of day features using sine/cosine for circularity
            hour = pd.DatetimeIndex(timestamps).hour
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            # Add hour features
            X_new = np.column_stack([X_new, hour_sin, hour_cos])
        
        if self.period_weekly:
            # Day of week features using sine/cosine for circularity
            day_of_week = pd.DatetimeIndex(timestamps).dayofweek
            day_sin = np.sin(2 * np.pi * day_of_week / 7)
            day_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            # Add day of week features
            X_new = np.column_stack([X_new, day_sin, day_cos])
        
        if self.period_monthly:
            # Day of month features using sine/cosine for circularity
            day_of_month = pd.DatetimeIndex(timestamps).day
            # Approximate with 30 days per month
            day_of_month_sin = np.sin(2 * np.pi * day_of_month / 30)
            day_of_month_cos = np.cos(2 * np.pi * day_of_month / 30)
            
            # Add day of month features
            X_new = np.column_stack([X_new, day_of_month_sin, day_of_month_cos])
        
        return X_new
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the periodic regression model.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Preprocess the data
            X_processed, y_processed = self.preprocess(X, y)
            
            # Create periodic features
            X_with_periodic = self._create_periodic_features(X_processed)
            
            # Log training start
            logger.info(f"Training {self.__class__.__name__} with periodic features")
            
            # Train the base model
            start_time = datetime.now()
            self.base_model.fit(X_with_periodic, y_processed)
            self.model = self.base_model  # Store the trained model
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate training metrics
            y_pred = self.model.predict(X_with_periodic)
            if self.scaler_y is not None:
                y_pred_original = self._inverse_scale_target(y_pred)
                y_original = self._inverse_scale_target(y_processed)
                self.metrics = self._calculate_metrics(y_original, y_pred_original)
            else:
                self.metrics = self._calculate_metrics(y_processed, y_pred)
            
            # Extract feature importances if available
            self._extract_feature_importances()
            
            # Update model metadata
            self.trained = True
            self.metadata = {
                'training_date': datetime.now().isoformat(),
                'samples': X.shape[0],
                'features': X.shape[1],
                'features_with_periodic': X_with_periodic.shape[1],
                'training_time': training_time,
                'model_type': self.__class__.__name__,
                'base_model_type': self.config.get('base_model_type'),
                'config': self.config.to_dict()
            }
            
            logger.info(f"Periodic model training completed. Metrics: {self.metrics}")
            
            return {
                'metrics': self.metrics,
                'metadata': self.metadata
            }
        
        except Exception as e:
            error_msg = f"Error during periodic model training: {str(e)}"
            logger.error(error_msg)
            raise TrainingError(error_msg) from e
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained periodic model.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        if not self.trained or self.model is None:
            raise InferenceError("Model is not trained. Call train() first.")
        
        try:
            # Preprocess input features
            X_processed, _ = self.preprocess(X)
            
            # Create periodic features
            X_with_periodic = self._create_periodic_features(X_processed)
            
            # Generate predictions
            predictions = self.model.predict(X_with_periodic)
            
            # Transform back to original scale if needed
            if self.scaler_y is not None:
                predictions = self._inverse_scale_target(predictions)
            
            return predictions
        
        except Exception as e:
            error_msg = f"Error during prediction with periodic model: {str(e)}"
            logger.error(error_msg)
            raise InferenceError(error_msg) from e


class AdaptiveRegressionModel(BaseRegressionModel):
    """Regression model that adapts to changing market conditions."""
    
    def _create_model(self):
        """Create and configure an adaptive regression model."""
        # Configure base models
        self.models = {}
        
        # Model for trending markets
        trend_model_type = self.config.get('trend_model_type', 'gbm')
        if trend_model_type == 'gbm':
            self.models['trend'] = GradientBoostingRegressor(
                n_estimators=self.config.get('trend_n_estimators', 100),
                learning_rate=self.config.get('trend_learning_rate', 0.1),
                max_depth=self.config.get('trend_max_depth', 5),
                random_state=self.config.get('random_state', 42)
            )
        else:
            self.models['trend'] = RandomForestRegressor(
                n_estimators=self.config.get('trend_n_estimators', 100),
                random_state=self.config.get('random_state', 42)
            )
        
        # Model for ranging markets
        range_model_type = self.config.get('range_model_type', 'svr')
        if range_model_type == 'svr':
            self.models['range'] = SVR(
                C=self.config.get('range_C', 1.0),
                epsilon=self.config.get('range_epsilon', 0.1),
                kernel=self.config.get('range_kernel', 'rbf')
            )
        else:
            self.models['range'] = Ridge(
                alpha=self.config.get('range_alpha', 1.0)
            )
        
        # Model for volatile markets
        volatile_model_type = self.config.get('volatile_model_type', 'rf')
        if volatile_model_type == 'rf':
            self.models['volatile'] = RandomForestRegressor(
                n_estimators=self.config.get('volatile_n_estimators', 100),
                max_depth=self.config.get('volatile_max_depth', 10),
                random_state=self.config.get('random_state', 42)
            )
        else:
            self.models['volatile'] = GradientBoostingRegressor(
                n_estimators=self.config.get('volatile_n_estimators', 100),
                random_state=self.config.get('random_state', 42)
            )
        
        # Initialize the market regime detector
        self.regime_threshold_trend = self.config.get('regime_threshold_trend', 0.6)
        self.regime_threshold_volatile = self.config.get('regime_threshold_volatile', 0.4)
        
        # The model will be selected during prediction based on market conditions
        self.model = None
    
    def _detect_market_regime(self, X: np.ndarray) -> str:
        """
        Detect the current market regime (trend, range, volatile).
        
        Args:
            X: Input features with market information
            
        Returns:
            Detected regime: 'trend', 'range', or 'volatile'
        """
        # Default implementation based on simple heuristics
        # In a real system, this would be more sophisticated
        
        # Check if regime features are available
        regime_col = self.config.get('regime_column', None)
        if regime_col is not None:
            try:
                if isinstance(regime_col, str) and hasattr(X, 'columns'):
                    regime_col_idx = list(X.columns).index(regime_col)
                else:
                    regime_col_idx = regime_col
                
                if hasattr(X, 'values'):
                    regime_value = X.values[-1, regime_col_idx]
                else:
                    regime_value = X[-1, regime_col_idx]
                
                if regime_value > self.regime_threshold_trend:
                    return 'trend'
                elif regime_value < self.regime_threshold_volatile:
                    return 'volatile'
                else:
                    return 'range'
            except (ValueError, IndexError) as e:
                logger.warning(f"Error detecting regime from column: {str(e)}")
        
        # Fallback: Use price movement characteristics
        price_col = self.config.get('price_column', -1)  # Default to last column
        try:
            if hasattr(X, 'values'):
                prices = X.values[:, price_col]
            else:
                prices = X[:, price_col]
            
            # Calculate recent price changes
            returns = np.diff(prices) / prices[:-1]
            
            # Calculate trend strength (absolute mean return)
            trend_strength = np.abs(np.mean(returns))
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns)
            
            # Determine regime based on trend and volatility
            if trend_strength > self.regime_threshold_trend:
                return 'trend'
            elif volatility > self.regime_threshold_volatile:
                return 'volatile'
            else:
                return 'range'
        except Exception as e:
            logger.warning(f"Error in fallback regime detection: {str(e)}")
            # Default to 'range' if detection fails
            return 'range'
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train all sub-models for different market regimes.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Preprocess the data
            X_processed, y_processed = self.preprocess(X, y)
            
            # Log training start
            logger.info(f"Training {self.__class__.__name__} with regime-specific models")
            
            # Train all models with the full dataset
            # In a more sophisticated implementation, we could segment the data by regime
            start_time = datetime.now()
            for regime, model in self.models.items():
                logger.info(f"Training {regime} model")
                model.fit(X_processed, y_processed)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # For evaluation, use a mix of predictions based on detected regimes
            regimes = [self._detect_market_regime(X_processed[i:i+1]) for i in range(len(X_processed))]
            y_pred = np.zeros_like(y_processed)
            
            for i, regime in enumerate(regimes):
                y_pred[i] = self.models[regime].predict(X_processed[i:i+1])[0]
            
            # Calculate training metrics
            if self.scaler_y is not None:
                y_pred_original = self._inverse_scale_target(y_pred)
                y_original = self._inverse_scale_target(y_processed)
                self.metrics = self._calculate_metrics(y_original, y_pred_original)
            else:
                self.metrics = self._calculate_metrics(y_processed, y_pred)
            
            # Update model metadata
            self.trained = True
            self.metadata = {
                'training_date': datetime.now().isoformat(),
                'samples': X.shape[0],
                'features': X.shape[1],
                'training_time': training_time,
                'model_type': self.__class__.__name__,
                'regime_models': list(self.models.keys()),
                'config': self.config.to_dict()
            }
            
            logger.info(f"Adaptive model training completed. Metrics: {self.metrics}")
            
            return {
                'metrics': self.metrics,
                'metadata': self.metadata
            }
        
        except Exception as e:
            error_msg = f"Error during adaptive model training: {str(e)}"
            logger.error(error_msg)
            raise TrainingError(error_msg) from e
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions by selecting appropriate model for market regime.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        if not self.trained or not self.models:
            raise InferenceError("Models are not trained. Call train() first.")
        
        try:
            # Preprocess input features
            X_processed, _ = self.preprocess(X)
            
            # Detect regime for each sample
            predictions = np.zeros((X_processed.shape[0],))
            
            for i in range(X_processed.shape[0]):
                # For batch prediction, use a window of recent data for regime detection
                sample = X_processed[i:i+1]
                regime = self._detect_market_regime(sample)
                predictions[i] = self.models[regime].predict(sample)[0]
            
            # Transform back to original scale if needed
            if self.scaler_y is not None:
                predictions = self._inverse_scale_target(predictions.reshape(-1, 1)).ravel()
            
            return predictions
        
        except Exception as e:
            error_msg = f"Error during prediction with adaptive model: {str(e)}"
            logger.error(error_msg)
            raise InferenceError(error_msg) from e


class AdvancedRegressionModel(BaseRegressionModel):
    """Simple extra trees regression model for experiments."""

    def _create_model(self):
        self.model = ExtraTreesRegressor(
            n_estimators=self.config.get('n_estimators', 200),
            random_state=self.config.get('random_state', 42),
        )


class QuantumInspiredRegressionModel(BaseRegressionModel):
    """
    Quantum-inspired regression model that uses quantum computing principles
    to explore a larger solution space. This is a classical simulation of
    quantum methods and doesn't require actual quantum hardware.
    """
    
    def _create_model(self):
        """Create and configure a quantum-inspired regression model."""
        # Use a classical ML model as the base implementation
        base_model_type = self.config.get('base_model_type', 'random_forest')
        
        if base_model_type == 'random_forest':
            self.base_model = RandomForestRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                random_state=self.config.get('random_state', 42)
            )
        elif base_model_type == 'gbm':
            self.base_model = GradientBoostingRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                random_state=self.config.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported base model type: {base_model_type}")
        
        # Quantum-inspired parameters
        self.n_qubits = self.config.get('n_qubits', 4)  # Number of quantum states to simulate
        self.n_iterations = self.config.get('n_iterations', 10)  # Number of quantum annealing iterations
        
        # The model will be an ensemble created during training
        self.model = None
    
    def _quantum_feature_expansion(self, X: np.ndarray) -> np.ndarray:
        """
        Expand features using quantum-inspired transformations.
        This simulates a quantum feature map without requiring quantum hardware.
        
        Args:
            X: Input features
            
        Returns:
            Expanded features
        """
        # Normalize features to [-1, 1] range for quantum simulation
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Prevent division by zero
        X_normalized = 2 * (X - X_min) / X_range - 1
        
        # Apply quantum-inspired feature expansion
        # This simulates the effect of applying quantum gates
        
        # Phase features (similar to rotation gates)
        sin_features = np.sin(np.pi * X_normalized)
        cos_features = np.cos(np.pi * X_normalized)
        
        # Entanglement-inspired features (pairwise interactions)
        n_features = X.shape[1]
        entangled_features = []
        
        for i in range(n_features):
            for j in range(i+1, min(i+self.n_qubits, n_features)):
                # Create features that capture interactions (simulating entanglement)
                entangled_features.append(X_normalized[:, i] * X_normalized[:, j])
                entangled_features.append(np.sin(np.pi * X_normalized[:, i] * X_normalized[:, j]))
        
        # Combine all features
        if entangled_features:
            entangled_array = np.column_stack(entangled_features)
            expanded_features = np.column_stack([X, sin_features, cos_features, entangled_array])
        else:
            expanded_features = np.column_stack([X, sin_features, cos_features])
        
        return expanded_features
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train the quantum-inspired regression model.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Preprocess the data
            X_processed, y_processed = self.preprocess(X, y)
            
            # Apply quantum-inspired feature expansion
            X_quantum = self._quantum_feature_expansion(X_processed)
            
            # Log training start
            logger.info(f"Training {self.__class__.__name__} with quantum-inspired features")
            
            # Train the base model with expanded features
            start_time = datetime.now()
            self.base_model.fit(X_quantum, y_processed)
            
            # The final model is the base model trained on quantum-expanded features
            self.model = self.base_model
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate training metrics
            y_pred = self.model.predict(X_quantum)
            if self.scaler_y is not None:
                y_pred_original = self._inverse_scale_target(y_pred)
                y_original = self._inverse_scale_target(y_processed)
                self.metrics = self._calculate_metrics(y_original, y_pred_original)
            else:
                self.metrics = self._calculate_metrics(y_processed, y_pred)
            
            # Extract feature importances if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances_ = self.model.feature_importances_
            
            # Update model metadata
            self.trained = True
            self.metadata = {
                'training_date': datetime.now().isoformat(),
                'samples': X.shape[0],
                'original_features': X.shape[1],
                'quantum_features': X_quantum.shape[1],
                'training_time': training_time,
                'model_type': self.__class__.__name__,
                'base_model_type': self.config.get('base_model_type'),
                'n_qubits': self.n_qubits,
                'config': self.config.to_dict()
            }
            
            logger.info(f"Quantum-inspired model training completed. Metrics: {self.metrics}")
            
            return {
                'metrics': self.metrics,
                'metadata': self.metadata
            }
        
        except Exception as e:
            error_msg = f"Error during quantum-inspired model training: {str(e)}"
            logger.error(error_msg)
            raise TrainingError(error_msg) from e
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the trained quantum-inspired model.
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        if not self.trained or self.model is None:
            raise InferenceError("Model is not trained. Call train() first.")
        
        try:
            # Preprocess input features
            X_processed, _ = self.preprocess(X)
            
            # Apply quantum-inspired feature expansion
            X_quantum = self._quantum_feature_expansion(X_processed)
            
            # Generate predictions
            predictions = self.model.predict(X_quantum)
            
            # Transform back to original scale if needed
            if self.scaler_y is not None:
                predictions = self._inverse_scale_target(predictions)
            
            return predictions
        
        except Exception as e:
            error_msg = f"Error during prediction with quantum-inspired model: {str(e)}"
            logger.error(error_msg)
            raise InferenceError(error_msg) from e
