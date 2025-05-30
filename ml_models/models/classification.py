#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Classification Models Module

This module provides advanced classification models for predicting market direction,
pattern completions, and trading signals. It includes sophisticated ensemble methods,
deep learning models, and traditional classification algorithms, all optimized for
high-accuracy predictions in financial markets.
"""

import os
import logging
import pickle
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union, Optional, Callable
from datetime import datetime
from pathlib import Path

# Machine learning imports
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers, Model, optimizers, callbacks, regularizers  # type: ignore
    from tensorflow.keras.models import Sequential, load_model  # type: ignore
    TENSORFLOW_AVAILABLE = True

except Exception:  # pragma: no cover - optional dependency
    tf = None  # type: ignore
    layers = Model = optimizers = callbacks = regularizers = None  # type: ignore
    Sequential = load_model = None  # type: ignore
    TENSORFLOW_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    log_loss, brier_score_loss, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    SMOTE = ADASYN = RandomUnderSampler = None  # type: ignore
    IMBLEARN_AVAILABLE = False

# Local imports
from common.logger import get_logger
from common.utils import timing_decorator, calculate_sharpe, calculate_sortino
from common.exceptions import ModelTrainingError, ModelPredictionError
from ml_models.models.base_model import BaseModel

# Initialize logger
logger = get_logger(__name__)

class ClassificationModel(BaseModel):
    """
    Base class for classification models in the QuantumSpectre trading system.
    Provides common functionality for all classification model types.
    """
    
    MODEL_TYPE = "classification"
    
    def __init__(
        self, 
        name: str, 
        model_type: str,
        model_config: Dict[str, Any],
        asset_id: str = None,
        timeframe: str = None,
        feature_columns: List[str] = None,
        target_column: str = 'target',
        scaling: bool = True,
        class_weights: Dict[int, float] = None,
        random_state: int = 42,
        model_path: str = None,
        use_gpu: bool = True,
        **kwargs
    ):
        """
        Initialize a classification model with the specified parameters.
        
        Args:
            name: Unique name for the model
            model_type: Type of model (e.g., 'random_forest', 'xgboost', 'lstm')
            model_config: Dictionary of model-specific configuration parameters
            asset_id: ID of the asset this model is for (if asset-specific)
            timeframe: Timeframe this model is for (e.g., '1h', '1d')
            feature_columns: List of feature column names for training/prediction
            target_column: Name of the target column for prediction
            scaling: Whether to apply feature scaling
            class_weights: Weights for different classes to handle imbalance
            random_state: Random seed for reproducibility
            model_path: Path to save/load the model
            use_gpu: Whether to use GPU acceleration when available
            **kwargs: Additional parameters passed to the model constructor
        """
        super().__init__(
            name=name,
            model_type=model_type,
            model_config=model_config,
            asset_id=asset_id,
            timeframe=timeframe,
            feature_columns=feature_columns,
            target_column=target_column,
            model_path=model_path,
            use_gpu=use_gpu,
            **kwargs
        )
        self.scaling = scaling
        self.class_weights = class_weights
        self.random_state = random_state
        self.scaler = StandardScaler() if scaling else None
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_multiclass = False
        self.num_classes = None
        self.classes_ = None
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize metrics for model evaluation."""
        self.metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'log_loss': 0.0,
            'confusion_matrix': None,
            'feature_importance': {},
            'training_duration': 0.0,
            'samples_processed': 0,
            'training_timestamp': None,
            'cross_val_scores': [],
        }
    
    @timing_decorator
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        validation_data: Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]] = None,
        sample_weight: np.ndarray = None,
        **kwargs
    ) -> None:
        """
        Train the classification model.
        
        Args:
            X: Training features
            y: Training target values
            validation_data: Optional tuple of (X_val, y_val) for validation
            sample_weight: Optional array of weights for training samples
            **kwargs: Additional training parameters
        
        Returns:
            None
        
        Raises:
            ModelTrainingError: If model training fails
        """
        try:
            start_time = time.time()
            logger.info(f"Starting training for classification model {self.name}")
            
            # Convert to DataFrame/Series if arrays
            X = pd.DataFrame(X, columns=self.feature_columns) if isinstance(X, np.ndarray) else X
            y = pd.Series(y, name=self.target_column) if isinstance(y, np.ndarray) else y
            
            # Save column order
            self.feature_columns = list(X.columns)
            
            # Check if multiclass and encode target if needed
            unique_classes = np.unique(y)
            self.num_classes = len(unique_classes)
            self.is_multiclass = self.num_classes > 2
            logger.info(f"Detected {self.num_classes} classes: {unique_classes}")
            
            if not np.issubdtype(y.dtype, np.number) or self.is_multiclass:
                logger.info(f"Encoding target variable with {self.num_classes} classes")
                y = pd.Series(self.label_encoder.fit_transform(y))
                self.classes_ = self.label_encoder.classes_
            else:
                self.classes_ = unique_classes
            
            # Handle class imbalance if needed
            if 'balance_method' in kwargs:
                X, y = self._handle_class_imbalance(X, y, kwargs['balance_method'])
                kwargs.pop('balance_method')
            
            # Scale features if requested
            X_train = X.copy()
            if self.scaling:
                logger.info(f"Scaling features for model {self.name}")
                X_train = pd.DataFrame(
                    self.scaler.fit_transform(X_train),
                    columns=X_train.columns
                )
            
            # Prepare validation data if provided
            if validation_data:
                X_val, y_val = validation_data
                X_val = pd.DataFrame(X_val, columns=self.feature_columns) if isinstance(X_val, np.ndarray) else X_val
                
                if self.scaling:
                    X_val = pd.DataFrame(
                        self.scaler.transform(X_val),
                        columns=X_val.columns
                    )
                
                if not np.issubdtype(y_val.dtype, np.number) or self.is_multiclass:
                    y_val = pd.Series(self.label_encoder.transform(y_val))
            
            # Initialize and train the specific model implementation
            self.model = self._create_model()
            self._train_model(X_train, y, validation_data=(X_val, y_val) if validation_data else None, 
                             sample_weight=sample_weight, **kwargs)
            
            # Calculate metrics
            self._calculate_training_metrics(X_train, y, validation_data)
            
            # Record training information
            self.metrics['training_duration'] = time.time() - start_time
            self.metrics['samples_processed'] = len(X)
            self.metrics['training_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Successfully trained model {self.name} in {self.metrics['training_duration']:.2f} seconds")
            logger.info(f"Model metrics: Accuracy={self.metrics['accuracy']:.4f}, F1={self.metrics['f1']:.4f}")
            
            # Save feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.metrics['feature_importance'] = dict(zip(
                    self.feature_columns, 
                    self.model.feature_importances_
                ))
            
            # Save the model
            if self.model_path:
                self.save()
                
        except Exception as e:
            logger.error(f"Error training classification model {self.name}: {str(e)}")
            raise ModelTrainingError(f"Failed to train model {self.name}: {str(e)}") from e
    
    def predict(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        return_probabilities: bool = False,
        threshold: float = 0.5,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions from the trained model.
        
        Args:
            X: Features for prediction
            return_probabilities: Whether to return probability estimates as well
            threshold: Probability threshold for binary predictions
            **kwargs: Additional prediction parameters
        
        Returns:
            Predicted classes, or tuple of (predicted classes, probabilities)
        
        Raises:
            ModelPredictionError: If prediction fails
        """
        try:
            logger.debug(f"Generating predictions with model {self.name}")
            
            # Convert to DataFrame if array
            X = pd.DataFrame(X, columns=self.feature_columns) if isinstance(X, np.ndarray) else X
            
            # Ensure correct feature order and handle missing features
            X = self._prepare_features_for_prediction(X)
            
            # Scale features if model was trained with scaling
            if self.scaling:
                X = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns
                )
            
            # Generate raw predictions
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
                
                # For binary classification, extract positive class probability
                if self.num_classes == 2 and probabilities.shape[1] == 2:
                    positive_probs = probabilities[:, 1]
                    predictions = (positive_probs >= threshold).astype(int)
                else:
                    predictions = np.argmax(probabilities, axis=1)
            else:
                predictions = self.model.predict(X)
                probabilities = None
            
            # Transform predictions back to original classes if encoded
            if hasattr(self, 'label_encoder') and self.label_encoder.classes_ is not None:
                predictions = self.label_encoder.inverse_transform(predictions)
            
            if return_probabilities:
                return predictions, probabilities
            else:
                return predictions
                
        except Exception as e:
            logger.error(f"Error generating predictions with model {self.name}: {str(e)}")
            raise ModelPredictionError(f"Failed to generate predictions with model {self.name}: {str(e)}") from e
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """
        Generate class probability estimates.
        
        Args:
            X: Features for prediction
            **kwargs: Additional prediction parameters
        
        Returns:
            Class probability estimates
        
        Raises:
            ModelPredictionError: If prediction fails
        """
        try:
            if not hasattr(self.model, 'predict_proba'):
                raise ModelPredictionError(f"Model {self.name} does not support probability predictions")
                
            # Convert to DataFrame if array
            X = pd.DataFrame(X, columns=self.feature_columns) if isinstance(X, np.ndarray) else X
            
            # Ensure correct feature order and handle missing features
            X = self._prepare_features_for_prediction(X)
            
            # Scale features if model was trained with scaling
            if self.scaling:
                X = pd.DataFrame(
                    self.scaler.transform(X),
                    columns=X.columns
                )
            
            # Return probability estimates
            return self.model.predict_proba(X)
            
        except Exception as e:
            logger.error(f"Error generating probability predictions with model {self.name}: {str(e)}")
            raise ModelPredictionError(f"Failed to generate probability predictions with model {self.name}: {str(e)}") from e
    
    def _create_model(self) -> Any:
        """
        Create and configure the underlying model based on model_type.
        
        Returns:
            Configured model instance
        
        Raises:
            ValueError: If the model type is unsupported
        """
        logger.info(f"Creating {self.model_type} model")
        
        # Configure TensorFlow if using GPU
        if self.use_gpu:
            self._configure_gpu()
        
        # Create and return the appropriate model based on model_type
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.model_config.get('n_estimators', 100),
                max_depth=self.model_config.get('max_depth', None),
                min_samples_split=self.model_config.get('min_samples_split', 2),
                min_samples_leaf=self.model_config.get('min_samples_leaf', 1),
                max_features=self.model_config.get('max_features', 'sqrt'),
                class_weight=self.class_weights,
                random_state=self.random_state,
                n_jobs=self.model_config.get('n_jobs', -1)
            )
        
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.model_config.get('n_estimators', 100),
                learning_rate=self.model_config.get('learning_rate', 0.1),
                max_depth=self.model_config.get('max_depth', 3),
                min_samples_split=self.model_config.get('min_samples_split', 2),
                min_samples_leaf=self.model_config.get('min_samples_leaf', 1),
                subsample=self.model_config.get('subsample', 1.0),
                max_features=self.model_config.get('max_features', 'sqrt'),
                random_state=self.random_state
            )
        
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=self.model_config.get('n_estimators', 100),
                learning_rate=self.model_config.get('learning_rate', 0.1),
                max_depth=self.model_config.get('max_depth', 6),
                min_child_weight=self.model_config.get('min_child_weight', 1),
                gamma=self.model_config.get('gamma', 0),
                subsample=self.model_config.get('subsample', 0.8),
                colsample_bytree=self.model_config.get('colsample_bytree', 0.8),
                objective='binary:logistic' if self.num_classes == 2 else 'multi:softprob',
                num_class=self.num_classes if self.is_multiclass else None,
                tree_method='gpu_hist' if self.use_gpu else 'hist',
                random_state=self.random_state,
                n_jobs=self.model_config.get('n_jobs', -1)
            )
        
        elif self.model_type == 'lightgbm':
            return LGBMClassifier(
                n_estimators=self.model_config.get('n_estimators', 100),
                learning_rate=self.model_config.get('learning_rate', 0.1),
                max_depth=self.model_config.get('max_depth', -1),
                num_leaves=self.model_config.get('num_leaves', 31),
                min_child_samples=self.model_config.get('min_child_samples', 20),
                subsample=self.model_config.get('subsample', 1.0),
                colsample_bytree=self.model_config.get('colsample_bytree', 1.0),
                reg_alpha=self.model_config.get('reg_alpha', 0.0),
                reg_lambda=self.model_config.get('reg_lambda', 0.0),
                objective='binary' if self.num_classes == 2 else 'multiclass',
                num_class=self.num_classes if self.is_multiclass else None,
                class_weight=self.class_weights,
                random_state=self.random_state,
                n_jobs=self.model_config.get('n_jobs', -1),
                device='gpu' if self.use_gpu else 'cpu'
            )
        
        elif self.model_type == 'catboost':
            return CatBoostClassifier(
                iterations=self.model_config.get('iterations', 100),
                learning_rate=self.model_config.get('learning_rate', 0.1),
                depth=self.model_config.get('depth', 6),
                l2_leaf_reg=self.model_config.get('l2_leaf_reg', 3),
                random_strength=self.model_config.get('random_strength', 1),
                rsm=self.model_config.get('rsm', None),
                loss_function='Logloss' if self.num_classes == 2 else 'MultiClass',
                class_weights=self.class_weights,
                random_seed=self.random_state,
                thread_count=self.model_config.get('thread_count', -1),
                task_type='GPU' if self.use_gpu else 'CPU',
                verbose=False
            )
        
        elif self.model_type == 'svm':
            return SVC(
                C=self.model_config.get('C', 1.0),
                kernel=self.model_config.get('kernel', 'rbf'),
                degree=self.model_config.get('degree', 3),
                gamma=self.model_config.get('gamma', 'scale'),
                probability=True,
                class_weight=self.class_weights,
                random_state=self.random_state
            )
        
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                C=self.model_config.get('C', 1.0),
                penalty=self.model_config.get('penalty', 'l2'),
                solver=self.model_config.get('solver', 'lbfgs'),
                max_iter=self.model_config.get('max_iter', 1000),
                class_weight=self.class_weights,
                random_state=self.random_state,
                n_jobs=self.model_config.get('n_jobs', -1),
                multi_class='auto'
            )
        
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=self.model_config.get('hidden_layer_sizes', (100,)),
                activation=self.model_config.get('activation', 'relu'),
                solver=self.model_config.get('solver', 'adam'),
                alpha=self.model_config.get('alpha', 0.0001),
                batch_size=self.model_config.get('batch_size', 'auto'),
                learning_rate=self.model_config.get('learning_rate', 'constant'),
                learning_rate_init=self.model_config.get('learning_rate_init', 0.001),
                max_iter=self.model_config.get('max_iter', 200),
                shuffle=self.model_config.get('shuffle', True),
                random_state=self.random_state
            )
        
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=self.model_config.get('n_neighbors', 5),
                weights=self.model_config.get('weights', 'uniform'),
                algorithm=self.model_config.get('algorithm', 'auto'),
                leaf_size=self.model_config.get('leaf_size', 30),
                p=self.model_config.get('p', 2),
                n_jobs=self.model_config.get('n_jobs', -1)
            )
        
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(
                criterion=self.model_config.get('criterion', 'gini'),
                max_depth=self.model_config.get('max_depth', None),
                min_samples_split=self.model_config.get('min_samples_split', 2),
                min_samples_leaf=self.model_config.get('min_samples_leaf', 1),
                max_features=self.model_config.get('max_features', None),
                class_weight=self.class_weights,
                random_state=self.random_state
            )
        
        elif self.model_type == 'qda':
            return QuadraticDiscriminantAnalysis(
                reg_param=self.model_config.get('reg_param', 0.0)
            )
        
        elif self.model_type == 'naive_bayes':
            return GaussianNB(
                var_smoothing=self.model_config.get('var_smoothing', 1e-9)
            )
        
        elif self.model_type == 'nn_classifier':
            return self._create_neural_network_classifier()
        
        elif self.model_type == 'voting_classifier':
            return self._create_voting_classifier()
        
        elif self.model_type == 'calibrated_classifier':
            base_classifier = self._get_base_classifier()
            return CalibratedClassifierCV(
                base_estimator=base_classifier,
                method=self.model_config.get('calibration_method', 'sigmoid'),
                cv=self.model_config.get('cv', 5)
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _train_model(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_data: Tuple[pd.DataFrame, pd.Series] = None,
        sample_weight: np.ndarray = None,
        **kwargs
    ) -> None:
        """
        Train the underlying model with the given data.
        
        Args:
            X: Training features
            y: Training target values
            validation_data: Optional tuple of (X_val, y_val) for validation
            sample_weight: Optional array of weights for training samples
            **kwargs: Additional training parameters
        """
        logger.info(f"Training model {self.name} with {len(X)} samples")
        
        # For standard sklearn-like models
        if hasattr(self.model, 'fit') and self.model_type not in ['nn_classifier']:
            if sample_weight is not None:
                self.model.fit(X, y, sample_weight=sample_weight)
            else:
                self.model.fit(X, y)
        
        # For neural network models
        elif self.model_type == 'nn_classifier':
            # Prepare validation data if provided
            validation_split = self.model_config.get('validation_split', 0.2)
            validation_data_keras = None
            
            if validation_data:
                validation_data_keras = (validation_data[0], 
                                         self._to_categorical(validation_data[1]) if self.is_multiclass else validation_data[1])
                validation_split = 0  # Use provided validation data instead of splitting
            
            # Convert y to categorical format for multi-class problems
            y_train = self._to_categorical(y) if self.is_multiclass else y
            
            # Configure callbacks
            callbacks_list = self._get_keras_callbacks()
            
            # Train the model
            self.model.fit(
                X, y_train,
                epochs=self.model_config.get('epochs', 100),
                batch_size=self.model_config.get('batch_size', 32),
                validation_split=validation_split,
                validation_data=validation_data_keras,
                callbacks=callbacks_list,
                verbose=self.model_config.get('verbose', 1),
                sample_weight=sample_weight
            )
        
        # Cross-validation if requested
        if kwargs.get('cross_val', False):
            self._perform_cross_validation(X, y, kwargs.get('cv', 5), sample_weight)
    
    def _calculate_training_metrics(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_data: Tuple[pd.DataFrame, pd.Series] = None
    ) -> None:
        """
        Calculate and store performance metrics after training.
        
        Args:
            X: Training features
            y: Training target values
            validation_data: Optional validation data to use for metrics
        """
        # Use validation data for metrics if provided
        if validation_data:
            X_eval, y_eval = validation_data
        else:
            # Otherwise use training data (less ideal)
            X_eval, y_eval = X, y
        
        # Generate predictions for evaluation
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_eval)
            if self.num_classes == 2:
                y_pred = (y_proba[:, 1] >= 0.5).astype(int)
            else:
                y_pred = np.argmax(y_proba, axis=1)
        else:
            y_pred = self.model.predict(X_eval)
            y_proba = None
        
        # Calculate classification metrics
        self.metrics['accuracy'] = accuracy_score(y_eval, y_pred)
        
        if self.is_multiclass:
            self.metrics['precision'] = precision_score(y_eval, y_pred, average='weighted')
            self.metrics['recall'] = recall_score(y_eval, y_pred, average='weighted')
            self.metrics['f1'] = f1_score(y_eval, y_pred, average='weighted')
        else:
            self.metrics['precision'] = precision_score(y_eval, y_pred)
            self.metrics['recall'] = recall_score(y_eval, y_pred)
            self.metrics['f1'] = f1_score(y_eval, y_pred)
        
        self.metrics['confusion_matrix'] = confusion_matrix(y_eval, y_pred).tolist()
        
        # Calculate AUC if probabilities are available
        if y_proba is not None:
            if self.num_classes == 2:
                self.metrics['auc'] = roc_auc_score(y_eval, y_proba[:, 1])
                self.metrics['log_loss'] = log_loss(y_eval, y_proba[:, 1])
            else:
                # One-vs-Rest AUC for multiclass
                self.metrics['auc'] = roc_auc_score(
                    pd.get_dummies(y_eval), y_proba, 
                    average='weighted', multi_class='ovr'
                )
                self.metrics['log_loss'] = log_loss(y_eval, y_proba)
        
        # Generate classification report
        self.metrics['classification_report'] = classification_report(y_eval, y_pred, output_dict=True)
        
        # Log detailed metrics
        logger.info(f"Model {self.name} metrics: ")
        logger.info(f"  Accuracy: {self.metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {self.metrics['precision']:.4f}")
        logger.info(f"  Recall: {self.metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {self.metrics['f1']:.4f}")
        if 'auc' in self.metrics and self.metrics['auc'] > 0:
            logger.info(f"  AUC-ROC: {self.metrics['auc']:.4f}")
    
    def _handle_class_imbalance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        method: str = 'smote',
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply resampling techniques to handle class imbalance.
        
        Args:
            X: Features
            y: Target labels
            method: Resampling method ('smote', 'adasyn', 'undersample')
            random_state: Random seed for reproducibility
        
        Returns:
            Resampled X and y
        """
        if random_state is None:
            random_state = self.random_state
            
        logger.info(f"Handling class imbalance with method: {method}")
        
        # Count original class distribution
        class_counts = y.value_counts()
        logger.info(f"Original class distribution: {class_counts.to_dict()}")
        
        if method.lower() == 'smote':
            sampler = SMOTE(random_state=random_state)
        elif method.lower() == 'adasyn':
            sampler = ADASYN(random_state=random_state)
        elif method.lower() == 'undersample':
            sampler = RandomUnderSampler(random_state=random_state)
        else:
            logger.warning(f"Unknown resampling method: {method}, using SMOTE instead")
            sampler = SMOTE(random_state=random_state)
        
        # Apply resampling
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        # Log the new class distribution
        new_class_counts = y_resampled.value_counts()
        logger.info(f"Resampled class distribution: {new_class_counts.to_dict()}")
        logger.info(f"Resampled data shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def _perform_cross_validation(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cv: int = 5,
        sample_weight: np.ndarray = None
    ) -> None:
        """
        Perform cross-validation and store results.
        
        Args:
            X: Features
            y: Target values
            cv: Number of folds
            sample_weight: Optional sample weights
        """
        logger.info(f"Performing {cv}-fold cross-validation for model {self.name}")
        
        # Initialize cross-validation
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=kfold, 
            scoring='accuracy',
            fit_params={'sample_weight': sample_weight} if sample_weight is not None else None
        )
        
        # Store and log results
        self.metrics['cross_val_scores'] = cv_scores.tolist()
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    def _create_neural_network_classifier(self) -> Any:
        """
        Create a neural network classifier using TensorFlow/Keras.
        
        Returns:
            Configured neural network model
        """
        logger.info("Creating neural network classifier")
        
        # Get model configuration parameters
        input_dim = len(self.feature_columns)
        hidden_layers = self.model_config.get('hidden_layers', [64, 32])
        dropout_rate = self.model_config.get('dropout_rate', 0.2)
        activation = self.model_config.get('activation', 'relu')
        learning_rate = self.model_config.get('learning_rate', 0.001)
        l2_reg = self.model_config.get('l2_reg', 0.001)
        
        # Define the model architecture
        model = Sequential()
        
        # Input layer
        model.add(layers.Dense(
            hidden_layers[0], 
            input_dim=input_dim,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg)
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(
                units, 
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg)
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        if self.is_multiclass:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
        else:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        
        # Compile the model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        # Display model summary
        model.summary(print_fn=logger.info)
        
        return model
    
    def _create_voting_classifier(self) -> VotingClassifier:
        """
        Create a voting ensemble classifier.
        
        Returns:
            Configured voting classifier
        """
        estimators = []
        
        # Add configured estimators
        for estimator_config in self.model_config.get('estimators', []):
            name = estimator_config['name']
            estimator_type = estimator_config['type']
            estimator_params = estimator_config.get('params', {})
            
            # Create the base classifier
            if estimator_type == 'random_forest':
                clf = RandomForestClassifier(random_state=self.random_state, **estimator_params)
            elif estimator_type == 'xgboost':
                clf = XGBClassifier(random_state=self.random_state, **estimator_params)
            elif estimator_type == 'lightgbm':
                clf = LGBMClassifier(random_state=self.random_state, **estimator_params)
            elif estimator_type == 'catboost':
                clf = CatBoostClassifier(random_seed=self.random_state, **estimator_params)
            elif estimator_type == 'logistic_regression':
                clf = LogisticRegression(random_state=self.random_state, **estimator_params)
            elif estimator_type == 'svm':
                clf = SVC(probability=True, random_state=self.random_state, **estimator_params)
            else:
                logger.warning(f"Unsupported estimator type: {estimator_type}, skipping")
                continue
                
            estimators.append((name, clf))
        
        # Create and return the voting classifier
        voting_type = self.model_config.get('voting', 'soft')
        weights = self.model_config.get('weights', None)
        
        return VotingClassifier(
            estimators=estimators,
            voting=voting_type,
            weights=weights,
            n_jobs=self.model_config.get('n_jobs', -1)
        )
    
    def _get_base_classifier(self) -> Any:
        """
        Get a base classifier for calibration or other meta-models.
        
        Returns:
            Base classifier instance
        """
        base_config = self.model_config.get('base_classifier', {})
        base_type = base_config.get('type', 'random_forest')
        
        # Create a temporary config with just the base classifier settings
        temp_config = {'model_type': base_type, 'model_config': base_config.get('params', {})}
        
        # Save current model type
        current_type = self.model_type
        
        # Temporarily set model type to base type
        self.model_type = base_type
        
        # Create the base classifier
        base_clf = self._create_model()
        
        # Restore original model type
        self.model_type = current_type
        
        return base_clf
    
    def _get_keras_callbacks(self) -> List[Any]:
        """
        Configure callbacks for Keras neural network training.
        
        Returns:
            List of Keras callbacks
        """
        callbacks_list = []
        
        # Early stopping
        if self.model_config.get('early_stopping', True):
            callbacks_list.append(callbacks.EarlyStopping(
                monitor=self.model_config.get('monitor', 'val_loss'),
                patience=self.model_config.get('patience', 10),
                restore_best_weights=True
            ))
        
        # Learning rate reduction
        if self.model_config.get('reduce_lr', True):
            callbacks_list.append(callbacks.ReduceLROnPlateau(
                monitor=self.model_config.get('lr_monitor', 'val_loss'),
                factor=self.model_config.get('lr_factor', 0.5),
                patience=self.model_config.get('lr_patience', 5),
                min_lr=self.model_config.get('min_lr', 1e-6)
            ))
        
        # Model checkpoint
        if self.model_path and self.model_config.get('checkpoint', True):
            checkpoint_path = os.path.join(
                os.path.dirname(self.model_path),
                f"{os.path.basename(self.model_path).split('.')[0]}_best.h5"
            )
            callbacks_list.append(callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor=self.model_config.get('checkpoint_monitor', 'val_loss'),
                save_best_only=True,
                mode=self.model_config.get('checkpoint_mode', 'min')
            ))
        
        # TensorBoard logging
        if self.model_config.get('tensorboard', False):
            log_dir = self.model_config.get('tensorboard_dir', './logs')
            callbacks_list.append(callbacks.TensorBoard(
                log_dir=os.path.join(log_dir, f"{self.name}_{int(time.time())}"),
                histogram_freq=1
            ))
        
        return callbacks_list
    
    def _to_categorical(self, y: pd.Series) -> np.ndarray:
        """
        Convert integer class labels to one-hot encoded matrix.
        
        Args:
            y: Class labels
        
        Returns:
            One-hot encoded matrix
        """
        if self.is_multiclass:
            return tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        return y
    
    def save(self) -> None:
        """
        Save the model and its metadata to disk.
        
        Raises:
            IOError: If saving fails
        """
        if not self.model_path:
            logger.warning(f"No model path specified for model {self.name}, skipping save")
            return
            
        try:
            logger.info(f"Saving model {self.name} to {self.model_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # For Keras neural network models
            if self.model_type == 'nn_classifier':
                model_file = f"{self.model_path}.h5"
                scaler_file = f"{self.model_path}_scaler.pkl"
                metadata_file = f"{self.model_path}_metadata.pkl"
                
                # Save the model
                self.model.save(model_file)
                
                # Save the scaler if used
                if self.scaling and self.scaler:
                    with open(scaler_file, 'wb') as f:
                        pickle.dump(self.scaler, f)
                
                # Save metadata including label encoder
                metadata = {
                    'name': self.name,
                    'model_type': self.model_type,
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'scaling': self.scaling,
                    'is_multiclass': self.is_multiclass,
                    'num_classes': self.num_classes,
                    'classes_': self.classes_.tolist() if self.classes_ is not None else None,
                    'metrics': self.metrics,
                    'label_encoder': self.label_encoder if hasattr(self, 'label_encoder') else None,
                    'model_config': self.model_config,
                    'creation_time': datetime.now().isoformat()
                }
                
                with open(metadata_file, 'wb') as f:
                    pickle.dump(metadata, f)
                    
            # For other scikit-learn based models
            else:
                # Save the entire model with metadata
                model_data = {
                    'model': self.model,
                    'name': self.name,
                    'model_type': self.model_type,
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'scaling': self.scaling,
                    'scaler': self.scaler,
                    'is_multiclass': self.is_multiclass,
                    'num_classes': self.num_classes,
                    'classes_': self.classes_,
                    'metrics': self.metrics,
                    'label_encoder': self.label_encoder if hasattr(self, 'label_encoder') else None,
                    'model_config': self.model_config,
                    'creation_time': datetime.now().isoformat()
                }
                
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                    
            logger.info(f"Successfully saved model {self.name}")
                
        except Exception as e:
            logger.error(f"Error saving model {self.name}: {str(e)}")
            raise IOError(f"Failed to save model {self.name}: {str(e)}") from e
    
    @classmethod
    def load(cls, model_path: str, use_gpu: bool = True) -> 'ClassificationModel':
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            use_gpu: Whether to use GPU for inference
            
        Returns:
            Loaded ClassificationModel instance
            
        Raises:
            IOError: If loading fails
        """
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Check if this is a Keras model (has .h5 extension)
            if model_path.endswith('.h5') or os.path.exists(f"{model_path}.h5"):
                # Load Keras model
                keras_path = model_path if model_path.endswith('.h5') else f"{model_path}.h5"
                metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.pkl" if model_path.endswith('.h5') else f"{model_path}_metadata.pkl"
                scaler_path = f"{os.path.splitext(model_path)[0]}_scaler.pkl" if model_path.endswith('.h5') else f"{model_path}_scaler.pkl"
                
                # Load the model
                model = load_model(keras_path)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Load scaler if exists
                scaler = None
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                
                # Create instance with loaded metadata
                instance = cls(
                    name=metadata['name'],
                    model_type=metadata['model_type'],
                    model_config=metadata['model_config'],
                    feature_columns=metadata['feature_columns'],
                    target_column=metadata['target_column'],
                    scaling=metadata['scaling'],
                    model_path=model_path,
                    use_gpu=use_gpu
                )
                
                # Set loaded attributes
                instance.model = model
                instance.scaler = scaler
                instance.is_multiclass = metadata['is_multiclass']
                instance.num_classes = metadata['num_classes']
                instance.metrics = metadata['metrics']
                
                # Set label encoder
                if 'label_encoder' in metadata and metadata['label_encoder'] is not None:
                    instance.label_encoder = metadata['label_encoder']
                    
                # Set classes
                if 'classes_' in metadata and metadata['classes_'] is not None:
                    instance.classes_ = np.array(metadata['classes_'])
                
            else:
                # Load scikit-learn type model
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Create instance with loaded metadata
                instance = cls(
                    name=model_data['name'],
                    model_type=model_data['model_type'],
                    model_config=model_data['model_config'],
                    feature_columns=model_data['feature_columns'],
                    target_column=model_data['target_column'],
                    scaling=model_data['scaling'],
                    model_path=model_path,
                    use_gpu=use_gpu
                )
                
                # Set loaded attributes
                instance.model = model_data['model']
                instance.scaler = model_data['scaler']
                instance.is_multiclass = model_data['is_multiclass']
                instance.num_classes = model_data['num_classes']
                instance.classes_ = model_data['classes_']
                instance.metrics = model_data['metrics']
                
                # Set label encoder if exists
                if 'label_encoder' in model_data and model_data['label_encoder'] is not None:
                    instance.label_encoder = model_data['label_encoder']
            
            logger.info(f"Successfully loaded model {instance.name}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise IOError(f"Failed to load model from {model_path}: {str(e)}") from e
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance if available for the model.
        
        Returns:
            Dictionary mapping feature names to importance values
        """
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_columns, self.model.feature_importances_))
        elif 'feature_importance' in self.metrics:
            return self.metrics['feature_importance']
        else:
            logger.warning(f"Feature importance not available for model {self.name}")
            return {}
    
    def calibrate(self, X: pd.DataFrame, y: pd.Series, method: str = 'sigmoid', cv: int = 5) -> None:
        """
        Calibrate the model's probability estimates.
        
        Args:
            X: Calibration features
            y: True labels
            method: Calibration method ('sigmoid' or 'isotonic')
            cv: Number of cross-validation folds
        """
        try:
            logger.info(f"Calibrating model {self.name} using method: {method}")
            
            # Scale features if needed
            X_calib = X.copy()
            if self.scaling and self.scaler:
                X_calib = pd.DataFrame(
                    self.scaler.transform(X_calib),
                    columns=X_calib.columns
                )
            
            # Create calibrated classifier
            calibrated_model = CalibratedClassifierCV(
                base_estimator=self.model,
                method=method,
                cv=cv
            )
            
            # Fit the calibrated model
            calibrated_model.fit(X_calib, y)
            
            # Replace the original model with calibrated version
            self.model = calibrated_model
            
            logger.info(f"Successfully calibrated model {self.name}")
            
        except Exception as e:
            logger.error(f"Error calibrating model {self.name}: {str(e)}")
            raise ModelTrainingError(f"Failed to calibrate model {self.name}: {str(e)}") from e
    
    def optimize_hyperparameters(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        cv: int = 5,
        scoring: str = 'accuracy',
        n_jobs: int = -1,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using grid search.
        
        Args:
            X: Training features
            y: Target values
            param_grid: Parameter grid to search
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            
        Returns:
            Dictionary with best parameters and results
        """
        try:
            logger.info(f"Starting hyperparameter optimization for model {self.name}")
            
            # Create base model for optimization
            base_model = self._create_model()
            
            # Create grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                return_train_score=True
            )
            
            # Prepare data
            X_train = X.copy()
            if self.scaling:
                X_train = pd.DataFrame(
                    self.scaler.fit_transform(X_train),
                    columns=X_train.columns
                )
            
            # Fit grid search
            grid_search.fit(X_train, y)
            
            # Get results
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best {scoring} score: {best_score:.4f}")
            
            # Update model with best parameters
            self.model_config.update(best_params)
            self.model = grid_search.best_estimator_
            
            # Return optimization results
            return {
                'best_params': best_params,
                'best_score': best_score,
                'cv_results': grid_search.cv_results_
            }
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters for model {self.name}: {str(e)}")
            raise ModelTrainingError(f"Failed to optimize hyperparameters for model {self.name}: {str(e)}") from e

# Factory function for creating classification models
def create_classification_model(
    name: str,
    model_type: str,
    model_config: Dict[str, Any],
    **kwargs
) -> ClassificationModel:
    """
    Factory function to create a classification model.
    
    Args:
        name: Model name
        model_type: Type of model to create
        model_config: Model configuration parameters
        **kwargs: Additional parameters for the model
        
    Returns:
        Instantiated ClassificationModel
        
    Raises:
        ValueError: If invalid model type is specified
    """
    return ClassificationModel(
        name=name,
        model_type=model_type,
        model_config=model_config,
        **kwargs
    )
