#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Feature Importance Module

This module provides advanced feature importance analysis tools to understand
which features contribute most to model predictions, allowing for model optimization
and deeper market understanding.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
import shap
try:
    import eli5  # type: ignore
    ELI5_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    eli5 = None  # type: ignore
    ELI5_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "eli5 not available; some feature importance methods disabled"
    )
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
try:
    from boruta import BorutaPy  # type: ignore
    BORUTA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    BorutaPy = None  # type: ignore
    BORUTA_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Boruta package not available; Boruta-based feature selection disabled"
    )
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from common.logger import get_logger
from common.constants import FEATURE_IMPORTANCE_CONFIG
from common.async_utils import run_in_threadpool
from common.exceptions import ModelNotSupportedError, InvalidFeatureFormatError

logger = get_logger(__name__)

class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analyzer with multiple methods for robust analysis.
    Supports various models and importance methods.
    """
    
    def __init__(self, model_type: str = "classification", random_state: int = 42):
        """
        Initialize the feature importance analyzer.
        
        Args:
            model_type: Type of model - 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.supported_methods = {
            "permutation": self._permutation_importance,
            "shap": self._shap_importance,
            "feature_importance": self._model_feature_importance,
            "mutual_info": self._mutual_information,
            "boruta": self._boruta_importance,
            "eli5": self._eli5_importance
        }
        logger.info(f"Initialized FeatureImportanceAnalyzer with model_type={model_type}")
        
    async def analyze(self, 
                     model: Any, 
                     X: Union[pd.DataFrame, np.ndarray], 
                     y: Union[pd.Series, np.ndarray],
                     method: str = "permutation",
                     n_repeats: int = 10,
                     n_top_features: int = 20,
                     feature_names: Optional[List[str]] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Analyze feature importance using the specified method.
        
        Args:
            model: Trained model object
            X: Feature data
            y: Target data
            method: Feature importance method to use
            n_repeats: Number of times to repeat permutation importance
            n_top_features: Number of top features to return
            feature_names: Optional list of feature names
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dictionary with importance analysis results
        """
        if method not in self.supported_methods:
            raise ValueError(f"Method {method} not supported. Choose from: {list(self.supported_methods.keys())}")
            
        # Validate input data
        X, feature_names = self._validate_and_prepare_data(X, feature_names)
        
        # Run importance analysis in threadpool to avoid blocking event loop
        importance_result = await run_in_threadpool(
            self.supported_methods[method],
            model=model,
            X=X,
            y=y,
            n_repeats=n_repeats,
            feature_names=feature_names,
            **kwargs
        )
        
        # Sort and limit to top N features
        sorted_importance = self._sort_and_limit_features(importance_result, n_top_features)
        
        logger.info(f"Completed feature importance analysis using {method} method")
        return {
            "method": method,
            "importance": sorted_importance,
            "model_type": self.model_type,
            "n_features": len(feature_names),
            "n_top_features": min(n_top_features, len(feature_names)),
            "raw_results": importance_result
        }
    
    def _validate_and_prepare_data(self, 
                                  X: Union[pd.DataFrame, np.ndarray],
                                  feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Validate and prepare input data for feature importance analysis.
        
        Args:
            X: Feature data
            feature_names: Optional list of feature names
            
        Returns:
            Tuple of (X as numpy array, feature names list)
        """
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            if feature_names is None:
                # Generate default feature names if not provided
                feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
                
        # Verify dimensions
        if len(feature_names) != X_array.shape[1]:
            raise InvalidFeatureFormatError(
                f"Feature names count ({len(feature_names)}) doesn't match X dimensions ({X_array.shape[1]})"
            )
            
        return X_array, feature_names
    
    def _sort_and_limit_features(self,
                                result: Dict[str, Any],
                                n_top_features: int) -> Dict[str, float]:
        """
        Sort features by importance and limit to top N.
        
        Args:
            result: Dictionary with importance results
            n_top_features: Number of top features to return
            
        Returns:
            Dictionary of {feature_name: importance_value} for top features
        """
        importance_dict = result.get("importance_dict", {})
        if not importance_dict:
            logger.warning("No importance dictionary found in results")
            return {}
            
        # Sort by importance (descending)
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top N features
        top_features = dict(sorted_features[:min(n_top_features, len(sorted_features))])
        
        return top_features
    
    def _permutation_importance(self,
                               model: Any,
                               X: np.ndarray,
                               y: np.ndarray,
                               n_repeats: int = 10,
                               feature_names: List[str] = None,
                               **kwargs) -> Dict[str, Any]:
        """
        Calculate permutation importance.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data
            n_repeats: Number of times to repeat permutation importance
            feature_names: List of feature names
            
        Returns:
            Dictionary with permutation importance results
        """
        try:
            # Run permutation importance
            results = permutation_importance(
                model, X, y, 
                n_repeats=n_repeats, 
                random_state=self.random_state,
                **kwargs
            )
            
            # Create importance dictionary
            importance_dict = {feature_names[i]: results.importances_mean[i] 
                              for i in range(len(feature_names))}
            
            return {
                "importance_dict": importance_dict,
                "importances_mean": results.importances_mean,
                "importances_std": results.importances_std,
                "feature_names": feature_names
            }
            
        except Exception as e:
            logger.error(f"Error in permutation importance: {str(e)}")
            raise
    
    def _shap_importance(self,
                        model: Any,
                        X: np.ndarray,
                        y: np.ndarray = None,
                        feature_names: List[str] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Calculate SHAP importance values.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data (unused, kept for API consistency)
            feature_names: List of feature names
            
        Returns:
            Dictionary with SHAP importance results
        """
        try:
            # Sample data if too large
            sample_size = min(500, X.shape[0])
            if X.shape[0] > sample_size:
                indices = np.random.choice(X.shape[0], sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
                
            # Create explainer based on model type
            if isinstance(model, (XGBRegressor, XGBClassifier, LGBMRegressor, LGBMClassifier)):
                explainer = shap.TreeExplainer(model)
            else:
                # Default to KernelExplainer for other model types
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 100))
                
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class output
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # For multi-class, take the mean absolute SHAP value across classes
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            
            # Calculate mean absolute SHAP value for each feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Create importance dictionary
            importance_dict = {feature_names[i]: mean_abs_shap[i] 
                              for i in range(len(feature_names))}
            
            return {
                "importance_dict": importance_dict,
                "mean_abs_shap": mean_abs_shap,
                "feature_names": feature_names,
                "sample_size": sample_size
            }
            
        except Exception as e:
            logger.error(f"Error in SHAP importance: {str(e)}")
            raise
    
    def _model_feature_importance(self,
                                 model: Any,
                                 X: np.ndarray,
                                 y: np.ndarray = None,
                                 feature_names: List[str] = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        Extract built-in feature importance from model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            X: Feature data (unused, kept for API consistency)
            y: Target data (unused, kept for API consistency)
            feature_names: List of feature names
            
        Returns:
            Dictionary with model's feature importance
        """
        try:
            # Check if model has feature_importances_ attribute
            if not hasattr(model, 'feature_importances_'):
                raise ModelNotSupportedError(
                    f"Model of type {type(model).__name__} does not support built-in feature importance"
                )
                
            # Get importance values
            importances = model.feature_importances_
            
            # Create importance dictionary
            importance_dict = {feature_names[i]: importances[i] 
                              for i in range(len(feature_names))}
            
            return {
                "importance_dict": importance_dict,
                "importances": importances,
                "feature_names": feature_names
            }
            
        except Exception as e:
            logger.error(f"Error in model feature importance: {str(e)}")
            raise
    
    def _mutual_information(self,
                           model: Any,
                           X: np.ndarray,
                           y: np.ndarray,
                           feature_names: List[str] = None,
                           **kwargs) -> Dict[str, Any]:
        """
        Calculate mutual information between features and target.
        
        Args:
            model: Trained model (unused, kept for API consistency)
            X: Feature data
            y: Target data
            feature_names: List of feature names
            
        Returns:
            Dictionary with mutual information results
        """
        try:
            # Select appropriate mutual info function based on model type
            if self.model_type == "classification":
                mi_func = mutual_info_classif
            else:
                mi_func = mutual_info_regression
                
            # Calculate mutual information
            mi_values = mi_func(X, y, random_state=self.random_state)
            
            # Create importance dictionary
            importance_dict = {feature_names[i]: mi_values[i] 
                              for i in range(len(feature_names))}
            
            return {
                "importance_dict": importance_dict,
                "mi_values": mi_values,
                "feature_names": feature_names
            }
            
        except Exception as e:
            logger.error(f"Error in mutual information: {str(e)}")
            raise
    
    def _boruta_importance(self,
                          model: Any,
                          X: np.ndarray,
                          y: np.ndarray,
                          feature_names: List[str] = None,
                          max_iter: int = 100,
                          **kwargs) -> Dict[str, Any]:
        """
        Perform Boruta feature selection and importance ranking.
        
        Args:
            model: Trained model or model class
            X: Feature data
            y: Target data
            feature_names: List of feature names
            max_iter: Maximum iterations for Boruta
            
        Returns:
            Dictionary with Boruta feature selection results
        """
        try:
            # Create appropriate base estimator if model is a class
            if isinstance(model, type):
                if self.model_type == "classification":
                    estimator = RandomForestClassifier(n_jobs=-1, random_state=self.random_state)
                else:
                    estimator = RandomForestRegressor(n_jobs=-1, random_state=self.random_state)
            else:
                estimator = model
                
            # Initialize Boruta
            boruta = BorutaPy(
                estimator=estimator,
                n_estimators='auto',
                max_iter=max_iter,
                random_state=self.random_state
            )
            
            # Fit Boruta
            boruta.fit(X, y)
            
            # Get importance ranks and confirmed flags
            ranks = boruta.ranking_
            confirmed = boruta.support_
            
            # Create dictionaries for feature status and ranking
            confirmed_features = [feature_names[i] for i in range(len(feature_names)) if confirmed[i]]
            tentative_features = [feature_names[i] for i in range(len(feature_names)) 
                                 if not confirmed[i] and boruta.support_weak_[i]]
            rejected_features = [feature_names[i] for i in range(len(feature_names)) 
                               if not confirmed[i] and not boruta.support_weak_[i]]
            
            # Invert ranking for importance (lower rank = more important)
            max_rank = np.max(ranks)
            importance_values = max_rank - ranks + 1
            
            # Create importance dictionary based on inverted ranks
            importance_dict = {feature_names[i]: float(importance_values[i]) / max_rank
                              for i in range(len(feature_names))}
            
            return {
                "importance_dict": importance_dict,
                "confirmed_features": confirmed_features,
                "tentative_features": tentative_features,
                "rejected_features": rejected_features,
                "rankings": {feature_names[i]: int(ranks[i]) for i in range(len(feature_names))},
                "feature_names": feature_names
            }
            
        except Exception as e:
            logger.error(f"Error in Boruta importance: {str(e)}")
            raise
    
    def _eli5_importance(self,
                        model: Any,
                        X: np.ndarray,
                        y: np.ndarray = None,
                        feature_names: List[str] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Calculate feature importance using ELI5.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target data (unused, kept for API consistency)
            feature_names: List of feature names
            
        Returns:
            Dictionary with ELI5 feature importance results
        """
        try:
            # Get explanation from ELI5
            explanation = eli5.explain_weights(model, feature_names=feature_names)
            
            # Extract weights from explanation
            weights = {}
            for feature in explanation.feature_weights.pos:
                weights[feature.feature] = feature.weight
            for feature in explanation.feature_weights.neg:
                weights[feature.feature] = feature.weight
                
            # Normalize weights to [0, 1] range if they exist
            if weights:
                max_abs_weight = max(abs(w) for w in weights.values())
                if max_abs_weight > 0:
                    weights = {f: abs(w) / max_abs_weight for f, w in weights.items()}
            
            return {
                "importance_dict": weights,
                "method": "eli5",
                "feature_names": feature_names
            }
            
        except Exception as e:
            logger.error(f"Error in ELI5 importance: {str(e)}")
            raise

    async def generate_importance_plot(self,
                                      importance_result: Dict[str, Any],
                                      output_path: str = None,
                                      return_figure: bool = False,
                                      plot_type: str = "horizontal_bar",
                                      title: str = "Feature Importance Analysis",
                                      **kwargs) -> Optional[plt.Figure]:
        """
        Generate feature importance visualization.
        
        Args:
            importance_result: Result from feature importance analysis
            output_path: Path to save plot (if None, plot is not saved)
            return_figure: Whether to return the matplotlib figure
            plot_type: Type of plot to generate
            title: Plot title
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib figure if return_figure is True
        """
        # Run in threadpool to avoid blocking event loop
        return await run_in_threadpool(
            self._generate_plot,
            importance_result=importance_result,
            output_path=output_path,
            return_figure=return_figure,
            plot_type=plot_type,
            title=title,
            **kwargs
        )
        
    def _generate_plot(self,
                      importance_result: Dict[str, Any],
                      output_path: str = None,
                      return_figure: bool = False,
                      plot_type: str = "horizontal_bar",
                      title: str = "Feature Importance Analysis",
                      figsize: Tuple[int, int] = (10, 8),
                      color: str = "#1f77b4",
                      **kwargs) -> Optional[plt.Figure]:
        """
        Generate feature importance visualization (non-async implementation).
        
        Args:
            importance_result: Result from feature importance analysis
            output_path: Path to save plot
            return_figure: Whether to return the matplotlib figure
            plot_type: Type of plot to generate
            title: Plot title
            figsize: Figure size
            color: Bar color
            **kwargs: Additional plotting parameters
            
        Returns:
            Matplotlib figure if return_figure is True
        """
        try:
            # Extract importance dictionary
            importance_dict = importance_result.get("importance", {})
            if not importance_dict:
                logger.warning("No importance data available for plotting")
                return None
                
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Sort data for plotting
            features = list(importance_dict.keys())
            values = list(importance_dict.values())
            sorted_idx = np.argsort(values)
            
            method = importance_result.get("method", "unknown")
            
            # Generate appropriate plot
            if plot_type == "horizontal_bar":
                y_pos = np.arange(len(features))
                ax.barh(y_pos, values, align='center', color=color, alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.invert_yaxis()  # Highest values at the top
                ax.set_xlabel('Importance')
                
            elif plot_type == "vertical_bar":
                x_pos = np.arange(len(features))
                ax.bar(x_pos, values, align='center', color=color, alpha=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(features, rotation=45, ha='right')
                ax.set_ylabel('Importance')
                
            elif plot_type == "pie":
                # Normalize values for pie chart
                norm_values = np.array(values) / sum(values)
                ax.pie(norm_values, labels=features, autopct='%1.1f%%', 
                      shadow=True, startangle=90)
                ax.axis('equal')  # Equal aspect ratio for circular pie
                
            else:
                logger.warning(f"Unknown plot type: {plot_type}, using horizontal bar")
                y_pos = np.arange(len(features))
                ax.barh(y_pos, values, align='center', color=color, alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.invert_yaxis()
            
            # Add title and subtitle
            ax.set_title(f"{title}\nMethod: {method.capitalize()}", fontsize=14)
            
            # Add timestamp
            plt.figtext(0.95, 0.01, f"Generated by QuantumSpectre Elite", 
                       ha='right', fontsize=8, style='italic')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if output path provided
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved feature importance plot to {output_path}")
                
            if return_figure:
                return fig
            else:
                plt.close(fig)
                return None

        except Exception as e:
            logger.error(f"Error generating feature importance plot: {str(e)}")
            if return_figure:
                return None


def calculate_feature_importance(model: Any, feature_names: List[str], model_type: str, method: str = "permutation") -> Dict[str, Any]:
    """Convenience wrapper to compute feature importance synchronously."""
    analyzer = FeatureImportanceAnalyzer(model_type)
    dummy_X = np.zeros((1, len(feature_names)))
    dummy_y = np.zeros(1)
    try:
        result = analyzer._model_feature_importance(model, dummy_X, dummy_y, feature_names)
        return result
    except Exception:
        return {}
