#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Machine Learning Models Module Initialization

This module provides the machine learning infrastructure for the QuantumSpectre Elite
Trading System, enabling advanced predictive capabilities through a diverse
set of specialized ML models optimized for different market patterns and conditions.
"""

import os
import logging
from typing import Dict, List, Any, Set, Optional, Union, Tuple
from .hyperopt import HyperOptService

# Set up module level logger
logger = logging.getLogger(__name__)

# Define module version
__version__ = '1.0.0'

# Import core components to make them available at package level
from ml_models.model_manager import ModelManager
try:
    from ml_models.prediction import PredictionEngine
except Exception as exc:  # pragma: no cover - optional dependency
    PredictionEngine = None  # type: ignore
    logger.warning("PredictionEngine unavailable: %s", exc)
from ml_models.feature_importance import FeatureImportanceAnalyzer

try:
    from ml_models.training import ModelTrainer
except Exception as exc:  # pragma: no cover - optional dependency
    ModelTrainer = None  # type: ignore
    logger.warning("ModelTrainer unavailable: %s", exc)

RLTradingAgent = None  # type: ignore
if os.environ.get("QS_DISABLE_RL_TRAINER") != "1":
    try:
        from ml_models.rl.trainer import RLTradingAgent  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        RLTradingAgent = None  # type: ignore
        logger.warning("RLTradingAgent unavailable: %s", exc)
from ml_models.rl import RLAgent, DQNAgent, PPOAgent

# Model type constants
MODEL_TYPE_REGRESSION = 'regression'
MODEL_TYPE_CLASSIFICATION = 'classification'
MODEL_TYPE_TIME_SERIES = 'time_series'
MODEL_TYPE_DEEP_LEARNING = 'deep_learning'
MODEL_TYPE_ENSEMBLE = 'ensemble'

# Default feature sets
DEFAULT_TECHNICAL_FEATURES = [
    'rsi_14', 'macd', 'macd_signal', 'macd_hist', 
    'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
    'atr_14', 'adx_14', 'cci_20', 'stoch_k', 'stoch_d',
    'williams_r', 'obv', 'ichimoku_a', 'ichimoku_b',
    'keltner_upper', 'keltner_middle', 'keltner_lower',
    'parabolic_sar', 'momentum_10', 'rate_of_change'
]

DEFAULT_VOLUME_FEATURES = [
    'volume', 'volume_sma_20', 'volume_ema_10', 
    'volume_oscillator', 'on_balance_volume',
    'chaikin_oscillator', 'accumulation_distribution',
    'money_flow_index', 'volume_price_trend',
    'normalized_volume', 'relative_volume'
]

DEFAULT_MARKET_STRUCTURE_FEATURES = [
    'support_level_1', 'support_level_2', 'resistance_level_1', 
    'resistance_level_2', 'swing_high', 'swing_low',
    'market_regime', 'trend_strength', 'market_phase',
    'volatility_regime', 'liquidity_index'
]

DEFAULT_PATTERN_FEATURES = [
    'chart_pattern_score', 'harmonic_pattern_score',
    'candlestick_pattern_score', 'support_resistance_score',
    'fibonacci_level', 'elliott_wave_position',
    'wyckoff_phase', 'supply_demand_zone'
]

DEFAULT_SENTIMENT_FEATURES = [
    'news_sentiment_score', 'social_sentiment_score',
    'market_sentiment_index', 'fear_greed_index',
    'institutional_positioning', 'retail_flow'
]

# Default hyperparameter search spaces
DEFAULT_HYPERPARAMETER_SPACES = {
    MODEL_TYPE_REGRESSION: {
        'xgboost': {
            'n_estimators': [50, 100, 200, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 8, 10],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1, 10, 100],
            'reg_lambda': [0, 0.1, 1, 10, 100]
        },
        'lightgbm': {
            'num_leaves': [31, 63, 127, 255],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 200, 500, 1000],
            'max_depth': [-1, 5, 10, 15, 20],
            'min_child_samples': [5, 10, 20, 50, 100],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1, 10],
            'reg_lambda': [0, 0.1, 1, 10]
        }
    },
    MODEL_TYPE_CLASSIFICATION: {
        'xgboost': {
            'n_estimators': [50, 100, 200, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 8, 10],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'scale_pos_weight': [1, 2.5, 5, 7.5, 10],
            'max_delta_step': [0, 1, 5, 10]
        },
        'catboost': {
            'iterations': [50, 100, 200, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9],
            'border_count': [32, 64, 128, 255],
            'bagging_temperature': [0, 1, 10],
            'random_strength': [0, 1, 10]
        }
    },
    MODEL_TYPE_TIME_SERIES: {
        'prophet': {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_range': [0.8, 0.9, 0.95]
        },
        'arima': {
            'p': [0, 1, 2, 3, 4, 5],
            'd': [0, 1, 2],
            'q': [0, 1, 2, 3, 4, 5]
        }
    },
    MODEL_TYPE_DEEP_LEARNING: {
        'lstm': {
            'lstm_units': [32, 64, 128, 256],
            'lstm_layers': [1, 2, 3],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            'recurrent_dropout': [0.0, 0.1, 0.2, 0.3],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64, 128, 256],
            'optimizer': ['adam', 'rmsprop', 'sgd']
        },
        'gru': {
            'gru_units': [32, 64, 128, 256],
            'gru_layers': [1, 2, 3],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            'recurrent_dropout': [0.0, 0.1, 0.2, 0.3],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64, 128, 256],
            'optimizer': ['adam', 'rmsprop', 'sgd']
        }
    }
}

# Model performance metrics to track
REGRESSION_METRICS = [
    'mean_squared_error', 'mean_absolute_error', 'r2_score',
    'mean_absolute_percentage_error', 'median_absolute_error'
]

CLASSIFICATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score', 
    'roc_auc', 'precision_recall_auc', 'log_loss'
]

TIME_SERIES_METRICS = [
    'mean_squared_error', 'mean_absolute_error', 
    'mean_absolute_percentage_error', 'symmetric_mean_absolute_percentage_error',
    'mean_absolute_scaled_error'
]

# Global registry for models
model_registry = {}

# Initialize model manager instance for global use
model_manager = None

def init_ml_infrastructure(config=None):
    """
    Initialize the ML infrastructure including model manager, trainers, and predictors.
    
    Args:
        config (dict, optional): ML configuration dictionary
        
    Returns:
        ModelManager: The initialized model manager instance
    """
    global model_manager
    
    logger.info("Initializing ML infrastructure")
    
    if config is None:
        # Load default configuration
        config = {
            'models_dir': os.path.join(os.path.dirname(__file__), 'saved_models'),
            'enable_gpu': True,
            'gpu_memory_limit': 0.8,
            'use_mixed_precision': True,
            'default_model_type': MODEL_TYPE_ENSEMBLE,
            'auto_optimization': True,
            'feature_selection': True,
            'hyperparameter_tuning': True
        }
    
    # Create model directory if it doesn't exist
    os.makedirs(config['models_dir'], exist_ok=True)
    
    # Initialize model manager
    model_manager = ModelManager(config)
    logger.info(f"ML infrastructure initialized with config: {config}")
    
    return model_manager

def register_model(model_name: str, model_type: str, model_config: Dict[str, Any]):
    """
    Register a model configuration in the global registry
    
    Args:
        model_name (str): Unique name for the model
        model_type (str): Type of model (regression, classification, etc.)
        model_config (dict): Configuration for the model
        
    Returns:
        bool: True if registration was successful
    """
    global model_registry
    
    if model_name in model_registry:
        logger.warning(f"Model '{model_name}' already exists in registry. Overwriting.")
    
    model_registry[model_name] = {
        'type': model_type,
        'config': model_config,
        'registered_at': __import__('datetime').datetime.now().isoformat()
    }
    
    logger.info(f"Registered model '{model_name}' of type '{model_type}'")
    return True

def get_registered_models() -> Dict[str, Dict[str, Any]]:
    """
    Get all registered models
    
    Returns:
        dict: Dictionary of registered models
    """
    return model_registry

def get_default_feature_set(asset_type: str = None, timeframe: str = None) -> List[str]:
    """
    Get default feature set based on asset type and timeframe
    
    Args:
        asset_type (str, optional): Asset type (crypto, forex, etc.)
        timeframe (str, optional): Timeframe (1m, 5m, 15m, etc.)
        
    Returns:
        list: List of default features to use
    """
    # Combine default features from different categories
    features = []
    features.extend(DEFAULT_TECHNICAL_FEATURES)
    features.extend(DEFAULT_VOLUME_FEATURES)
    features.extend(DEFAULT_MARKET_STRUCTURE_FEATURES)
    features.extend(DEFAULT_PATTERN_FEATURES)
    features.extend(DEFAULT_SENTIMENT_FEATURES)
    
    # Adjust features based on asset type if specified
    if asset_type:
        if asset_type.lower() == 'crypto':
            # Add crypto-specific features
            features.extend([
                'network_hash_rate', 'active_addresses',
                'exchange_inflow', 'exchange_outflow',
                'realized_price', 'mvrv_ratio'
            ])
        elif asset_type.lower() == 'forex':
            # Add forex-specific features
            features.extend([
                'interest_rate_differential', 'economic_surprise_index',
                'central_bank_sentiment', 'cot_positioning'
            ])
    
    # Adjust features based on timeframe if specified
    if timeframe:
        if timeframe.lower() in ['1m', '5m']:
            # Remove slower indicators for very short timeframes
            features = [f for f in features if not any(
                x in f for x in ['ichimoku', 'elliott_wave', 'wyckoff']
            )]
        elif timeframe.lower() in ['1d', '1w']:
            # Add longer-term indicators for higher timeframes
            features.extend([
                'monthly_pivot', 'quarterly_pivot',
                'yearly_open', 'yearly_high', 'yearly_low'
            ])
    
    return features

