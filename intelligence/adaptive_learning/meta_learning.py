

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Meta Learning Module

This module implements a sophisticated meta-learning system that learns how to learn
trading strategies, adapting to changing market conditions and optimizing parameter
learning rates and strategy selection based on historical performance.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    import torch.optim as optim  # type: ignore
    from torch.utils.data import DataLoader, Dataset, TensorDataset  # type: ignore
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    optim = None  # type: ignore
    DataLoader = Dataset = TensorDataset = None  # type: ignore
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available; meta-learning features are disabled")
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
import json
import asyncio
import concurrent.futures
import random
from scipy.stats import norm
from collections import defaultdict, deque

# Internal imports
from common.logger import get_logger
from common.utils import timer, chunks, make_async
from common.exceptions import ModelTrainingError, InsufficientDataError
from data_storage.market_data import MarketDataRepository
from data_storage.models.strategy_data import StrategyPerformance
from intelligence.adaptive_learning.reinforcement import ReinforcementLearner
from intelligence.adaptive_learning.genetic import GeneticOptimizer
from intelligence.adaptive_learning.bayesian import BayesianOptimizer
from ml_models.model_manager import ModelManager

logger = get_logger(__name__)

class MetaStrategy:
    """Represents a strategy with meta-parameters to control learning rates and adaptivity"""
    
    def __init__(self, 
                 strategy_id: str, 
                 strategy_name: str,
                 strategy_type: str,
                 default_parameters: Dict[str, Any],
                 meta_parameters: Dict[str, Any] = None):
        """
        Initialize a meta-strategy
        
        Args:
            strategy_id: Unique identifier
            strategy_name: Human-readable name
            strategy_type: Type of strategy (e.g., 'trend', 'mean_reversion', etc.)
            default_parameters: Default strategy parameters
            meta_parameters: Parameters controlling learning rates, adaptivity, etc.
        """
        self.strategy_id = strategy_id
        self.strategy_name = strategy_name
        self.strategy_type = strategy_type
        self.default_parameters = default_parameters
        
        # Set default meta-parameters if none provided
        if meta_parameters is None:
            meta_parameters = {
                'learning_rate': 0.01,
                'adaptation_rate': 0.05,
                'exploration_factor': 0.2,
                'memory_length': 500,
                'meta_learning_rate': 0.001,
                'performance_history_weight': 0.8,
                'environment_sensitivity': 0.5,
                'strategy_mutation_probability': 0.05
            }
        self.meta_parameters = meta_parameters
        
        # Performance tracking
        self.performance_history = deque(maxlen=self.meta_parameters['memory_length'])
        self.environment_history = deque(maxlen=self.meta_parameters['memory_length'])
        self.parameter_history = deque(maxlen=self.meta_parameters['memory_length'])
        
        # Current parameters (initialized to defaults)
        self.current_parameters = default_parameters.copy()
        
        # Performance metrics
        self.performance_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'expectancy': 0.0,
            'recovery_factor': 0.0
        }
        
        # Meta-learning state
        self.last_update_time = datetime.now()
        self.update_count = 0
        self.environment_states = []
        self.last_environment_state = None
        
    def update_performance(self, 
                           performance_data: Dict[str, float], 
                           environment_state: Dict[str, float],
                           parameters_used: Dict[str, Any]) -> None:
        """
        Update performance history with new data
        
        Args:
            performance_data: Dictionary of performance metrics
            environment_state: Market environment state information
            parameters_used: Strategy parameters used during this period
        """
        self.performance_history.append(performance_data)
        self.environment_history.append(environment_state)
        self.parameter_history.append(parameters_used)
        
        # Update aggregate performance metrics with exponential weighting
        alpha = self.meta_parameters['performance_history_weight']
        for key, value in performance_data.items():
            if key in self.performance_metrics:
                old_value = self.performance_metrics[key]
                self.performance_metrics[key] = alpha * value + (1 - alpha) * old_value
        
        self.last_update_time = datetime.now()
        self.update_count += 1
        self.last_environment_state = environment_state
        
        # Add environment state to history if significantly different
        self._update_environment_states(environment_state)
        
    def _update_environment_states(self, new_state: Dict[str, float]) -> None:
        """
        Update environment state history with clustering to identify distinct regimes
        
        Args:
            new_state: New environment state to potentially add to history
        """
        # Simplified environment state clustering
        if not self.environment_states:
            self.environment_states.append(new_state)
            return
            
        # Check if the new state is significantly different from existing states
        is_similar = False
        for state in self.environment_states:
            similarity = self._calculate_state_similarity(state, new_state)
            if similarity > 0.8:  # 80% similarity threshold
                is_similar = True
                # Update existing state with weighted average
                for key in state:
                    if key in new_state:
                        state[key] = 0.9 * state[key] + 0.1 * new_state[key]
                break
                
        if not is_similar:
            # Limit the number of environment states we track
            if len(self.environment_states) >= 10:
                # Replace the least recently used state
                oldest_idx = 0
                self.environment_states[oldest_idx] = new_state
            else:
                self.environment_states.append(new_state)
    
    def _calculate_state_similarity(self, state1: Dict[str, float], 
                                   state2: Dict[str, float]) -> float:
        """
        Calculate similarity between two environment states
        
        Args:
            state1: First environment state
            state2: Second environment state
            
        Returns:
            Similarity score (0-1)
        """
        common_keys = [k for k in state1 if k in state2]
        if not common_keys:
            return 0.0
            
        similarities = []
        for key in common_keys:
            # Normalize by typical ranges of these values
            if key == 'volatility':
                # Volatility typically ranges from near 0 to ~0.5
                max_diff = 0.5
            elif key == 'trend_strength':
                # Trend strength typically -1 to 1
                max_diff = 2.0
            else:
                max_diff = 1.0
                
            diff = abs(state1[key] - state2[key])
            similarity = max(0, 1 - (diff / max_diff))
            similarities.append(similarity)
            
        return sum(similarities) / len(similarities)
        
    def adapt_parameters(self) -> Dict[str, Any]:
        """
        Adapt strategy parameters based on performance history and environment
        
        Returns:
            New parameter set for the strategy
        """
        if len(self.performance_history) < 5:
            # Not enough data to adapt yet
            return self.current_parameters.copy()
            
        # Find most similar historical environment to current
        if self.last_environment_state:
            best_parameters = self._find_best_parameters_for_environment(
                self.last_environment_state
            )
            
            # Compute parameter adjustments based on meta-parameters
            new_parameters = {}
            for key, value in self.current_parameters.items():
                if key in best_parameters:
                    # Calculate adaptive learning rate based on parameter sensitivity
                    param_sensitivity = self._estimate_parameter_sensitivity(key)
                    effective_lr = self.meta_parameters['learning_rate'] * param_sensitivity
                    
                    # Move toward best known parameters
                    new_value = value + effective_lr * (best_parameters[key] - value)
                    
                    # Add exploration noise based on exploration factor
                    exploration_noise = (
                        np.random.normal(0, 1) * 
                        self.meta_parameters['exploration_factor'] * 
                        abs(value) * 0.1  # Scale noise by parameter value
                    )
                    new_value += exploration_noise
                    
                    new_parameters[key] = new_value
                else:
                    new_parameters[key] = value
                    
            # Apply parameter constraints (e.g., keep parameters in valid ranges)
            new_parameters = self._apply_parameter_constraints(new_parameters)
            
            # Update current parameters
            self.current_parameters = new_parameters
            
        return self.current_parameters.copy()
    
    def _estimate_parameter_sensitivity(self, parameter_name: str) -> float:
        """
        Estimate how sensitive performance is to changes in this parameter
        
        Args:
            parameter_name: Name of parameter to evaluate
            
        Returns:
            Sensitivity score (0-1)
        """
        if len(self.parameter_history) < 10 or len(self.performance_history) < 10:
            return 0.5  # Default middle sensitivity with insufficient data
            
        # Extract parameter values and corresponding performance metrics
        param_values = []
        performance_values = []
        
        for params, perf in zip(self.parameter_history, self.performance_history):
            if parameter_name in params:
                param_values.append(params[parameter_name])
                
                # Use expectancy as the performance metric
                if 'expectancy' in perf:
                    performance_values.append(perf['expectancy'])
                elif 'win_rate' in perf:
                    performance_values.append(perf['win_rate'])
                else:
                    # Use first available metric
                    performance_values.append(next(iter(perf.values())))
        
        if len(param_values) < 5:
            return 0.5  # Not enough variation to calculate
            
        # Calculate correlation between parameter values and performance
        try:
            correlation = np.corrcoef(param_values, performance_values)[0, 1]
            
            # Convert to absolute value for sensitivity
            sensitivity = abs(correlation)
            
            # If NaN (e.g., no variation in parameters), return default
            if np.isnan(sensitivity):
                return 0.5
                
            return min(1.0, max(0.1, sensitivity))  # Bound between 0.1 and 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating parameter sensitivity: {e}")
            return 0.5
            
    def _find_best_parameters_for_environment(self, 
                                             current_environment: Dict[str, float]) -> Dict[str, Any]:
        """
        Find the best performing parameters for an environment similar to the current one
        
        Args:
            current_environment: The current market environment state
            
        Returns:
            Best parameters for this environment type
        """
        if not self.environment_history or not self.parameter_history or not self.performance_history:
            return self.default_parameters.copy()
            
        # Calculate similarities to current environment
        similarities = []
        for env in self.environment_history:
            similarity = self._calculate_state_similarity(current_environment, env)
            similarities.append(similarity)
            
        # Weight performance by environment similarity
        weighted_performance = []
        for sim, perf in zip(similarities, self.performance_history):
            # Get expectancy or other aggregate performance metric
            if 'expectancy' in perf:
                performance_value = perf['expectancy']
            elif 'win_rate' in perf:
                performance_value = perf['win_rate']
            else:
                # Use first available metric
                performance_value = next(iter(perf.values()))
                
            # Apply exponential weighting by similarity
            weight = np.exp(5 * (sim - 1))  # Exponential scaling of similarity
            weighted_performance.append((weight, performance_value))
            
        # Find the parameters corresponding to the highest weighted performance
        best_idx = np.argmax([w * p for w, p in weighted_performance])
        
        # Return a copy of the best parameters
        return self.parameter_history[best_idx].copy()
        
    def _apply_parameter_constraints(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply constraints to ensure parameters remain in valid ranges
        
        Args:
            parameters: Unconstrained parameters
            
        Returns:
            Constrained parameters
        """
        constrained = parameters.copy()
        
        # Apply general constraints based on parameter names
        for key, value in parameters.items():
            # Constrain probability parameters to [0, 1]
            if 'probability' in key or 'threshold' in key:
                constrained[key] = min(1.0, max(0.0, value))
                
            # Constrain period parameters to be positive integers
            elif 'period' in key or 'window' in key or 'length' in key:
                constrained[key] = max(1, int(round(value)))
                
            # Constrain multiplier parameters to be positive
            elif 'multiplier' in key or 'factor' in key:
                constrained[key] = max(0.001, value)
                
            # Ensure lookup values don't go negative
            elif 'lookup' in key or 'lookback' in key:
                constrained[key] = max(1, int(round(value)))
        
        return constrained
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get the current performance summary for this meta-strategy
        
        Returns:
            Dictionary of performance metrics
        """
        return self.performance_metrics.copy()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert meta-strategy to dictionary for serialization
        
        Returns:
            Dictionary representation
        """
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.strategy_name,
            'strategy_type': self.strategy_type,
            'default_parameters': self.default_parameters,
            'meta_parameters': self.meta_parameters,
            'current_parameters': self.current_parameters,
            'performance_metrics': self.performance_metrics,
            'update_count': self.update_count,
            'last_update_time': self.last_update_time.isoformat(),
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaStrategy':
        """
        Create a meta-strategy from dictionary representation
        
        Args:
            data: Dictionary representation
            
        Returns:
            Constructed MetaStrategy object
        """
        strategy = cls(
            strategy_id=data['strategy_id'],
            strategy_name=data['strategy_name'],
            strategy_type=data['strategy_type'],
            default_parameters=data['default_parameters'],
            meta_parameters=data['meta_parameters'],
        )
        
        strategy.current_parameters = data['current_parameters']
        strategy.performance_metrics = data['performance_metrics']
        strategy.update_count = data['update_count']
        strategy.last_update_time = datetime.fromisoformat(data['last_update_time'])
        
        return strategy


class MetaLearningSystem:
    """
    Advanced meta-learning system that learns how to learn and adapt trading strategies
    """
    
    def __init__(self, 
                 model_manager: ModelManager,
                 market_data_repo: MarketDataRepository,
                 strategy_performance_repo: Optional[Any] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the meta-learning system
        
        Args:
            model_manager: Model manager for ML models
            market_data_repo: Repository for market data access
            strategy_performance_repo: Repository for strategy performance data
            config: Configuration options
        """
        self.model_manager = model_manager
        self.market_data_repo = market_data_repo
        self.strategy_performance_repo = strategy_performance_repo
        
        # Default configuration
        self._default_config = {
            'meta_learning_rate': 0.001,
            'meta_batch_size': 32,
            'inner_learning_rate': 0.01,
            'meta_train_steps': 1000,
            'inner_train_steps': 5,
            'environment_classification_interval': 24,  # hours
            'memory_replay_size': 10000,
            'strategy_refresh_interval': 4,  # hours
            'environment_history_length': 50,
            'performance_metrics_window': 100,  # trades
            'parameter_adaptation_interval': 20,  # trades
            'meta_optimizer': 'adam',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_gpu_acceleration': torch.cuda.is_available(),
            'save_interval': 24  # hours
        }
        
        # Merge provided config with defaults
        self.config = self._default_config.copy()
        if config:
            self.config.update(config)
            
        # Setup auxiliary optimization methods
        self.reinforcement_learner = ReinforcementLearner()
        self.genetic_optimizer = GeneticOptimizer()
        self.bayesian_optimizer = BayesianOptimizer()
        
        # Initialize collections
        self.strategies: Dict[str, MetaStrategy] = {}
        self.environment_models = {}  # Models for classifying market environments
        self.adaptation_models = {}  # Models for learning adaptation rules
        self.environment_clusters = []  # Discovered environment clusters
        
        # Performance tracking
        self.system_performance = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'strategy_adaptation_impact': 0.0,
            'environment_detection_accuracy': 0.0
        }
        
        # System state
        self.is_initialized = False
        self.last_environment_detection = datetime.now() - timedelta(days=1)
        self.last_strategy_refresh = datetime.now() - timedelta(days=1)
        self.last_save = datetime.now() - timedelta(days=1)
        self.current_environment = None
        
        # Memory replay buffer for meta-learning
        self.memory_replay = deque(maxlen=self.config['memory_replay_size'])
        
        # Setup meta-learning model
        self._setup_meta_learning_model()
        
        logger.info("MetaLearningSystem initialized")
        
    def _setup_meta_learning_model(self) -> None:
        """Set up the meta-learning neural network model"""
        # Define simplified MAML-inspired model for learning adaptation rules
        self.meta_model = nn.Sequential(
            nn.Linear(20, 128),  # Input: concat of environment features and strategy params
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # Output: parameter adjustments for key strategy parameters
        ).to(self.config['device'])
        
        # Define optimizer
        if self.config['meta_optimizer'] == 'adam':
            self.meta_optimizer = optim.Adam(
                self.meta_model.parameters(), 
                lr=self.config['meta_learning_rate']
            )
        else:
            self.meta_optimizer = optim.SGD(
                self.meta_model.parameters(),
                lr=self.config['meta_learning_rate']
            )
            
        # Environment classification model
        self.environment_classifier = nn.Sequential(
            nn.Linear(15, 64),  # Input: market features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # Output: probabilities for environment classes
        ).to(self.config['device'])
        
        self.env_classifier_optimizer = optim.Adam(
            self.environment_classifier.parameters(),
            lr=0.001
        )
        
    async def initialize(self) -> None:
        """Initialize the system with data loading and model preparation"""
        if self.is_initialized:
            return
            
        try:
            # Load saved strategies and models if available
            await self.load_state()
            
            # Initialize environment classification
            await self._initialize_environment_classification()
            
            # Flag as initialized
            self.is_initialized = True
            logger.info("Meta-learning system initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize meta-learning system: {e}")
            raise
            
    async def _initialize_environment_classification(self) -> None:
        """Initialize the environment classification system"""
        try:
            # Load historical market data to identify environment types
            lookback_days = 90  # Use last 90 days of data for initial clustering
            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback_days)
            
            # Get market features for multiple timeframes to improve environment detection
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            market_features = {}
            
            for timeframe in timeframes:
                features = await self.market_data_repo.get_market_features(
                    symbol='BTC/USDT',  # Use Bitcoin as reference
                    start_time=start_time,
                    end_time=end_time,
                    timeframe=timeframe
                )
                
                if features is not None and not features.empty:
                    # Resample to hourly for alignment
                    if timeframe.endswith('m'):
                        features = features.resample('1h').last()
                    elif timeframe == '4h':
                        features = features.resample('4h').last()
                    elif timeframe == '1d':
                        features = features.resample('1d').last()
                        
                    market_features[timeframe] = features
            
            if not market_features:
                logger.warning("No market features available for environment classification")
                return
                
            # Perform environment clustering
            await self._cluster_market_environments(market_features)
            
            # Train environment classifier
            await self._train_environment_classifier(market_features)
            
            logger.info(f"Identified {len(self.environment_clusters)} distinct market environment types")
            
        except Exception as e:
            logger.error(f"Error initializing environment classification: {e}")
            raise
            
    async def _cluster_market_environments(self, 
                                          market_features: Dict[str, pd.DataFrame]) -> None:
        """
        Cluster market data into distinct environment types
        
        Args:
            market_features: Dictionary of market features by timeframe
        """
        try:
            # Use 1h timeframe as base for environment clustering
            features_1h = market_features.get('1h')
            if features_1h is None or features_1h.empty:
                logger.warning("No 1h features available for environment clustering")
                return
                
            # Extract key features for environment detection
            env_features = features_1h[['volatility', 'trend_strength', 'volume_profile', 
                                        'momentum', 'market_regime', 'support_resistance_proximity']].copy()
            
            # Fill missing values
            env_features = env_features.fillna(method='ffill').fillna(0)
            
            # Normalize features
            scaler = StandardScaler()
            normalized = scaler.fit_transform(env_features.values)
            
            # Cluster environments (using K-means for simplicity)
            from sklearn.cluster import KMeans
            
            # Try different numbers of clusters and use elbow method
            max_clusters = min(10, len(normalized) // 10)  # At most 10 clusters
            inertias = []
            
            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(normalized)
                inertias.append(kmeans.inertia_)
            
            # Simple elbow detection
            inertia_diffs = np.diff(inertias)
            inertia_diffs2 = np.diff(inertia_diffs)
            optimal_clusters = np.argmax(inertia_diffs2) + 3  # +3 because we start at 2 and diff twice
            
            # Use at least 3 and at most 8 clusters
            optimal_clusters = max(3, min(8, optimal_clusters))
            
            # Final clustering
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(normalized)
            
            # Analyze environment clusters
            cluster_data = []
            
            for i in range(optimal_clusters):
                cluster_indices = np.where(clusters == i)[0]
                cluster_features = env_features.iloc[cluster_indices]
                
                cluster_info = {
                    'id': i,
                    'size': len(cluster_indices),
                    'center': kmeans.cluster_centers_[i].tolist(),
                    'mean': cluster_features.mean().to_dict(),
                    'std': cluster_features.std().to_dict(),
                    'volatility_range': [
                        cluster_features['volatility'].min(),
                        cluster_features['volatility'].max()
                    ],
                    'trend_strength_range': [
                        cluster_features['trend_strength'].min(),
                        cluster_features['trend_strength'].max()
                    ],
                    'timestamps': features_1h.index[cluster_indices].tolist()
                }
                
                # Assign descriptive name based on features
                if cluster_info['mean']['volatility'] > 0.3:
                    if cluster_info['mean']['trend_strength'] > 0.5:
                        cluster_info['name'] = "Volatile Uptrend"
                    elif cluster_info['mean']['trend_strength'] < -0.5:
                        cluster_info['name'] = "Volatile Downtrend"
                    else:
                        cluster_info['name'] = "Volatile Sideways"
                else:
                    if cluster_info['mean']['trend_strength'] > 0.5:
                        cluster_info['name'] = "Calm Uptrend"
                    elif cluster_info['mean']['trend_strength'] < -0.5:
                        cluster_info['name'] = "Calm Downtrend"
                    else:
                        cluster_info['name'] = "Calm Sideways"
                        
                cluster_data.append(cluster_info)
            
            # Store environment clusters
            self.environment_clusters = cluster_data
            
            # Save scaler for future normalization
            self.environment_scaler = scaler
            
            # Save kmeans model
            self.environment_kmeans = kmeans
            
            logger.info(f"Market environment clustering completed with {optimal_clusters} clusters")
            
        except Exception as e:
            logger.error(f"Error clustering market environments: {e}")
            raise
            
    async def _train_environment_classifier(self, 
                                           market_features: Dict[str, pd.DataFrame]) -> None:
        """
        Train a classifier to identify market environments in real-time
        
        Args:
            market_features: Dictionary of market features by timeframe
        """
        features_1h = market_features.get('1h')
        if features_1h is None or features_1h.empty or not hasattr(self, 'environment_kmeans'):
            logger.warning("Missing data for environment classifier training")
            return
            
        try:
            # Extract features for training
            X = features_1h[['volatility', 'trend_strength', 'volume_profile', 
                              'momentum', 'market_regime', 'support_resistance_proximity']].copy()
            X = X.fillna(method='ffill').fillna(0)
            
            # Get cluster labels from KMeans
            X_normalized = self.environment_scaler.transform(X.values)
            y = self.environment_kmeans.predict(X_normalized)
            
            # Convert to tensors
            X_tensor = torch.tensor(X_normalized, dtype=torch.float32).to(self.config['device'])
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.config['device'])
            
            # Create dataset
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Train classifier
            self.environment_classifier.train()
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(50):  # 50 epochs
                total_loss = 0.0
                correct = 0
                total = 0
                
                for inputs, labels in dataloader:
                    self.env_classifier_optimizer.zero_grad()
                    
                    outputs = self.environment_classifier(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    self.env_classifier_optimizer.step()
                    
                    total_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                if (epoch + 1) % 10 == 0:
                    accuracy = 100 * correct / total
                    logger.debug(f"Environment classifier epoch {epoch+1}: "
                                 f"Loss = {total_loss/len(dataloader):.4f}, "
                                 f"Accuracy = {accuracy:.2f}%")
            
            # Evaluate final accuracy
            self.environment_classifier.eval()
            with torch.no_grad():
                outputs = self.environment_classifier(X_tensor)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = 100 * (predicted == y_tensor).sum().item() / len(y_tensor)
            
            self.system_performance['environment_detection_accuracy'] = accuracy
            logger.info(f"Environment classifier training completed with {accuracy:.2f}% accuracy")
            
        except Exception as e:
            logger.error(f"Error training environment classifier: {e}")
            raise
            
    async def register_strategy(self, 
                              strategy_id: str, 
                              strategy_name: str,
                              strategy_type: str,
                              default_parameters: Dict[str, Any],
                              meta_parameters: Dict[str, Any] = None) -> MetaStrategy:
        """
        Register a strategy with the meta-learning system
        
        Args:
            strategy_id: Unique strategy identifier
            strategy_name: Human-readable name
            strategy_type: Type of strategy
            default_parameters: Default strategy parameters
            meta_parameters: Meta-learning parameters
            
        Returns:
            The registered MetaStrategy object
        """
        if strategy_id in self.strategies:
            logger.warning(f"Strategy {strategy_id} already registered, updating instead")
            
        meta_strategy = MetaStrategy(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            default_parameters=default_parameters,
            meta_parameters=meta_parameters
        )
        
        self.strategies[strategy_id] = meta_strategy
        logger.info(f"Registered strategy {strategy_name} with ID {strategy_id}")
        
        return meta_strategy
        
    async def update_strategy_performance(self, 
                                        strategy_id: str,
                                        performance_data: Dict[str, float],
                                        environment_state: Dict[str, float],
                                        parameters_used: Dict[str, Any]) -> None:
        """
        Update performance data for a strategy
        
        Args:
            strategy_id: Strategy identifier
            performance_data: Performance metrics
            environment_state: Market environment state
            parameters_used: Strategy parameters used
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not registered, cannot update performance")
            return
            
        meta_strategy = self.strategies[strategy_id]
        meta_strategy.update_performance(
            performance_data=performance_data,
            environment_state=environment_state,
            parameters_used=parameters_used
        )
        
        # Add to memory replay for meta-learning
        self.memory_replay.append({
            'strategy_id': strategy_id,
            'performance': performance_data,
            'environment': environment_state,
            'parameters': parameters_used
        })
        
        # Check if we should adapt parameters
        if (meta_strategy.update_count % self.config['parameter_adaptation_interval'] == 0 and
                meta_strategy.update_count > 5):
            # Adapt parameters and return them
            await self._adapt_strategy_parameters(strategy_id)
        
        # Update system performance metrics with exponential weighting
        alpha = 0.1  # Weight for new data
        for key in ['win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown']:
            if key in performance_data and key in self.system_performance:
                old_value = self.system_performance[key]
                self.system_performance[key] = alpha * performance_data[key] + (1 - alpha) * old_value
        
        # Check if periodic tasks should run
        now = datetime.now()
        
        # Environment detection
        env_hours = (now - self.last_environment_detection).total_seconds() / 3600
        if env_hours >= self.config['environment_classification_interval']:
            await self._detect_current_environment()
            self.last_environment_detection = now
            
        # Strategy refresh (meta-learning update)
        refresh_hours = (now - self.last_strategy_refresh).total_seconds() / 3600
        if refresh_hours >= self.config['strategy_refresh_interval'] and len(self.memory_replay) > 100:
            await self._update_meta_learning()
            self.last_strategy_refresh = now
            
        # Save state
        save_hours = (now - self.last_save).total_seconds() / 3600
        if save_hours >= self.config['save_interval']:
            await self.save_state()
            self.last_save = now
            
    async def _adapt_strategy_parameters(self, strategy_id: str) -> Dict[str, Any]:
        """
        Adapt parameters for a strategy using meta-learning and other techniques
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Adapted parameters
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not registered, cannot adapt parameters")
            return {}
            
        meta_strategy = self.strategies[strategy_id]
        
        # First, let the meta-strategy apply its own parameter adaptation
        adapted_params = meta_strategy.adapt_parameters()
        
        # If we have a current environment classification and enough training data,
        # apply meta-learned adaptations
        if self.current_environment is not None and len(self.memory_replay) > 50:
            try:
                # Apply neural network meta-model to suggest parameter adjustments
                env_features = torch.tensor(
                    [list(self.current_environment.values())], 
                    dtype=torch.float32
                ).to(self.config['device'])
                
                # Extract key parameters as feature vector
                param_keys = sorted(adapted_params.keys())[:10]  # Use up to 10 key parameters
                param_values = [adapted_params.get(k, 0.0) for k in param_keys]
                
                # Normalize parameters (simple scaling)
                param_values = [max(-10, min(10, p)) / 10.0 for p in param_values]
                
                # Pad if needed
                while len(param_values) < 10:
                    param_values.append(0.0)
                    
                param_tensor = torch.tensor(
                    [param_values], 
                    dtype=torch.float32
                ).to(self.config['device'])
                
                # Combine environment and parameter features
                combined_features = torch.cat([env_features, param_tensor], dim=1)
                
                # Get adjustment recommendations from meta-model
                with torch.no_grad():
                    adjustments = self.meta_model(combined_features).cpu().numpy()[0]
                
                # Apply adjustments to parameters
                for i, key in enumerate(param_keys[:min(10, len(param_keys))]):
                    # Scale adjustment factor based on current value
                    scale_factor = max(0.001, abs(adapted_params[key]) * 0.1)
                    
                    # Apply bounded adjustment
                    adjustment = max(-0.5, min(0.5, adjustments[i])) * scale_factor
                    adapted_params[key] += adjustment
                
                # Apply parameter constraints
                adapted_params = meta_strategy._apply_parameter_constraints(adapted_params)
                
                # Reinforcement learning fine-tuning
                if meta_strategy.update_count > 20:
                    rl_params = self.reinforcement_learner.suggest_parameters(
                        strategy_id=strategy_id,
                        current_parameters=adapted_params,
                        performance_history=list(meta_strategy.performance_history),
                        environment_state=self.current_environment
                    )
                    
                    # Blend RL suggestions (30% weight)
                    for key in adapted_params:
                        if key in rl_params:
                            adapted_params[key] = 0.7 * adapted_params[key] + 0.3 * rl_params[key]
                
                # Update the meta-strategy with the final adapted parameters
                meta_strategy.current_parameters = adapted_params
                
                logger.debug(f"Meta-learning parameter adaptation applied for strategy {strategy_id}")
                
            except Exception as e:
                logger.warning(f"Error applying meta-learned adaptations: {e}")
                # Fall back to just using the meta-strategy's own adaptation
        
        return adapted_params
        
    async def _detect_current_environment(self) -> Dict[str, float]:
        """
        Detect the current market environment
        
        Returns:
            Environment state dictionary
        """
        try:
            # Get latest market features
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)  # Last 24 hours
            
            features = await self.market_data_repo.get_market_features(
                symbol='BTC/USDT',  # Use Bitcoin as reference
                start_time=start_time,
                end_time=end_time,
                timeframe='1h'
            )
            
            if features is None or features.empty:
                logger.warning("No market features available for environment detection")
                return self.current_environment
                
            # Extract key features for environment detection
            env_features = features[['volatility', 'trend_strength', 'volume_profile', 
                                     'momentum', 'market_regime', 'support_resistance_proximity']].copy()
            
            # Use the most recent data point
            latest_features = env_features.iloc[-1].to_dict()
            
            # Normalize features using the environment scaler
            if hasattr(self, 'environment_scaler'):
                feature_vector = np.array([list(latest_features.values())])
                normalized = self.environment_scaler.transform(feature_vector)
                
                # Use the classifier to determine environment class
                if hasattr(self, 'environment_classifier'):
                    self.environment_classifier.eval()
                    with torch.no_grad():
                        tensor_input = torch.tensor(normalized, dtype=torch.float32).to(self.config['device'])
                        outputs = self.environment_classifier(tensor_input)
                        _, predicted = torch.max(outputs.data, 1)
                        env_class = predicted.item()
                        
                        # Get environment details from clusters
                        if self.environment_clusters and env_class < len(self.environment_clusters):
                            env_info = self.environment_clusters[env_class]
                            logger.info(f"Current market environment detected: {env_info['name']}")
                            
                            # Augment latest features with cluster info
                            latest_features['environment_class'] = env_class
                            latest_features['environment_name'] = env_info['name']
                else:
                    logger.warning("Environment classifier not available")
            else:
                logger.warning("Environment scaler not available")
            
            # Update current environment
            self.current_environment = latest_features
            return latest_features
            
        except Exception as e:
            logger.error(f"Error detecting current market environment: {e}")
            if self.current_environment is None:
                # Create a default environment state if none exists
                self.current_environment = {
                    'volatility': 0.2,
                    'trend_strength': 0.0,
                    'volume_profile': 0.5,
                    'momentum': 0.0,
                    'market_regime': 0.5,
                    'support_resistance_proximity': 0.5
                }
            return self.current_environment
            
    async def _update_meta_learning(self) -> None:
        """Update the meta-learning model using collected experience"""
        if len(self.memory_replay) < 100:
            logger.debug("Not enough data for meta-learning update")
            return
            
        try:
            # Prepare training data from replay memory
            train_data = []
            
            for entry in self.memory_replay:
                strategy_id = entry['strategy_id']
                if strategy_id not in self.strategies:
                    continue
                    
                meta_strategy = self.strategies[strategy_id]
                
                # Input: environment features + strategy parameters
                env_features = list(entry['environment'].values())[:10]  # Up to 10 env features
                
                # Extract key parameters (up to 10)
                param_keys = sorted(entry['parameters'].keys())[:10]
                param_values = [entry['parameters'].get(k, 0.0) for k in param_keys]
                
                # Normalize parameter values
                param_values = [max(-10, min(10, p)) / 10.0 for p in param_values]
                
                # Ensure we have 10 parameter values (pad with zeros if needed)
                while len(param_values) < 10:
                    param_values.append(0.0)
                    
                # Combine features
                input_features = env_features + param_values
                
                # Target: performance metrics
                perf_metrics = [
                    entry['performance'].get('win_rate', 0.5),
                    entry['performance'].get('expectancy', 0.0),
                    entry['performance'].get('sharpe_ratio', 0.0)
                ]
                
                # Map performance to parameter adjustments (simplified)
                adjustments = []
                
                # Calculate mean performance metric (normalized)
                perf_mean = (perf_metrics[0] + max(0, min(1, perf_metrics[1] * 2 + 0.5)) + 
                             max(0, min(1, perf_metrics[2] * 0.5 + 0.5))) / 3.0
                
                # Generate target adjustments
                for i, key in enumerate(param_keys[:min(10, len(param_keys))]):
                    # Positive performance = reinforce current direction
                    # Negative performance = reverse direction
                    if len(meta_strategy.parameter_history) > 1:
                        prev_params = list(meta_strategy.parameter_history)[-2]
                        if key in prev_params:
                            prev_value = prev_params[key]
                            current_value = entry['parameters'][key]
                            
                            # Direction of last change
                            last_change = current_value - prev_value
                            
                            # Scale based on performance
                            adjustment_factor = (perf_mean - 0.5) * 2  # -1 to 1
                            
                            # If positive performance, reinforce change direction
                            # If negative performance, reverse change direction
                            if abs(last_change) > 1e-6:  # Non-zero change
                                adjustment = adjustment_factor * min(1.0, abs(last_change) / abs(current_value))
                            else:
                                # No previous change, use small random adjustment
                                adjustment = adjustment_factor * 0.1 * random.choice([-1, 1])
                        else:
                            adjustment = 0.0
                    else:
                        # No history, use small random adjustment based on performance
                        adjustment = (perf_mean - 0.5) * 0.2 * random.choice([-1, 1])
                        
                    adjustments.append(adjustment)
                
                # Pad adjustments to 10 if needed
                while len(adjustments) < 10:
                    adjustments.append(0.0)
                
                train_data.append((input_features, adjustments))
            
            # Convert to tensors
            X = torch.tensor([item[0] for item in train_data], dtype=torch.float32).to(self.config['device'])
            y = torch.tensor([item[1] for item in train_data], dtype=torch.float32).to(self.config['device'])
            
            # Create dataset and loader
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config['meta_batch_size'], 
                shuffle=True
            )
            
            # Train meta-model
            self.meta_model.train()
            criterion = nn.MSELoss()
            
            for epoch in range(self.config['meta_train_steps']):
                total_loss = 0.0
                
                for inputs, targets in dataloader:
                    self.meta_optimizer.zero_grad()
                    
                    outputs = self.meta_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    self.meta_optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 100 == 0:
                    logger.debug(f"Meta-learning epoch {epoch+1}: Loss = {total_loss/len(dataloader):.6f}")
            
            logger.info("Meta-learning model updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating meta-learning model: {e}")
        
    async def get_strategy_parameters(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get the current optimal parameters for a strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Current strategy parameters
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not registered, returning empty parameters")
            return {}
            
        meta_strategy = self.strategies[strategy_id]
        return meta_strategy.current_parameters.copy()
        
    async def save_state(self) -> None:
        """Save the current state of the meta-learning system"""
        try:
            # Save strategies
            strategies_data = {}
            for strategy_id, strategy in self.strategies.items():
                strategies_data[strategy_id] = strategy.to_dict()
                
            # Save environment clusters
            env_clusters = []
            for cluster in self.environment_clusters:
                # Remove timestamp lists which may be too large
                cluster_copy = cluster.copy()
                if 'timestamps' in cluster_copy:
                    del cluster_copy['timestamps']
                env_clusters.append(cluster_copy)
                
            # Create state dictionary
            state = {
                'strategies': strategies_data,
                'environment_clusters': env_clusters,
                'system_performance': self.system_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to JSON file
            with open('meta_learning_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
            # Save models
            torch.save(self.meta_model.state_dict(), 'meta_model.pth')
            torch.save(self.environment_classifier.state_dict(), 'env_classifier.pth')
            
            logger.info("Meta-learning system state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving meta-learning system state: {e}")
            
    async def load_state(self) -> None:
        """Load the saved state of the meta-learning system"""
        try:
            # Check if state file exists
            if not os.path.exists('meta_learning_state.json'):
                logger.info("No saved state found, initializing new meta-learning system")
                return
                
            # Load state from JSON file
            with open('meta_learning_state.json', 'r') as f:
                state = json.load(f)
                
            # Load strategies
            strategies_data = state.get('strategies', {})
            for strategy_id, strategy_data in strategies_data.items():
                self.strategies[strategy_id] = MetaStrategy.from_dict(strategy_data)
                
            # Load environment clusters
            self.environment_clusters = state.get('environment_clusters', [])
            
            # Load system performance
            self.system_performance = state.get('system_performance', self.system_performance)
            
            # Load models if they exist
            if os.path.exists('meta_model.pth') and os.path.exists('env_classifier.pth'):
                self.meta_model.load_state_dict(torch.load('meta_model.pth'))
                self.environment_classifier.load_state_dict(torch.load('env_classifier.pth'))
                
            logger.info(f"Loaded meta-learning state with {len(self.strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Error loading meta-learning system state: {e}")
            logger.info("Initializing new meta-learning system")


if __name__ == "__main__":
    # Simple test code
    async def run_test():
        from ml_models.model_manager import ModelManager
        from data_storage.market_data import MarketDataRepository
        
        # Create mock dependencies
        model_manager = ModelManager()
        market_data_repo = MarketDataRepository()
        
        # Create meta-learning system
        meta_learning = MetaLearningSystem(
            model_manager=model_manager,
            market_data_repo=market_data_repo
        )
        
        # Initialize
        await meta_learning.initialize()
        
        # Register a test strategy
        await meta_learning.register_strategy(
            strategy_id="test_strat_1",
            strategy_name="Test Strategy 1",
            strategy_type="momentum",
            default_parameters={
                'period': 14,
                'threshold': 0.5,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'risk_factor': 1.0
            }
        )
        
        # Print registered strategy
        strategy_params = await meta_learning.get_strategy_parameters("test_strat_1")
        logger.info(f"Initial strategy parameters: {strategy_params}")
        
        # Simulate some performance updates
        for i in range(20):
            # Simulate performance data
            performance_data = {
                'win_rate': 0.6 + (random.random() - 0.5) * 0.2,
                'profit_factor': 1.5 + (random.random() - 0.5) * 0.5,
                'sharpe_ratio': 1.2 + (random.random() - 0.5) * 0.4,
                'max_drawdown': 0.1 + (random.random() - 0.5) * 0.05,
                'expectancy': 0.02 + (random.random() - 0.5) * 0.01
            }
            
            # Simulate environment state
            environment_state = {
                'volatility': 0.2 + (random.random() - 0.5) * 0.1,
                'trend_strength': 0.3 + (random.random() - 0.5) * 0.6,
                'volume_profile': 0.5 + (random.random() - 0.5) * 0.2,
                'momentum': 0.1 + (random.random() - 0.5) * 0.4,
                'market_regime': 0.6 + (random.random() - 0.5) * 0.4,
                'support_resistance_proximity': 0.4 + (random.random() - 0.5) * 0.3
            }
            
            # Get current parameters to use
            current_params = await meta_learning.get_strategy_parameters("test_strat_1")
            
            # Update strategy performance
            await meta_learning.update_strategy_performance(
                strategy_id="test_strat_1",
                performance_data=performance_data,
                environment_state=environment_state,
                parameters_used=current_params
            )
            
            # Print adapted parameters
            if (i + 1) % 5 == 0:
                adapted_params = await meta_learning.get_strategy_parameters("test_strat_1")
                logger.info(f"Adapted parameters after {i+1} updates: {adapted_params}")
        
        # Save state
        await meta_learning.save_state()
    
    # Run test
    asyncio.run(run_test())

