#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Online Learning System

This module implements real-time learning capabilities that allow the system
to continuously adapt and improve based on new market data and trading outcomes.
It provides both supervised and reinforcement learning approaches for strategy optimization.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import threading
import queue
import logging
import traceback
from collections import deque, defaultdict
import asyncio
import concurrent.futures

# ML/AI imports
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.regularizers import l1_l2
    TENSORFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    tf = None
    layers = models = optimizers = callbacks = None
    l1_l2 = None
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    optim = None
    TORCH_AVAILABLE = False
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Internal imports
from common.logger import get_logger
from common.utils import (
    create_directory_if_not_exists, 
    load_json_file, 
    save_json_file,
    get_timestamp, 
    time_function
)
from common.constants import (
    ONLINE_LEARNING_INTERVAL, 
    MODEL_SAVE_INTERVAL, 
    MIN_SAMPLES_FOR_TRAINING,
    FEATURE_IMPORTANCE_THRESHOLD,
    LEARNING_RATES,
    REINFORCEMENT_DECAY_FACTOR,
    BATCH_SIZES
)
from common.metrics import MetricsCollector
from common.exceptions import (
    ModelTrainingError, 
    InsufficientDataError, 
    ModelSaveError,
    FeatureEngineeringError
)
from common.redis_client import RedisClient
from common.db_client import DatabaseClient
from common.async_utils import run_in_executor

from data_storage.market_data import MarketDataRepository
from data_storage.models.strategy_data import StrategyPerformance
from feature_service.feature_extraction import FeatureExtractor
from ml_models.models.classification import ClassificationModelFactory
from ml_models.models.regression import RegressionModelFactory
from ml_models.hardware.gpu import GPUAccelerator

logger = get_logger(__name__)

class OnlineLearner:
    """
    Online learning system that continuously adapts and improves trading strategies 
    based on new market data and feedback from trading outcomes.
    
    This class implements both supervised and reinforcement learning approaches
    to optimize trading strategies in real-time.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        redis_client: RedisClient,
        db_client: DatabaseClient,
        market_data_repo: MarketDataRepository,
        feature_extractor: FeatureExtractor,
        metrics_collector: MetricsCollector
    ):
        """
        Initialize the OnlineLearner.
        
        Args:
            config: Configuration dictionary
            redis_client: Redis client for caching and pub/sub
            db_client: Database client for persistent storage
            market_data_repo: Repository for accessing market data
            feature_extractor: Feature extraction service
            metrics_collector: Metrics collection service
        """
        self.config = config
        self.redis_client = redis_client
        self.db_client = db_client
        self.market_data_repo = market_data_repo
        self.feature_extractor = feature_extractor
        self.metrics_collector = metrics_collector
        
        # Initialize GPU acceleration if available
        self.gpu_accelerator = GPUAccelerator()
        self.devices = self.gpu_accelerator.get_available_devices()
        if self.devices:
            logger.info(f"Using GPU acceleration for online learning: {self.devices}")
        else:
            logger.info("No GPU devices available, using CPU for online learning")
            
        # Data buffers for training
        self.training_buffers = defaultdict(lambda: defaultdict(lambda: deque(maxlen=config.get("buffer_size", 100000))))
        
        # Model factories
        self.classification_factory = ClassificationModelFactory()
        self.regression_factory = RegressionModelFactory()
        
        # Training queues
        self.training_queue = queue.PriorityQueue()
        self.feedback_queue = queue.Queue()
        
        # Model persistence path
        self.model_dir = os.path.join(
            config.get("model_storage_path", "models"),
            "online_learner"
        )
        create_directory_if_not_exists(self.model_dir)
        
        # Performance tracking
        self.performance_metrics = defaultdict(lambda: defaultdict(list))
        
        # Active training tasks
        self.active_training_tasks = {}
        self.task_lock = threading.Lock()
        
        # For reinforcement learning
        self.reward_history = defaultdict(lambda: defaultdict(list))
        self.action_history = defaultdict(lambda: defaultdict(list))
        self.state_history = defaultdict(lambda: defaultdict(list))
        
        # Initialize the learning models
        self._initialize_models()
        
        logger.info("Online Learner initialized successfully")
    
    async def start(self):
        """Start the online learning system"""
        logger.info("Starting Online Learner")
        
        # Start worker threads
        self.running = True
        self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.training_thread.start()
        
        self.feedback_thread = threading.Thread(target=self._feedback_worker, daemon=True)
        self.feedback_thread.start()
        
        # Start periodic tasks
        asyncio.create_task(self._run_periodic_training())
        asyncio.create_task(self._run_model_persistence())
        asyncio.create_task(self._run_performance_evaluation())
        
        await self._subscribe_to_events()
        
        logger.info("Online Learner started successfully")
    
    async def stop(self):
        """Stop the online learning system"""
        logger.info("Stopping Online Learner")
        self.running = False
        
        # Wait for worker threads to finish
        if hasattr(self, 'training_thread'):
            self.training_thread.join(timeout=5.0)
        
        if hasattr(self, 'feedback_thread'):
            self.feedback_thread.join(timeout=5.0)
            
        # Save all models before stopping
        await self._persist_all_models()
        
        logger.info("Online Learner stopped successfully")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant Redis events for online learning"""
        pubsub = self.redis_client.client.pubsub()
        
        # Subscribe to market data updates
        await pubsub.subscribe("market_data_update")
        
        # Subscribe to trading signals
        await pubsub.subscribe("trading_signal")
        
        # Subscribe to trade executions
        await pubsub.subscribe("trade_execution")
        
        # Subscribe to trade outcomes
        await pubsub.subscribe("trade_outcome")
        
        # Process messages in background
        asyncio.create_task(self._process_pubsub_messages(pubsub))
        
        logger.info("Subscribed to Redis events for online learning")
    
    async def _process_pubsub_messages(self, pubsub):
        """Process messages from Redis pub/sub"""
        while self.running:
            try:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    channel = message['channel'].decode('utf-8')
                    data = pickle.loads(message['data'])
                    
                    if channel == "market_data_update":
                        await self._handle_market_data_update(data)
                    elif channel == "trading_signal":
                        await self._handle_trading_signal(data)
                    elif channel == "trade_execution":
                        await self._handle_trade_execution(data)
                    elif channel == "trade_outcome":
                        await self._handle_trade_outcome(data)
                
                await asyncio.sleep(0.01)  # Small delay to prevent tight loop
            except Exception as e:
                logger.error(f"Error processing pub/sub message: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1.0)  # Delay before retry
    
    async def _handle_market_data_update(self, data):
        """Handle market data updates for continuous learning"""
        try:
            # Extract metadata
            symbol = data.get('symbol')
            timeframe = data.get('timeframe')
            platform = data.get('platform')
            
            # Store in training buffer
            key = f"{platform}:{symbol}:{timeframe}"
            
            # Get features
            features = await self.feature_extractor.extract_features(data)
            
            # Add to training buffer
            self.training_buffers[key]['features'].append(features)
            
            # If we have labels for this data point, add them too
            if 'labels' in data:
                self.training_buffers[key]['labels'].append(data['labels'])
            
            # Update metrics
            self.metrics_collector.increment_counter(
                "online_learner.market_data_updates",
                1,
                {"platform": platform, "symbol": symbol, "timeframe": timeframe}
            )
        except Exception as e:
            logger.error(f"Error handling market data update: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _handle_trading_signal(self, data):
        """Handle trading signals for reinforcement learning state tracking"""
        try:
            # Extract metadata
            symbol = data.get('symbol')
            timeframe = data.get('timeframe')
            platform = data.get('platform')
            strategy = data.get('strategy')
            
            key = f"{platform}:{symbol}:{timeframe}"
            
            # Store state for reinforcement learning
            if 'features' in data:
                self.state_history[key][strategy].append(data['features'])
            
            # Store action for reinforcement learning
            if 'action' in data:
                self.action_history[key][strategy].append(data['action'])
            
            # Update metrics
            self.metrics_collector.increment_counter(
                "online_learner.trading_signals",
                1,
                {"platform": platform, "symbol": symbol, "timeframe": timeframe, "strategy": strategy}
            )
        except Exception as e:
            logger.error(f"Error handling trading signal: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _handle_trade_execution(self, data):
        """Handle trade executions for strategy performance tracking"""
        try:
            # We don't need to do much here, just record the execution for later matching
            trade_id = data.get('trade_id')
            
            # Store in Redis for quick lookup when we get the outcome
            await self.redis_client.set_with_expiry(
                f"trade_execution:{trade_id}", 
                pickle.dumps(data),
                expiry_seconds=86400  # 24 hours
            )
        except Exception as e:
            logger.error(f"Error handling trade execution: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _handle_trade_outcome(self, data):
        """Handle trade outcomes for reinforcement learning rewards"""
        try:
            # Extract metadata
            trade_id = data.get('trade_id')
            symbol = data.get('symbol')
            timeframe = data.get('timeframe')
            platform = data.get('platform')
            strategy = data.get('strategy')
            outcome = data.get('outcome', {})
            
            # Get the original execution data
            execution_data_bytes = await self.redis_client.get(f"trade_execution:{trade_id}")
            if execution_data_bytes:
                execution_data = pickle.loads(execution_data_bytes)
                
                # Combine execution and outcome data
                feedback_data = {
                    **execution_data,
                    'outcome': outcome
                }
                
                # Push to feedback queue for processing
                self.feedback_queue.put(feedback_data)
                
                # Calculate reward for reinforcement learning
                reward = self._calculate_reward(outcome)
                
                # Store reward for reinforcement learning
                key = f"{platform}:{symbol}:{timeframe}"
                self.reward_history[key][strategy].append(reward)
                
                # Update metrics
                self.metrics_collector.increment_counter(
                    "online_learner.trade_outcomes",
                    1,
                    {
                        "platform": platform, 
                        "symbol": symbol, 
                        "timeframe": timeframe, 
                        "strategy": strategy,
                        "success": outcome.get('successful', False)
                    }
                )
                
                # If we have enough data, schedule a training task
                if (len(self.state_history[key][strategy]) > MIN_SAMPLES_FOR_TRAINING and
                    len(self.action_history[key][strategy]) > MIN_SAMPLES_FOR_TRAINING and
                    len(self.reward_history[key][strategy]) > MIN_SAMPLES_FOR_TRAINING):
                    
                    # Schedule reinforcement learning task
                    self._schedule_training_task(
                        priority=1,  # High priority
                        task_type="reinforcement",
                        platform=platform,
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy=strategy
                    )
        except Exception as e:
            logger.error(f"Error handling trade outcome: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _calculate_reward(self, outcome):
        """Calculate reward for reinforcement learning based on trade outcome"""
        # Basic implementation - use profit/loss as reward
        if 'pnl' in outcome:
            return float(outcome['pnl'])
        
        # If we don't have PnL, use success/failure
        if outcome.get('successful', False):
            return 1.0
        else:
            return -1.0
    
    def _initialize_models(self):
        """Initialize or load existing models"""
        try:
            logger.info("Initializing online learning models")
            self.models = defaultdict(dict)
            
            # Check if we have saved models
            if os.path.exists(self.model_dir):
                for model_file in os.listdir(self.model_dir):
                    if model_file.endswith('.pkl'):
                        model_path = os.path.join(self.model_dir, model_file)
                        try:
                            # Parse model_file to get metadata
                            parts = model_file.replace('.pkl', '').split('_')
                            if len(parts) >= 4:
                                model_type = parts[0]
                                platform = parts[1]
                                symbol = parts[2]
                                timeframe = parts[3]
                                if len(parts) >= 5:
                                    strategy = '_'.join(parts[4:])
                                else:
                                    strategy = 'default'
                                
                                # Load the model
                                with open(model_path, 'rb') as f:
                                    model_data = pickle.load(f)
                                
                                key = f"{platform}:{symbol}:{timeframe}"
                                self.models[key][strategy] = model_data
                                
                                logger.info(f"Loaded model: {model_file}")
                        except Exception as e:
                            logger.error(f"Error loading model {model_file}: {str(e)}")
            
            logger.info("Online learning models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing online learning models: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _run_periodic_training(self):
        """Run periodic training for all strategies"""
        while self.running:
            try:
                # Check all training buffers
                for key, buffer in self.training_buffers.items():
                    if len(buffer['features']) > MIN_SAMPLES_FOR_TRAINING and len(buffer['labels']) > MIN_SAMPLES_FOR_TRAINING:
                        # Parse key
                        platform, symbol, timeframe = key.split(':')
                        
                        # Schedule supervised learning task
                        self._schedule_training_task(
                            priority=2,  # Medium priority
                            task_type="supervised",
                            platform=platform,
                            symbol=symbol,
                            timeframe=timeframe,
                            strategy="default"
                        )
                
                # Sleep for the configured interval
                await asyncio.sleep(ONLINE_LEARNING_INTERVAL)
            except Exception as e:
                logger.error(f"Error in periodic training: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)  # Sleep for a minute on error
    
    async def _run_model_persistence(self):
        """Periodically save models to disk"""
        while self.running:
            try:
                await self._persist_all_models()
                
                # Sleep for the configured interval
                await asyncio.sleep(MODEL_SAVE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in model persistence: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(300)  # Sleep for 5 minutes on error
    
    async def _persist_all_models(self):
        """Save all models to disk"""
        logger.info("Persisting all online learning models")
        
        for key, strategies in self.models.items():
            platform, symbol, timeframe = key.split(':')
            
            for strategy, model_data in strategies.items():
                try:
                    model_type = model_data.get('type', 'unknown')
                    model_file = f"{model_type}_{platform}_{symbol}_{timeframe}_{strategy}.pkl"
                    model_path = os.path.join(self.model_dir, model_file)
                    
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_data, f)
                    
                    logger.debug(f"Saved model: {model_file}")
                except Exception as e:
                    logger.error(f"Error saving model {model_file}: {str(e)}")
        
        logger.info("All models persisted successfully")
    
    async def _run_performance_evaluation(self):
        """Periodically evaluate model performance"""
        while self.running:
            try:
                # Evaluate all models
                for key, strategies in self.models.items():
                    platform, symbol, timeframe = key.split(':')
                    
                    for strategy, model_data in strategies.items():
                        # Skip if model doesn't have evaluation data
                        if 'evaluation' not in model_data:
                            continue
                        
                        # Get the latest performance metrics
                        metrics = model_data['evaluation']
                        
                        # Store in database for long-term tracking
                        await run_in_executor(
                            self.db_client.insert,
                            "model_performance",
                            {
                                "platform": platform,
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "strategy": strategy,
                                "model_type": model_data.get('type', 'unknown'),
                                "accuracy": metrics.get('accuracy', 0.0),
                                "precision": metrics.get('precision', 0.0),
                                "recall": metrics.get('recall', 0.0),
                                "f1_score": metrics.get('f1', 0.0),
                                "mse": metrics.get('mse', 0.0),
                                "mae": metrics.get('mae', 0.0),
                                "r2": metrics.get('r2', 0.0),
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                
                # Sleep for an hour between evaluations
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Error in performance evaluation: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(600)  # Sleep for 10 minutes on error
    
    def _schedule_training_task(self, priority, task_type, platform, symbol, timeframe, strategy):
        """Schedule a training task"""
        # Check if this exact task is already in the queue or running
        task_key = f"{task_type}_{platform}_{symbol}_{timeframe}_{strategy}"
        
        with self.task_lock:
            if task_key in self.active_training_tasks:
                # Task already running or scheduled
                return
            
            # Mark task as active
            self.active_training_tasks[task_key] = time.time()
        
        # Create task
        task = {
            'type': task_type,
            'platform': platform,
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy': strategy,
            'timestamp': time.time()
        }
        
        # Add to queue with priority
        self.training_queue.put((priority, task))
        
        logger.debug(f"Scheduled {task_type} training task for {platform}:{symbol}:{timeframe}:{strategy}")
    
    def _training_worker(self):
        """Worker thread for processing training tasks"""
        logger.info("Training worker thread started")
        
        while self.running:
            try:
                # Get task from queue with 1 second timeout
                try:
                    priority, task = self.training_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process task
                task_type = task['type']
                platform = task['platform']
                symbol = task['symbol']
                timeframe = task['timeframe']
                strategy = task['strategy']
                
                key = f"{platform}:{symbol}:{timeframe}"
                task_key = f"{task_type}_{platform}_{symbol}_{timeframe}_{strategy}"
                
                try:
                    if task_type == "supervised":
                        self._run_supervised_learning(key, strategy)
                    elif task_type == "reinforcement":
                        self._run_reinforcement_learning(key, strategy)
                    
                    # Update metrics
                    self.metrics_collector.increment_counter(
                        "online_learner.training_tasks_completed",
                        1,
                        {
                            "task_type": task_type,
                            "platform": platform,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "strategy": strategy
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing training task {task_key}: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Update error metrics
                    self.metrics_collector.increment_counter(
                        "online_learner.training_task_errors",
                        1,
                        {
                            "task_type": task_type,
                            "platform": platform,
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "strategy": strategy,
                            "error": str(e)[:100]  # Truncate long error messages
                        }
                    )
                finally:
                    # Mark task as complete
                    with self.task_lock:
                        if task_key in self.active_training_tasks:
                            del self.active_training_tasks[task_key]
                    
                    # Mark task as done in queue
                    self.training_queue.task_done()
            except Exception as e:
                logger.error(f"Error in training worker: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(5)  # Sleep for 5 seconds on error
        
        logger.info("Training worker thread stopped")
    
    def _feedback_worker(self):
        """Worker thread for processing trade feedback"""
        logger.info("Feedback worker thread started")
        
        while self.running:
            try:
                # Get feedback from queue with 1 second timeout
                try:
                    feedback = self.feedback_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process feedback
                self._process_trade_feedback(feedback)
                
                # Mark feedback as done
                self.feedback_queue.task_done()
            except Exception as e:
                logger.error(f"Error in feedback worker: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(5)  # Sleep for 5 seconds on error
        
        logger.info("Feedback worker thread stopped")
    
    def _process_trade_feedback(self, feedback):
        """Process trade feedback and generate training data"""
        try:
            # Extract metadata
            symbol = feedback.get('symbol')
            timeframe = feedback.get('timeframe')
            platform = feedback.get('platform')
            strategy = feedback.get('strategy')
            
            key = f"{platform}:{symbol}:{timeframe}"
            
            # Generate label from outcome
            outcome = feedback.get('outcome', {})
            success = outcome.get('successful', False)
            pnl = outcome.get('pnl', 0.0)
            direction = feedback.get('direction', 'none')
            
            # For classification: 1 for successful trade, 0 for unsuccessful
            if success:
                classification_label = 1
            else:
                classification_label = 0
            
            # For regression: the actual PnL
            regression_label = float(pnl)
            
            # If we have features in the feedback, use them for training
            if 'features' in feedback:
                features = feedback['features']
                
                # Add to training buffer with both classification and regression labels
                self.training_buffers[key]['features'].append(features)
                self.training_buffers[key]['labels'].append({
                    'classification': classification_label,
                    'regression': regression_label,
                    'direction': direction
                })
                
                # If we have enough data, schedule a training task
                if (len(self.training_buffers[key]['features']) > MIN_SAMPLES_FOR_TRAINING and
                    len(self.training_buffers[key]['labels']) > MIN_SAMPLES_FOR_TRAINING):
                    
                    # Schedule supervised learning task
                    self._schedule_training_task(
                        priority=1,  # High priority
                        task_type="supervised",
                        platform=platform,
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy=strategy
                    )
            
            # Update metrics
            self.metrics_collector.increment_counter(
                "online_learner.trade_feedback_processed",
                1,
                {
                    "platform": platform,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "success": success
                }
            )
        except Exception as e:
            logger.error(f"Error processing trade feedback: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _run_supervised_learning(self, key, strategy):
        """Run supervised learning for a specific key and strategy"""
        # Get the training data
        features = list(self.training_buffers[key]['features'])
        labels = list(self.training_buffers[key]['labels'])
        
        if len(features) < MIN_SAMPLES_FOR_TRAINING or len(labels) < MIN_SAMPLES_FOR_TRAINING:
            logger.warning(f"Insufficient data for training {key}:{strategy}, skipping")
            return
        
        # Convert to numpy arrays
        X = np.array(features)
        
        # Extract classification labels
        y_classification = np.array([label.get('classification', 0) for label in labels])
        
        # Extract regression labels
        y_regression = np.array([label.get('regression', 0.0) for label in labels])
        
        # Extract direction labels
        y_direction = np.array([label.get('direction', 'none') for label in labels])
        
        # Train classification model
        classification_model = self._train_classification_model(X, y_classification)
        
        # Train regression model
        regression_model = self._train_regression_model(X, y_regression)
        
        # Train direction model (if we have direction labels)
        direction_model = None
        unique_directions = set(y_direction)
        if len(unique_directions) > 1 and 'none' not in unique_directions:
            direction_model = self._train_direction_model(X, y_direction)
        
        # Store models
        self.models[key][strategy] = {
            'type': 'supervised',
            'classification': classification_model,
            'regression': regression_model,
            'direction': direction_model,
            'last_updated': datetime.now().isoformat(),
            'samples_used': len(features),
            'evaluation': {
                'accuracy': classification_model.get('metrics', {}).get('accuracy', 0.0),
                'precision': classification_model.get('metrics', {}).get('precision', 0.0),
                'recall': classification_model.get('metrics', {}).get('recall', 0.0),
                'f1': classification_model.get('metrics', {}).get('f1', 0.0),
                'mse': regression_model.get('metrics', {}).get('mse', 0.0),
                'mae': regression_model.get('metrics', {}).get('mae', 0.0),
                'r2': regression_model.get('metrics', {}).get('r2', 0.0)
            }
        }
        
        logger.info(f"Completed supervised learning for {key}:{strategy}")
        
        # Log metrics
        metrics = self.models[key][strategy]['evaluation']
        logger.debug(f"Model metrics for {key}:{strategy}: {metrics}")
    
    def _train_classification_model(self, X, y):
        """Train a classification model for trade success prediction"""
        # Split into training and testing sets
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            break  # Just use the last split
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try multiple model types and select the best one
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, 
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42
            ),
            "gradient_boosting": self.classification_factory.create_model(
                model_type="gradient_boosting",
                input_dim=X_train.shape[1]
            ),
            "neural_network": self.classification_factory.create_model(
                model_type="neural_network",
                input_dim=X_train.shape[1]
            )
        }
        
        best_model = None
        best_score = -np.inf
        best_predictions = None
        best_model_type = None
        
        for model_type, model in models.items():
            # Train the model
            if model_type == "neural_network":
                # For neural network models
                if isinstance(model, dict) and 'model' in model:
                    nn_model = model['model']
                    if hasattr(nn_model, 'fit'):
                        nn_model.fit(
                            X_train_scaled, 
                            y_train,
                            epochs=20,
                            batch_size=64,
                            verbose=0
                        )
                        
                    y_pred = nn_model.predict(X_test_scaled)
                    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                        # For multi-class output
                        y_pred = np.argmax(y_pred, axis=1)
                    else:
                        # For binary output
                        y_pred = (y_pred > 0.5).astype(int)
                else:
                    # Skip if model creation failed
                    continue
            else:
                # For scikit-learn models
                model.fit(X_train_scaled, y_train)
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_scaled)
                    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
                else:
                    y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            # Calculate weighted score (emphasize precision and recall)
            score = (accuracy + 2*precision + 2*recall + f1) / 6
            
            if score > best_score:
                best_score = score
                best_model = model
                best_predictions = y_pred
                best_model_type = model_type
        
        if best_model is None:
            raise ModelTrainingError("Failed to train any classification model")
        
        # Re-calculate metrics for the best model
        accuracy = accuracy_score(y_test, best_predictions)
        precision = precision_score(y_test, best_predictions, average='binary', zero_division=0)
        recall = recall_score(y_test, best_predictions, average='binary', zero_division=0)
        f1 = f1_score(y_test, best_predictions, average='binary', zero_division=0)
        
        # Get feature importances if available
        feature_importances = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = best_model.feature_importances_
        
        # Return the model and its metadata
        return {
            'model': best_model,
            'scaler': scaler,
            'model_type': best_model_type,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            },
            'feature_importances': feature_importances
        }
    
    def _train_regression_model(self, X, y):
        """Train a regression model for PnL prediction"""
        # Split into training and testing sets
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            break  # Just use the last split
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try multiple model types and select the best one
        models = {
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_split=2,
                learning_rate=0.1,
                loss='huber',
                random_state=42
            ),
            "neural_network": self.regression_factory.create_model(
                model_type="neural_network",
                input_dim=X_train.shape[1]
            )
        }
        
        best_model = None
        best_score = np.inf  # Lower is better for MSE
        best_predictions = None
        best_model_type = None
        
        for model_type, model in models.items():
            try:
                # Train the model
                if model_type == "neural_network":
                    # For neural network models
                    if isinstance(model, dict) and 'model' in model:
                        nn_model = model['model']
                        if hasattr(nn_model, 'fit'):
                            nn_model.fit(
                                X_train_scaled, 
                                y_train,
                                epochs=20,
                                batch_size=64,
                                verbose=0
                            )
                            
                        y_pred = nn_model.predict(X_test_scaled).flatten()
                    else:
                        # Skip if model creation failed
                        continue
                else:
                    # For scikit-learn models
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Use MSE as the primary score
                score = mse
                
                if score < best_score:
                    best_score = score
                    best_model = model
                    best_predictions = y_pred
                    best_model_type = model_type
            except Exception as e:
                logger.error(f"Error training regression model {model_type}: {str(e)}")
        
        if best_model is None:
            raise ModelTrainingError("Failed to train any regression model")
        
        # Re-calculate metrics for the best model
        mse = mean_squared_error(y_test, best_predictions)
        mae = mean_absolute_error(y_test, best_predictions)
        r2 = r2_score(y_test, best_predictions)
        
        # Get feature importances if available
        feature_importances = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = best_model.feature_importances_
        
        # Return the model and its metadata
        return {
            'model': best_model,
            'scaler': scaler,
            'model_type': best_model_type,
            'metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2)
            },
            'feature_importances': feature_importances
        }
    
    def _train_direction_model(self, X, y):
        """Train a model to predict trade direction (buy/sell)"""
        # Convert labels to categorical
        from sklearn.preprocessing import LabelEncoder
        
        # Encode direction labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        # Split into training and testing sets
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_encoded[train_index], y_encoded[test_index]
            break  # Just use the last split
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create a random forest classifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get feature importances
        feature_importances = None
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        
        # Return the model and its metadata
        return {
            'model': model,
            'scaler': scaler,
            'encoder': encoder,
            'model_type': 'random_forest',
            'metrics': {
                'accuracy': float(accuracy)
            },
            'feature_importances': feature_importances,
            'classes': list(encoder.classes_)
        }
    
    def _run_reinforcement_learning(self, key, strategy):
        """Run reinforcement learning for a specific key and strategy"""
        # Get the reinforcement learning data
        states = list(self.state_history[key][strategy])
        actions = list(self.action_history[key][strategy])
        rewards = list(self.reward_history[key][strategy])
        
        if len(states) < MIN_SAMPLES_FOR_TRAINING or len(actions) < MIN_SAMPLES_FOR_TRAINING or len(rewards) < MIN_SAMPLES_FOR_TRAINING:
            logger.warning(f"Insufficient data for reinforcement learning {key}:{strategy}, skipping")
            return
        
        # Make sure all lists have the same length
        min_length = min(len(states), len(actions), len(rewards))
        states = states[:min_length]
        actions = actions[:min_length]
        rewards = rewards[:min_length]
        
        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        
        # Train a Q-learning model or DQN depending on data size
        if len(states) > 1000:
            # Use DQN for larger datasets
            model = self._train_dqn(states, actions, rewards)
        else:
            # Use Q-learning for smaller datasets
            model = self._train_q_learning(states, actions, rewards)
        
        # Store model
        self.models[key][strategy] = {
            'type': 'reinforcement',
            'model': model,
            'last_updated': datetime.now().isoformat(),
            'samples_used': min_length
        }
        
        logger.info(f"Completed reinforcement learning for {key}:{strategy}")
    
    def _train_q_learning(self, states, actions, rewards):
        """Train a Q-learning model"""
        # Q-learning for discrete state and action spaces
        # For simplicity, we'll discretize the continuous state space
        from sklearn.cluster import KMeans
        
        # Discretize states using KMeans clustering
        n_clusters = min(100, len(states))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(states)
        
        # Map each state to its cluster index
        state_clusters = kmeans.predict(states)
        
        # Find unique actions
        unique_actions = np.unique(actions)
        n_actions = len(unique_actions)
        
        # Initialize Q-table
        q_table = np.zeros((n_clusters, n_actions))
        
        # Map actions to indices
        action_to_index = {action: i for i, action in enumerate(unique_actions)}
        action_indices = np.array([action_to_index[action] for action in actions])
        
        # Learning parameters
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        
        # Train Q-table
        for i in range(len(state_clusters) - 1):
            state = state_clusters[i]
            action = action_indices[i]
            reward = rewards[i]
            next_state = state_clusters[i + 1]
            
            # Q-learning update rule
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]))
        
        # Return the model
        return {
            'type': 'q_learning',
            'q_table': q_table,
            'kmeans': kmeans,
            'unique_actions': unique_actions,
            'action_to_index': action_to_index
        }
    
    def _train_dqn(self, states, actions, rewards):
        """Train a Deep Q-Network model"""
        # For simplicity, we'll implement a basic DQN here
        # In a real implementation, this would be much more sophisticated
        
        # Find unique actions
        unique_actions = np.unique(actions)
        n_actions = len(unique_actions)
        
        # Map actions to indices
        action_to_index = {action: i for i, action in enumerate(unique_actions)}
        action_indices = np.array([action_to_index[action] for action in actions])
        
        # Parameters
        input_dim = states.shape[1]
        
        # Create a DQN model using TensorFlow
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(n_actions, activation='linear')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        # Create target arrays for DQN training
        # For each state, the target is the reward plus the discounted max Q-value of the next state
        gamma = 0.9  # Discount factor
        targets = np.zeros((len(states), n_actions))
        
        # Fill in targets for the taken actions
        for i in range(len(states) - 1):
            action = action_indices[i]
            reward = rewards[i]
            next_state = states[i + 1]
            
            # Predict Q-values for next state
            next_q_values = model.predict(next_state.reshape(1, -1), verbose=0)[0]
            
            # Set target for the action taken to be the reward plus discounted max future reward
            targets[i] = model.predict(states[i].reshape(1, -1), verbose=0)[0]
            targets[i, action] = reward + gamma * np.max(next_q_values)
        
        # Train the model
        model.fit(
            states, 
            targets,
            epochs=10,
            batch_size=64,
            verbose=0
        )
        
        # Return the model
        return {
            'type': 'dqn',
            'model': model,
            'unique_actions': unique_actions,
            'action_to_index': action_to_index
        }
    
    async def get_prediction(self, symbol, timeframe, platform, strategy, features):
        """Get a prediction from the trained models"""
        key = f"{platform}:{symbol}:{timeframe}"
        
        # Check if we have models for this key and strategy
        if key not in self.models or strategy not in self.models[key]:
            # Try to use default strategy if available
            if key in self.models and 'default' in self.models[key]:
                strategy = 'default'
            else:
                # No model available
                return {
                    'success_probability': 0.5,  # Default probability
                    'estimated_pnl': 0.0,        # Default PnL
                    'recommended_direction': 'none',  # Default direction
                    'confidence': 0.0            # Default confidence
                }
        
        model_data = self.models[key][strategy]
        model_type = model_data.get('type', 'unknown')
        
        if model_type == 'supervised':
            # Get predictions from supervised models
            return await self._get_supervised_prediction(model_data, features)
        elif model_type == 'reinforcement':
            # Get predictions from reinforcement learning models
            return await self._get_reinforcement_prediction(model_data, features)
        else:
            # Unknown model type
            return {
                'success_probability': 0.5,
                'estimated_pnl': 0.0,
                'recommended_direction': 'none',
                'confidence': 0.0
            }
    
    async def _get_supervised_prediction(self, model_data, features):
        """Get predictions from supervised learning models"""
        # Convert features to numpy array
        X = np.array([features])
        
        # Get classification model
        classification_model = model_data.get('classification', {})
        if not classification_model or 'model' not in classification_model or 'scaler' not in classification_model:
            # No valid classification model
            success_probability = 0.5
        else:
            # Scale features
            scaler = classification_model['scaler']
            X_scaled = scaler.transform(X)
            
            # Get model and model type
            model = classification_model['model']
            model_type = classification_model.get('model_type', 'unknown')
            
            # Get prediction
            if model_type == 'neural_network':
                # For neural network models
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_scaled, verbose=0)
                    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                        # For multi-class output
                        success_probability = float(y_pred[0, 1])
                    else:
                        # For binary output
                        success_probability = float(y_pred[0])
                else:
                    success_probability = 0.5
            else:
                # For scikit-learn models
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_scaled)
                    success_probability = float(y_pred_proba[0, 1])
                else:
                    y_pred = model.predict(X_scaled)
                    success_probability = float(y_pred[0])
        
        # Get regression model
        regression_model = model_data.get('regression', {})
        if not regression_model or 'model' not in regression_model or 'scaler' not in regression_model:
            # No valid regression model
            estimated_pnl = 0.0
        else:
            # Scale features
            scaler = regression_model['scaler']
            X_scaled = scaler.transform(X)
            
            # Get model and model type
            model = regression_model['model']
            model_type = regression_model.get('model_type', 'unknown')
            
            # Get prediction
            if model_type == 'neural_network':
                # For neural network models
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_scaled, verbose=0)
                    estimated_pnl = float(y_pred[0])
                else:
                    estimated_pnl = 0.0
            else:
                # For scikit-learn models
                y_pred = model.predict(X_scaled)
                estimated_pnl = float(y_pred[0])
        
        # Get direction model
        direction_model = model_data.get('direction', {})
        if not direction_model or 'model' not in direction_model or 'scaler' not in direction_model or 'encoder' not in direction_model:
            # No valid direction model
            recommended_direction = 'none'
        else:
            # Scale features
            scaler = direction_model['scaler']
            X_scaled = scaler.transform(X)
            
            # Get model
            model = direction_model['model']
            
            # Get prediction
            y_pred = model.predict(X_scaled)
            
            # Decode direction
            encoder = direction_model['encoder']
            recommended_direction = encoder.inverse_transform([y_pred[0]])[0]
        
        # Calculate confidence based on model metrics
        metrics = model_data.get('evaluation', {})
        accuracy = metrics.get('accuracy', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1 = metrics.get('f1', 0.0)
        
        # Weight metrics for confidence calculation
        confidence = (accuracy + 2*precision + recall + f1) / 5
        
        # Return predictions
        return {
            'success_probability': float(success_probability),
            'estimated_pnl': float(estimated_pnl),
            'recommended_direction': str(recommended_direction),
            'confidence': float(confidence)
        }
    
    async def _get_reinforcement_prediction(self, model_data, features):
        """Get predictions from reinforcement learning models"""
        # Convert features to numpy array
        state = np.array([features])
        
        model = model_data.get('model', {})
        model_type = model.get('type', 'unknown')
        
        if model_type == 'q_learning':
            # Get model components
            q_table = model.get('q_table', [])
            kmeans = model.get('kmeans')
            unique_actions = model.get('unique_actions', [])
            
            if q_table is None or kmeans is None or unique_actions is None or len(unique_actions) == 0:
                # Invalid model components
                return {
                    'success_probability': 0.5,
                    'estimated_pnl': 0.0,
                    'recommended_direction': 'none',
                    'confidence': 0.0
                }
            
            # Predict cluster for state
            state_cluster = kmeans.predict(state)[0]
            
            # Get Q-values for this state
            q_values = q_table[state_cluster]
            
            # Get best action and its Q-value
            best_action_idx = np.argmax(q_values)
            best_q_value = q_values[best_action_idx]
            
            # Convert action index to actual action
            best_action = unique_actions[best_action_idx]
            
            # Convert Q-value to success probability using sigmoid
            success_probability = 1.0 / (1.0 + np.exp(-best_q_value))
            
            # Estimated PnL is the Q-value
            estimated_pnl = best_q_value
            
            # Direction is 'buy' or 'sell' based on the action
            if isinstance(best_action, (int, float)):
                recommended_direction = 'buy' if best_action > 0 else 'sell'
            else:
                recommended_direction = str(best_action)
            
            # Confidence is based on the relative difference between best and second-best Q-value
            if len(q_values) > 1:
                sorted_q_values = np.sort(q_values)[::-1]  # Sort in descending order
                second_best_q_value = sorted_q_values[1]
                confidence = (best_q_value - second_best_q_value) / max(abs(best_q_value), 1.0)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            else:
                confidence = 0.5
        
        elif model_type == 'dqn':
            # Get model components
            dqn_model = model.get('model')
            unique_actions = model.get('unique_actions', [])
            
            if dqn_model is None or unique_actions is None or len(unique_actions) == 0:
                # Invalid model components
                return {
                    'success_probability': 0.5,
                    'estimated_pnl': 0.0,
                    'recommended_direction': 'none',
                    'confidence': 0.0
                }
            
            # Get Q-values from DQN
            q_values = dqn_model.predict(state, verbose=0)[0]
            
            # Get best action and its Q-value
            best_action_idx = np.argmax(q_values)
            best_q_value = q_values[best_action_idx]
            
            # Convert action index to actual action
            best_action = unique_actions[best_action_idx]
            
            # Convert Q-value to success probability using sigmoid
            success_probability = 1.0 / (1.0 + np.exp(-best_q_value))
            
            # Estimated PnL is the Q-value
            estimated_pnl = best_q_value
            
            # Direction is 'buy' or 'sell' based on the action
            if isinstance(best_action, (int, float)):
                recommended_direction = 'buy' if best_action > 0 else 'sell'
            else:
                recommended_direction = str(best_action)
            
            # Confidence is based on the relative difference between best and second-best Q-value
            if len(q_values) > 1:
                sorted_q_values = np.sort(q_values)[::-1]  # Sort in descending order
                second_best_q_value = sorted_q_values[1]
                confidence = (best_q_value - second_best_q_value) / max(abs(best_q_value), 1.0)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            else:
                confidence = 0.5
        
        else:
            # Unknown model type
            return {
                'success_probability': 0.5,
                'estimated_pnl': 0.0,
                'recommended_direction': 'none',
                'confidence': 0.0
            }
        
        # Return predictions
        return {
            'success_probability': float(success_probability),
            'estimated_pnl': float(estimated_pnl),
            'recommended_direction': str(recommended_direction),
            'confidence': float(confidence)
        }
