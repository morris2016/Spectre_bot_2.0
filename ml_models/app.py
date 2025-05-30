#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
ML Models Service

This module provides the main service for machine learning model management,
training, and prediction in the QuantumSpectre Elite Trading System.
"""

import os
import time
import json
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

# Internal imports
from common.async_utils import run_in_threadpool
from common.exceptions import ModelNotFoundError, ModelTrainingError, ModelPredictionError
from common.metrics import MetricsCollector
from common.redis_client import RedisClient
from common.constants import ML_MODEL_TYPES, MODEL_REGISTRY_KEYS

# ML model imports
from .model_manager import ModelManager
from .training import ModelTrainer
from .prediction import ModelPredictor
from .feature_importance import FeatureImportanceAnalyzer

# Initialize logger
logger = logging.getLogger('quantumspectre.ml_models')

class MLModelService:
    """
    Main service for machine learning model management, training, and prediction.
    
    This service:
    1. Manages model lifecycle (creation, training, evaluation, deployment)
    2. Handles model versioning and storage
    3. Processes prediction requests
    4. Coordinates parallel model operations
    5. Implements model selection and ensemble strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ML model service.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self.model_manager = ModelManager(config.get('model_manager', {}))
        self.model_trainer = ModelTrainer(config.get('model_trainer', {}))
        self.model_predictor = ModelPredictor(config.get('model_predictor', {}))
        self.feature_analyzer = FeatureImportanceAnalyzer(config.get('feature_analyzer', {}))
        
        # Set up Redis client for communication
        redis_config = config.get('redis', {})
        self.redis = RedisClient(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            password=redis_config.get('password', None)
        )
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(
            max_workers=config.get('max_workers', os.cpu_count() * 2)
        )
        
        # Metrics collection
        self.metrics = MetricsCollector(namespace='ml_models')
        
        # Initialize event loop
        self.loop = None
        self.is_running = False
        
        # Subscription channels
        self.training_channel = config.get('training_channel', 'training_requests')
        self.prediction_channel = config.get('prediction_channel', 'prediction_requests')
        
        # Cache for frequently used models
        self.model_cache = {}
        self.cache_ttl = config.get('cache_ttl', 3600)  # Cache TTL in seconds
        self.max_cache_size = config.get('max_cache_size', 10)  # Max models in cache
        
        # Track model performance
        self.model_performance = {}
        
        logger.info("ML Models Service initialized")
        logger.info("ML Models Service could benefit from direct integration with asset-specific brain councils")
    
    async def start(self):
        """
        Start the ML model service, including subscriptions and background tasks.
        """
        self.loop = asyncio.get_running_loop()
        self.is_running = True
        
        # Initialize connections
        await self.redis.connect()
        await self.model_manager.initialize()
        
        # Start message consumers
        asyncio.create_task(self.training_consumer())
        asyncio.create_task(self.prediction_consumer())
        
        # Start background tasks
        asyncio.create_task(self.periodic_model_evaluation())
        asyncio.create_task(self.model_cache_management())
        
        logger.info("ML Models Service started")
    
    async def stop(self):
        """
        Stop the ML model service and clean up resources.
        """
        self.is_running = False
        
        # Close Redis connections
        await self.redis.close()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("ML Models Service stopped")
    
    async def training_consumer(self):
        """
        Consume and process model training requests.
        """
        logger.info(f"Starting training consumer on channel: {self.training_channel}")
        
        await self.redis.subscribe(self.training_channel)
        
        while self.is_running:
            try:
                message = await self.redis.get_message(self.training_channel)
                
                if message:
                    # Parse training request
                    try:
                        request = json.loads(message)
                        logger.info(f"Received training request: {request.get('id', 'unknown')}")
                        
                        # Process training request asynchronously
                        asyncio.create_task(self.process_training_request(request))
                        
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in training request: {message}")
                    
                else:
                    # No message, short sleep to prevent CPU spinning
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in training consumer: {str(e)}", exc_info=True)
                await asyncio.sleep(1)  # Sleep longer on error
    
    async def prediction_consumer(self):
        """
        Consume and process model prediction requests.
        """
        logger.info(f"Starting prediction consumer on channel: {self.prediction_channel}")
        
        await self.redis.subscribe(self.prediction_channel)
        
        while self.is_running:
            try:
                message = await self.redis.get_message(self.prediction_channel)
                
                if message:
                    # Parse prediction request
                    try:
                        request = json.loads(message)
                        logger.info(f"Received prediction request: {request.get('id', 'unknown')}")
                        
                        # Process prediction request asynchronously
                        asyncio.create_task(self.process_prediction_request(request))
                        
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in prediction request: {message}")
                    
                else:
                    # No message, short sleep to prevent CPU spinning
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in prediction consumer: {str(e)}", exc_info=True)
                await asyncio.sleep(1)  # Sleep longer on error
    
    async def process_training_request(self, request: Dict[str, Any]):
        """
        Process a model training request.
        
        Args:
            request: Training request dictionary with model specs and data references
        """
        request_id = request.get('id', 'unknown')
        response_channel = request.get('response_channel', 'training_responses')
        
        try:
            # Extract request details
            model_type = request.get('model_type')
            model_name = request.get('model_name')
            model_config = request.get('model_config', {})
            data_source = request.get('data_source', {})
            
            # Validate request
            if not model_type or not model_name:
                raise ValueError("Missing required fields: model_type, model_name")
            
            if model_type not in ML_MODEL_TYPES:
                raise ValueError(f"Invalid model_type: {model_type}")
            
            # Get training data
            training_data = await self._get_training_data(data_source)
            
            # Train model in thread pool to avoid blocking the event loop
            with self.metrics.measure_time(f'training_time.{model_type}'):
                model, metrics = await run_in_threadpool(
                    self.loop, 
                    self.model_trainer.train_model,
                    model_type=model_type,
                    model_name=model_name,
                    model_config=model_config,
                    training_data=training_data
                )
            
            # Save trained model
            await run_in_threadpool(
                self.loop,
                self.model_manager.save_model,
                model=model,
                model_name=model_name,
                model_type=model_type,
                metadata={'training_metrics': metrics}
            )
            
            # Update model performance tracking
            self.model_performance[model_name] = {
                'metrics': metrics,
                'timestamp': time.time()
            }
            
            # Add to cache if appropriate
            if model_type in ['classification', 'regression', 'time_series']:
                self._add_to_cache(model_name, model)
            
            # Send response
            response = {
                'id': request_id,
                'status': 'success',
                'model_name': model_name,
                'metrics': metrics
            }
            
            logger.info(f"Training completed for model {model_name} with metrics: {metrics}")
            logger.info(f"Model {model_name} could provide more value with asset-specific council integration")
            
        except Exception as e:
            logger.error(f"Error training model for request {request_id}: {str(e)}", exc_info=True)
            
            # Send error response
            response = {
                'id': request_id,
                'status': 'error',
                'error': str(e)
            }
        
        # Publish response
        await self.redis.publish(response_channel, json.dumps(response))
    
    async def process_prediction_request(self, request: Dict[str, Any]):
        """
        Process a model prediction request.
        
        Args:
            request: Prediction request dictionary with model name and input data
        """
        request_id = request.get('id', 'unknown')
        response_channel = request.get('response_channel', 'prediction_responses')
        
        try:
            # Extract request details
            model_name = request.get('model_name')
            ensemble_config = request.get('ensemble_config', None)
            data = request.get('data', {})
            
            # Validate request
            if not model_name and not ensemble_config:
                raise ValueError("Missing required field: model_name or ensemble_config")
            
            # Load data
            input_data = self._prepare_prediction_data(data)
            
            # Get predictions - either from a single model or an ensemble
            with self.metrics.measure_time('prediction_time'):
                if ensemble_config:
                    predictions = await self._get_ensemble_predictions(
                        ensemble_config, input_data)
                else:
                    predictions = await self._get_single_model_predictions(
                        model_name, input_data)
            
            # Send response
            response = {
                'id': request_id,
                'status': 'success',
                'predictions': predictions
            }
            
            logger.info(f"Prediction completed for request {request_id}")
            
        except Exception as e:
            logger.error(f"Error generating predictions for request {request_id}: {str(e)}", 
                       exc_info=True)
            
            # Send error response
            response = {
                'id': request_id,
                'status': 'error',
                'error': str(e)
            }
        
        # Publish response
        await self.redis.publish(response_channel, json.dumps(response))
    
    async def _get_training_data(self, data_source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve training data based on the specified data source.
        
        Args:
            data_source: Data source configuration
            
        Returns:
            Dictionary containing training data
        """
        source_type = data_source.get('type', 'redis')
        
        if source_type == 'redis':
            # Get data from Redis
            key = data_source.get('key')
            if not key:
                raise ValueError("Missing 'key' in Redis data source")
            
            data_json = await self.redis.get(key)
            if not data_json:
                raise ValueError(f"No data found in Redis for key: {key}")
            
            return json.loads(data_json)
            
        elif source_type == 'feature_service':
            # Request data from feature service
            request_channel = data_source.get('request_channel', 'feature_requests')
            response_channel = f"feature_responses:{time.time()}"
            
            # Build request
            request = {
                'id': f"train_{time.time()}",
                'response_channel': response_channel,
                'features': data_source.get('features', []),
                'filters': data_source.get('filters', {}),
                'time_range': data_source.get('time_range', {})
            }
            
            # Subscribe to response channel
            await self.redis.subscribe(response_channel)
            
            # Send request
            await self.redis.publish(request_channel, json.dumps(request))
            
            # Wait for response with timeout
            timeout = data_source.get('timeout', 60)
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                message = await self.redis.get_message(response_channel)
                
                if message:
                    # Unsubscribe from response channel
                    await self.redis.unsubscribe(response_channel)
                    
                    # Parse response
                    response = json.loads(message)
                    
                    if response.get('status') == 'error':
                        raise ValueError(f"Feature service error: {response.get('error')}")
                    
                    return response.get('data', {})
                
                await asyncio.sleep(0.1)
            
            # Timeout reached
            await self.redis.unsubscribe(response_channel)
            raise TimeoutError("Timeout waiting for feature service response")
            
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")
    
    def _prepare_prediction_data(self, data: Dict[str, Any]) -> Any:
        """
        Prepare input data for prediction.
        
        Args:
            data: Input data in various formats
            
        Returns:
            Prepared data in the format expected by models
        """
        data_format = data.get('format', 'dict')
        
        if data_format == 'dict':
            # Dictionary format, ready to use
            return data.get('values', {})
            
        elif data_format == 'dataframe':
            # Convert to DataFrame
            df_data = data.get('values', [])
            columns = data.get('columns', [])
            
            if columns:
                return pd.DataFrame(df_data, columns=columns)
            else:
                return pd.DataFrame(df_data)
                
        elif data_format == 'array':
            # Convert to numpy array
            return np.array(data.get('values', []))
            
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
    
    async def _get_single_model_predictions(
        self, model_name: str, input_data: Any) -> Dict[str, Any]:
        """
        Get predictions from a single model.
        
        Args:
            model_name: Name of the model to use
            input_data: Prepared input data
            
        Returns:
            Prediction results
        """
        # Check if model is in cache
        model = self._get_from_cache(model_name)
        
        # If not in cache, load from storage
        if model is None:
            try:
                model = await run_in_threadpool(
                    self.loop,
                    self.model_manager.load_model,
                    model_name=model_name
                )
                
                # Add to cache
                self._add_to_cache(model_name, model)
                
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}", exc_info=True)
                raise ModelNotFoundError(f"Could not load model {model_name}: {str(e)}")
        
        # Generate predictions
        try:
            predictions = await run_in_threadpool(
                self.loop,
                self.model_predictor.predict,
                model=model,
                data=input_data
            )
            
            # Convert numpy types to Python native types for JSON serialization
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()
            elif isinstance(predictions, dict):
                for k, v in predictions.items():
                    if isinstance(v, np.ndarray):
                        predictions[k] = v.tolist()
                    elif isinstance(v, np.number):
                        predictions[k] = v.item()
            
            return {
                'model': model_name,
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Error generating predictions with model {model_name}: {str(e)}", 
                       exc_info=True)
            raise ModelPredictionError(f"Prediction error with model {model_name}: {str(e)}")
    
    async def _get_ensemble_predictions(
        self, ensemble_config: Dict[str, Any], input_data: Any) -> Dict[str, Any]:
        """
        Get predictions from an ensemble of models.
        
        Args:
            ensemble_config: Ensemble configuration with models and weights
            input_data: Prepared input data
            
        Returns:
            Combined prediction results
        """
        ensemble_type = ensemble_config.get('type', 'weighted_average')
        models = ensemble_config.get('models', [])
        weights = ensemble_config.get('weights', [])
        
        if not models:
            raise ValueError("No models specified in ensemble configuration")
        
        # Default to equal weights if not provided
        if not weights:
            weights = [1.0 / len(models)] * len(models)
        elif len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Get predictions from each model in parallel
        prediction_tasks = []
        for model_name in models:
            task = self._get_single_model_predictions(model_name, input_data)
            prediction_tasks.append(task)
        
        model_predictions = await asyncio.gather(*prediction_tasks)
        
        # Combine predictions based on ensemble type
        if ensemble_type == 'weighted_average':
            return self._combine_weighted_average(model_predictions, weights)
        elif ensemble_type == 'majority_vote':
            return self._combine_majority_vote(model_predictions, weights)
        elif ensemble_type == 'stacking':
            return await self._combine_stacking(ensemble_config, model_predictions, input_data)
        else:
            raise ValueError(f"Unsupported ensemble type: {ensemble_type}")
    
    def _combine_weighted_average(
        self, model_predictions: List[Dict[str, Any]], weights: List[float]) -> Dict[str, Any]:
        """
        Combine predictions using weighted average.
        
        Args:
            model_predictions: List of prediction results from individual models
            weights: Model weights
            
        Returns:
            Combined prediction result
        """
        combined = {}
        
        # Extract raw predictions
        raw_predictions = []
        model_names = []
        
        for pred, weight in zip(model_predictions, weights):
            model_names.append(pred['model'])
            raw_predictions.append(pred['predictions'])
        
        # Handle different prediction formats
        if all(isinstance(p, list) for p in raw_predictions):
            # Arrays/lists of values - weighted average each element
            combined_preds = []
            
            # Ensure all predictions have the same length
            pred_lengths = [len(p) for p in raw_predictions]
            if len(set(pred_lengths)) > 1:
                raise ValueError("All models must produce the same number of predictions")
            
            # Weighted average each prediction
            for i in range(pred_lengths[0]):
                weighted_sum = sum(p[i] * w for p, w in zip(raw_predictions, weights))
                combined_preds.append(weighted_sum)
                
            combined['predictions'] = combined_preds
            
        elif all(isinstance(p, dict) for p in raw_predictions):
            # Dictionary format - weighted average each key
            combined_preds = {}
            
            # Get all keys
            all_keys = set()
            for p in raw_predictions:
                all_keys.update(p.keys())
            
            # Weighted average each key
            for key in all_keys:
                values = []
                key_weights = []
                
                for p, w in zip(raw_predictions, weights):
                    if key in p:
                        values.append(p[key])
                        key_weights.append(w)
                
                # Normalize weights for this key
                key_weight_sum = sum(key_weights)
                if key_weight_sum > 0:
                    key_weights = [w / key_weight_sum for w in key_weights]
                    
                    # Calculate weighted average
                    weighted_sum = sum(v * w for v, w in zip(values, key_weights))
                    combined_preds[key] = weighted_sum
            
            combined['predictions'] = combined_preds
            
        else:
            # Mixed formats or unsupported format
            raise ValueError("Inconsistent prediction formats from ensemble models")
        
        combined['ensemble'] = {
            'type': 'weighted_average',
            'models': model_names,
            'weights': weights
        }
        
        return combined
    
    def _combine_majority_vote(
        self, model_predictions: List[Dict[str, Any]], weights: List[float]) -> Dict[str, Any]:
        """
        Combine predictions using weighted majority vote.
        
        Args:
            model_predictions: List of prediction results from individual models
            weights: Model weights
            
        Returns:
            Combined prediction result
        """
        combined = {}
        
        # Extract raw predictions
        raw_predictions = []
        model_names = []
        
        for pred, weight in zip(model_predictions, weights):
            model_names.append(pred['model'])
            raw_predictions.append(pred['predictions'])
        
        # Handle different prediction formats
        if all(isinstance(p, list) for p in raw_predictions):
            # Arrays/lists of values - vote for each element
            combined_preds = []
            
            # Ensure all predictions have the same length
            pred_lengths = [len(p) for p in raw_predictions]
            if len(set(pred_lengths)) > 1:
                raise ValueError("All models must produce the same number of predictions")
            
            # Vote on each prediction
            for i in range(pred_lengths[0]):
                # Count votes for each unique prediction
                votes = {}
                for p, w in zip(raw_predictions, weights):
                    val = p[i]
                    votes[val] = votes.get(val, 0) + w
                
                # Find the value with the most votes
                winner = max(votes.items(), key=lambda x: x[1])
                combined_preds.append(winner[0])
                
            combined['predictions'] = combined_preds
            
        elif all(isinstance(p, dict) for p in raw_predictions):
            # Dictionary format - vote for each key
            combined_preds = {}
            
            # Get all keys
            all_keys = set()
            for p in raw_predictions:
                all_keys.update(p.keys())
            
            # Vote on each key
            for key in all_keys:
                votes = {}
                
                for p, w in zip(raw_predictions, weights):
                    if key in p:
                        val = p[key]
                        votes[val] = votes.get(val, 0) + w
                
                if votes:
                    # Find the value with the most votes
                    winner = max(votes.items(), key=lambda x: x[1])
                    combined_preds[key] = winner[0]
            
            combined['predictions'] = combined_preds
            
        else:
            # Mixed formats or unsupported format
            raise ValueError("Inconsistent prediction formats from ensemble models")
        
        combined['ensemble'] = {
            'type': 'majority_vote',
            'models': model_names,
            'weights': weights
        }
        
        return combined
    
    async def _combine_stacking(
        self, ensemble_config: Dict[str, Any], 
        model_predictions: List[Dict[str, Any]], 
        input_data: Any) -> Dict[str, Any]:
        """
        Combine predictions using a meta-model (stacking).
        
        Args:
            ensemble_config: Ensemble configuration
            model_predictions: List of prediction results from individual models
            input_data: Original input data
            
        Returns:
            Combined prediction result
        """
        meta_model_name = ensemble_config.get('meta_model')
        if not meta_model_name:
            raise ValueError("Meta model name required for stacking ensemble")
        
        # Extract raw predictions as features for meta-model
        base_predictions = []
        model_names = []
        
        for pred in model_predictions:
            model_names.append(pred['model'])
            
            # Convert predictions to feature vector
            if isinstance(pred['predictions'], list):
                base_predictions.append(pred['predictions'])
            elif isinstance(pred['predictions'], dict):
                # Flatten dictionary to vector
                # Use consistent ordering of keys
                keys = sorted(pred['predictions'].keys())
                values = [pred['predictions'][k] for k in keys]
                base_predictions.append(values)
        
        # Transpose to get features as columns
        meta_features = list(zip(*base_predictions))
        
        # Prepare meta-model input
        if isinstance(input_data, dict):
            # For dictionary input, create new dict with original features and model predictions
            meta_input = input_data.copy()
            
            # Add base model predictions as additional features
            for i, model_name in enumerate(model_names):
                for j, pred in enumerate(base_predictions[i]):
                    meta_input[f"{model_name}_pred_{j}"] = pred
                    
        elif isinstance(input_data, pd.DataFrame):
            # For DataFrame input, add predictions as new columns
            meta_input = input_data.copy()
            
            for i, model_name in enumerate(model_names):
                if len(base_predictions[i]) == len(meta_input):
                    meta_input[f"{model_name}_pred"] = base_predictions[i]
                else:
                    # Handle multiple predictions per row
                    for j in range(len(base_predictions[i])):
                        meta_input[f"{model_name}_pred_{j}"] = base_predictions[i][j]
                        
        elif isinstance(input_data, np.ndarray):
            # For numpy input, concatenate predictions as additional features
            base_preds_array = np.array(base_predictions).T
            meta_input = np.hstack((input_data, base_preds_array))
            
        else:
            # Cannot combine, just use the base predictions
            meta_input = np.array(base_predictions).T
        
        # Get meta-model prediction
        meta_result = await self._get_single_model_predictions(meta_model_name, meta_input)
        
        # Format final result
        result = {
            'predictions': meta_result['predictions'],
            'ensemble': {
                'type': 'stacking',
                'base_models': model_names,
                'meta_model': meta_model_name
            }
        }
        
        return result
    
    def _add_to_cache(self, model_name: str, model):
        """Add model to cache with timestamp."""
        # Enforce cache size limit
        if len(self.model_cache) >= self.max_cache_size:
            # Remove oldest model
            oldest_model = min(self.model_cache.items(), key=lambda x: x[1]['timestamp'])
            del self.model_cache[oldest_model[0]]
        
        # Add to cache
        self.model_cache[model_name] = {
            'model': model,
            'timestamp': time.time()
        }
    
    def _get_from_cache(self, model_name: str):
        """Get model from cache if available and not expired."""
        if model_name in self.model_cache:
            cache_entry = self.model_cache[model_name]
            
            # Check if entry is still valid
            if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                # Update timestamp to reflect recent use
                cache_entry['timestamp'] = time.time()
                return cache_entry['model']
            else:
                # Entry expired, remove from cache
                del self.model_cache[model_name]
        
        return None
    
    async def periodic_model_evaluation(self):
        """
        Periodically evaluate models on recent data to track performance drift.
        """
        evaluation_interval = self.config.get('evaluation_interval', 3600)  # 1 hour
        
        while self.is_running:
            try:
                # Wait for interval
                await asyncio.sleep(evaluation_interval)
                
                # Get list of active models
                active_models = await run_in_threadpool(
                    self.loop,
                    self.model_manager.list_models
                )
                
                # Skip if no models to evaluate
                if not active_models:
                    continue
                
                logger.info(f"Starting periodic evaluation of {len(active_models)} models")
                
                # Get evaluation data
                try:
                    eval_data = await self._get_training_data(
                        self.config.get('evaluation_data_source', {
                            'type': 'feature_service',
                            'time_range': {'last': '1d'}
                        })
                    )
                except Exception as e:
                    logger.error(f"Failed to get evaluation data: {str(e)}", exc_info=True)
                    continue
                
                # Evaluate each model
                for model_info in active_models:
                    model_name = model_info['name']
                    model_type = model_info['type']
                    
                    try:
                        # Load model
                        model = await run_in_threadpool(
                            self.loop,
                            self.model_manager.load_model,
                            model_name=model_name
                        )
                        
                        # Evaluate model
                        metrics = await run_in_threadpool(
                            self.loop,
                            self.model_trainer.evaluate_model,
                            model=model,
                            model_type=model_type,
                            data=eval_data
                        )
                        
                        # Update performance tracking
                        self.model_performance[model_name] = {
                            'metrics': metrics,
                            'timestamp': time.time()
                        }
                        
                        # Check for performance drift
                        if model_info.get('last_metrics'):
                            drift = self._calculate_performance_drift(
                                model_info['last_metrics'], metrics)
                            
                            if drift > self.config.get('drift_threshold', 0.2):
                                logger.warning(
                                    f"Performance drift detected for model {model_name}: {drift}")
                                
                                # Publish alert
                                await self.redis.publish(
                                    'model_alerts',
                                    json.dumps({
                                        'type': 'drift',
                                        'model_name': model_name,
                                        'drift': drift,
                                        'old_metrics': model_info['last_metrics'],
                                        'new_metrics': metrics
                                    })
                                )
                        
                        # Update model metadata
                        await run_in_threadpool(
                            self.loop,
                            self.model_manager.update_model_metadata,
                            model_name=model_name,
                            metadata={'last_metrics': metrics, 'last_eval': time.time()}
                        )
                        
                        logger.info(f"Evaluated model {model_name}: {metrics}")
                        
                    except Exception as e:
                        logger.error(
                            f"Error evaluating model {model_name}: {str(e)}", exc_info=True)
                
                logger.info("Completed periodic model evaluation")
                
            except Exception as e:
                logger.error(f"Error in periodic model evaluation: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Sleep longer on error
    
    async def model_cache_management(self):
        """
        Periodically clean up expired cache entries and preload frequently used models.
        """
        cache_check_interval = self.config.get('cache_check_interval', 300)  # 5 minutes
        
        while self.is_running:
            try:
                # Wait for interval
                await asyncio.sleep(cache_check_interval)
                
                # Remove expired entries
                current_time = time.time()
                expired_models = [
                    name for name, entry in self.model_cache.items()
                    if current_time - entry['timestamp'] > self.cache_ttl
                ]
                
                for name in expired_models:
                    del self.model_cache[name]
                
                # Preload frequently used models
                if len(self.model_cache) < self.max_cache_size:
                    # Get list of active models
                    active_models = await run_in_threadpool(
                        self.loop,
                        self.model_manager.list_models,
                        sort_by='last_used',
                        limit=self.max_cache_size
                    )
                    
                    # Preload models not already in cache
                    for model_info in active_models:
                        model_name = model_info['name']
                        
                        if (model_name not in self.model_cache and 
                                len(self.model_cache) < self.max_cache_size):
                            try:
                                model = await run_in_threadpool(
                                    self.loop,
                                    self.model_manager.load_model,
                                    model_name=model_name
                                )
                                
                                self._add_to_cache(model_name, model)
                                logger.debug(f"Preloaded model {model_name} into cache")
                                
                            except Exception as e:
                                logger.warning(
                                    f"Failed to preload model {model_name}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error in model cache management: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Sleep longer on error
    
    def _calculate_performance_drift(self, old_metrics: Dict[str, float], 
                                   new_metrics: Dict[str, float]) -> float:
        """
        Calculate performance drift between old and new metrics.
        
        Args:
            old_metrics: Previous performance metrics
            new_metrics: New performance metrics
            
        Returns:
            Drift score (0-1), higher means more drift
        """
        # Focus on common metrics
        common_metrics = set(old_metrics.keys()).intersection(new_metrics.keys())
        
        if not common_metrics:
            return 0.0
        
        # Calculate relative changes for each metric
        relative_changes = []
        
        for metric in common_metrics:
            old_value = old_metrics[metric]
            new_value = new_metrics[metric]
            
            # Skip if both values are zero
            if old_value == 0 and new_value == 0:
                continue
            
            # Handle division by zero
            if old_value == 0:
                relative_changes.append(1.0)  # Maximum change
            else:
                relative_change = abs((new_value - old_value) / old_value)
                relative_changes.append(min(relative_change, 1.0))  # Cap at 1.0
        
        # Return average change
        if not relative_changes:
            return 0.0
        
        return sum(relative_changes) / len(relative_changes)

# Create singleton instance
_instance = None

def get_ml_model_service(config: Dict[str, Any] = None) -> MLModelService:
    """
    Get the ML model service singleton instance.
    
    Args:
        config: Optional configuration (only used for first call)
        
    Returns:
        MLModelService instance
    """
    global _instance
    
    if _instance is None:
        if config is None:
            raise ValueError("Configuration required for first initialization")
        
        _instance = MLModelService(config)
    
    return _instance
