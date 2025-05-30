#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Machine Learning Model Manager

This module provides sophisticated model lifecycle management functions
for the QuantumSpectre Elite Trading System, handling model registration,
versioning, storage, loading, and metadata management. It serves as the 
central hub for all ML model operations across the system.
"""

import os
import json
import pickle
import shutil
import datetime
import hashlib
import logging
import threading
import warnings
from functools import wraps
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import joblib
try:  # Optional dependencies
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional
    h5py = None
try:
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover - optional
    tf = None
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None

from common.utils import generate_uuid, create_directory_if_not_exists, compress_data
from common.exceptions import (
    ModelNotFoundError, ModelVersionError, 
    ModelRegistrationError, ModelLoadError,
    InvalidModelStateError, ModelSaveError
)
from common.metrics import MetricsCollector
from common.db_client import DatabaseClient, get_db_client

# Set up module logger
logger = logging.getLogger(__name__)

# Suppress TensorFlow and PyTorch warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)

class ModelManager:
    """
    Sophisticated model manager class that handles all aspects of ML model lifecycle:
    - Registration and versioning
    - Training coordination
    - Persistence and serialization
    - Loading and deployment
    - Metadata tracking
    - Performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model manager with configuration
        
        Args:
            config (dict): Configuration dictionary with the following keys:
                - models_dir (str): Directory to store model files
                - enable_gpu (bool): Whether to enable GPU acceleration
                - gpu_memory_limit (float): Fraction of GPU memory to use
                - use_mixed_precision (bool): Whether to use mixed precision
                - default_model_type (str): Default type of model
                - auto_optimization (bool): Whether to enable auto-optimization
                - feature_selection (bool): Whether to enable feature selection
                - hyperparameter_tuning (bool): Whether to enable hyperparameter tuning
                - db_connection (dict, optional): Database connection parameters
        """
        self.config = config
        self.models_dir = config.get('models_dir', os.path.join(os.path.dirname(__file__), 'saved_models'))
        self.model_registry = {}  # In-memory model registry
        self.active_models = {}   # Currently loaded models
        self.model_locks = {}     # Thread locks for model updates
        self.model_metrics = {}   # Performance metrics for models
        
        # Set up metrics collector
        self.metrics = MetricsCollector(
            namespace="ml_models",
            subsystem="model_manager"
        )
        
        # Set up database client if provided
        if 'db_connection' in config:
            self._db_params = config['db_connection']
        else:
            self._db_params = {}
        self.db_client = None
        
        # Configure GPU usage if enabled
        if config.get('enable_gpu', True):
            self._configure_gpu(
                memory_limit=config.get('gpu_memory_limit', 0.8),
                use_mixed_precision=config.get('use_mixed_precision', True)
            )
        
        # Ensure models directory exists
        create_directory_if_not_exists(self.models_dir)
        
        # Load existing model registry from disk
        self._load_model_registry()

        logger.info(f"Model Manager initialized with {len(self.model_registry)} registered models")

    async def initialize(self, db_connector: Optional[DatabaseClient] = None) -> None:
        """Asynchronously obtain a database client and ensure tables exist."""
        if db_connector is not None:
            self.db_client = db_connector
        if self.db_client is None and "db_connection" in self.config:
            self.db_client = DatabaseClient(self.config["db_connection"])
        if self.db_client is None:
            self.db_client = await get_db_client(**self._db_params)
        if getattr(self.db_client, "pool", None) is None:
            await self.db_client.initialize()
        await self.db_client.create_tables()
    
    def _configure_gpu(self, memory_limit: float = 0.8, use_mixed_precision: bool = True):
        """
        Configure GPU usage for TensorFlow and PyTorch
        
        Args:
            memory_limit (float): Fraction of GPU memory to use
            use_mixed_precision (bool): Whether to use mixed precision
        """
        # TensorFlow GPU configuration
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                    if memory_limit < 1.0:
                        tf.config.experimental.set_virtual_device_configuration(
                            device,
                            [tf.config.experimental.VirtualDeviceConfiguration(
                                memory_limit=int(memory_limit * 1024 * 1024 * 1024)  # Convert to bytes
                            )]
                        )
                
                # Enable mixed precision if requested
                if use_mixed_precision:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    
                logger.info(f"TensorFlow configured to use {len(physical_devices)} GPUs with memory limit {memory_limit}")
            else:
                logger.warning("No GPUs found for TensorFlow")
        except Exception as e:
            logger.warning(f"Error configuring TensorFlow GPU: {str(e)}")
        
        # PyTorch GPU configuration
        try:
            if torch.cuda.is_available():
                # Set default device
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                
                # Set memory limit using environment variable
                total_memory = torch.cuda.get_device_properties(0).total_memory
                max_memory = int(total_memory * memory_limit)
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{max_memory // (1024 * 1024)}'
                
                # Enable autocast for mixed precision if requested
                if use_mixed_precision:
                    # Will be used with context managers when running models
                    pass
                
                logger.info(f"PyTorch configured to use GPU with memory limit {memory_limit}")
            else:
                logger.warning("No GPUs found for PyTorch")
        except Exception as e:
            logger.warning(f"Error configuring PyTorch GPU: {str(e)}")
    
    def _load_model_registry(self):
        """
        Load the model registry from disk
        """
        registry_path = os.path.join(self.models_dir, 'model_registry.json')
        
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
                logger.info(f"Loaded model registry with {len(self.model_registry)} models")
            except Exception as e:
                logger.error(f"Error loading model registry: {str(e)}")
                self.model_registry = {}
        else:
            logger.info("No existing model registry found, creating new one")
            self.model_registry = {}
            self._save_model_registry()
    
    def _save_model_registry(self):
        """
        Save the model registry to disk
        """
        registry_path = os.path.join(self.models_dir, 'model_registry.json')
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
            logger.debug("Model registry saved to disk")
        except Exception as e:
            logger.error(f"Error saving model registry: {str(e)}")
    
    def register_model(self, 
                      model_id: str, 
                      model_type: str, 
                      asset: str, 
                      timeframe: str,
                      description: str = "",
                      metadata: Dict[str, Any] = None,
                      version: str = "1.0.0") -> Dict[str, Any]:
        """
        Register a new model in the registry
        
        Args:
            model_id (str): Unique identifier for the model
            model_type (str): Type of model (classification, regression, etc.)
            asset (str): Asset this model is for (e.g., 'BTC/USD')
            timeframe (str): Timeframe this model is for (e.g., '5m')
            description (str, optional): Description of the model
            metadata (dict, optional): Additional metadata for the model
            version (str, optional): Version of the model
            
        Returns:
            dict: Model information dictionary
        """
        if model_id in self.model_registry:
            raise ModelRegistrationError(f"Model with ID '{model_id}' already exists")
        
        # Create model lock
        self.model_locks[model_id] = threading.RLock()
        
        # Create model information
        model_info = {
            'model_id': model_id,
            'model_type': model_type,
            'asset': asset,
            'timeframe': timeframe,
            'description': description,
            'created_at': datetime.datetime.now().isoformat(),
            'updated_at': datetime.datetime.now().isoformat(),
            'versions': {
                version: {
                    'status': 'registered',
                    'created_at': datetime.datetime.now().isoformat(),
                    'path': None,
                    'metrics': {},
                    'metadata': metadata or {}
                }
            },
            'current_version': version,
            'metadata': metadata or {}
        }
        
        # Add to registry
        self.model_registry[model_id] = model_info
        
        # Save registry
        self._save_model_registry()
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_id)
        create_directory_if_not_exists(model_dir)
        
        logger.info(f"Registered new model '{model_id}' of type '{model_type}' for asset '{asset}', timeframe '{timeframe}'")
        return model_info
    
    def save_model(self, 
                  model_id: str, 
                  model_object: Any, 
                  metrics: Dict[str, float] = None,
                  metadata: Dict[str, Any] = None,
                  version: str = None) -> Dict[str, Any]:
        """
        Save a trained model to disk
        
        Args:
            model_id (str): Model identifier
            model_object (any): Trained model object
            metrics (dict, optional): Performance metrics for the model
            metadata (dict, optional): Additional metadata to save
            version (str, optional): Version to save (defaults to current version)
            
        Returns:
            dict: Updated model information
        """
        if model_id not in self.model_registry:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")
        
        with self.model_locks[model_id]:
            model_info = self.model_registry[model_id]
            
            # Determine version to use
            if version is None:
                version = model_info['current_version']
            elif version not in model_info['versions']:
                # Create new version entry
                model_info['versions'][version] = {
                    'status': 'registered',
                    'created_at': datetime.datetime.now().isoformat(),
                    'path': None,
                    'metrics': {},
                    'metadata': {}
                }
                model_info['current_version'] = version
            
            version_info = model_info['versions'][version]
            
            # Create model directory
            model_dir = os.path.join(self.models_dir, model_id, version)
            create_directory_if_not_exists(model_dir)
            
            # Determine model save path
            model_path = os.path.join(model_dir, 'model')
            
            try:
                # Save model based on type
                if isinstance(model_object, (tf.keras.Model, tf.keras.Sequential)):
                    # TensorFlow model
                    tf_path = f"{model_path}.h5"
                    model_object.save(tf_path, save_format='h5')
                    save_path = tf_path
                    framework = 'tensorflow'
                
                elif isinstance(model_object, torch.nn.Module):
                    # PyTorch model
                    torch_path = f"{model_path}.pt"
                    torch.save(model_object.state_dict(), torch_path)
                    save_path = torch_path
                    framework = 'pytorch'
                
                elif hasattr(model_object, 'sklearn_version'):
                    # Scikit-learn model
                    sklearn_path = f"{model_path}.joblib"
                    joblib.dump(model_object, sklearn_path, compress=3)
                    save_path = sklearn_path
                    framework = 'sklearn'
                
                elif hasattr(model_object, 'xgb'):
                    # XGBoost model
                    xgboost_path = f"{model_path}.xgb"
                    model_object.save_model(xgboost_path)
                    save_path = xgboost_path
                    framework = 'xgboost'
                
                elif hasattr(model_object, 'booster_'):
                    # LightGBM model
                    lgbm_path = f"{model_path}.lgbm"
                    model_object.booster_.save_model(lgbm_path)
                    save_path = lgbm_path
                    framework = 'lightgbm'
                
                elif hasattr(model_object, 'get_params'):
                    # Generic model with get_params (scikit-learn API)
                    generic_path = f"{model_path}.pkl"
                    with open(generic_path, 'wb') as f:
                        pickle.dump(model_object, f)
                    save_path = generic_path
                    framework = 'generic'
                
                else:
                    # Unknown model type, use pickle
                    pickle_path = f"{model_path}.pkl"
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(model_object, f)
                    save_path = pickle_path
                    framework = 'unknown'
                
                # Update version information
                version_info['status'] = 'trained'
                version_info['path'] = save_path
                version_info['updated_at'] = datetime.datetime.now().isoformat()
                version_info['framework'] = framework
                
                # Add metrics if provided
                if metrics:
                    version_info['metrics'] = {**version_info.get('metrics', {}), **metrics}
                
                # Add metadata if provided
                if metadata:
                    version_info['metadata'] = {**version_info.get('metadata', {}), **metadata}
                
                # Calculate model hash for integrity verification
                with open(save_path, 'rb') as f:
                    model_hash = hashlib.sha256(f.read()).hexdigest()
                version_info['hash'] = model_hash
                
                # Update model info in registry
                model_info['updated_at'] = datetime.datetime.now().isoformat()
                self.model_registry[model_id] = model_info
                
                # Save registry
                self._save_model_registry()
                
                # Save model information to JSON for easy access
                info_path = os.path.join(model_dir, 'info.json')
                with open(info_path, 'w') as f:
                    json.dump(version_info, f, indent=2)
                
                logger.info(f"Saved model '{model_id}' version '{version}' to {save_path}")
                
                return model_info
                
            except Exception as e:
                error_msg = f"Error saving model '{model_id}': {str(e)}"
                logger.error(error_msg)
                raise ModelSaveError(error_msg) from e
    
    def load_model(self, 
                  model_id: str, 
                  version: str = None, 
                  force_reload: bool = False) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a model from disk
        
        Args:
            model_id (str): Model identifier
            version (str, optional): Version to load (defaults to current version)
            force_reload (bool): Force reload even if model is already loaded
            
        Returns:
            tuple: (model_object, model_info)
        """
        if model_id not in self.model_registry:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")
        
        with self.model_locks[model_id]:
            model_info = self.model_registry[model_id]
            
            # Determine version to use
            if version is None:
                version = model_info['current_version']
            elif version not in model_info['versions']:
                raise ModelVersionError(f"Version '{version}' not found for model '{model_id}'")
            
            version_info = model_info['versions'][version]
            
            # Check if model is already loaded
            model_key = f"{model_id}_{version}"
            if not force_reload and model_key in self.active_models:
                logger.debug(f"Using cached model '{model_id}' version '{version}'")
                return self.active_models[model_key], model_info
            
            # Check if model file exists
            model_path = version_info.get('path')
            if not model_path or not os.path.exists(model_path):
                raise ModelLoadError(f"Model file for '{model_id}' version '{version}' not found")
            
            try:
                # Verify model hash for integrity
                with open(model_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                
                if 'hash' in version_info and file_hash != version_info['hash']:
                    logger.warning(f"Model hash mismatch for '{model_id}' version '{version}'")
                
                # Load model based on framework
                framework = version_info.get('framework', 'unknown')
                
                if framework == 'tensorflow' or model_path.endswith('.h5'):
                    # TensorFlow model
                    model_object = tf.keras.models.load_model(model_path)
                
                elif framework == 'pytorch' or model_path.endswith('.pt'):
                    # PyTorch model - need model class for this
                    # This is a placeholder - in real implementation, we'd need to know the model class
                    # and initialize it before loading the state dict
                    model_class = version_info.get('metadata', {}).get('model_class')
                    if model_class:
                        # This assumes model_class is a string that can be imported
                        parts = model_class.split('.')
                        module_name = '.'.join(parts[:-1])
                        class_name = parts[-1]
                        module = __import__(module_name, fromlist=[class_name])
                        model_class = getattr(module, class_name)
                        
                        # Initialize model with saved params if available
                        model_params = version_info.get('metadata', {}).get('model_params', {})
                        model_object = model_class(**model_params)
                        model_object.load_state_dict(torch.load(model_path))
                    else:
                        raise ModelLoadError(f"Cannot load PyTorch model without model_class information")
                
                elif framework == 'sklearn' or model_path.endswith('.joblib'):
                    # Scikit-learn model
                    model_object = joblib.load(model_path)
                
                elif framework == 'xgboost' or model_path.endswith('.xgb'):
                    # XGBoost model
                    import xgboost as xgb
                    model_object = xgb.Booster()
                    model_object.load_model(model_path)
                
                elif framework == 'lightgbm' or model_path.endswith('.lgbm'):
                    # LightGBM model
                    import lightgbm as lgb
                    model_object = lgb.Booster(model_file=model_path)
                
                elif model_path.endswith('.pkl'):
                    # Pickle model
                    with open(model_path, 'rb') as f:
                        model_object = pickle.load(f)
                
                else:
                    raise ModelLoadError(f"Unknown model format for '{model_id}' at {model_path}")
                
                # Store in active models cache
                self.active_models[model_key] = model_object
                
                # Update access timestamp
                version_info['last_accessed'] = datetime.datetime.now().isoformat()
                self._save_model_registry()
                
                logger.info(f"Loaded model '{model_id}' version '{version}' from {model_path}")
                
                return model_object, model_info
                
            except Exception as e:
                error_msg = f"Error loading model '{model_id}' version '{version}': {str(e)}"
                logger.error(error_msg)
                raise ModelLoadError(error_msg) from e
    
    def predict(self, 
               model_id: str, 
               data: Union[pd.DataFrame, np.ndarray], 
               version: str = None,
               batch_size: int = None) -> np.ndarray:
        """
        Make predictions using a model
        
        Args:
            model_id (str): Model identifier
            data (DataFrame or ndarray): Input data for prediction
            version (str, optional): Model version to use
            batch_size (int, optional): Batch size for prediction
            
        Returns:
            ndarray: Predictions
        """
        # Load model
        model, model_info = self.load_model(model_id, version)
        
        # Get model framework
        version = version or model_info['current_version']
        framework = model_info['versions'][version].get('framework', 'unknown')
        
        # Start prediction timer
        start_time = datetime.datetime.now()
        
        try:
            # Make predictions based on framework
            if framework == 'tensorflow':
                # TensorFlow model
                if batch_size:
                    predictions = model.predict(data, batch_size=batch_size)
                else:
                    predictions = model.predict(data)
            
            elif framework == 'pytorch':
                # PyTorch model
                model.eval()  # Set to evaluation mode
                with torch.no_grad():
                    # Convert data to torch tensor if needed
                    if not isinstance(data, torch.Tensor):
                        data = torch.tensor(data.values if isinstance(data, pd.DataFrame) else data, 
                                           dtype=torch.float32)
                    
                    # Move to GPU if available
                    if torch.cuda.is_available():
                        data = data.cuda()
                        model = model.cuda()
                    
                    # Predict in batches if batch_size is specified
                    if batch_size:
                        predictions = []
                        for i in range(0, len(data), batch_size):
                            batch = data[i:i+batch_size]
                            outputs = model(batch)
                            predictions.append(outputs.cpu().numpy())
                        predictions = np.vstack(predictions)
                    else:
                        outputs = model(data)
                        predictions = outputs.cpu().numpy()
            
            elif framework in ['sklearn', 'xgboost', 'lightgbm', 'generic', 'unknown']:
                # Scikit-learn, XGBoost, LightGBM, or generic model
                if hasattr(model, 'predict_proba'):
                    predictions = model.predict_proba(data)
                else:
                    predictions = model.predict(data)
            
            else:
                raise ValueError(f"Unsupported model framework: {framework}")
            
            # Calculate prediction time
            prediction_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.metrics.observe_value(
                name="prediction_time_seconds",
                value=prediction_time,
                labels={"model_id": model_id, "version": version}
            )
            self.metrics.increment_counter(
                name="prediction_count",
                labels={"model_id": model_id, "version": version}
            )
            
            logger.debug(f"Made predictions with model '{model_id}' version '{version}' in {prediction_time:.4f} seconds")
            
            return predictions
            
        except Exception as e:
            error_msg = f"Error making predictions with model '{model_id}': {str(e)}"
            logger.error(error_msg)
            
            # Update error metrics
            self.metrics.increment_counter(
                name="prediction_error_count",
                labels={"model_id": model_id, "version": version, "error_type": type(e).__name__}
            )
            
            raise RuntimeError(error_msg) from e
    
    def delete_model(self, model_id: str, version: str = None) -> bool:
        """
        Delete a model from the registry and disk
        
        Args:
            model_id (str): Model identifier
            version (str, optional): Version to delete (if None, delete all versions)
            
        Returns:
            bool: True if deletion was successful
        """
        if model_id not in self.model_registry:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")
        
        with self.model_locks[model_id]:
            model_info = self.model_registry[model_id]
            
            if version is None:
                # Delete all versions
                model_dir = os.path.join(self.models_dir, model_id)
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                
                # Remove from registry
                del self.model_registry[model_id]
                
                # Remove from active models
                keys_to_remove = [k for k in self.active_models if k.startswith(f"{model_id}_")]
                for key in keys_to_remove:
                    del self.active_models[key]
                
                # Remove lock
                del self.model_locks[model_id]
                
                logger.info(f"Deleted model '{model_id}' with all versions")
            
            else:
                # Delete specific version
                if version not in model_info['versions']:
                    raise ModelVersionError(f"Version '{version}' not found for model '{model_id}'")
                
                # Delete version directory
                version_dir = os.path.join(self.models_dir, model_id, version)
                if os.path.exists(version_dir):
                    shutil.rmtree(version_dir)
                
                # Remove from registry
                del model_info['versions'][version]
                
                # Update current version if needed
                if model_info['current_version'] == version:
                    if model_info['versions']:
                        # Set to latest version
                        model_info['current_version'] = max(model_info['versions'].keys())
                    else:
                        # No more versions
                        model_info['current_version'] = None
                
                # Remove from active models
                model_key = f"{model_id}_{version}"
                if model_key in self.active_models:
                    del self.active_models[model_key]
                
                logger.info(f"Deleted model '{model_id}' version '{version}'")
            
            # Save registry
            self._save_model_registry()
            
            return True
    
    def create_model_version(self, 
                           model_id: str, 
                           version: str,
                           copy_from_version: str = None) -> Dict[str, Any]:
        """
        Create a new version for an existing model
        
        Args:
            model_id (str): Model identifier
            version (str): New version to create
            copy_from_version (str, optional): Version to copy from
            
        Returns:
            dict: Updated model information
        """
        if model_id not in self.model_registry:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")
        
        with self.model_locks[model_id]:
            model_info = self.model_registry[model_id]
            
            if version in model_info['versions']:
                raise ModelVersionError(f"Version '{version}' already exists for model '{model_id}'")
            
            # Create new version entry
            model_info['versions'][version] = {
                'status': 'registered',
                'created_at': datetime.datetime.now().isoformat(),
                'path': None,
                'metrics': {},
                'metadata': {}
            }
            
            # Copy from existing version if specified
            if copy_from_version:
                if copy_from_version not in model_info['versions']:
                    raise ModelVersionError(f"Source version '{copy_from_version}' not found for model '{model_id}'")
                
                source_version = model_info['versions'][copy_from_version]
                source_path = source_version.get('path')
                
                if source_path and os.path.exists(source_path):
                    # Create version directory
                    version_dir = os.path.join(self.models_dir, model_id, version)
                    create_directory_if_not_exists(version_dir)
                    
                    # Copy model file
                    dest_path = os.path.join(version_dir, os.path.basename(source_path))
                    shutil.copy2(source_path, dest_path)
                    
                    # Update version info
                    model_info['versions'][version] = {
                        **source_version,  # Copy all attributes
                        'status': 'copied',
                        'created_at': datetime.datetime.now().isoformat(),
                        'path': dest_path,
                        'copied_from': copy_from_version
                    }
                    
                    logger.info(f"Created new version '{version}' for model '{model_id}' by copying from '{copy_from_version}'")
                else:
                    logger.info(f"Created new version '{version}' for model '{model_id}' (source version has no model file to copy)")
            else:
                # Create version directory
                version_dir = os.path.join(self.models_dir, model_id, version)
                create_directory_if_not_exists(version_dir)
                
                logger.info(f"Created new version '{version}' for model '{model_id}'")
            
            # Update current version
            model_info['current_version'] = version
            model_info['updated_at'] = datetime.datetime.now().isoformat()
            
            # Save registry
            self._save_model_registry()
            
            return model_info
    
    def update_model_metadata(self, 
                             model_id: str, 
                             metadata: Dict[str, Any],
                             version: str = None) -> Dict[str, Any]:
        """
        Update metadata for a model
        
        Args:
            model_id (str): Model identifier
            metadata (dict): Metadata to update
            version (str, optional): Version to update (if None, update model-level metadata)
            
        Returns:
            dict: Updated model information
        """
        if model_id not in self.model_registry:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")
        
        with self.model_locks[model_id]:
            model_info = self.model_registry[model_id]
            
            if version is None:
                # Update model-level metadata
                model_info['metadata'] = {**model_info.get('metadata', {}), **metadata}
                model_info['updated_at'] = datetime.datetime.now().isoformat()
            else:
                # Update version-level metadata
                if version not in model_info['versions']:
                    raise ModelVersionError(f"Version '{version}' not found for model '{model_id}'")
                
                version_info = model_info['versions'][version]
                version_info['metadata'] = {**version_info.get('metadata', {}), **metadata}
                version_info['updated_at'] = datetime.datetime.now().isoformat()
            
            # Save registry
            self._save_model_registry()
            
            logger.info(f"Updated metadata for model '{model_id}'" + 
                       (f" version '{version}'" if version else ""))
            
            return model_info
    
    def update_model_metrics(self, 
                            model_id: str, 
                            metrics: Dict[str, float],
                            version: str = None) -> Dict[str, Any]:
        """
        Update performance metrics for a model
        
        Args:
            model_id (str): Model identifier
            metrics (dict): Performance metrics to update
            version (str, optional): Version to update (defaults to current version)
            
        Returns:
            dict: Updated model information
        """
        if model_id not in self.model_registry:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")
        
        with self.model_locks[model_id]:
            model_info = self.model_registry[model_id]
            
            # Determine version to use
            if version is None:
                version = model_info['current_version']
            elif version not in model_info['versions']:
                raise ModelVersionError(f"Version '{version}' not found for model '{model_id}'")
            
            version_info = model_info['versions'][version]
            
            # Update metrics
            version_info['metrics'] = {**version_info.get('metrics', {}), **metrics}
            version_info['updated_at'] = datetime.datetime.now().isoformat()
            
            # Save registry
            self._save_model_registry()
            
            # Update model information file
            version_dir = os.path.join(self.models_dir, model_id, version)
            if os.path.exists(version_dir):
                info_path = os.path.join(version_dir, 'info.json')
                with open(info_path, 'w') as f:
                    json.dump(version_info, f, indent=2)
            
            logger.info(f"Updated metrics for model '{model_id}' version '{version}'")
            
            return model_info
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a model
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            dict: Model information
        """
        if model_id not in self.model_registry:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")
        
        return self.model_registry[model_id]
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered models
        
        Returns:
            dict: Mapping of model_id to model information
        """
        return self.model_registry
    
    def find_models(self, 
                   asset: str = None, 
                   timeframe: str = None,
                   model_type: str = None) -> List[Dict[str, Any]]:
        """
        Find models matching specific criteria
        
        Args:
            asset (str, optional): Asset to match
            timeframe (str, optional): Timeframe to match
            model_type (str, optional): Model type to match
            
        Returns:
            list: List of matching model information dictionaries
        """
        matches = []
        
        for model_id, model_info in self.model_registry.items():
            # Match asset if specified
            if asset and model_info['asset'] != asset:
                continue
            
            # Match timeframe if specified
            if timeframe and model_info['timeframe'] != timeframe:
                continue
            
            # Match model type if specified
            if model_type and model_info['model_type'] != model_type:
                continue
            
            matches.append(model_info)
        
        return matches
    
    def get_best_model(self, 
                      asset: str, 
                      timeframe: str,
                      model_type: str = None,
                      metric: str = 'accuracy') -> Tuple[str, str, Dict[str, Any]]:
        """
        Find the best performing model for a specific asset and timeframe
        
        Args:
            asset (str): Asset to match
            timeframe (str): Timeframe to match
            model_type (str, optional): Model type to match
            metric (str, optional): Metric to use for comparison
            
        Returns:
            tuple: (model_id, version, model_info) of the best model
        """
        # Find matching models
        matches = self.find_models(asset, timeframe, model_type)
        
        if not matches:
            raise ModelNotFoundError(f"No models found for asset='{asset}', timeframe='{timeframe}'"
                                    + (f", model_type='{model_type}'" if model_type else ""))
        
        # Find best model version based on metric
        best_model_id = None
        best_version = None
        best_score = float('-inf')
        best_info = None
        
        for model_info in matches:
            model_id = model_info['model_id']
            
            for version, version_info in model_info['versions'].items():
                if 'metrics' in version_info and metric in version_info['metrics']:
                    score = version_info['metrics'][metric]
                    
                    if score > best_score:
                        best_score = score
                        best_model_id = model_id
                        best_version = version
                        best_info = model_info
        
        if best_model_id is None:
            raise ModelNotFoundError(f"No models with metric '{metric}' found for asset='{asset}', timeframe='{timeframe}'"
                                    + (f", model_type='{model_type}'" if model_type else ""))
        
        return best_model_id, best_version, best_info
    
    def export_model(self, 
                    model_id: str, 
                    version: str = None,
                    export_format: str = 'default',
                    output_path: str = None) -> str:
        """
        Export a model to a specific format
        
        Args:
            model_id (str): Model identifier
            version (str, optional): Version to export (defaults to current version)
            export_format (str, optional): Format to export to ('default', 'onnx', 'pickle', 'tensorflow_js')
            output_path (str, optional): Path to export to (if None, generates one)
            
        Returns:
            str: Path to exported model
        """
        # Load model
        model, model_info = self.load_model(model_id, version)
        
        # Determine version
        if version is None:
            version = model_info['current_version']
        
        # Get model framework
        framework = model_info['versions'][version].get('framework', 'unknown')
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{model_id}_{version}_{timestamp}"
            output_path = os.path.join(self.models_dir, 'exports', filename)
            create_directory_if_not_exists(os.path.dirname(output_path))
        
        try:
            # Export based on format and framework
            if export_format == 'default':
                # Use native format for the model framework
                if framework == 'tensorflow':
                    tf.keras.models.save_model(model, output_path)
                    export_path = output_path
                
                elif framework == 'pytorch':
                    torch.save(model.state_dict(), f"{output_path}.pt")
                    export_path = f"{output_path}.pt"
                
                elif framework in ['sklearn', 'generic', 'unknown']:
                    with open(f"{output_path}.pkl", 'wb') as f:
                        pickle.dump(model, f)
                    export_path = f"{output_path}.pkl"
                
                elif framework == 'xgboost':
                    model.save_model(f"{output_path}.xgb")
                    export_path = f"{output_path}.xgb"
                
                elif framework == 'lightgbm':
                    model.booster_.save_model(f"{output_path}.lgbm")
                    export_path = f"{output_path}.lgbm"
                
                else:
                    raise ValueError(f"Unsupported model framework: {framework}")
            
            elif export_format == 'onnx':
                # Export to ONNX format
                if framework == 'tensorflow':
                    import tf2onnx
                    input_signature = [tf.TensorSpec([None, model.input_shape[1:]], tf.float32)]
                    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
                    onnx_path = f"{output_path}.onnx"
                    with open(onnx_path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                    export_path = onnx_path
                
                elif framework == 'pytorch':
                    import torch.onnx
                    # Get input shape from metadata
                    input_shape = model_info['versions'][version].get('metadata', {}).get('input_shape')
                    if not input_shape:
                        raise ValueError("Input shape not found in model metadata")
                    
                    # Create dummy input
                    dummy_input = torch.randn(input_shape)
                    onnx_path = f"{output_path}.onnx"
                    torch.onnx.export(model, dummy_input, onnx_path)
                    export_path = onnx_path
                
                elif framework in ['sklearn', 'xgboost', 'lightgbm']:
                    import skl2onnx
                    from skl2onnx.common.data_types import FloatTensorType
                    
                    # Get input shape from metadata
                    input_shape = model_info['versions'][version].get('metadata', {}).get('input_shape')
                    if not input_shape:
                        raise ValueError("Input shape not found in model metadata")
                    
                    initial_type = [('float_input', FloatTensorType([None, input_shape[-1]]))]
                    onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
                    
                    onnx_path = f"{output_path}.onnx"
                    with open(onnx_path, "wb") as f:
                        f.write(onnx_model.SerializeToString())
                    export_path = onnx_path
                
                else:
                    raise ValueError(f"ONNX export not supported for framework: {framework}")
            
            elif export_format == 'pickle':
                # Export to pickle format
                with open(f"{output_path}.pkl", 'wb') as f:
                    pickle.dump(model, f)
                export_path = f"{output_path}.pkl"
            
            elif export_format == 'tensorflow_js':
                # Export to TensorFlow.js format
                if framework == 'tensorflow':
                    import tensorflowjs as tfjs
                    tfjs_path = f"{output_path}_tfjs"
                    tfjs.converters.save_keras_model(model, tfjs_path)
                    export_path = tfjs_path
                else:
                    raise ValueError(f"TensorFlow.js export only supported for TensorFlow models")
            
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            logger.info(f"Exported model '{model_id}' version '{version}' to {export_path}")
            
            return export_path
            
        except Exception as e:
            error_msg = f"Error exporting model '{model_id}': {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def import_model(self, 
                    model_path: str, 
                    model_id: str,
                    model_type: str,
                    asset: str,
                    timeframe: str,
                    description: str = "",
                    metadata: Dict[str, Any] = None,
                    version: str = "1.0.0") -> Dict[str, Any]:
        """
        Import an existing model file into the registry
        
        Args:
            model_path (str): Path to model file
            model_id (str): Unique identifier for the model
            model_type (str): Type of model (classification, regression, etc.)
            asset (str): Asset this model is for (e.g., 'BTC/USD')
            timeframe (str): Timeframe this model is for (e.g., '5m')
            description (str, optional): Description of the model
            metadata (dict, optional): Additional metadata for the model
            version (str, optional): Version of the model
            
        Returns:
            dict: Model information dictionary
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        # Register model
        model_info = self.register_model(
            model_id=model_id,
            model_type=model_type,
            asset=asset,
            timeframe=timeframe,
            description=description,
            metadata=metadata,
            version=version
        )
        
        # Determine model framework
        if model_path.endswith('.h5'):
            framework = 'tensorflow'
        elif model_path.endswith('.pt'):
            framework = 'pytorch'
        elif model_path.endswith('.joblib'):
            framework = 'sklearn'
        elif model_path.endswith('.xgb'):
            framework = 'xgboost'
        elif model_path.endswith('.lgbm'):
            framework = 'lightgbm'
        elif model_path.endswith('.pkl'):
            framework = 'generic'
        else:
            framework = 'unknown'
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_id, version)
        create_directory_if_not_exists(model_dir)
        
        # Copy model file
        dest_path = os.path.join(model_dir, os.path.basename(model_path))
        shutil.copy2(model_path, dest_path)
        
        # Calculate model hash
        with open(dest_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Update version information
        version_info = model_info['versions'][version]
        version_info['status'] = 'imported'
        version_info['path'] = dest_path
        version_info['framework'] = framework
        version_info['hash'] = model_hash
        
        # Save registry
        self._save_model_registry()
        
        # Save model information to JSON
        info_path = os.path.join(model_dir, 'info.json')
        with open(info_path, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        logger.info(f"Imported model '{model_id}' version '{version}' from {model_path}")
        
        return model_info
    
    def set_current_version(self, model_id: str, version: str) -> Dict[str, Any]:
        """
        Set the current version for a model
        
        Args:
            model_id (str): Model identifier
            version (str): Version to set as current
            
        Returns:
            dict: Updated model information
        """
        if model_id not in self.model_registry:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry")
        
        with self.model_locks[model_id]:
            model_info = self.model_registry[model_id]
            
            if version not in model_info['versions']:
                raise ModelVersionError(f"Version '{version}' not found for model '{model_id}'")
            
            model_info['current_version'] = version
            model_info['updated_at'] = datetime.datetime.now().isoformat()
            
            # Save registry
            self._save_model_registry()
            
            logger.info(f"Set current version of model '{model_id}' to '{version}'")
            
            return model_info
    
    def cleanup_resources(self):
        """
        Clean up resources used by the model manager
        """
        # Save registry in case of changes
        self._save_model_registry()
        
        # Clear active models
        self.active_models.clear()
        
        # Release GPU memory
        try:
            # TensorFlow cleanup
            if 'tf' in sys.modules:
                tf.keras.backend.clear_session()
            
            # PyTorch cleanup
            if 'torch' in sys.modules and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error during GPU memory cleanup: {str(e)}")
        
        logger.info("Model manager resources cleaned up")

