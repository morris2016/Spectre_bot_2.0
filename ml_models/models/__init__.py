#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Machine Learning Models Package

This package contains all the machine learning model implementations and abstractions
used by the QuantumSpectre Elite Trading System for predictive modeling and analysis.
"""

import importlib
import inspect
from typing import Dict, List, Type, Any, Optional, Set, Tuple
import logging

from common.logger import get_logger
from common.constants import MODEL_REGISTRY_CONFIG

logger = get_logger(__name__)

# Global model registry
MODEL_REGISTRY = {}
MODEL_CATEGORIES = {
    "regression": [],
    "classification": [],
    "time_series": [],
    "deep_learning": [],
    "ensemble": []
}

class ModelRegistrationError(Exception):
    """Exception raised for errors in model registration process."""
    pass

class ModelCategoryNotFoundError(Exception):
    """Exception raised when an invalid model category is specified."""
    pass

class ModelNotFoundError(Exception):
    """Exception raised when a model is not found in the registry."""
    pass

def register_model(name: str, 
                  model_class: Type,
                  categories: List[str] = None,
                  description: str = "",
                  parameters: Dict[str, Any] = None,
                  requires_gpu: bool = False,
                  supports_online_learning: bool = False,
                  version: str = "1.0.0") -> None:
    """
    Register a model in the global model registry.
    
    Args:
        name: Unique name for the model
        model_class: The model class
        categories: List of categories the model belongs to
        description: Description of the model
        parameters: Default parameters for the model
        requires_gpu: Whether the model requires GPU for optimal performance
        supports_online_learning: Whether the model supports online learning
        version: Model version string
    
    Raises:
        ModelRegistrationError: If registration fails
    """
    if not categories:
        # Infer categories based on class name if not provided
        categories = []
        class_name = model_class.__name__.lower()
        if "regressor" in class_name:
            categories.append("regression")
        if "classifier" in class_name:
            categories.append("classification")
        if any(x in class_name for x in ["rnn", "lstm", "gru", "cnn", "transformer"]):
            categories.append("deep_learning")
        if any(x in class_name for x in ["time", "series", "forecast"]):
            categories.append("time_series")
        if any(x in class_name for x in ["ensemble", "boost", "stack", "bag"]):
            categories.append("ensemble")
        
        # If we couldn't infer any category, use "other"
        if not categories:
            categories = ["other"]
    
    # Validate categories
    for category in categories:
        if category not in MODEL_CATEGORIES and category != "other":
            raise ModelCategoryNotFoundError(f"Invalid model category: {category}")
    
    # Check if model already exists
    if name in MODEL_REGISTRY:
        logger.warning(f"Model '{name}' already registered, overwriting")
    
    # Register model
    MODEL_REGISTRY[name] = {
        "class": model_class,
        "categories": categories,
        "description": description,
        "parameters": parameters or {},
        "requires_gpu": requires_gpu,
        "supports_online_learning": supports_online_learning,
        "version": version
    }
    
    # Add to category lists
    for category in categories:
        if category != "other":
            if name not in MODEL_CATEGORIES[category]:
                MODEL_CATEGORIES[category].append(name)
    
    logger.info(f"Registered model '{name}' in categories {categories}")

def get_model_class(name: str) -> Type:
    """
    Get a model class by name from the registry.
    
    Args:
        name: Name of the model
        
    Returns:
        The model class
        
    Raises:
        ModelNotFoundError: If model is not found
    """
    if name not in MODEL_REGISTRY:
        raise ModelNotFoundError(f"Model '{name}' not found in registry")
    
    return MODEL_REGISTRY[name]["class"]

def get_model_info(name: str) -> Dict[str, Any]:
    """
    Get model information by name.
    
    Args:
        name: Name of the model
        
    Returns:
        Dictionary with model information
        
    Raises:
        ModelNotFoundError: If model is not found
    """
    if name not in MODEL_REGISTRY:
        raise ModelNotFoundError(f"Model '{name}' not found in registry")
    
    return MODEL_REGISTRY[name]

def get_models_by_category(category: str) -> List[str]:
    """
    Get list of model names in a specific category.
    
    Args:
        category: Model category name
        
    Returns:
        List of model names
        
    Raises:
        ModelCategoryNotFoundError: If category is not valid
    """
    if category not in MODEL_CATEGORIES:
        raise ModelCategoryNotFoundError(f"Invalid model category: {category}")
    
    return MODEL_CATEGORIES[category]

def get_all_models() -> Dict[str, Dict[str, Any]]:
    """
    Get all registered models.
    
    Returns:
        Dictionary of all models and their information
    """
    return MODEL_REGISTRY

def get_model_categories() -> List[str]:
    """
    Get all available model categories.
    
    Returns:
        List of model category names
    """
    return list(MODEL_CATEGORIES.keys())

def init_model_registry() -> None:
    """
    Initialize the model registry by automatically registering models from submodules.
    """
    from ml_models.models import regression, classification, time_series, ensemble

    modules = [regression, classification, time_series]

    try:
        from ml_models.models import deep_learning
        modules.append(deep_learning)
    except Exception as e:  # pragma: no cover - optional dependency
        logger.warning(f"Deep learning models not available: {e}")

    modules.append(ensemble)
    
    # Number of models registered
    count = 0
    
    # Process each module
    for module in modules:
        # Find all classes in the module
        for name, obj in inspect.getmembers(module):
            # Check if it's a class defined in this module (not imported)
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                try:
                    # Check if class has a META attribute with registration info
                    if hasattr(obj, 'META'):
                        meta = obj.META
                        register_model(
                            name=meta.get('name', name),
                            model_class=obj,
                            categories=meta.get('categories', None),
                            description=meta.get('description', ''),
                            parameters=meta.get('parameters', {}),
                            requires_gpu=meta.get('requires_gpu', False),
                            supports_online_learning=meta.get('supports_online_learning', False),
                            version=meta.get('version', '1.0.0')
                        )
                        count += 1
                except Exception as e:
                    logger.error(f"Error registering model {name}: {str(e)}")
    
    logger.info(f"Model registry initialized with {count} models")

def create_model_instance(name: str, **kwargs) -> Any:
    """
    Create an instance of a model from the registry with given parameters.
    
    Args:
        name: Name of the model
        **kwargs: Parameters to pass to the model constructor
        
    Returns:
        Instantiated model
        
    Raises:
        ModelNotFoundError: If model is not found
    """
    # Get model class
    model_class = get_model_class(name)
    
    # Get default parameters
    default_params = MODEL_REGISTRY[name].get("parameters", {}).copy()
    
    # Update with provided parameters
    default_params.update(kwargs)
    
    # Create instance
    try:
        model = model_class(**default_params)
        logger.debug(f"Created model instance for '{name}'")
        return model
    except Exception as e:
        logger.error(f"Error creating model instance for '{name}': {str(e)}")
        raise

def get_recommended_models(task: str, 
                          data_size: str = "medium", 
                          complexity: str = "medium",
                          supports_online_learning: bool = False) -> List[str]:
    """
    Get recommended models for a specific task and constraints.
    
    Args:
        task: Task type ('regression', 'classification', 'time_series')
        data_size: Size of the dataset ('small', 'medium', 'large')
        complexity: Desired model complexity ('low', 'medium', 'high')
        supports_online_learning: Whether online learning is required
        
    Returns:
        List of recommended model names
    """
    # Validate task
    if task not in MODEL_CATEGORIES:
        raise ModelCategoryNotFoundError(f"Invalid task type: {task}")
    
    # Get models for the task
    models = get_models_by_category(task)
    
    # Filter based on other criteria
    results = []
    for model_name in models:
        model_info = get_model_info(model_name)
        
        # Filter by online learning support if required
        if supports_online_learning and not model_info["supports_online_learning"]:
            continue
            
        # Filter by complexity and data size based on recommendations
        recommendations = MODEL_REGISTRY_CONFIG.get("recommendations", {})
        model_recommendations = recommendations.get(model_name, {})
        
        recommended_data_sizes = model_recommendations.get("data_size", ["small", "medium", "large"])
        recommended_complexity = model_recommendations.get("complexity", ["low", "medium", "high"])
        
        if data_size in recommended_data_sizes and complexity in recommended_complexity:
            results.append(model_name)
    
    return results

# Initialize model registry when the package is imported
init_model_registry()
