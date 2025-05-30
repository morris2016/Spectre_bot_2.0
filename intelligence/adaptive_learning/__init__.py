

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Adaptive Learning Package

This package provides the adaptive learning capabilities for the QuantumSpectre Elite
Trading System, allowing strategies to evolve and improve over time based on market
performance and changing conditions.
"""

from typing import Dict, List, Set, Union, Any, Type
import logging
import importlib
import pkgutil
import inspect
import os
import sys
from pathlib import Path

# Import core components
from common.logger import get_logger

logger = get_logger(__name__)

# Dictionary to store all adaptive learning components
ADAPTIVE_LEARNERS = {}

def register_adaptive_learner(name: str, learner_class: Type) -> None:
    """
    Register an adaptive learner in the system.
    
    Args:
        name: Unique name for the learner
        learner_class: Reference to the learner class
    """
    if name in ADAPTIVE_LEARNERS:
        logger.warning(f"Overwriting existing adaptive learner: {name}")
    
    ADAPTIVE_LEARNERS[name] = learner_class
    logger.debug(f"Registered adaptive learner: {name}")

def get_adaptive_learner(name: str) -> Type:
    """
    Get an adaptive learner by name.
    
    Args:
        name: Name of the learner to retrieve
        
    Returns:
        The learner class
        
    Raises:
        KeyError: If the learner is not found
    """
    if name not in ADAPTIVE_LEARNERS:
        raise KeyError(f"Adaptive learner not found: {name}")
    
    return ADAPTIVE_LEARNERS[name]

def list_adaptive_learners() -> List[str]:
    """
    List all registered adaptive learners.
    
    Returns:
        List of learner names
    """
    return list(ADAPTIVE_LEARNERS.keys())

def _discover_adaptive_learners() -> None:
    """
    Automatically discover and register all adaptive learners in the package.
    """
    package_dir = Path(__file__).parent
    for (_, module_name, _) in pkgutil.iter_modules([str(package_dir)]):
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
        except ImportError as exc:  # pragma: no cover - optional deps
            logger.warning(f"Failed to import adaptive learner module {module_name}: {exc}")
            continue

        for item_name in dir(module):
            item = getattr(module, item_name)

            if inspect.isclass(item) and hasattr(item, '_is_adaptive_learner'):
                register_adaptive_learner(item_name, item)

# Auto-discover learners when the package is imported
# Auto-discover learners when the package is imported unless disabled
if os.environ.get("QS_DISABLE_ADAPTIVE_DISCOVERY") != "1":
    _discover_adaptive_learners()

# Define base class for all adaptive learners
class BaseAdaptiveLearner:
    """Base class for all adaptive learning components."""
    
    # Class attribute to mark this as an adaptive learner for auto-discovery
    _is_adaptive_learner = True
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adaptive learner.
        
        Args:
            config: Configuration dictionary for the learner
        """
        self.config = config or {}
        self.is_trained = False
        self.version = 1
        self.performance_history = []
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        self.logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")
    
    def train(self, training_data: Any) -> None:
        """
        Train the adaptive learner on historical data.
        
        Args:
            training_data: Data to train the learner on
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def update(self, new_data: Any) -> None:
        """
        Update the learner with new data.
        
        Args:
            new_data: New data to update the learner with
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def predict(self, input_data: Any) -> Any:
        """
        Make a prediction using the learner.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            The prediction result
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """
        Evaluate the learner on test data.
        
        Args:
            test_data: Data to evaluate the learner on
            
        Returns:
            Dictionary of performance metrics
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def save(self, filepath: str) -> None:
        """
        Save the learner to a file.
        
        Args:
            filepath: Path to save the learner to
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement save()")
    
    def load(self, filepath: str) -> None:
        """
        Load the learner from a file.
        
        Args:
            filepath: Path to load the learner from
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement load()")
    
    def reset(self) -> None:
        """
        Reset the learner to its initial state.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement reset()")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the learner's current parameters.
        
        Returns:
            Dictionary of learner parameters
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_parameters()")
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set the learner's parameters.
        
        Args:
            parameters: Dictionary of parameters to set
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement set_parameters()")
    
    def record_performance(self, metrics: Dict[str, float]) -> None:
        """
        Record the learner's performance for future analysis.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        self.performance_history.append({
            'version': self.version,
            'metrics': metrics
        })
        self.logger.debug(f"Recorded performance for version {self.version}: {metrics}")
        self.version += 1


class AdaptiveLearningManager:
    """
    Adaptive Learning Manager
    
    This class coordinates adaptive learning across various types of learners
    and provides a unified interface for adaptive learning capabilities.
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adaptive learning manager.
        
        Args:
            config: Configuration for adaptive learning
        """
        self.config = config or {}
        self.learners = {}
        self.active_learners = set()
        
        # Initialize learners
        self._initialize_learners()
        
        logger.info("Adaptive Learning Manager initialized")
    
    def _initialize_learners(self):
        """Initialize all registered adaptive learners."""
        for name, learner_class in ADAPTIVE_LEARNERS.items():
            try:
                # Get learner-specific config
                learner_config = self.config.get(name, {})
                
                # Create an instance of the learner
                learner = learner_class(learner_config)
                
                # Store the learner instance
                self.learners[name] = learner
                
                logger.debug(f"Initialized adaptive learner: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize adaptive learner '{name}': {str(e)}")
    
    async def train_learner(self, name: str, training_data: Any) -> Dict[str, Any]:
        """
        Train a specific adaptive learner.
        
        Args:
            name: Name of the learner to train
            training_data: Data to train the learner on
            
        Returns:
            Training results
        """
        if name not in self.learners:
            logger.error(f"Adaptive learner '{name}' not found")
            return {'success': False, 'error': f"Learner '{name}' not found"}
        
        try:
            learner = self.learners[name]
            learner.train(training_data)
            
            # Mark as active
            self.active_learners.add(name)
            
            return {'success': True, 'learner': name}
        except Exception as e:
            logger.error(f"Error training learner '{name}': {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def update_learner(self, name: str, new_data: Any) -> Dict[str, Any]:
        """
        Update a specific adaptive learner with new data.
        
        Args:
            name: Name of the learner to update
            new_data: New data to update the learner with
            
        Returns:
            Update results
        """
        if name not in self.learners:
            logger.error(f"Adaptive learner '{name}' not found")
            return {'success': False, 'error': f"Learner '{name}' not found"}
        
        try:
            learner = self.learners[name]
            learner.update(new_data)
            
            return {'success': True, 'learner': name}
        except Exception as e:
            logger.error(f"Error updating learner '{name}': {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def predict(self, name: str, input_data: Any) -> Dict[str, Any]:
        """
        Make a prediction using a specific adaptive learner.
        
        Args:
            name: Name of the learner to use
            input_data: Input data for prediction
            
        Returns:
            Prediction results
        """
        if name not in self.learners:
            logger.error(f"Adaptive learner '{name}' not found")
            return {'success': False, 'error': f"Learner '{name}' not found"}
        
        try:
            learner = self.learners[name]
            prediction = learner.predict(input_data)
            
            return {
                'success': True,
                'learner': name,
                'prediction': prediction
            }
        except Exception as e:
            logger.error(f"Error making prediction with learner '{name}': {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_active_learners(self) -> List[str]:
        """
        Get the names of all active learners.
        
        Returns:
            List of active learner names
        """
        return list(self.active_learners)
    
    def get_available_learners(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available learners.
        
        Returns:
            Dictionary mapping learner names to information about them
        """
        result = {}
        
        for name, learner in self.learners.items():
            result[name] = {
                'is_trained': getattr(learner, 'is_trained', False),
                'version': getattr(learner, 'version', 1),
                'active': name in self.active_learners,
                'type': learner.__class__.__name__
            }
        
        return result


# Create a singleton instance of the adaptive learning manager
adaptive_learning_manager = AdaptiveLearningManager({
    'genetic': {
        'population_size': 100,
        'mutation_rate': 0.01,
        'crossover_rate': 0.7
    },
    'bayesian': {
        'prior_strength': 0.5,
        'mcmc_samples': 1000
    },
    'reinforcement': {
        'learning_rate': 0.001,
        'discount_factor': 0.99,
        'exploration_rate': 0.1
    },
    'meta_learning': {
        'meta_learning_rate': 0.01,
        'inner_learning_rate': 0.1,
        'adaptation_steps': 5
    }
})

# Update module exports
__all__ = [
    'BaseAdaptiveLearner',
    'register_adaptive_learner',
    'get_adaptive_learner',
    'list_adaptive_learners',
    'adaptive_learning_manager'
]
