#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Intelligence System Module Initialization

This module initializes the Intelligence system components, which are responsible for:
- Advanced pattern recognition
- Loophole detection and exploitation
- Adaptive learning and continuous improvement
- Predictive modeling and signal generation

The Intelligence system is the core of the QuantumSpectre Elite Trading System's
high win rate potential, continuously evolving and adapting to market conditions.
"""

import os
import sys
import importlib
from typing import Dict, List, Set, Any, Optional, Type, Callable
import logging

# Internal imports
from common.logger import get_logger
from common.constants import INTELLIGENCE_MODULES, DEFAULT_MODULE_CONFIG

# Initialize module logger
logger = get_logger("intelligence")

# Component registry
pattern_recognizers = {}
loophole_detectors = {}
adaptive_learners = {}
intelligence_components = {}

class QuantumSpectreIntelligence:
    """
    Main intelligence system interface providing access to all intelligence components.
    """
    
    def __init__(self):
        """
        Initialize the QuantumSpectreIntelligence system interface.
        """
        self.initialized = False
        self.active_components = set()
        self.component_status = {}
        self.component_errors = {}
        self.version = "1.0.0"
        self.support_vector = "advanced"
        logger.info(f"QuantumSpectreIntelligence {self.version} initialized")
    
    def status(self) -> dict:
        """
        Return the current status of the intelligence system.
        
        Returns:
            dict: Status information about all intelligence components
        """
        return {
            "initialized": self.initialized,
            "active_components": list(self.active_components),
            "component_status": self.component_status,
            "component_errors": self.component_errors,
            "version": self.version,
            "support_vector": self.support_vector
        }


def register_pattern_recognizer(name: str) -> Callable:
    """
    Decorator to register pattern recognition components.
    
    Args:
        name: Unique identifier for the pattern recognizer
        
    Returns:
        Decorator function for registration
    """
    def decorator(cls):
        if name in pattern_recognizers:
            logger.warning(f"Pattern recognizer '{name}' already registered. Overwriting.")
        pattern_recognizers[name] = cls
        intelligence_components[f"pattern_recognizer.{name}"] = cls
        logger.debug(f"Registered pattern recognizer: {name}")
        return cls
    return decorator


def register_loophole_detector(name: str) -> Callable:
    """
    Decorator to register loophole detection components.
    
    Args:
        name: Unique identifier for the loophole detector
        
    Returns:
        Decorator function for registration
    """
    def decorator(cls):
        if name in loophole_detectors:
            logger.warning(f"Loophole detector '{name}' already registered. Overwriting.")
        loophole_detectors[name] = cls
        intelligence_components[f"loophole_detector.{name}"] = cls
        logger.debug(f"Registered loophole detector: {name}")
        return cls
    return decorator


def register_adaptive_learner(name: str) -> Callable:
    """
    Decorator to register adaptive learning components.
    
    Args:
        name: Unique identifier for the adaptive learner
        
    Returns:
        Decorator function for registration
    """
    def decorator(cls):
        if name in adaptive_learners:
            logger.warning(f"Adaptive learner '{name}' already registered. Overwriting.")
        adaptive_learners[name] = cls
        intelligence_components[f"adaptive_learner.{name}"] = cls
        logger.debug(f"Registered adaptive learner: {name}")
        return cls
    return decorator


def get_pattern_recognizer(name: str) -> Any:
    """
    Retrieve a registered pattern recognizer by name.
    
    Args:
        name: The name of the pattern recognizer to retrieve
        
    Returns:
        The requested pattern recognizer class
        
    Raises:
        KeyError: If the requested pattern recognizer is not registered
    """
    if name not in pattern_recognizers:
        raise KeyError(f"Pattern recognizer '{name}' not found")
    return pattern_recognizers[name]


def get_loophole_detector(name: str) -> Any:
    """
    Retrieve a registered loophole detector by name.
    
    Args:
        name: The name of the loophole detector to retrieve
        
    Returns:
        The requested loophole detector class
        
    Raises:
        KeyError: If the requested loophole detector is not registered
    """
    if name not in loophole_detectors:
        raise KeyError(f"Loophole detector '{name}' not found")
    return loophole_detectors[name]


def get_adaptive_learner(name: str) -> Any:
    """
    Retrieve a registered adaptive learner by name.
    
    Args:
        name: The name of the adaptive learner to retrieve
        
    Returns:
        The requested adaptive learner class
        
    Raises:
        KeyError: If the requested adaptive learner is not registered
    """
    if name not in adaptive_learners:
        raise KeyError(f"Adaptive learner '{name}' not found")
    return adaptive_learners[name]


def get_component(name: str) -> Any:
    """
    Retrieve any registered intelligence component by full name.
    
    Args:
        name: The full name of the component to retrieve
        
    Returns:
        The requested component class
        
    Raises:
        KeyError: If the requested component is not registered
    """
    if name not in intelligence_components:
        raise KeyError(f"Intelligence component '{name}' not found")
    return intelligence_components[name]


def list_pattern_recognizers() -> List[str]:
    """
    List all registered pattern recognizers.
    
    Returns:
        List of registered pattern recognizer names
    """
    return list(pattern_recognizers.keys())


def list_loophole_detectors() -> List[str]:
    """
    List all registered loophole detectors.
    
    Returns:
        List of registered loophole detector names
    """
    return list(loophole_detectors.keys())


def list_adaptive_learners() -> List[str]:
    """
    List all registered adaptive learners.
    
    Returns:
        List of registered adaptive learner names
    """
    return list(adaptive_learners.keys())


def list_all_components() -> List[str]:
    """
    List all registered intelligence components.
    
    Returns:
        List of all registered intelligence component names
    """
    return list(intelligence_components.keys())


# Import all submodules to ensure registration
def _import_all_modules():
    """
    Dynamically import all intelligence submodules to trigger registration.
    """
    for module_name in INTELLIGENCE_MODULES:
        try:
            importlib.import_module(f"intelligence.{module_name}")
            logger.debug(f"Imported intelligence module: {module_name}")
        except ImportError as e:
            logger.error(f"Failed to import intelligence module {module_name}: {e}")


# Create singleton instance
intelligence = QuantumSpectreIntelligence()

# Run module imports on initialization
_import_all_modules()

# Export public interface
__all__ = [
    'intelligence',
    'register_pattern_recognizer',
    'register_loophole_detector',
    'register_adaptive_learner',
    'get_pattern_recognizer',
    'get_loophole_detector',
    'get_adaptive_learner',
    'get_component',
    'list_pattern_recognizers',
    'list_loophole_detectors',
    'list_adaptive_learners',
    'list_all_components',
    'pattern_recognizers',
    'loophole_detectors',
    'adaptive_learners',
    'intelligence_components'
]
