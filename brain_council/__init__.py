#!/usr/bin/env python3
"""
Brain Council Package Initialization

This module initializes the Brain Council package which coordinates
the various strategy brains and generates high-confidence trading signals.
The Brain Council is a sophisticated decision-making system that weighs
signals from various specialized brains to produce optimal trading decisions.
"""

import os
import sys
import importlib
from typing import Dict, List, Any, Type, Optional, Set
import logging

# Configure package logger
logger = logging.getLogger("brain_council")

# Brain Council version
__version__ = "1.0.0"

# Dictionary to store all registered brain council classes
REGISTERED_COUNCILS = {}

# Dictionary to store all initialized brain council instances
COUNCIL_INSTANCES = {}

# Set for tracking loaded modules to prevent duplicate loading
LOADED_MODULES = set()

def register_council(council_cls):
    """
    Decorator to register brain council classes for dynamic loading
    
    Args:
        council_cls: The brain council class to register
        
    Returns:
        The original class with registration side effect
    """
    council_name = council_cls.__name__
    if council_name in REGISTERED_COUNCILS:
        logger.warning(f"Council {council_name} already registered, overwriting")
    
    REGISTERED_COUNCILS[council_name] = council_cls
    logger.debug(f"Registered council: {council_name}")
    return council_cls

def get_council(council_name: str, config: Dict = None) -> Any:
    """
    Get or create an instance of a brain council
    
    Args:
        council_name: Name of the council to instantiate
        config: Optional configuration for the council
        
    Returns:
        Instance of the requested brain council
        
    Raises:
        ValueError: If the requested council is not registered
    """
    if council_name not in REGISTERED_COUNCILS:
        raise ValueError(f"Brain council '{council_name}' not registered")
    
    # Return existing instance if already created
    if council_name in COUNCIL_INSTANCES:
        return COUNCIL_INSTANCES[council_name]
    
    # Create new instance with provided config
    config = config or {}
    instance = REGISTERED_COUNCILS[council_name](**config)
    COUNCIL_INSTANCES[council_name] = instance
    logger.info(f"Instantiated brain council: {council_name}")
    
    return instance

def get_available_councils() -> List[str]:
    """
    Get list of all registered brain councils
    
    Returns:
        List of brain council names
    """
    return list(REGISTERED_COUNCILS.keys())

def discover_and_load_councils(package_dir: Optional[str] = None) -> None:
    """
    Automatically discover and load all brain council modules
    
    Args:
        package_dir: Optional directory to search for modules
    """
    # Default to current package directory if not specified
    if package_dir is None:
        package_dir = os.path.dirname(__file__)
    
    # Find all Python files in the directory
    for filename in os.listdir(package_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]  # Remove .py extension
            module_path = f"brain_council.{module_name}"
            
            # Skip already loaded modules
            if module_path in LOADED_MODULES:
                continue
                
            try:
                # Import the module to trigger registration
                importlib.import_module(module_path)
                LOADED_MODULES.add(module_path)
                logger.debug(f"Loaded council module: {module_path}")
            except ImportError as e:
                logger.error(f"Failed to import council module {module_path}: {e}")

# Automatically discover and load council modules
discover_and_load_councils()

# Import core components to make them available at package level
from brain_council.base_council import BaseCouncil
from brain_council.voting_system import VotingSystem
from brain_council.weighting_system import WeightingSystem
from brain_council.performance_tracker import PerformanceTracker
from brain_council.signal_generator import SignalGenerator

# Import specific councils to register them
from brain_council.master_council import MasterCouncil
from brain_council.timeframe_council import TimeframeCouncil
from brain_council.asset_council import AssetCouncil
from brain_council.regime_council import RegimeCouncil

# Define what's available for import
__all__ = [
    'BaseCouncil',
    'VotingSystem',
    'WeightingSystem',
    'PerformanceTracker',
    'SignalGenerator',
    'MasterCouncil',
    'TimeframeCouncil',
    'AssetCouncil',
    'RegimeCouncil',
    'register_council',
    'get_council',
    'get_available_councils',
    'discover_and_load_councils'
]
