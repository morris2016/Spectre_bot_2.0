#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Execution Engine Module Initialization

This module serves as the initialization for the execution engine that handles signal processing,
order creation, submission, tracking, and management for the QuantumSpectre Elite Trading System.
"""

import os
import sys
import logging
from typing import Dict, List, Set, Any, Optional, Type, Callable

__version__ = "1.0.0"

# Setup module level logger
logger = logging.getLogger(__name__)

# Registry for execution components
EXECUTION_COMPONENT_REGISTRY = {}

# Trading signal handlers
SIGNAL_HANDLERS = {}

# Order history tracking
ORDER_TRACKER = {}

# Active positions
ACTIVE_POSITIONS = {}

# Execution strategy mapping
EXECUTION_STRATEGIES = {}

# Platform-specific execution adapters
PLATFORM_ADAPTERS = {
    'binance': None,
    'deriv': None
}

# Define exports
__all__ = [
    'OrderManager',
    'PositionManager',
    'RiskManager',
    'AdaptiveExecutor',
    'ExecutionEngine',
    'Microstructure',
    'CapitalManagement',
    'OrderType',
    'OrderStatus',
    'OrderDirection',
    'ExecutionStrategy',
    'ExecutionPriority',
    'PositionStatus',
    'register_execution_component',
    'register_signal_handler',
    'get_execution_component',
    'get_signal_handler'
]

def register_execution_component(component_name: str, component_class: Type) -> None:
    """
    Register an execution component for use in the system
    
    Args:
        component_name: Name identifier for the component
        component_class: The class reference for the component
    """
    global EXECUTION_COMPONENT_REGISTRY
    if component_name in EXECUTION_COMPONENT_REGISTRY:
        logger.warning(f"Overwriting existing execution component: {component_name}")
    
    EXECUTION_COMPONENT_REGISTRY[component_name] = component_class
    logger.debug(f"Registered execution component: {component_name}")

def get_execution_component(component_name: str) -> Optional[Type]:
    """
    Retrieve a registered execution component by name
    
    Args:
        component_name: Name identifier for the component
        
    Returns:
        The component class or None if not found
    """
    return EXECUTION_COMPONENT_REGISTRY.get(component_name)

def register_signal_handler(signal_type: str, handler_func: Callable) -> None:
    """
    Register a handler function for a specific signal type
    
    Args:
        signal_type: Type of trading signal
        handler_func: Function to handle this signal type
    """
    global SIGNAL_HANDLERS
    if signal_type in SIGNAL_HANDLERS:
        logger.warning(f"Overwriting existing signal handler for: {signal_type}")
    
    SIGNAL_HANDLERS[signal_type] = handler_func
    logger.debug(f"Registered signal handler for: {signal_type}")

def get_signal_handler(signal_type: str) -> Optional[Callable]:
    """
    Retrieve a registered signal handler by signal type
    
    Args:
        signal_type: Type of trading signal
        
    Returns:
        The handler function or None if not found
    """
    return SIGNAL_HANDLERS.get(signal_type)

# Import execution components
from execution_engine.order_manager import OrderManager
from execution_engine.position_manager import PositionManager
from execution_engine.risk_manager import RiskManager
from execution_engine.adaptive_executor import AdaptiveExecutor
from execution_engine.microstructure import MicrostructureAnalyzer as Microstructure
from execution_engine.capital_management import CapitalManagement
from execution_engine.app import ExecutionEngine

# Import enums
from execution_engine.constants import (
    OrderType, OrderStatus, OrderDirection, 
    ExecutionStrategy, ExecutionPriority, PositionStatus
)

# Initialize platform adapters
def initialize_platform_adapters():
    """Initialize the execution adapters for different trading platforms"""
    from execution_engine.platform_adapters.binance_adapter import BinanceAdapter
    from execution_engine.platform_adapters.deriv_adapter import DerivAdapter
    
    global PLATFORM_ADAPTERS
    
    PLATFORM_ADAPTERS['binance'] = BinanceAdapter()
    PLATFORM_ADAPTERS['deriv'] = DerivAdapter()
    logger.info("Platform adapters initialized for Binance and Deriv")

# Auto-register execution components
def _register_default_components():
    """Register default execution components"""
    register_execution_component('order_manager', OrderManager)
    register_execution_component('position_manager', PositionManager)
    register_execution_component('risk_manager', RiskManager)
    register_execution_component('adaptive_executor', AdaptiveExecutor)
    register_execution_component('microstructure', Microstructure)
    register_execution_component('capital_management', CapitalManagement)
    logger.debug("Default execution components registered")

# Initialize default components on module import
_register_default_components()
