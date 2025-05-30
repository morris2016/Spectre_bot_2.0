

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Monitoring System Initialization

This module initializes the monitoring system components for real-time tracking
of system health, performance, and trading metrics.
"""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Set, Any, Optional

# Version
__version__ = "1.0.0"

# Initialize logger
logger = logging.getLogger("monitoring")

# Import monitoring components (using lazy imports to avoid circular dependencies)
def get_metrics_collector():
    """Return the metrics collector instance."""
    from monitoring.metrics_collector import MetricsCollector
    return MetricsCollector()

def get_alerting_system():
    """Return the alerting system instance."""
    from monitoring.alerting import AlertingSystem
    return AlertingSystem()

def get_performance_tracker():
    """Return the performance tracker instance."""
    from monitoring.performance_tracker import PerformanceTracker
    return PerformanceTracker()

def get_system_health_monitor():
    """Return the system health monitor instance."""
    from monitoring.system_health import SystemHealthMonitor
    return SystemHealthMonitor()

def get_log_analyzer():
    """Return the log analyzer instance."""
    from monitoring.log_analyzer import LogAnalyzer
    return LogAnalyzer()

# Component registry
_component_registry = {
    "metrics_collector": None,
    "alerting_system": None,
    "performance_tracker": None,
    "system_health_monitor": None,
    "log_analyzer": None
}

def get_component(component_name: str) -> Any:
    """
    Get a component instance by name, initializing it if necessary.
    
    Args:
        component_name: Name of the component to retrieve
        
    Returns:
        The requested component instance
        
    Raises:
        ValueError: If the component name is not recognized
    """
    if component_name not in _component_registry:
        raise ValueError(f"Unknown component: {component_name}")
        
    # Initialize if not already done
    if _component_registry[component_name] is None:
        if component_name == "metrics_collector":
            _component_registry[component_name] = get_metrics_collector()
        elif component_name == "alerting_system":
            _component_registry[component_name] = get_alerting_system()
        elif component_name == "performance_tracker":
            _component_registry[component_name] = get_performance_tracker()
        elif component_name == "system_health_monitor":
            _component_registry[component_name] = get_system_health_monitor()
        elif component_name == "log_analyzer":
            _component_registry[component_name] = get_log_analyzer()
            
    return _component_registry[component_name]

async def initialize_monitoring(config: Dict[str, Any]) -> None:
    """
    Initialize all monitoring components with the provided configuration.
    
    Args:
        config: Configuration dictionary for monitoring components
    """
    logger.info("Initializing monitoring system components")
    
    # Initialize all components
    for component_name in _component_registry:
        component = get_component(component_name)
        if hasattr(component, "initialize"):
            init_func = component.initialize
            if asyncio.iscoroutinefunction(init_func):
                await init_func(config.get(component_name, {}))
            else:
                init_func(config.get(component_name, {}))
            result = component.initialize(config.get(component_name, {}))
            if asyncio.iscoroutine(result):
                await result
            
    logger.info("Monitoring system initialization complete")

def shutdown_monitoring() -> None:
    """Gracefully shut down all monitoring components."""
    logger.info("Shutting down monitoring system components")
    
    # Shutdown all initialized components
    for component_name, component in _component_registry.items():
        if component is not None and hasattr(component, "shutdown"):
            try:
                component.shutdown()
                logger.debug(f"Successfully shut down {component_name}")
            except Exception as e:
                logger.error(f"Error shutting down {component_name}: {str(e)}")
                
    logger.info("Monitoring system shutdown complete")

# Register monitoring exporters and handlers
def register_exporters(config: Dict[str, Any]) -> None:
    """
    Register metric exporters based on configuration.
    
    Args:
        config: Configuration dictionary for exporters
    """
    metrics_collector = get_component("metrics_collector")
    if hasattr(metrics_collector, "register_exporters"):
        metrics_collector.register_exporters(config.get("exporters", {}))

def register_alert_handlers(config: Dict[str, Any]) -> None:
    """
    Register alert handlers based on configuration.
    
    Args:
        config: Configuration dictionary for alert handlers
    """
    alerting_system = get_component("alerting_system")
    if hasattr(alerting_system, "register_handlers"):
        alerting_system.register_handlers(config.get("alert_handlers", {}))

# Export monitoring components
__all__ = [
    "initialize_monitoring",
    "shutdown_monitoring",
    "get_component",
    "register_exporters",
    "register_alert_handlers",
]

