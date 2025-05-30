"""
QuantumSpectre Elite Trading System
Common Module

This module provides shared functionality, utilities, and constants used throughout
the QuantumSpectre Elite Trading System. It serves as a foundation for all other
system components.
"""

import logging
import importlib
from typing import Dict, List, Any, Callable
from pathlib import Path

# Version information
__version__ = "1.0.0"
__author__ = "QuantumSpectre Team"
__license__ = "MIT"

# Module registry for automatic component discovery and management

_registry = {
    "utils": set(),
    "constants": set(),
    "exceptions": set(),
    "metrics": set(),
    "loggers": set(),
    "clients": set(),
    "models": set(),
    "services": set(),
    "strategies": set(),
    "patterns": set(),
    "indicators": set(),
    "hooks": set(),
    "plugins": set(),
}


class Registry:
    """Registry for system components with automatic registration."""

    @classmethod
    def register(cls, category: str, name: str = None) -> Callable:
        """
        Decorator for registering components.

        Args:
            category: Component category
            name: Optional component name

        Returns:
            Decorator function
        """

        def decorator(obj: Any) -> Any:
            nonlocal name
            if name is None:
                name = obj.__name__

            if category in _registry:
                _registry[category].add((name, obj))
            else:
                _registry[category] = {(name, obj)}

            return obj

        return decorator

    @classmethod
    def get(cls, category: str, name: str = None) -> Any:
        """
        Get a registered component.

        Args:
            category: Component category
            name: Component name (optional - returns all if None)

        Returns:
            Component or dictionary of components
        """
        if category not in _registry:
            return None

        if name is None:
            # Return dictionary of all components in this category
            return {name: obj for name, obj in _registry[category]}

        # Find specific component
        for comp_name, obj in _registry[category]:
            if comp_name == name:
                return obj

        return None

    @classmethod
    def list(cls, category: str = None) -> Dict[str, List[str]]:
        """
        List registered components.

        Args:
            category: Component category (optional - lists all if None)

        Returns:
            Dictionary of component names by category
        """
        result = {}

        if category is not None:
            if category in _registry:
                result[category] = [name for name, _ in _registry[category]]
        else:
            for cat, components in _registry.items():
                result[cat] = [name for name, _ in components]

        return result

    @classmethod
    def unregister(cls, category: str, name: str) -> bool:
        """
        Unregister a component.

        Args:
            category: Component category
            name: Component name

        Returns:
            True if component was unregistered, False if not found
        """
        if category not in _registry:
            return False

        for comp_name, obj in list(_registry[category]):
            if comp_name == name:
                _registry[category].remove((comp_name, obj))
                return True

        return False


def register_utils(name: str = None) -> Callable:
    """Register utility functions."""
    return Registry.register("utils", name)


def register_constants(name: str = None) -> Callable:
    """Register constants."""
    return Registry.register("constants", name)


def register_exception(name: str = None) -> Callable:
    """Register exceptions."""
    return Registry.register("exceptions", name)


def register_metric(name: str = None) -> Callable:
    """Register metrics."""
    return Registry.register("metrics", name)


def register_logger(name: str = None) -> Callable:
    """Register loggers."""
    return Registry.register("loggers", name)


def register_client(name: str = None) -> Callable:
    """Register clients."""
    return Registry.register("clients", name)


def register_model(name: str = None) -> Callable:
    """Register models."""
    return Registry.register("models", name)


def register_service(name: str = None) -> Callable:
    """Register services."""
    return Registry.register("services", name)


def register_strategy(name: str = None) -> Callable:
    """Register trading strategies."""
    return Registry.register("strategies", name)


def register_pattern(name: str = None) -> Callable:
    """Register trading patterns."""
    return Registry.register("patterns", name)


def register_indicator(name: str = None) -> Callable:
    """Register technical indicators."""
    return Registry.register("indicators", name)


def register_hook(name: str = None) -> Callable:
    """Register system hooks."""
    return Registry.register("hooks", name)


def register_plugin(name: str = None) -> Callable:
    """Register plugins."""
    return Registry.register("plugins", name)


def discover_modules() -> None:
    """
    Automatically discover and import modules in the common package.
    This ensures all components are properly registered.
    """
    common_dir = Path(__file__).parent
    for py_file in common_dir.glob("*.py"):
        if py_file.stem == "__init__":
            continue

        module_name = f"common.{py_file.stem}"
        try:
            importlib.import_module(module_name)
        except Exception as e:
            # Log but continue

            logging.warning(f"Failed to import module {module_name}: {str(e)}")


# Automatically discover modules when package is imported
discover_modules()


# Export version info
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "Registry",
    "register_utils",
    "register_constants",
    "register_exception",
    "register_metric",
    "register_logger",
    "register_client",
    "register_model",
    "register_service",
    "register_strategy",
    "register_pattern",
    "register_indicator",
    "register_hook",
    "register_plugin",
]
