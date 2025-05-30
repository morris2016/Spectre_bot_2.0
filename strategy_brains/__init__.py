#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Strategy Brains Module

This module contains the specialized trading strategy brains that form 
the foundation of the system's trading intelligence. Each brain specializes
in a specific trading approach and is optimized for particular market
conditions and assets.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Type, Optional

from common.logger import get_logger
from common.async_utils import AsyncTaskManager
from common.metrics import MetricsCollector

logger = get_logger(__name__)

# Strategy brain registry
_strategy_brains = {}
_asset_specific_brains = {}
_platform_specific_brains = {
    'binance': {},
    'deriv': {}
}

def register_strategy_brain(strategy_cls):
    """
    Decorator to register a strategy brain class.
    """
    strategy_name = strategy_cls.__name__
    if strategy_name in _strategy_brains:
        logger.warning(f"Strategy brain {strategy_name} already registered. Overwriting.")
    
    _strategy_brains[strategy_name] = strategy_cls
    logger.info(f"Registered strategy brain: {strategy_name}")
    
    # Register for specific assets if specified
    if hasattr(strategy_cls, 'SUPPORTED_ASSETS') and strategy_cls.SUPPORTED_ASSETS:
        for asset in strategy_cls.SUPPORTED_ASSETS:
            if asset not in _asset_specific_brains:
                _asset_specific_brains[asset] = []
            _asset_specific_brains[asset].append(strategy_name)
            logger.debug(f"Registered {strategy_name} for asset: {asset}")
    
    # Register for specific platforms if specified
    if hasattr(strategy_cls, 'PLATFORM') and strategy_cls.PLATFORM in _platform_specific_brains:
        platform = strategy_cls.PLATFORM
        _platform_specific_brains[platform][strategy_name] = strategy_cls
        logger.debug(f"Registered {strategy_name} for platform: {platform}")
    
    return strategy_cls

def get_all_strategy_brains() -> Dict[str, Type]:
    """
    Get all registered strategy brains.
    
    Returns:
        Dict[str, Type]: Dictionary mapping strategy names to strategy classes
    """
    return _strategy_brains.copy()

def get_strategy_brain(name: str) -> Optional[Type]:
    """
    Get a specific strategy brain by name.
    
    Args:
        name: Name of the strategy brain
        
    Returns:
        Type or None: The strategy brain class if found, None otherwise
    """
    return _strategy_brains.get(name)

def get_asset_specific_brains(asset: str) -> List[Type]:
    """
    Get all strategy brains specialized for a specific asset.
    
    Args:
        asset: The asset symbol (e.g., 'BTC/USDT')
        
    Returns:
        List[Type]: List of strategy brain classes specialized for the asset
    """
    brain_names = _asset_specific_brains.get(asset, [])
    return [_strategy_brains[name] for name in brain_names if name in _strategy_brains]

def get_platform_specific_brains(platform: str) -> Dict[str, Type]:
    """
    Get all strategy brains specific to a trading platform.
    
    Args:
        platform: The platform name ('binance' or 'deriv')
        
    Returns:
        Dict[str, Type]: Dictionary of strategy brains for the platform
    """
    if platform not in _platform_specific_brains:
        logger.warning(f"Unknown platform: {platform}")
        return {}
    return _platform_specific_brains[platform].copy()

# Import all strategy brain implementations to register them
from . import base_brain
from . import momentum_brain
from . import mean_reversion_brain
try:
    from . import breakout_brain
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning(f"breakout_brain not available: {e}")
try:
    from . import volatility_brain
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning(f"volatility_brain not available: {e}")
try:
    from . import pattern_brain
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning(f"pattern_brain not available: {e}")
try:
    from . import sentiment_brain
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning(f"sentiment_brain not available: {e}")
from . import order_flow_brain
try:
    from . import market_structure_brain
    from . import statistical_brain
    from . import onchain_brain
    from . import regime_brain
    from . import adaptive_brain
    from . import trend_brain
    from . import swing_brain
    from . import scalping_brain
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning(f"some optional brains not available: {e}")
try:
    from . import ml_brain
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning(f"ml_brain not available: {e}")
try:
    from . import reinforcement_brain
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning(f"reinforcement_brain not available: {e}")
try:
    from . import onchain_brain
    from . import regime_brain
    from . import adaptive_brain
    from . import trend_brain
    from . import swing_brain
    from . import scalping_brain
    from . import arbitrage_brain
    from . import correlation_brain
    from . import divergence_brain
    from . import ensemble_brain
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning(f"additional brains not available: {e}")

# Import only works when files are there, during development you can comment out
# the imports that don't exist yet

# Register metrics
metrics = MetricsCollector.get_instance()
metrics.register_gauge("strategy_brains_total", "Total number of strategy brains", 
                      lambda: len(_strategy_brains))
metrics.register_gauge("asset_specific_brains_total", "Total number of asset-specific brain mappings", 
                      lambda: sum(len(brains) for brains in _asset_specific_brains.values()))
