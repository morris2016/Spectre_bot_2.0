#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Backtester Module Initialization

This module provides comprehensive backtesting capabilities for the QuantumSpectre
Elite Trading System, enabling strategy development, validation, and optimization.
"""

import os
import sys
import importlib
from typing import Dict, List, Any, Optional, Type, Union, Set
import logging

# Initialize module logger
logger = logging.getLogger(__name__)

# Version information
__version__ = '1.0.0'
__author__ = 'QuantumSpectre Team'

# Available backtest modes
BACKTEST_MODES = {
    'standard': 'Standard historical backtesting with OHLCV data',
    'tick': 'Tick-by-tick backtesting with order book reconstruction',
    'monte_carlo': 'Monte Carlo simulation with randomized parameters',
    'walk_forward': 'Walk-forward analysis with optimization',
    'scenario': 'Scenario-based testing with custom market conditions',
    'stress': 'Stress testing with extreme market conditions'
}

# Available backtest metrics
BACKTEST_METRICS = {
    'total_return': 'Total return percentage',
    'annualized_return': 'Annualized return percentage',
    'sharpe_ratio': 'Sharpe ratio (risk-adjusted return)',
    'sortino_ratio': 'Sortino ratio (downside risk-adjusted return)',
    'max_drawdown': 'Maximum drawdown percentage',
    'win_rate': 'Percentage of winning trades',
    'profit_factor': 'Gross profit / Gross loss',
    'calmar_ratio': 'Annualized return / Maximum drawdown',
    'recovery_factor': 'Total return / Maximum drawdown',
    'ulcer_index': 'Measure of drawdown severity',
    'omega_ratio': 'Probability-weighted ratio of gains vs losses',
    'expectancy': 'Average profit/loss per trade',
    'avg_win': 'Average winning trade amount',
    'avg_loss': 'Average losing trade amount',
    'avg_win_loss_ratio': 'Average win / Average loss',
    'max_consecutive_wins': 'Maximum consecutive winning trades',
    'max_consecutive_losses': 'Maximum consecutive losing trades',
    'longest_winning_streak': 'Longest winning streak duration',
    'longest_losing_streak': 'Longest losing streak duration',
    'time_in_market': 'Percentage of time with open positions',
    'trade_duration': 'Average trade duration',
    'turnover': 'Total trading volume',
    'kelly_criterion': 'Kelly criterion percentage',
    'information_ratio': 'Information ratio vs benchmark',
    'treynor_ratio': 'Treynor ratio (excess return per unit of market risk)',
    'benchmark_correlation': 'Correlation with benchmark',
    'beta': 'Beta (systematic risk)',
    'alpha': 'Alpha (excess return over benchmark)',
    'distribution_metrics': 'Return distribution metrics (skewness, kurtosis)',
    'regime_performance': 'Performance breakdowns by market regime',
    'drawdown_analysis': 'Detailed drawdown analysis',
    'trade_analytics': 'Detailed trade-by-trade analytics'
}

# Public exports
__all__ = [
    'BacktestEngine',
    'BacktestDataProvider',
    'Performance',
    'Optimization',
    'Scenario',
    'Report',
    'BACKTEST_MODES',
    'BACKTEST_METRICS'
]

# Dynamic imports for public API
from backtester.engine import BacktestEngine
from backtester.data_provider import BacktestDataProvider
from backtester.performance import Performance
from backtester.optimization import Optimization
from backtester.scenario import Scenario
from backtester.report import Report

# Component registry for dependency injection
_components = {}

def register_component(name: str, component: Any) -> None:
    """Register a component in the backtester module."""
    _components[name] = component
    logger.debug(f"Registered backtester component: {name}")

def get_component(name: str) -> Any:
    """Get a registered component by name."""
    if name not in _components:
        raise KeyError(f"Component not found: {name}")
    return _components[name]

def has_component(name: str) -> bool:
    """Check if a component is registered."""
    return name in _components

# Auto-discover and register components
def _discover_components() -> None:
    """Auto-discover and register components from the backtester module."""
    component_modules = [
        'engine',
        'data_provider',
        'performance',
        'optimization',
        'scenario',
        'report'
    ]
    
    for module_name in component_modules:
        try:
            module = importlib.import_module(f'backtester.{module_name}')
            if hasattr(module, 'register_components'):
                module.register_components()
                logger.debug(f"Registered components from module: backtester.{module_name}")
        except ImportError as e:
            logger.warning(f"Could not import module backtester.{module_name}: {e}")
        except Exception as e:
            logger.error(f"Error registering components from backtester.{module_name}: {e}")

# Initialize component discovery
_discover_components()

# Configuration validation
def validate_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate the backtester configuration.
    
    Args:
        config: Backtester configuration dictionary
        
    Returns:
        bool: True if the configuration is valid, False otherwise
    """
    required_fields = {
        'data_source': str,
        'strategy': str,
        'timeframe': str,
        'start_date': str,
        'end_date': str,
    }
    
    # Check required fields
    for field, field_type in required_fields.items():
        if field not in config:
            logger.error(f"Missing required field in backtester config: {field}")
            return False
        if not isinstance(config[field], field_type):
            logger.error(f"Invalid type for field {field} in backtester config. Expected {field_type.__name__}")
            return False
    
    # Check mode
    if 'mode' in config and config['mode'] not in BACKTEST_MODES:
        logger.error(f"Invalid backtest mode: {config['mode']}. Available modes: {list(BACKTEST_MODES.keys())}")
        return False
    
    # Check metrics
    if 'metrics' in config:
        if not isinstance(config['metrics'], list):
            logger.error("Metrics must be a list")
            return False
        
        invalid_metrics = [m for m in config['metrics'] if m not in BACKTEST_METRICS]
        if invalid_metrics:
            logger.error(f"Invalid backtest metrics: {invalid_metrics}")
            return False
    
    return True

# Module initialization status
_initialized = False

def initialize() -> None:
    """Initialize the backtester module."""
    global _initialized
    if _initialized:
        logger.warning("Backtester module already initialized")
        return
    
    # Perform initialization tasks
    logger.info("Initializing backtester module")
    
    # Register with the metrics system
    try:
        from common.metrics import register_metrics_provider
        register_metrics_provider('backtester', BACKTEST_METRICS)
        logger.debug("Registered backtester metrics provider")
    except ImportError:
        logger.warning("Could not register metrics provider - common.metrics not available")
    
    _initialized = True
    logger.info(f"Backtester module initialized (version {__version__})")

# Auto-initialize when imported
initialize()
