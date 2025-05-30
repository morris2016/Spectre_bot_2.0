

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Walk Forward Analysis Module

This module provides advanced walk forward analysis capabilities for strategy validation,
parameter optimization, and robustness testing through out-of-sample validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import asyncio
import logging
import time
import json
import os
from enum import Enum
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

from common.logger import get_logger
from common.utils import serialize_dataframe, deserialize_dataframe
from common.exceptions import ValidationError, OptimizationError
from data_storage.time_series import TimeSeriesStorage
from backtester.engine import BacktestEngine
from backtester.optimization import ParameterOptimizer
from backtester.performance import PerformanceAnalyzer

logger = get_logger(__name__)


class WalkForwardMethod(Enum):
    """Enumeration of walk forward analysis methods."""
    ANCHORED = "anchored"      # Training window expands, validation window slides
    ROLLING = "rolling"        # Both training and validation windows slide
    EXPANDING = "expanding"    # Training window expands, validation window expands
    SLIDING = "sliding"        # Fixed training and validation windows that slide
    ANCHORED_MONTE_CARLO = "anchored_monte_carlo"  # Anchored with random validation periods


@dataclass
class WalkForwardConfig:
    """Configuration for walk forward analysis."""
    method: WalkForwardMethod = WalkForwardMethod.ROLLING
    start_date: str = None
    end_date: str = None
    training_window: int = 180  # Days for training (in-sample) period
    validation_window: int = 90  # Days for validation (out-of-sample) period
    step_size: int = 30  # Days to step forward
    min_training_window: int = 90  # Minimum training window for anchored method
    optimization_metric: str = "sharpe_ratio"  # Metric to optimize
    target_parameters: List[str] = field(default_factory=list)  # Parameters to optimize
    metric_threshold: float = 0.0  # Minimum metric value for validation
    monte_carlo_samples: int = 0  # Use Monte Carlo validation (if > 0)
    robustness_test: bool = True  # Test parameter robustness
    parameter_variation: float = 0.1  # Parameter variation for robustness testing
    adaptive_parameters: bool = True  # Allow parameter adaptation between periods
    
    def __post_init__(self):
        """Validate the configuration."""
        if self.start_date is None or self.end_date is None:
            raise ValueError("Start and end dates must be specified")
        
        if self.method in [WalkForwardMethod.ROLLING, WalkForwardMethod.SLIDING]:
            if self.training_window <= 0 or self.validation_window <= 0:
                raise ValueError("Training and validation windows must be positive")
        
        if self.method == WalkForwardMethod.ANCHORED_MONTE_CARLO:
            if self.monte_carlo_samples <= 0:
                raise ValueError("Monte Carlo samples must be positive for anchored_monte_carlo method")
        
        if self.target_parameters is None or len(self.target_parameters) == 0:
            logger.warning("No target parameters specified for optimization")
        
        # Convert date strings to datetime objects for internal use
        self._start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        self._end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        
        # Ensure total time period is sufficient
        total_days = (self._end_date - self._start_date).days
        if self.method != WalkForwardMethod.ANCHORED_MONTE_CARLO:
            min_required = self.training_window + self.validation_window
            if total_days < min_required:
                raise ValueError(f"Total time period ({total_days} days) must be at least training + validation windows ({min_required} days)")
        
        # Adjust step size if it's too large
        if self.step_size > self.validation_window:
            logger.warning(f"Step size ({self.step_size}) is larger than validation window ({self.validation_window}), some data will not be validated")


class WalkForwardAnalyzer:
    """
    Advanced Walk Forward Analysis for trading strategy validation and optimization.
    
    This class provides sophisticated walk forward analysis capabilities including:
    - Multiple walk forward methods (rolling, anchored, expanding, sliding)
    - Parameter optimization for each in-sample period
    - Out-of-sample performance validation
    - Parameter stability analysis
    - Robustness testing through parameter variation
    - Monte Carlo validation option
    - Comprehensive performance reporting and visualization
    """
    
    def __init__(self, backtest_engine: BacktestEngine, optimizer: ParameterOptimizer,
                config: Optional[WalkForwardConfig] = None):
        """
        Initialize the Walk Forward Analyzer.
        
        Args:
            backtest_engine: The backtesting engine to use
            optimizer: Parameter optimizer for in-sample optimization
            config: Configuration for walk forward analysis
        """
        self.backtest_engine = backtest_engine
        self.optimizer = optimizer
        self.config = config or WalkForwardConfig()
        
        self._time_series_store = TimeSeriesStore()
        self._perf_analyzer = PerformanceAnalyzer()
        self._results_cache = {}
        
        # Pre-calculate the test periods based on the method
        self.periods = self._calculate_periods()
        
        logger.info(f"Initialized Walk Forward Analyzer with method {self.config.method.value}")
        logger.info(f"Created {len(self.periods)} analysis periods")
    
    def _calculate_periods(self) -> List[Dict[str, Any]]:
        """
        Calculate the in-sample and out-of-sample periods based on the configuration.
        
        Returns:
            List of dictionaries containing the start and end dates for each period
        """
        periods = []
        
        start_date = self.config._start_date
        end_date = self.config._end_date
        total_days = (end_date - start_date).days
        
        if self.config.method == WalkForwardMethod.ROLLING:
            # Rolling window: both training and validation windows slide forward
            current_start = start_date
            while True:
                train_end = current_start + timedelta(days=self.config.training_window)
                valid_start = train_end + timedelta(days=1)
                valid_end = valid_start + timedelta(days=self.config.validation_window - 1)
                
                if valid_end > end_date:
                    break
                    
                periods.append({
                    'train_start': current_start.strftime("%Y-%m-%d"),
                    'train_end': train_end.strftime("%Y-%m-%d"),
                    'valid_start': valid_start.strftime("%Y-%m-%d"),
                    'valid_end': valid_end.strftime("%Y-%m-%d"),
                    'period_index': len(periods)
                })
                
                current_start += timedelta(days=self.config.step_size)
        
        elif self.config.method == WalkForwardMethod.ANCHORED:
            # Anchored window: training window expands, validation window slides
            train_start = start_date
            valid_start = train_start + timedelta(days=self.config.min_training_window)
            
            while True:
                valid_end = valid_start + timedelta(days=self.config.validation_window - 1)
                train_end = valid_start - timedelta(days=1)
                
                if valid_end > end_date:
                    break
                    
                periods.append({
                    'train_start': train_start.strftime("%Y-%m-%d"),
                    'train_end': train_end.strftime("%Y-%m-%d"),
                    'valid_start': valid_start.strftime("%Y-%m-%d"),
                    'valid_end': valid_end.strftime("%Y-%m-%d"),
                    'period_index': len(periods)
                })
                
                valid_start += timedelta(days=self.config.step_size)
        
        elif self.config.method == WalkForwardMethod.EXPANDING:
            # Expanding window: both training and validation windows expand
            train_start = start_date
            train_days = self.config.min_training_window
            
            while True:
                train_end = train_start + timedelta(days=train_days)
                valid_start = train_end + timedelta(days=1)
                valid_end = valid_start + timedelta(days=self.config.validation_window - 1)
                
                if valid_end > end_date:
                    break
                    
                periods.append({
                    'train_start': train_start.strftime("%Y-%m-%d"),
                    'train_end': train_end.strftime("%Y-%m-%d"),
                    'valid_start': valid_start.strftime("%Y-%m-%d"),
                    'valid_end': valid_end.strftime("%Y-%m-%d"),
                    'period_index': len(periods)
                })
                
                train_days += self.config.step_size
                # Validate next validation window fits
                if train_start + timedelta(days=train_days + self.config.validation_window) > end_date:
                    break
        
        elif self.config.method == WalkForwardMethod.SLIDING:
            # Sliding window: fixed training and validation windows that slide
            current_start = start_date
            while True:
                train_end = current_start + timedelta(days=self.config.training_window - 1)
                valid_start = train_end + timedelta(days=1)
                valid_end = valid_start + timedelta(days=self.config.validation_window - 1)
                
                if valid_end > end_date:
                    break
                    
                periods.append({
                    'train_start': current_start.strftime("%Y-%m-%d"),
                    'train_end': train_end.strftime("%Y-%m-%d"),
                    'valid_start': valid_start.strftime("%Y-%m-%d"),
                    'valid_end': valid_end.strftime("%Y-%m-%d"),
                    'period_index': len(periods)
                })
                
                current_start += timedelta(days=self.config.step_size)
        
        elif self.config.method == WalkForwardMethod.ANCHORED_MONTE_CARLO:
            # Anchored with random validation periods
            train_start = start_date
            min_start = train_start + timedelta(days=self.config.min_training_window)
            max_start = end_date - timedelta(days=self.config.validation_window)
            
            # Calculate valid range for random validation start dates
            if min_start >= max_start:
                raise ValueError("Not enough data for anchored Monte Carlo analysis")
            
            valid_range_days = (max_start - min_start).days
            
            # Generate random validation periods
            np.random.seed(42)  # For reproducibility
            for i in range(self.config.monte_carlo_samples):
                # Random validation start within valid range
                random_offset = np.random.randint(0, valid_range_days)
                valid_start = min_start + timedelta(days=random_offset)
                valid_end = valid_start + timedelta(days=self.config.validation_window - 1)
                train_end = valid_start - timedelta(days=1)
                
                periods.append({
                    'train_start': train_start.strftime("%Y-%m-%d"),
                    'train_end': train_end.strftime("%Y-%m-%d"),
                    'valid_start': valid_start.strftime("%Y-%m-%d"),
                    'valid_end': valid_end.strftime("%Y-%m-%d"),
                    'period_index': len(periods),
                    'monte_carlo_index': i
                })
        
        # Validate we have at least one period
        if not periods:
            raise ValueError("No valid walk forward periods could be generated with the given configuration")
        
        return periods
    
    async def run_analysis(self, strategy_id: str, parameters: Dict[str, Any],
                         assets: List[str], platform: str) -> Dict[str, Any]:
        """
        Run the walk forward analysis for a strategy.
        
        Args:
            strategy_id: Identifier of the strategy to analyze
            parameters: Initial strategy parameters
            assets: List of assets to include in the analysis
            platform: Trading platform (Binance/Deriv)
            
        Returns:
            Dictionary containing the analysis results
        """
        logger.info(f"Running walk forward analysis for strategy {strategy_id} with method {self.config.method.value}")
        
        # Track overall performance and parameter evolution
        overall_results = {
            'strategy_id': strategy_id,
            'method': self.config.method.value,
            'periods': [],
            'parameters': [],
            'full_equity_curve': [],
            'metrics': {},
            'parameter_stability': {},
            'robustness_results': [] if self.config.robustness_test else None
        }
        
        # Initialize parameters for the first period
        current_params = parameters.copy()
        full_equity_data = []
        
        # Process each period
        for period_idx, period in enumerate(self.periods):
            logger.info(f"Processing period {period_idx+1}/{len(self.periods)}: "
                      f"Training {period['train_start']} to {period['train_end']}, "
                      f"Validation {period['valid_start']} to {period['valid_end']}")
            
            # Step 1: Optimize parameters using in-sample data
            if self.config.target_parameters and len(self.config.target_parameters) > 0:
                optimization_result = await self.optimizer.optimize(
                    strategy_id=strategy_id,
                    base_parameters=current_params,
                    target_parameters=self.config.target_parameters,
                    start_date=period['train_start'],
                    end_date=period['train_end'],
                    assets=assets,
                    platform=platform,
                    optimization_metric=self.config.optimization_metric
                )
                
                optimized_params = optimization_result['best_parameters']
                train_metrics = optimization_result['best_metrics']
                train_equity = optimization_result['best_equity_curve']
            else:
                # If no parameters to optimize, just run a backtest with current parameters
                train_result = self.backtest_engine.run_backtest(
                    strategy_id=strategy_id,
                    parameters=current_params,
                    start_date=period['train_start'],
                    end_date=period['train_end'],
                    assets=assets,
                    platform=platform
                )
                
                optimized_params = current_params.copy()
                train_metrics = self._perf_analyzer.calculate_metrics(train_result)
                train_equity = train_result['equity_curve']
            
            # Step 2: Validate optimized parameters on out-of-sample data
            valid_result = self.backtest_engine.run_backtest(
                strategy_id=strategy_id,
                parameters=optimized_params,
                start_date=period['valid_start'],
                end_date=period['valid_end'],
                assets=assets,
                platform=platform
            )
            
            valid_metrics = self._perf_analyzer.calculate_metrics(valid_result)
            valid_equity = valid_result['equity_curve']
            
            # Step 3: Run robustness tests if configured
            robustness_results = None
            if self.config.robustness_test:
                robustness_results = await self._run_robustness_test(
                    strategy_id=strategy_id,
                    parameters=optimized_params,
                    start_date=period['valid_start'],
                    end_date=period['valid_end'],
                    assets=assets,
                    platform=platform
                )
            
            # Step 4: Store period results
            period_result = {
                'period': period,
                'training_parameters': optimized_params,
                'training_metrics': train_metrics,
                'validation_metrics': valid_metrics,
                'parameter_changes': self._calculate_parameter_changes(
                    current_params, optimized_params
                ),
                'robustness': robustness_results
            }
            
            overall_results['periods'].append(period_result)
            overall_results['parameters'].append(optimized_params)
            
            # Add equity curve data to full series
            full_equity_data.extend(valid_equity.to_dict('records'))
            
            # Update parameters for next period if adaptive
            if self.config.adaptive_parameters:
                current_params = optimized_params.copy()
        
        # Create full equity curve
        if full_equity_data:
            full_equity_curve = pd.DataFrame(full_equity_data)
            full_equity_curve.set_index('timestamp', inplace=True)
            full_equity_curve.sort_index(inplace=True)
            overall_results['full_equity_curve'] = serialize_dataframe(full_equity_curve)
            
            # Calculate overall metrics
            overall_metrics = self._perf_analyzer.calculate_metrics({
                'equity_curve': full_equity_curve,
                'trades': [t for p in overall_results['periods'] for t in valid_result['trades']]
            })
            overall_results['metrics'] = overall_metrics
        
        # Calculate parameter stability
        overall_results['parameter_stability'] = self._analyze_parameter_stability(
            overall_results['parameters']
        )
        
        # Add analysis metadata
        overall_results['metadata'] = {
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': {k: v if not isinstance(v, Enum) else v.value 
                      for k, v in vars(self.config).items()
                      if not k.startswith('_')}
        }
        
        # Cache results
        cache_key = f"{strategy_id}_{platform}_{self.config.method.value}"
        self._results_cache[cache_key] = overall_results
        
        logger.info(f"Completed walk forward analysis for strategy {strategy_id}")
        
        return overall_results
    
    async def _run_robustness_test(self, strategy_id: str, parameters: Dict[str, Any],
                                  start_date: str, end_date: str, assets: List[str],
                                  platform: str) -> Dict[str, Any]:
        """
        Run robustness tests by varying parameters slightly and evaluating performance.
        
        Args:
            strategy_id: Strategy identifier
            parameters: Parameters to test robustness for
            start_date: Start date for testing
            end_date: End date for testing
            assets: List of assets
            platform: Trading platform
            
        Returns:
            Dictionary containing robustness test results
        """
        logger.debug(f"Running robustness tests for period {start_date} to {end_date}")
        
        # Number of variations to test
        num_variations = 10
        
        # Generate parameter variations
        variations = []
        for i in range(num_variations):
            varied_params = self._generate_parameter_variation(parameters)
            variations.append(varied_params)
        
        # Run backtests for all variations in parallel
        robustness_results = []
        
        # Create tasks for all variations
        tasks = []
        for i, params in enumerate(variations):
            tasks.append(self._run_robustness_backtest(
                strategy_id=strategy_id,
                parameters=params,
                start_date=start_date,
                end_date=end_date,
                assets=assets,
                platform=platform,
                variation_idx=i
            ))
        
        # Execute all tasks
        robustness_results = await asyncio.gather(*tasks)
        
        # Calculate robustness statistics
        base_result = self.backtest_engine.run_backtest(
            strategy_id=strategy_id,
            parameters=parameters,
            start_date=start_date,
            end_date=end_date,
            assets=assets,
            platform=platform
        )
        base_metrics = self._perf_analyzer.calculate_metrics(base_result)
        
        # Extract key metrics for comparison
        metric_values = {
            'total_return': [r['metrics']['total_return'] for r in robustness_results],
            'sharpe_ratio': [r['metrics']['sharpe_ratio'] for r in robustness_results],
            'max_drawdown': [r['metrics']['max_drawdown'] for r in robustness_results],
            'win_rate': [r['metrics']['win_rate'] for r in robustness_results]
        }
        
        # Calculate statistics
        robustness_stats = {}
        for metric, values in metric_values.items():
            robustness_stats[metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std_dev': np.std(values),
                'min': min(values),
                'max': max(values),
                'base_value': base_metrics[metric],
                'variation_coefficient': np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf')
            }
        
        return {
            'variation_results': robustness_results,
            'statistics': robustness_stats,
            'robustness_score': self._calculate_robustness_score(robustness_stats)
        }
    
    async def _run_robustness_backtest(self, strategy_id: str, parameters: Dict[str, Any],
                                     start_date: str, end_date: str, assets: List[str],
                                     platform: str, variation_idx: int) -> Dict[str, Any]:
        """Run a single robustness backtest with varied parameters."""
        try:
            result = self.backtest_engine.run_backtest(
                strategy_id=strategy_id,
                parameters=parameters,
                start_date=start_date,
                end_date=end_date,
                assets=assets,
                platform=platform
            )
            
            metrics = self._perf_analyzer.calculate_metrics(result)
            
            return {
                'variation_idx': variation_idx,
                'parameters': parameters,
                'metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in robustness backtest {variation_idx}: {str(e)}")
            return {
                'variation_idx': variation_idx,
                'parameters': parameters,
                'success': False,
                'error': str(e)
            }
    
    def _generate_parameter_variation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameter variations based on configuration."""
        varied_params = {}
        
        for key, value in parameters.items():
            # Skip non-numeric parameters
            if not isinstance(value, (int, float)):
                varied_params[key] = value
                continue
                
            # Generate variation based on parameter type
            if isinstance(value, int):
                # Calculate variation range for integers
                variation_range = max(1, int(value * self.config.parameter_variation))
                varied_value = value + np.random.randint(-variation_range, variation_range + 1)
                # Ensure non-negative for parameters that logically can't be negative
                if key.lower() in ['period', 'length', 'window', 'size', 'threshold']:
                    varied_value = max(1, varied_value)
            else:
                # Variation for floating point parameters
                variation = value * self.config.parameter_variation
                varied_value = value + np.random.uniform(-variation, variation)
                # Ensure non-negative for parameters that logically can't be negative
                if key.lower() in ['threshold', 'ratio', 'factor', 'size', 'rate']:
                    varied_value = max(0.0, varied_value)
            
            varied_params[key] = varied_value
        
        return varied_params
    
    def _calculate_parameter_changes(self, old_params: Dict[str, Any], 
                                   new_params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the percentage changes between old and new parameter values."""
        changes = {}
        
        for key in new_params:
            if key in old_params:
                old_val = old_params[key]
                new_val = new_params[key]
                
                # Only calculate changes for numeric parameters
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    if old_val != 0:
                        # Calculate percentage change
                        pct_change = (new_val - old_val) / abs(old_val)
                        changes[key] = pct_change
                    else:
                        # Handle division by zero case
                        changes[key] = float('inf') if new_val > 0 else float('-inf') if new_val < 0 else 0.0
        
        return changes
    
    def _analyze_parameter_stability(self, parameter_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the stability of parameters across walk forward periods.
        
        Args:
            parameter_sets: List of parameter dictionaries from each period
            
        Returns:
            Dictionary containing stability analysis results
        """
        if not parameter_sets or len(parameter_sets) <= 1:
            return {'error': "Insufficient parameter sets for stability analysis"}
        
        # Extract parameters present in all sets
        common_params = set(parameter_sets[0].keys())
        for params in parameter_sets[1:]:
            common_params.intersection_update(params.keys())
        
        # Filter to numeric parameters
        numeric_params = {}
        for param in common_params:
            if all(isinstance(params[param], (int, float)) for params in parameter_sets):
                numeric_params[param] = [params[param] for params in parameter_sets]
        
        # Calculate stability metrics for each parameter
        stability_results = {}
        for param, values in numeric_params.items():
            # Convert to numpy array
            values_arr = np.array(values)
            
            # Calculate statistics
            stability_results[param] = {
                'mean': float(np.mean(values_arr)),
                'median': float(np.median(values_arr)),
                'std_dev': float(np.std(values_arr)),
                'min': float(np.min(values_arr)),
                'max': float(np.max(values_arr)),
                'range': float(np.max(values_arr) - np.min(values_arr)),
                'coefficient_of_variation': float(np.std(values_arr) / np.mean(values_arr)) if np.mean(values_arr) != 0 else float('inf'),
                'stability_score': self._calculate_stability_score(values_arr)
            }
        
        # Calculate overall stability score
        stability_scores = [metrics['stability_score'] for param, metrics in stability_results.items()]
        overall_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        return {
            'parameter_metrics': stability_results,
            'overall_stability': overall_stability
        }
    
    def _calculate_stability_score(self, values: np.ndarray) -> float:
        """
        Calculate a stability score for a parameter (0-1 where 1 is most stable).
        
        Uses a combination of coefficient of variation and trend analysis.
        """
        if len(values) <= 1:
            return 1.0  # Only one value is perfectly stable
        
        # Normalize the values to [0, 1] range for fair comparison
        if np.max(values) != np.min(values):
            normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
        else:
            return 1.0  # All values are the same (perfect stability)
        
        # Calculate coefficient of variation (lower is more stable)
        cv = np.std(normalized_values) / np.mean(normalized_values) if np.mean(normalized_values) != 0 else float('inf')
        cv_score = np.exp(-3 * cv)  # Convert to 0-1 score (exponential decay)
        
        # Check for trend (unstable if strong trend exists)
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, normalized_values)
        r_squared = r_value ** 2
        
        # Strong trend indicates instability (r² close to 1)
        trend_penalty = r_squared * abs(slope) * 10  # Scale the penalty
        
        # Calculate final score (bounded to [0, 1])
        stability_score = max(0.0, min(1.0, cv_score - trend_penalty))
        
        return float(stability_score)
    
    def _calculate_robustness_score(self, robustness_stats: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate an overall robustness score based on the statistics of varied backtests.
        
        Higher score means more robust (less sensitive to parameter changes).
        """
        # Weight factors for different metrics
        weights = {
            'total_return': 0.25,
            'sharpe_ratio': 0.3,
            'max_drawdown': 0.25,
            'win_rate': 0.2
        }
        
        # Calculate individual scores (higher is better)
        scores = {}
        
        # Total return: higher mean relative to std_dev is better
        if robustness_stats['total_return']['mean'] > 0:
            cv_return = robustness_stats['total_return']['std_dev'] / robustness_stats['total_return']['mean']
            scores['total_return'] = np.exp(-3 * cv_return)  # Convert to 0-1 score
        else:
            scores['total_return'] = 0.0  # Negative mean return gets zero score
        
        # Sharpe ratio: higher mean relative to std_dev is better
        if robustness_stats['sharpe_ratio']['mean'] > 0:
            cv_sharpe = robustness_stats['sharpe_ratio']['std_dev'] / robustness_stats['sharpe_ratio']['mean']
            scores['sharpe_ratio'] = np.exp(-3 * cv_sharpe)  # Convert to 0-1 score
        else:
            scores['sharpe_ratio'] = 0.0  # Negative mean Sharpe gets zero score
        
        # Max drawdown: lower cv is better (less variation in downside risk)
        cv_drawdown = robustness_stats['max_drawdown']['std_dev'] / robustness_stats['max_drawdown']['mean'] if robustness_stats['max_drawdown']['mean'] != 0 else float('inf')
        scores['max_drawdown'] = np.exp(-3 * cv_drawdown)  # Convert to 0-1 score
        
        # Win rate: higher mean and lower std_dev is better
        cv_win_rate = robustness_stats['win_rate']['std_dev'] / robustness_stats['win_rate']['mean'] if robustness_stats['win_rate']['mean'] != 0 else float('inf')
        scores['win_rate'] = np.exp(-3 * cv_win_rate)  # Convert to 0-1 score
        
        # Calculate weighted average score
        weighted_score = sum(scores[metric] * weight for metric, weight in weights.items())
        
        # Add bonus for high 80%+ win rate probability
        win_rates = robustness_stats['win_rate']
        percent_above_80 = sum(1 for val in win_rates if val >= 0.8) / len(win_rates)
        win_rate_bonus = percent_above_80 * 0.2  # Up to 0.2 bonus
        
        # Calculate final score (bounded to [0, 1])
        final_score = min(1.0, weighted_score + win_rate_bonus)
        
        return float(final_score)
    
    def generate_report(self, analysis_results: Dict[str, Any], output_dir: str = None) -> str:
        """Generate a comprehensive HTML report from the walk forward analysis results."""
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'reports', 'walk_forward')
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create report filename based on timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_path = os.path.join(output_dir, f"walk_forward_report_{timestamp}.html")
        
        # Generate plots and save them
        self._generate_plots(analysis_results, output_dir)
        
        # Create HTML report
        html_content = self._generate_html_report(analysis_results, output_dir)
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated Walk Forward report at {report_path}")
        return report_path
    
    def _generate_plots(self, analysis: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Generate plots for the walk forward analysis report."""
        plot_paths = {}
        
        # Ensure the images directory exists
        img_dir = os.path.join(output_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        
        # Common timestamp for all plot files
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        try:
            # Plot 1: Full equity curve
            if 'full_equity_curve' in analysis and analysis['full_equity_curve']:
                equity_curve = deserialize_dataframe(analysis['full_equity_curve'])
                
                plt.figure(figsize=(12, 6))
                plt.plot(equity_curve.index, equity_curve['equity'], color='blue', linewidth=2)
                
                # Add vertical lines for period boundaries
                for period in analysis['periods']:
                    valid_start = datetime.strptime(period['period']['valid_start'], "%Y-%m-%d")
                    plt.axvline(x=valid_start, color='green', linestyle='--', alpha=0.5)
                
                plt.title('Walk Forward Equity Curve')
                plt.xlabel('Date')
                plt.ylabel('Equity')
                plt.grid(True, alpha=0.3)
                
                # Format x-axis dates
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.gcf().autofmt_xdate()
                
                equity_plot_path = os.path.join(img_dir, f"equity_curve_{timestamp}.png")
                plt.savefig(equity_plot_path)
                plt.close()
                plot_paths['equity_curve'] = equity_plot_path
                
            # Plot 2: Parameter evolution
            if 'parameters' in analysis and len(analysis['parameters']) > 0:
                # Get common parameters across all periods
                common_params = set(analysis['parameters'][0].keys())
                for params in analysis['parameters'][1:]:
                    common_params.intersection_update(params.keys())
                
                # Filter to numeric parameters
                numeric_params = {}
                for param in common_params:
                    if all(isinstance(params[param], (int, float)) for params in analysis['parameters']):
                        numeric_params[param] = [params[param] for params in analysis['parameters']]
                
                if numeric_params:
                    # Create a plot for each parameter
                    plt.figure(figsize=(12, 8))
                    
                    # Calculate rows and columns for subplots
                    n_params = len(numeric_params)
                    n_cols = min(3, n_params)
                    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
                    
                    for i, (param, values) in enumerate(numeric_params.items(), 1):
                        plt.subplot(n_rows, n_cols, i)
                        plt.plot(range(1, len(values) + 1), values, 'o-', linewidth=2)
                        plt.title(param)
                        plt.xlabel('Period')
                        plt.ylabel('Value')
                        plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    params_plot_path = os.path.join(img_dir, f"parameter_evolution_{timestamp}.png")
                    plt.savefig(params_plot_path)
                    plt.close()
                    plot_paths['parameter_evolution'] = params_plot_path
                
            # Plot 3: In-sample vs out-of-sample performance
            if 'periods' in analysis and analysis['periods']:
                # Extract metrics for comparison
                in_sample_returns = [p['training_metrics']['total_return'] for p in analysis['periods']]
                out_sample_returns = [p['validation_metrics']['total_return'] for p in analysis['periods']]
                
                in_sample_sharpe = [p['training_metrics'].get('sharpe_ratio', 0) for p in analysis['periods']]
                out_sample_sharpe = [p['validation_metrics'].get('sharpe_ratio', 0) for p in analysis['periods']]
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Plot returns
                x = range(1, len(in_sample_returns) + 1)
                ax1.plot(x, in_sample_returns, 'o-', color='blue', label='In-sample')
                ax1.plot(x, out_sample_returns, 'o-', color='green', label='Out-of-sample')
                ax1.set_title('Total Return: In-sample vs Out-of-sample')
                ax1.set_xlabel('Period')
                ax1.set_ylabel('Return')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Plot Sharpe ratio
                ax2.plot(x, in_sample_sharpe, 'o-', color='blue', label='In-sample')
                ax2.plot(x, out_sample_sharpe, 'o-', color='green', label='Out-of-sample')
                ax2.set_title('Sharpe Ratio: In-sample vs Out-of-sample')
                ax2.set_xlabel('Period')
                ax2.set_ylabel('Sharpe Ratio')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                comparison_plot_path = os.path.join(img_dir, f"insample_vs_outsample_{timestamp}.png")
                plt.savefig(comparison_plot_path)
                plt.close()
                plot_paths['insample_vs_outsample'] = comparison_plot_path
            
            # Plot 4: Parameter stability
            if 'parameter_stability' in analysis and 'parameter_metrics' in analysis['parameter_stability']:
                param_metrics = analysis['parameter_stability']['parameter_metrics']
                
                # Extract stability scores
                params = list(param_metrics.keys())
                stability_scores = [param_metrics[p]['stability_score'] for p in params]
                
                # Sort by stability score
                sorted_indices = np.argsort(stability_scores)
                sorted_params = [params[i] for i in sorted_indices]
                sorted_scores = [stability_scores[i] for i in sorted_indices]
                
                # Create bar chart
                plt.figure(figsize=(12, 6))
                colors = ['#ff9999' if score < 0.5 else '#66b3ff' if score < 0.75 else '#99ff99' for score in sorted_scores]
                
                plt.barh(sorted_params, sorted_scores, color=colors)
                plt.title('Parameter Stability Scores')
                plt.xlabel('Stability Score (higher is better)')
                plt.xlim(0, 1)
                plt.grid(True, alpha=0.3)
                
                stability_plot_path = os.path.join(img_dir, f"parameter_stability_{timestamp}.png")
                plt.savefig(stability_plot_path)
                plt.close()
                plot_paths['parameter_stability'] = stability_plot_path
            
            # Plot 5: Robustness heatmap (if robustness testing was done)
            if analysis.get('robustness_results') and len(analysis['periods']) > 0:
                # Create a heatmap of robustness scores across periods
                robustness_scores = [p.get('robustness', {}).get('robustness_score', 0) 
                                    for p in analysis['periods'] if 'robustness' in p]
                
                if robustness_scores:
                    plt.figure(figsize=(10, 6))
                    plt.bar(range(1, len(robustness_scores) + 1), robustness_scores, color='purple')
                    plt.title('Strategy Robustness by Period')
                    plt.xlabel('Period')
                    plt.ylabel('Robustness Score (higher is better)')
                    plt.ylim(0, 1)
                    plt.grid(True, alpha=0.3)
                    
                    robustness_plot_path = os.path.join(img_dir, f"robustness_scores_{timestamp}.png")
                    plt.savefig(robustness_plot_path)
                    plt.close()
                    plot_paths['robustness_scores'] = robustness_plot_path
        
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            
        return plot_paths
    
    def _generate_html_report(self, analysis: Dict[str, Any], output_dir: str) -> str:
        """Generate HTML report content from the analysis results."""
        # This would normally generate a complete HTML report with tables, charts, etc.
        # For brevity, we'll just return a simple template here
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate average win rate metrics if available
        is_metrics = [p['training_metrics'].get('win_rate', 0) for p in analysis['periods']]
        os_metrics = [p['validation_metrics'].get('win_rate', 0) for p in analysis['periods']]
        
        avg_is_win_rate = sum(is_metrics) / len(is_metrics) if is_metrics else 0
        avg_os_win_rate = sum(os_metrics) / len(os_metrics) if os_metrics else 0
        
        # Count periods with 80%+ win rate
        is_80plus = sum(1 for x in is_metrics if x >= 0.8)
        os_80plus = sum(1 for x in os_metrics if x >= 0.8)
        
        is_80plus_pct = is_80plus / len(is_metrics) if is_metrics else 0
        os_80plus_pct = os_80plus / len(os_metrics) if os_metrics else 0
        
        period_rows = []
        for i, p in enumerate(analysis.get("periods", [])):
            period_rows.append(
                f"<tr>"
                f"<td>{i + 1}</td>"
                f"<td>{p['period']['train_start']} to {p['period']['train_end']}</td>"
                f"<td>{p['period']['valid_start']} to {p['period']['valid_end']}</td>"
                f"<td>{p['training_metrics'].get('total_return', 0):.2%}</td>"
                f"<td>{p['validation_metrics'].get('total_return', 0):.2%}</td>"
                f"<td>{p['training_metrics'].get('win_rate', 0):.2%}</td>"
                f"<td>{p['validation_metrics'].get('win_rate', 0):.2%}</td>"
                f"</tr>"
            )

        param_rows = []
        for param, metrics in analysis.get("parameter_stability", {}).get("parameter_metrics", {}).items():
            param_rows.append(
                f"<tr>"
                f"<td>{param}</td>"
                f"<td>{metrics['mean']:.4f}</td>"
                f"<td>{metrics['median']:.4f}</td>"
                f"<td>{metrics['std_dev']:.4f}</td>"
                f"<td>{metrics['min']:.4f}</td>"
                f"<td>{metrics['max']:.4f}</td>"
                f"<td>{metrics['stability_score']:.2f}</td>"
                f"</tr>"
            )

        robustness = (
            "High"
            if avg_os_win_rate >= 0.75 and os_80plus_pct >= 0.7 and analysis.get("parameter_stability", {}).get("overall_stability", 0) > 0.7
            else "Medium"
            if avg_os_win_rate >= 0.65 and os_80plus_pct >= 0.5 and analysis.get("parameter_stability", {}).get("overall_stability", 0) > 0.5
            else "Low"
        )

        preferred_market = (
            "Trending markets"
            if avg_os_win_rate >= 0.7 and any("trend" in str(p).lower() for p in analysis.get("parameters", [{}])[0].keys())
            else "Volatile markets"
            if any("volatility" in str(p).lower() for p in analysis.get("parameters", [{}])[0].keys())
            else "Rangebound markets"
            if any("range" in str(p).lower() for p in analysis.get("parameters", [{}])[0].keys())
            else "Various market conditions"
        )

        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Walk Forward Analysis Report</h1>
            <p>Generated on: {timestamp}</p>

            <h2>Analysis Overview</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Strategy ID</td><td>{analysis.get('strategy_id', 'N/A')}</td></tr>
                <tr><td>Method</td><td>{analysis.get('method', 'N/A')}</td></tr>
                <tr><td>Number of Periods</td><td>{len(analysis.get('periods', []))}</td></tr>
                <tr><td>Avg. In-Sample Win Rate</td><td>{avg_is_win_rate:.2%}</td></tr>
                <tr><td>Avg. Out-of-Sample Win Rate</td><td>{avg_os_win_rate:.2%}</td></tr>
                <tr><td>In-Sample Periods with 80%+ Win Rate</td><td>{is_80plus} ({is_80plus_pct:.2%})</td></tr>
                <tr><td>Out-of-Sample Periods with 80%+ Win Rate</td><td>{os_80plus} ({os_80plus_pct:.2%})</td></tr>
                <tr><td>Overall Stability Score</td><td>{analysis.get('parameter_stability', {}).get('overall_stability', 0):.2f}</td></tr>
            </table>

            <h2>Overall Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Return</td><td>{analysis.get('metrics', {}).get('total_return', 0):.2%}</td></tr>
                <tr><td>Annualized Return</td><td>{analysis.get('metrics', {}).get('annualized_return', 0):.2%}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{analysis.get('metrics', {}).get('sharpe_ratio', 0):.2f}</td></tr>
                <tr><td>Max Drawdown</td><td>{analysis.get('metrics', {}).get('max_drawdown', 0):.2%}</td></tr>
                <tr><td>Win Rate</td><td>{analysis.get('metrics', {}).get('win_rate', 0):.2%}</td></tr>
                <tr><td>Profit Factor</td><td>{analysis.get('metrics', {}).get('profit_factor', 0):.2f}</td></tr>
            </table>

            <h2>Period-by-Period Comparison</h2>
            <table>
                <tr>
                    <th>Period</th>
                    <th>Training Window</th>
                    <th>Validation Window</th>
                    <th>In-Sample Return</th>
                    <th>Out-of-Sample Return</th>
                    <th>In-Sample Win Rate</th>
                    <th>Out-of-Sample Win Rate</th>
                </tr>
                {''.join(period_rows)}
            </table>

            <h2>Parameter Stability Analysis</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Stability Score</th>
                </tr>
                {''.join(param_rows)}
            </table>

            <h2>Strategy Robustness</h2>
            <p>{'' if 'robustness_results' in analysis else ''}</p>
            <p>Overall assessment of strategy robustness: {robustness}</p>

            <h2>Conclusion and Recommendations</h2>
            <p>Based on the walk forward analysis, the following recommendations can be made:</p>
            <ul>
                <li>Probability of achieving 80%+ win rate in out-of-sample periods: {os_80plus_pct:.2%}</li>
                <li>Parameter stability: {robustness}</li>
                <li>Strategy performs best in: {preferred_market}</li>
            </ul>

            <p>Optimal Parameters: Consider using the most stable parameters identified in the analysis for live trading.</p>
            <p>Risk Management: Based on observed drawdowns, it's recommended to use proper position sizing to limit risk to an acceptable level.</p>

            <h3>Disclaimer</h3>
            <p>Past performance, even in walk forward analysis, does not guarantee future results.</p>
            <p>Markets change over time, and strategies may need to be adapted as conditions evolve.</p>
            <p>Always use proper risk management when deploying any trading strategy.</p>
        </body>
        </html>
        """

        return html_content


if __name__ == "__main__":
    # Example usage
    from backtester.engine import BacktestEngine
    from backtester.optimization import ParameterOptimizer
    
    backtest_engine = BacktestEngine()
    optimizer = ParameterOptimizer(backtest_engine)
    
    config = WalkForwardConfig(
        method=WalkForwardMethod.ROLLING,
        start_date="2022-01-01",
        end_date="2022-12-31",
        training_window=90,
        validation_window=30,
        step_size=30,
        target_parameters=["ema_short", "ema_long", "rsi_period"]
    )
    
    analyzer = WalkForwardAnalyzer(backtest_engine, optimizer, config)
    
    # Example parameters
    example_parameters = {
        "ema_short": 10,
        "ema_long": 20,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "stop_loss": 0.02,
        "take_profit": 0.04
    }
    
    async def run_example():
        results = await analyzer.run_analysis(
            strategy_id="momentum_strategy",
            parameters=example_parameters,
            assets=["BTC/USDT"],
            platform="Binance"
        )
        
        report_path = analyzer.generate_report(results)
        logger.info(f"Report generated at: {report_path}")
    
    # Run the example in an asyncio event loop
    # asyncio.run(run_example())

