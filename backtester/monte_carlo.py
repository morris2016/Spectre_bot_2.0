
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Monte Carlo Simulation Module

This module provides advanced Monte Carlo simulation capabilities for strategy validation,
risk assessment, and robustness testing. It allows for comprehensive analysis of strategy
performance across a wide range of market scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass
import json
import os
import time

from common.logger import get_logger
from common.utils import serialize_dataframe, deserialize_dataframe
from common.exceptions import SimulationError
from data_storage.time_series import TimeSeriesStorage
from backtester.engine import BacktestEngine
from backtester.performance import PerformanceAnalyzer

logger = get_logger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulations."""
    num_simulations: int = 1000
    confidence_interval: float = 0.95
    random_seed: Optional[int] = None
    parameter_variation: float = 0.1  # Parameter variation range (±10%)
    price_variation: float = 0.05     # Price data variation range (±5%)
    volume_variation: float = 0.2     # Volume data variation range (±20%)
    spread_variation: float = 0.3     # Spread variation range (±30%)
    latency_variation: float = 0.5    # Execution latency variation (±50%)
    stochastic_fills: bool = True     # Use stochastic fill probabilities
    sequence_bootstrapping: bool = True  # Use sequence bootstrapping
    scenario_weights: Dict[str, float] = None  # Scenario weights
    cpu_cores: int = -1  # Use all available cores if -1
    
    def __post_init__(self):
        """Validate the configuration."""
        if self.num_simulations < 100:
            logger.warning(f"Low simulation count ({self.num_simulations}), results may not be statistically significant")
        
        if self.confidence_interval < 0.5 or self.confidence_interval > 0.99:
            raise ValueError(f"Confidence interval must be between 0.5 and 0.99, got {self.confidence_interval}")
        
        # Initialize default scenario weights if None
        if self.scenario_weights is None:
            self.scenario_weights = {
                "normal": 0.6,
                "volatile": 0.2,
                "trending": 0.1,
                "sideways": 0.1
            }
        
        # Validate scenario weights sum to 1.0
        if abs(sum(self.scenario_weights.values()) - 1.0) > 0.001:
            raise ValueError(f"Scenario weights must sum to 1.0, got {sum(self.scenario_weights.values())}")


class MonteCarloSimulator:
    """
    Advanced Monte Carlo simulation engine for strategy robustness testing.
    
    This class provides sophisticated Monte Carlo simulation capabilities including:
    - Strategy parameter perturbation
    - Market data variation
    - Execution quality simulation
    - Bootstrapping of historical sequences
    - Various market regime simulations
    - Multi-asset correlation preservation
    """
    
    def __init__(self, backtest_engine: BacktestEngine, config: Optional[MonteCarloConfig] = None):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            backtest_engine: The backtesting engine to use for simulations
            config: Configuration for the Monte Carlo simulations
        """
        self.backtest_engine = backtest_engine
        self.config = config or MonteCarloConfig()
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            
        self._time_series_store = TimeSeriesStore()
        self._results_cache = {}
        self._perf_analyzer = PerformanceAnalyzer()
        
        # Set CPU cores for parallel processing
        self._cpu_cores = self.config.cpu_cores
        if self._cpu_cores == -1:
            import multiprocessing
            self._cpu_cores = multiprocessing.cpu_count()
        
        logger.info(f"Initialized Monte Carlo simulator with {self.config.num_simulations} simulations "
                   f"and {self._cpu_cores} CPU cores")
    
    async def run_simulations(self, strategy_id: str, parameters: Dict[str, Any],
                             start_date: str, end_date: str, assets: List[str],
                             platform: str) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations for a strategy with parameter and market variations.
        
        Args:
            strategy_id: Identifier of the strategy to test
            parameters: Strategy parameters
            start_date: Start date for the simulation
            end_date: End date for the simulation
            assets: List of assets to include in the simulation
            platform: Trading platform (Binance/Deriv)
            
        Returns:
            Dictionary containing the simulation results
        """
        logger.info(f"Running {self.config.num_simulations} Monte Carlo simulations for strategy {strategy_id}")
        
        # Generate simulation tasks
        simulation_tasks = []
        for i in range(self.config.num_simulations):
            # Create a copy of parameters with variation
            params_variation = self._generate_parameter_variation(parameters)
            
            # Generate scenario for this simulation
            scenario = self._select_scenario_by_weight()
            
            # Create simulation task
            sim_task = {
                'sim_id': i,
                'strategy_id': strategy_id,
                'parameters': params_variation,
                'start_date': start_date,
                'end_date': end_date,
                'assets': assets,
                'platform': platform,
                'scenario': scenario
            }
            simulation_tasks.append(sim_task)
        
        # Execute simulations in parallel
        start_time = time.time()
        results = await self._execute_simulations_parallel(simulation_tasks)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Completed {self.config.num_simulations} Monte Carlo simulations in {elapsed_time:.2f} seconds")
        
        # Analyze results
        analysis = self._analyze_simulation_results(results)
        
        # Cache results
        cache_key = f"{strategy_id}_{start_date}_{end_date}_{platform}"
        self._results_cache[cache_key] = {
            'raw_results': results,
            'analysis': analysis,
            'timestamp': time.time()
        }
        
        return analysis
    
    async def _execute_simulations_parallel(self, simulation_tasks: List[Dict]) -> List[Dict]:
        """Execute simulation tasks in parallel using ProcessPoolExecutor."""
        results = []
        
        # Split tasks into batches to avoid memory issues
        batch_size = max(1, self.config.num_simulations // (self._cpu_cores * 2))
        batches = [simulation_tasks[i:i + batch_size] 
                  for i in range(0, len(simulation_tasks), batch_size)]
        
        logger.debug(f"Split {len(simulation_tasks)} tasks into {len(batches)} batches of {batch_size}")
        
        for batch_idx, batch in enumerate(batches):
            logger.debug(f"Processing batch {batch_idx+1}/{len(batches)}")
            
            with ProcessPoolExecutor(max_workers=self._cpu_cores) as executor:
                # Create a list of futures
                futures = [
                    executor.submit(self._execute_single_simulation, task)
                    for task in batch
                ]
                
                # Get results as they complete
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                        if len(results) % 50 == 0:
                            logger.debug(f"Completed {len(results)}/{self.config.num_simulations} simulations")
                    except Exception as e:
                        logger.error(f"Simulation failed: {str(e)}")
                        # Add a failed result to maintain the count
                        results.append({
                            'success': False,
                            'error': str(e)
                        })
        
        return results
    
    def _execute_single_simulation(self, task: Dict) -> Dict:
        """Execute a single simulation with the given parameters and scenario."""
        try:
            # Apply scenario adjustments to market data
            modified_data = self._apply_scenario_to_data(task['assets'], task['start_date'], 
                                                        task['end_date'], task['scenario'])
            
            # Run backtest with modified data and parameters
            backtest_result = self.backtest_engine.run_backtest(
                strategy_id=task['strategy_id'],
                parameters=task['parameters'],
                start_date=task['start_date'],
                end_date=task['end_date'],
                assets=task['assets'],
                platform=task['platform'],
                custom_data=modified_data,
                simulation_mode=True
            )
            
            # Calculate performance metrics
            metrics = self._perf_analyzer.calculate_metrics(backtest_result)
            
            return {
                'sim_id': task['sim_id'],
                'strategy_id': task['strategy_id'],
                'scenario': task['scenario'],
                'parameters': task['parameters'],
                'trades': backtest_result['trades'],
                'equity_curve': serialize_dataframe(backtest_result['equity_curve']),
                'metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in simulation {task['sim_id']}: {str(e)}")
            return {
                'sim_id': task['sim_id'],
                'strategy_id': task['strategy_id'],
                'scenario': task['scenario'],
                'success': False,
                'error': str(e)
            }
    
    def _generate_parameter_variation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameter variations based on configuration."""
        if not self.config.parameter_variation:
            return parameters.copy()
        
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
    
    def _select_scenario_by_weight(self) -> str:
        """Select a scenario based on configured weights."""
        scenarios = list(self.config.scenario_weights.keys())
        weights = list(self.config.scenario_weights.values())
        
        return np.random.choice(scenarios, p=weights)
    
    def _apply_scenario_to_data(self, assets: List[str], start_date: str, 
                               end_date: str, scenario: str) -> Dict[str, pd.DataFrame]:
        """
        Apply a specific scenario to market data.
        
        This creates data variations based on the selected scenario:
        - normal: Small random variations
        - volatile: Increased volatility
        - trending: Enhanced trends
        - sideways: Reduced volatility, muted trends
        
        Returns:
            Dictionary mapping asset to modified DataFrames
        """
        modified_data = {}
        
        for asset in assets:
            # Fetch original data
            original_df = self._time_series_store.get_ohlcv_data(
                asset=asset, 
                start_date=start_date, 
                end_date=end_date
            )
            
            if original_df is None or original_df.empty:
                raise SimulationError(f"No data available for asset {asset} in date range {start_date} to {end_date}")
            
            # Create copy for modification
            df = original_df.copy()
            
            # Apply different modifications based on scenario
            if scenario == "normal":
                df = self._apply_normal_variations(df)
            elif scenario == "volatile":
                df = self._apply_volatile_variations(df)
            elif scenario == "trending":
                df = self._apply_trending_variations(df)
            elif scenario == "sideways":
                df = self._apply_sideways_variations(df)
            else:
                logger.warning(f"Unknown scenario: {scenario}, applying normal variations")
                df = self._apply_normal_variations(df)
            
            # Ensure data consistency
            df = self._ensure_data_consistency(df)
            
            modified_data[asset] = df
            
        return modified_data
    
    def _apply_normal_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normal variations to price and volume data."""
        # Small random price variations
        price_variation = self.config.price_variation * 0.5  # Reduced variation for normal scenario
        
        # Apply variations to OHLC (preserving relationships)
        close_variation = 1 + np.random.uniform(-price_variation, price_variation, len(df))
        df['close'] = df['close'] * close_variation
        
        # Maintain OHLC relationships
        for i in range(len(df)):
            # Calculate original price range and center
            original_range = df['high'].iloc[i] - df['low'].iloc[i]
            original_center = (df['high'].iloc[i] + df['low'].iloc[i]) / 2
            
            # Calculate new center based on close variation
            new_center = original_center * close_variation[i]
            
            # Apply range variation
            range_variation = 1 + np.random.uniform(-price_variation, price_variation)
            new_range = original_range * range_variation
            
            # Update high and low while preserving relationships
            df.loc[df.index[i], 'high'] = new_center + (new_range / 2)
            df.loc[df.index[i], 'low'] = new_center - (new_range / 2)
            
            # Ensure open is within high-low range
            df.loc[df.index[i], 'open'] = df['open'].iloc[i] * close_variation[i]
            df.loc[df.index[i], 'open'] = min(df['high'].iloc[i], max(df['low'].iloc[i], df['open'].iloc[i]))
        
        # Volume variations
        volume_variation = self.config.volume_variation
        df['volume'] = df['volume'] * (1 + np.random.uniform(-volume_variation, volume_variation, len(df)))
        df['volume'] = df['volume'].astype(int).clip(1)  # Ensure positive integers
        
        return df
    
    def _apply_volatile_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply high volatility variations to price and volume data."""
        # Enhanced price variations
        price_variation = self.config.price_variation * 2.0  # Increased variation for volatile scenario
        
        # Apply stronger variations to OHLC (preserving relationships)
        close_variation = 1 + np.random.uniform(-price_variation, price_variation, len(df))
        df['close'] = df['close'] * close_variation
        
        # Maintain OHLC relationships with enhanced ranges
        for i in range(len(df)):
            # Calculate original price range and center
            original_range = df['high'].iloc[i] - df['low'].iloc[i]
            original_center = (df['high'].iloc[i] + df['low'].iloc[i]) / 2
            
            # Calculate new center based on close variation
            new_center = original_center * close_variation[i]
            
            # Apply enhanced range variation for volatility
            range_variation = 1 + np.random.uniform(0, price_variation * 3)  # Prefer expanded ranges
            new_range = original_range * range_variation
            
            # Update high and low with wider ranges
            df.loc[df.index[i], 'high'] = new_center + (new_range / 2)
            df.loc[df.index[i], 'low'] = new_center - (new_range / 2)
            
            # Ensure open is within high-low range
            df.loc[df.index[i], 'open'] = df['open'].iloc[i] * close_variation[i]
            df.loc[df.index[i], 'open'] = min(df['high'].iloc[i], max(df['low'].iloc[i], df['open'].iloc[i]))
        
        # Enhanced volume variations
        volume_variation = self.config.volume_variation * 2.0
        df['volume'] = df['volume'] * (1 + np.random.uniform(-volume_variation, volume_variation * 1.5, len(df)))
        df['volume'] = df['volume'].astype(int).clip(1)  # Ensure positive integers
        
        return df
    
    def _apply_trending_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply trending variations to price data."""
        # First apply normal variations
        df = self._apply_normal_variations(df)
        
        # Then add a trending component
        # Randomly decide if the trend is up or down
        trend_direction = 1 if np.random.random() > 0.5 else -1
        trend_strength = np.random.uniform(0.002, 0.005)  # Daily trend factor
        
        # Apply cumulative trend
        trend_factors = np.array([1.0 + (trend_direction * trend_strength * i) for i in range(len(df))])
        
        # Apply trend to close prices
        df['close'] = df['close'] * trend_factors
        
        # Adjust OHLC to maintain relationships
        for i in range(len(df)):
            factor = trend_factors[i]
            df.loc[df.index[i], 'open'] = df['open'].iloc[i] * factor
            df.loc[df.index[i], 'high'] = df['high'].iloc[i] * factor
            df.loc[df.index[i], 'low'] = df['low'].iloc[i] * factor
        
        # Trending markets often have increasing volume in the trend direction
        volume_trend = np.array([1.0 + (trend_direction * 0.01 * i) for i in range(len(df))])
        df['volume'] = df['volume'] * volume_trend
        df['volume'] = df['volume'].astype(int).clip(1)
        
        return df
    
    def _apply_sideways_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sideways market variations with reduced volatility and range."""
        # Reduced price variations
        price_variation = self.config.price_variation * 0.3  # Reduced variation
        
        # Apply small variations to OHLC (preserving relationships)
        close_variation = 1 + np.random.uniform(-price_variation, price_variation, len(df))
        df['close'] = df['close'] * close_variation
        
        # Calculate mean price for centering the sideways range
        mean_price = df['close'].mean()
        
        # Pull prices toward the mean to create sideways behavior
        pull_strength = np.random.uniform(0.1, 0.3)  # How strongly to pull toward mean
        df['close'] = df['close'] * (1 - pull_strength) + mean_price * pull_strength
        
        # Maintain OHLC relationships with reduced ranges
        for i in range(len(df)):
            # Calculate original price range and center
            original_range = df['high'].iloc[i] - df['low'].iloc[i]
            
            # Apply reduced range variation for sideways market
            range_variation = 1 - np.random.uniform(0, 0.5)  # Compress ranges
            new_range = original_range * range_variation
            
            # Center around the close price
            new_center = df['close'].iloc[i]
            
            # Update high and low with narrower ranges
            df.loc[df.index[i], 'high'] = new_center + (new_range / 2)
            df.loc[df.index[i], 'low'] = new_center - (new_range / 2)
            
            # Ensure open is within high-low range
            df.loc[df.index[i], 'open'] = min(df['high'].iloc[i], max(df['low'].iloc[i], df['open'].iloc[i]))
        
        # Reduced volume variations
        df['volume'] = df['volume'] * np.random.uniform(0.5, 0.8, len(df))
        df['volume'] = df['volume'].astype(int).clip(1)
        
        return df
    
    def _ensure_data_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data consistency: high >= open/close >= low, and volumes > 0."""
        # Ensure high is maximum
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        
        # Ensure low is minimum
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Handle any inversion of high/low caused by randomization
        if (df['high'] < df['low']).any():
            # Swap high and low where inverted
            inverted = df['high'] < df['low']
            high_temp = df.loc[inverted, 'high'].copy()
            df.loc[inverted, 'high'] = df.loc[inverted, 'low']
            df.loc[inverted, 'low'] = high_temp
        
        # Ensure open and close are within high-low range
        df['open'] = df['open'].clip(df['low'], df['high'])
        df['close'] = df['close'].clip(df['low'], df['high'])
        
        # Ensure positive volumes
        df['volume'] = df['volume'].clip(1)
        
        return df
    
    def _analyze_simulation_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze Monte Carlo simulation results.
        
        This function calculates various statistics and distributions from the simulation results:
        - Success rate of simulations
        - Distribution of returns
        - Confidence intervals
        - Drawdown statistics
        - Best/worst case scenarios
        - Risk-adjusted metrics distribution
        """
        # Filter out failed simulations
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            raise SimulationError("All simulations failed, cannot perform analysis")
        
        success_rate = len(successful_results) / len(results)
        logger.info(f"Simulation success rate: {success_rate:.2%}")
        
        # Extract metrics from successful simulations
        returns = [r['metrics']['total_return'] for r in successful_results]
        sharpe_ratios = [r['metrics']['sharpe_ratio'] for r in successful_results]
        max_drawdowns = [r['metrics']['max_drawdown'] for r in successful_results]
        win_rates = [r['metrics']['win_rate'] for r in successful_results]
        profit_factors = [r['metrics']['profit_factor'] for r in successful_results]
        
        # Calculate confidence intervals
        alpha = 1 - self.config.confidence_interval
        ci_lower_idx = int(np.floor(alpha/2 * len(returns)))
        ci_upper_idx = int(np.ceil((1-alpha/2) * len(returns)))
        
        # Sort arrays for percentile calculations
        returns_sorted = sorted(returns)
        drawdowns_sorted = sorted(max_drawdowns)
        win_rates_sorted = sorted(win_rates)
        
        # Group results by scenario type
        scenario_results = {}
        for r in successful_results:
            scenario = r['scenario']
            if scenario not in scenario_results:
                scenario_results[scenario] = []
            scenario_results[scenario].append(r)
        
        # Calculate statistics by scenario
        scenario_stats = {}
        for scenario, res_list in scenario_results.items():
            sc_returns = [r['metrics']['total_return'] for r in res_list]
            sc_drawdowns = [r['metrics']['max_drawdown'] for r in res_list]
            sc_win_rates = [r['metrics']['win_rate'] for r in res_list]
            
            scenario_stats[scenario] = {
                'count': len(res_list),
                'avg_return': np.mean(sc_returns),
                'median_return': np.median(sc_returns),
                'avg_drawdown': np.mean(sc_drawdowns),
                'avg_win_rate': np.mean(sc_win_rates),
                'return_std': np.std(sc_returns),
                'worst_return': min(sc_returns),
                'best_return': max(sc_returns)
            }
        
        # Find best and worst performing simulations
        best_sim_idx = returns.index(max(returns))
        worst_sim_idx = returns.index(min(returns))
        
        best_sim = successful_results[best_sim_idx]
        worst_sim = successful_results[worst_sim_idx]
        
        # Calculate key statistics
        analysis = {
            'simulation_count': len(results),
            'successful_count': len(successful_results),
            'success_rate': success_rate,
            
            'return_statistics': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std_dev': np.std(returns),
                'min': min(returns),
                'max': max(returns),
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'confidence_interval': [returns_sorted[ci_lower_idx], returns_sorted[ci_upper_idx]],
                'percentiles': {
                    '5': np.percentile(returns, 5),
                    '25': np.percentile(returns, 25),
                    '50': np.percentile(returns, 50),
                    '75': np.percentile(returns, 75),
                    '95': np.percentile(returns, 95)
                }
            },
            
            'drawdown_statistics': {
                'mean': np.mean(max_drawdowns),
                'median': np.median(max_drawdowns),
                'std_dev': np.std(max_drawdowns),
                'max': max(max_drawdowns),
                'min': min(max_drawdowns),
                'confidence_interval': [
                    np.percentile(max_drawdowns, 100 * alpha/2),
                    np.percentile(max_drawdowns, 100 * (1-alpha/2))
                ],
                'percentiles': {
                    '5': np.percentile(max_drawdowns, 5),
                    '25': np.percentile(max_drawdowns, 25),
                    '50': np.percentile(max_drawdowns, 50),
                    '75': np.percentile(max_drawdowns, 75),
                    '95': np.percentile(max_drawdowns, 95)
                }
            },
            
            'win_rate_statistics': {
                'mean': np.mean(win_rates),
                'median': np.median(win_rates),
                'std_dev': np.std(win_rates),
                'min': min(win_rates),
                'max': max(win_rates),
                'confidence_interval': [
                    np.percentile(win_rates, 100 * alpha/2),
                    np.percentile(win_rates, 100 * (1-alpha/2))
                ],
                'percentiles': {
                    '5': np.percentile(win_rates, 5),
                    '25': np.percentile(win_rates, 25),
                    '50': np.percentile(win_rates, 50),
                    '75': np.percentile(win_rates, 75),
                    '95': np.percentile(win_rates, 95)
                },
                '80_percent_threshold': np.sum(np.array(win_rates) >= 0.8) / len(win_rates)
            },
            
            'risk_adjusted_statistics': {
                'sharpe_ratio': {
                    'mean': np.mean(sharpe_ratios),
                    'median': np.median(sharpe_ratios),
                    'std_dev': np.std(sharpe_ratios),
                    'min': min(sharpe_ratios),
                    'max': max(sharpe_ratios)
                },
                'profit_factor': {
                    'mean': np.mean(profit_factors),
                    'median': np.median(profit_factors),
                    'std_dev': np.std(profit_factors),
                    'min': min(profit_factors),
                    'max': max(profit_factors)
                },
                'return_to_drawdown': {
                    'mean': np.mean([r/d if d != 0 else float('inf') for r, d in zip(returns, max_drawdowns)]),
                    'median': np.median([r/d if d != 0 else float('inf') for r, d in zip(returns, max_drawdowns)])
                }
            },
            
            'scenario_analysis': scenario_stats,
            
            'best_simulation': {
                'sim_id': best_sim['sim_id'],
                'scenario': best_sim['scenario'],
                'metrics': best_sim['metrics'],
                'parameters': best_sim['parameters']
            },
            
            'worst_simulation': {
                'sim_id': worst_sim['sim_id'],
                'scenario': worst_sim['scenario'],
                'metrics': worst_sim['metrics'],
                'parameters': worst_sim['parameters']
            },
            
            'profit_probability': np.mean(np.array(returns) > 0),
            'target_return_probability': {
                '5_percent': np.mean(np.array(returns) > 0.05),
                '10_percent': np.mean(np.array(returns) > 0.10),
                '20_percent': np.mean(np.array(returns) > 0.20),
                '50_percent': np.mean(np.array(returns) > 0.50)
            },
            
            'ruin_probability': np.mean(np.array(max_drawdowns) > 0.5),
            
            'timestamp': time.time()
        }
        
        # Calculate risk of ruin based on probability of specified drawdown
        analysis['risk_of_ruin'] = {
            'drawdown_30_percent': np.mean(np.array(max_drawdowns) > 0.3),
            'drawdown_50_percent': np.mean(np.array(max_drawdowns) > 0.5),
            'drawdown_80_percent': np.mean(np.array(max_drawdowns) > 0.8),
            'drawdown_90_percent': np.mean(np.array(max_drawdowns) > 0.9)
        }
        
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any], output_dir: str = None) -> str:
        """Generate a comprehensive HTML report from the simulation results."""
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'reports', 'monte_carlo')
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create report filename based on timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_path = os.path.join(output_dir, f"monte_carlo_report_{timestamp}.html")
        
        # Generate plots and save them
        self._generate_plots(analysis, output_dir)
        
        # Create HTML report
        html_content = self._generate_html_report(analysis, output_dir)
        
        # Write report to file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated Monte Carlo report at {report_path}")
        return report_path
    
    def _generate_plots(self, analysis: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Generate plots for the Monte Carlo analysis report."""
        plot_paths = {}
        
        # Ensure the images directory exists
        img_dir = os.path.join(output_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        
        # Common timestamp for all plot files
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Plot 1: Return Distribution
        plt.figure(figsize=(10, 6))
        returns = [r for r in self._results_cache[list(self._results_cache.keys())[0]]['raw_results'] 
                  if r.get('success', False)]
        returns = [r['metrics']['total_return'] for r in returns]
        
        plt.hist(returns, bins=50, alpha=0.7, color='blue')
        plt.axvline(analysis['return_statistics']['mean'], color='red', linestyle='dashed', linewidth=2)
        plt.axvline(analysis['return_statistics']['confidence_interval'][0], color='green', linestyle='dashed', linewidth=2)
        plt.axvline(analysis['return_statistics']['confidence_interval'][1], color='green', linestyle='dashed', linewidth=2)
        plt.title('Distribution of Returns')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        return_plot_path = os.path.join(img_dir, f"return_distribution_{timestamp}.png")
        plt.savefig(return_plot_path)
        plt.close()
        plot_paths['return_distribution'] = return_plot_path
        
        # Plot 2: Drawdown Distribution
        plt.figure(figsize=(10, 6))
        drawdowns = [r['metrics']['max_drawdown'] for r in returns if r.get('success', False)]
        
        plt.hist(drawdowns, bins=50, alpha=0.7, color='red')
        plt.axvline(analysis['drawdown_statistics']['mean'], color='blue', linestyle='dashed', linewidth=2)
        plt.axvline(analysis['drawdown_statistics']['confidence_interval'][0], color='green', linestyle='dashed', linewidth=2)
        plt.axvline(analysis['drawdown_statistics']['confidence_interval'][1], color='green', linestyle='dashed', linewidth=2)
        plt.title('Distribution of Maximum Drawdowns')
        plt.xlabel('Drawdown (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        drawdown_plot_path = os.path.join(img_dir, f"drawdown_distribution_{timestamp}.png")
        plt.savefig(drawdown_plot_path)
        plt.close()
        plot_paths['drawdown_distribution'] = drawdown_plot_path
        
        # Plot 3: Win Rate Distribution
        plt.figure(figsize=(10, 6))
        win_rates = [r['metrics']['win_rate'] for r in returns if r.get('success', False)]
        
        plt.hist(win_rates, bins=50, alpha=0.7, color='green')
        plt.axvline(analysis['win_rate_statistics']['mean'], color='blue', linestyle='dashed', linewidth=2)
        plt.axvline(0.8, color='red', linestyle='dashed', linewidth=2)  # 80% win rate threshold
        plt.title('Distribution of Win Rates')
        plt.xlabel('Win Rate (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        win_rate_plot_path = os.path.join(img_dir, f"win_rate_distribution_{timestamp}.png")
        plt.savefig(win_rate_plot_path)
        plt.close()
        plot_paths['win_rate_distribution'] = win_rate_plot_path
        
        # Plot 4: Scenario Comparison
        plt.figure(figsize=(12, 8))
        scenarios = list(analysis['scenario_analysis'].keys())
        avg_returns = [analysis['scenario_analysis'][s]['avg_return'] for s in scenarios]
        avg_drawdowns = [analysis['scenario_analysis'][s]['avg_drawdown'] for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width/2, avg_returns, width, label='Avg Return')
        rects2 = ax.bar(x + width/2, avg_drawdowns, width, label='Avg Drawdown')
        
        ax.set_ylabel('Value')
        ax.set_title('Average Return and Drawdown by Scenario')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.legend()
        
        # Add labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
                
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        
        scenario_plot_path = os.path.join(img_dir, f"scenario_comparison_{timestamp}.png")
        plt.savefig(scenario_plot_path)
        plt.close()
        plot_paths['scenario_comparison'] = scenario_plot_path
        
        return plot_paths
    
    def _generate_html_report(self, analysis: Dict[str, Any], output_dir: str) -> str:
        """Generate HTML report content from the analysis results."""
        # This would normally generate a complete HTML report with tables, charts, etc.
        # For brevity, we'll just return a simple template here
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        
        
        
            Monte Carlo Simulation Report - {timestamp}
            
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; text-align: left; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .section {{ margin-bottom: 30px; }}
                .highlight {{ background-color: #ffffcc; }}
                .img-container {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; }}
            
        
        
            Monte Carlo Simulation Report
            Generated on: {timestamp}
            
            
                Simulation Overview
                
                    MetricValue
                    Total Simulations{analysis['simulation_count']}
                    Successful Simulations{analysis['successful_count']}
                    Success Rate{analysis['success_rate']:.2%}
                    Mean Return{analysis['return_statistics']['mean']:.2%}
                    Median Return{analysis['return_statistics']['median']:.2%}
                    Return Standard Deviation{analysis['return_statistics']['std_dev']:.2%}
                    Mean Maximum Drawdown{analysis['drawdown_statistics']['mean']:.2%}
                    Mean Win Rate{analysis['win_rate_statistics']['mean']:.2%}
                    Probability of 80%+ Win Rate{analysis['win_rate_statistics']['80_percent_threshold']:.2%}
                    Probability of Profit{analysis['profit_probability']:.2%}
                
            
            
            
                Return Distribution
                
                    
                
                
                    PercentileReturn
                    5th{analysis['return_statistics']['percentiles']['5']:.2%}
                    25th{analysis['return_statistics']['percentiles']['25']:.2%}
                    50th (Median){analysis['return_statistics']['percentiles']['50']:.2%}
                    75th{analysis['return_statistics']['percentiles']['75']:.2%}
                    95th{analysis['return_statistics']['percentiles']['95']:.2%}
                    Confidence Interval ({self.config.confidence_interval:.0%})
                        {analysis['return_statistics']['confidence_interval'][0]:.2%} to {analysis['return_statistics']['confidence_interval'][1]:.2%}
                
            
            
            
                Risk Analysis
                
                    Risk MetricValue
                    Maximum Drawdown Range ({self.config.confidence_interval:.0%} CI)
                        {analysis['drawdown_statistics']['confidence_interval'][0]:.2%} to {analysis['drawdown_statistics']['confidence_interval'][1]:.2%}
                    Risk of 30% Drawdown{analysis['risk_of_ruin']['drawdown_30_percent']:.2%}
                    Risk of 50% Drawdown{analysis['risk_of_ruin']['drawdown_50_percent']:.2%}
                    Risk of 80% Drawdown{analysis['risk_of_ruin']['drawdown_80_percent']:.2%}
                    Risk of 90% Drawdown{analysis['risk_of_ruin']['drawdown_90_percent']:.2%}
                
            
            
            
                Scenario Analysis
                
                    {''.join([f"" for s, stats in analysis['scenario_analysis'].items()])}
                
                    
                        Scenario
                        Count
                        Avg Return
                        Avg Drawdown
                        Avg Win Rate
                        Best Return
                        Worst Return
                    {s}{stats['count']}{stats['avg_return']:.2%}{stats['avg_drawdown']:.2%}{stats['avg_win_rate']:.2%}{stats['best_return']:.2%}{stats['worst_return']:.2%}
            
            
            
                Best Performing Simulation
                
                    MetricValue
                    Simulation ID{analysis['best_simulation']['sim_id']}
                    Scenario{analysis['best_simulation']['scenario']}
                    Total Return{analysis['best_simulation']['metrics']['total_return']:.2%}
                    Win Rate{analysis['best_simulation']['metrics']['win_rate']:.2%}
                    Max Drawdown{analysis['best_simulation']['metrics']['max_drawdown']:.2%}
                    Sharpe Ratio{analysis['best_simulation']['metrics']['sharpe_ratio']:.2f}
                
                
                Best Simulation Parameters
                
                    {''.join([f"" for k, v in analysis['best_simulation']['parameters'].items()])}
                
                    ParameterValue{k}{v}
            
            
            
                Recommendations
                Based on the Monte Carlo simulations, the following recommendations can be made:
                
                    Probability of achieving 80%+ win rate: {analysis['win_rate_statistics']['80_percent_threshold']:.2%}
                    Expected return range ({self.config.confidence_interval:.0%} confidence): {analysis['return_statistics']['confidence_interval'][0]:.2%} to {analysis['return_statistics']['confidence_interval'][1]:.2%}
                    Expected maximum drawdown: {analysis['drawdown_statistics']['mean']:.2%}
                    Best performing scenario: {max(analysis['scenario_analysis'].items(), key=lambda x: x[1]['avg_return'])[0]}
                
                
                Strategy Robustness: 
                    {
                        "High" if analysis['profit_probability'] > 0.8 and analysis['win_rate_statistics']['80_percent_threshold'] > 0.7 
                        else "Medium" if analysis['profit_probability'] > 0.6 and analysis['win_rate_statistics']['80_percent_threshold'] > 0.5 
                        else "Low"
                    }
                
                
                Parameter Recommendation: Consider using the parameters from the best performing simulation.
            
            
            
                Disclaimer
                Monte Carlo simulation results are based on random variations of historical data and strategy parameters. 
                Past performance and simulations do not guarantee future results. These simulations should be used as one 
                of many tools in the decision-making process.
            
        
        
        """
        
        return html_content


if __name__ == "__main__":
    # Example usage
    from backtester.engine import BacktestEngine
    
    backtest_engine = BacktestEngine()
    config = MonteCarloConfig(num_simulations=100)  # Reduced for demonstration
    
    simulator = MonteCarloSimulator(backtest_engine, config)
    
    # This would be replaced with actual parameters in production
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
        results = await simulator.run_simulations(
            strategy_id="momentum_strategy",
            parameters=example_parameters,
            start_date="2022-01-01",
            end_date="2022-12-31",
            assets=["BTC/USDT"],
            platform="Binance"
        )
        
        report_path = simulator.generate_report(results)
        logger.info(f"Report generated at: {report_path}")
    
    # Run the example in an asyncio event loop
    # asyncio.run(run_example())

