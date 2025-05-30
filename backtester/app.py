#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Backtester Application Service

This module implements the main backtester service that orchestrates the backtesting
process, including data loading, strategy testing, optimization, performance analysis, 
and reporting.
"""

import os
import sys
import time
import asyncio
import logging
import threading
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import uuid
import signal
import traceback
import contextlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Internal imports
from config import Config
from common.logger import get_logger
from common.async_utils import AsyncTaskManager, create_task_group
from common.metrics import MetricsCollector
from common.exceptions import (
    BacktestConfigError, BacktestDataError, BacktestStrategyError, 
    BacktestExecutionError, BacktestOptimizationError
)
from common.utils import TimeFrame, format_timestamp, parse_timeframe

from backtester.engine import BacktestEngine
from backtester.data_provider import DataProvider
from backtester.performance import Performance
from backtester.optimization import Optimization
from backtester.scenario import Scenario
from backtester.report import Report

# Initialize module logger
logger = get_logger('backtester.app')

class BacktesterService:
    """
    Main backtester service for the QuantumSpectre Elite Trading System.
    
    This service orchestrates the backtesting process, including:
    - Loading and preparing historical data
    - Executing backtests with various strategies and parameters
    - Analyzing performance results
    - Optimizing strategy parameters
    - Running scenario tests
    - Generating reports
    
    The service supports multiple backtest modes, including standard OHLCV backtesting,
    tick-by-tick simulation, walk-forward optimization, and Monte Carlo simulations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backtester service.
        
        Args:
            config: Service configuration dictionary (optional)
        """
        self.config = config or {}
        self.base_config = Config.get_section('backtester', {})
        self.config = {**self.base_config, **self.config}
        
        self.data_dir = Path(self.config.get('data_dir', 'data/backtest'))
        self.results_dir = Path(self.config.get('results_dir', 'results/backtest'))
        self.max_workers = self.config.get('max_workers', mp.cpu_count())
        self.default_mode = self.config.get('default_mode', 'standard')
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics = MetricsCollector('backtester')
        self.task_manager = AsyncTaskManager()
        self.active_tasks = {}
        self.results_cache = {}
        self._running = False
        self._initialized = False
        self._stop_event = threading.Event()
        
        # Component instances - lazy initialized
        self._engine = None
        self._data_provider = None
        self._performance = None
        self._optimization = None
        self._scenario = None
        self._report = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info("Backtester service initialized")
    
    @property
    def engine(self) -> BacktestEngine:
        """Get the backtest engine instance, lazy initializing if necessary."""
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = BacktestEngine(
                        config=self.config.get('engine', {}),
                        metrics=self.metrics
                    )
        return self._engine
    
    @property
    def data_provider(self) -> DataProvider:
        """Get the data provider instance, lazy initializing if necessary."""
        if self._data_provider is None:
            with self._lock:
                if self._data_provider is None:
                    self._data_provider = DataProvider(
                        config=self.config.get('data_provider', {}),
                        data_dir=self.data_dir
                    )
        return self._data_provider
    
    @property
    def performance(self) -> Performance:
        """Get the performance analyzer instance, lazy initializing if necessary."""
        if self._performance is None:
            with self._lock:
                if self._performance is None:
                    self._performance = Performance(
                        config=self.config.get('performance', {})
                    )
        return self._performance
    
    @property
    def optimization(self) -> Optimization:
        """Get the optimization instance, lazy initializing if necessary."""
        if self._optimization is None:
            with self._lock:
                if self._optimization is None:
                    self._optimization = Optimization(
                        config=self.config.get('optimization', {})
                    )
        return self._optimization
    
    @property
    def scenario(self) -> Scenario:
        """Get the scenario instance, lazy initializing if necessary."""
        if self._scenario is None:
            with self._lock:
                if self._scenario is None:
                    self._scenario = Scenario(
                        config=self.config.get('scenario', {})
                    )
        return self._scenario
    
    @property
    def report(self) -> Report:
        """Get the report instance, lazy initializing if necessary."""
        if self._report is None:
            with self._lock:
                if self._report is None:
                    self._report = Report(
                        config=self.config.get('report', {}),
                        results_dir=self.results_dir
                    )
        return self._report
    
    async def start(self) -> None:
        """Start the backtester service."""
        if self._running:
            logger.warning("Backtester service is already running")
            return
        
        logger.info("Starting backtester service")
        self._running = True
        self._stop_event.clear()
        
        # Initialize all components
        await self._initialize_components()
        
        # Start background tasks
        await self._start_background_tasks()
        
        self._initialized = True
        logger.info("Backtester service started")
    
    async def stop(self) -> None:
        """Stop the backtester service."""
        if not self._running:
            logger.warning("Backtester service is not running")
            return
        
        logger.info("Stopping backtester service")
        self._running = False
        self._stop_event.set()
        
        # Cancel active tasks
        await self._cancel_active_tasks()
        
        # Stop task manager
        await self.task_manager.shutdown()
        
        logger.info("Backtester service stopped")
    
    async def _initialize_components(self) -> None:
        """Initialize all component instances."""
        # Access properties to trigger lazy initialization
        _ = self.engine
        _ = self.data_provider
        _ = self.performance
        _ = self.optimization
        _ = self.scenario
        _ = self.report
        
        # Initialize components if they have async initialization
        init_tasks = []
        
        for component_name in ['engine', 'data_provider', 'performance', 
                               'optimization', 'scenario', 'report']:
            component = getattr(self, component_name)
            if hasattr(component, 'initialize') and asyncio.iscoroutinefunction(component.initialize):
                init_tasks.append(component.initialize())
        
        if init_tasks:
            await asyncio.gather(*init_tasks)
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        # Start metrics collection task
        self.task_manager.create_task(
            self._collect_metrics(),
            name="backtest_metrics_collection"
        )
        
        # Start cache maintenance task
        self.task_manager.create_task(
            self._maintain_results_cache(),
            name="backtest_cache_maintenance"
        )
    
    async def _cancel_active_tasks(self) -> None:
        """Cancel all active backtest tasks."""
        logger.info(f"Canceling {len(self.active_tasks)} active backtest tasks")
        
        # Make a copy of task IDs to avoid modification during iteration
        task_ids = list(self.active_tasks.keys())
        
        for task_id in task_ids:
            await self.cancel_backtest(task_id)
    
    async def _collect_metrics(self) -> None:
        """Background task to collect and report metrics."""
        while self._running:
            try:
                # Collect component metrics
                metrics = {
                    'active_tasks': len(self.active_tasks),
                    'results_cache_size': len(self.results_cache),
                    'data_provider': self.data_provider.get_metrics() if hasattr(self.data_provider, 'get_metrics') else {},
                    'engine': self.engine.get_metrics() if hasattr(self.engine, 'get_metrics') else {}
                }
                
                # Report metrics
                self.metrics.report_many(metrics)
                
                # Wait for next collection interval or until stopped
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _maintain_results_cache(self) -> None:
        """Background task to maintain the results cache."""
        cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour default
        
        while self._running:
            try:
                # Get current time
                now = time.time()
                
                # Find expired cache entries
                expired = [
                    key for key, (timestamp, _) in self.results_cache.items()
                    if now - timestamp > cache_ttl
                ]
                
                # Remove expired entries
                for key in expired:
                    del self.results_cache[key]
                
                if expired:
                    logger.debug(f"Removed {len(expired)} expired entries from results cache")
                
                # Wait for next maintenance interval or until stopped
                await asyncio.sleep(300)  # Clean cache every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error maintaining cache: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def run_backtest(self, config: Dict[str, Any]) -> str:
        """
        Run a backtest with the given configuration.
        
        Args:
            config: Backtest configuration
            
        Returns:
            str: Task ID for the backtest
        """
        if not self._initialized:
            raise RuntimeError("Backtester service not initialized. Call start() first.")
        
        # Validate configuration
        self._validate_backtest_config(config)
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Create and start the backtest task
        task = self.task_manager.create_task(
            self._execute_backtest(task_id, config),
            name=f"backtest_{task_id}"
        )
        
        # Store task
        self.active_tasks[task_id] = {
            'task': task,
            'config': config,
            'start_time': time.time(),
            'status': 'running'
        }
        
        logger.info(f"Started backtest task {task_id}")
        return task_id
    
    async def get_backtest_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a backtest task.
        
        Args:
            task_id: Task ID from run_backtest
            
        Returns:
            dict: Task status information
        """
        if task_id not in self.active_tasks and task_id not in self.results_cache:
            raise KeyError(f"Backtest task not found: {task_id}")
        
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            elapsed = time.time() - task_info['start_time']
            
            return {
                'task_id': task_id,
                'status': task_info['status'],
                'running': True,
                'elapsed': elapsed,
                'progress': task_info.get('progress', 0)
            }
        
        # Task is completed and in cache
        timestamp, result = self.results_cache[task_id]
        
        status_info = {
            'task_id': task_id,
            'status': 'completed' if not isinstance(result, Exception) else 'failed',
            'running': False,
            'completed_at': timestamp
        }
        
        if isinstance(result, Exception):
            status_info['error'] = str(result)
        
        return status_info
    
    async def get_backtest_result(self, task_id: str) -> Dict[str, Any]:
        """
        Get the result of a completed backtest.
        
        Args:
            task_id: Task ID from run_backtest
            
        Returns:
            dict: Backtest results
        """
        if task_id not in self.results_cache:
            if task_id in self.active_tasks:
                raise RuntimeError(f"Backtest task {task_id} is still running")
            raise KeyError(f"Backtest task not found: {task_id}")
        
        _, result = self.results_cache[task_id]
        
        if isinstance(result, Exception):
            raise result
        
        return result
    
    async def cancel_backtest(self, task_id: str) -> None:
        """
        Cancel a running backtest.
        
        Args:
            task_id: Task ID from run_backtest
        """
        if task_id not in self.active_tasks:
            if task_id in self.results_cache:
                logger.warning(f"Backtest {task_id} is already completed and cannot be canceled")
                return
            raise KeyError(f"Backtest task not found: {task_id}")
        
        task_info = self.active_tasks[task_id]
        
        # Cancel the task
        if not task_info['task'].done():
            task_info['task'].cancel()
            
            # Wait for the task to be canceled
            try:
                await asyncio.wait_for(
                    asyncio.shield(task_info['task']), 
                    timeout=5.0
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        
        # Update task status
        task_info['status'] = 'canceled'
        
        # Store the result in cache
        self.results_cache[task_id] = (
            time.time(),
            BacktestExecutionError(f"Backtest {task_id} was canceled")
        )
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        logger.info(f"Canceled backtest task {task_id}")
    
    async def run_optimization(self, config: Dict[str, Any]) -> str:
        """
        Run a parameter optimization for a strategy.
        
        Args:
            config: Optimization configuration
            
        Returns:
            str: Task ID for the optimization
        """
        if not self._initialized:
            raise RuntimeError("Backtester service not initialized. Call start() first.")
        
        # Validate configuration
        self._validate_optimization_config(config)
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Create and start the optimization task
        task = self.task_manager.create_task(
            self._execute_optimization(task_id, config),
            name=f"optimization_{task_id}"
        )
        
        # Store task
        self.active_tasks[task_id] = {
            'task': task,
            'config': config,
            'start_time': time.time(),
            'status': 'running',
            'type': 'optimization'
        }
        
        logger.info(f"Started optimization task {task_id}")
        return task_id
    
    async def run_scenario(self, config: Dict[str, Any]) -> str:
        """
        Run a scenario test with customized market conditions.
        
        Args:
            config: Scenario configuration
            
        Returns:
            str: Task ID for the scenario test
        """
        if not self._initialized:
            raise RuntimeError("Backtester service not initialized. Call start() first.")
        
        # Validate configuration
        self._validate_scenario_config(config)
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Create and start the scenario task
        task = self.task_manager.create_task(
            self._execute_scenario(task_id, config),
            name=f"scenario_{task_id}"
        )
        
        # Store task
        self.active_tasks[task_id] = {
            'task': task,
            'config': config,
            'start_time': time.time(),
            'status': 'running',
            'type': 'scenario'
        }
        
        logger.info(f"Started scenario test task {task_id}")
        return task_id
    
    async def generate_report(self, result_ids: List[str], 
                             report_config: Dict[str, Any]) -> str:
        """
        Generate a report from backtest results.
        
        Args:
            result_ids: List of backtest task IDs to include in the report
            report_config: Report configuration
            
        Returns:
            str: Report ID
        """
        if not self._initialized:
            raise RuntimeError("Backtester service not initialized. Call start() first.")
        
        # Get results for all provided task IDs
        results = []
        for task_id in result_ids:
            try:
                result = await self.get_backtest_result(task_id)
                results.append((task_id, result))
            except Exception as e:
                logger.error(f"Error getting result for task {task_id}: {e}")
                raise ValueError(f"Cannot generate report: task {task_id} failed - {e}")
        
        # Generate report
        report_id = str(uuid.uuid4())
        report_path = self.results_dir / f"report_{report_id}.html"
        
        await self.report.generate(
            results=results,
            config=report_config,
            output_path=str(report_path)
        )
        
        logger.info(f"Generated report {report_id}")
        return report_id
    
    async def get_report(self, report_id: str) -> str:
        """
        Get the path to a generated report.
        
        Args:
            report_id: Report ID from generate_report
            
        Returns:
            str: Path to the report file
        """
        report_path = self.results_dir / f"report_{report_id}.html"
        
        if not report_path.exists():
            raise FileNotFoundError(f"Report {report_id} not found")
        
        return str(report_path)
    
    async def _execute_backtest(self, task_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a backtest task.
        
        Args:
            task_id: Task ID
            config: Backtest configuration
            
        Returns:
            dict: Backtest results
        """
        try:
            logger.info(f"Executing backtest {task_id}")
            
            # Update progress
            def update_progress(progress: float) -> None:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]['progress'] = progress
            
            # Get backtest mode
            mode = config.get('mode', self.default_mode)
            
            # Load data
            symbol = config['symbol']
            timeframe = config['timeframe']
            start_date = config['start_date']
            end_date = config['end_date']
            
            # Get data source
            data_source = config.get('data_source', 'default')
            
            # Load historical data
            data = await self.data_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                source=data_source
            )
            
            if not data or len(data) == 0:
                raise BacktestDataError(f"No data available for {symbol} {timeframe} from {start_date} to {end_date}")
            
            # Get strategy class and parameters
            strategy_name = config['strategy']
            strategy_params = config.get('parameters', {})
            
            # Run backtest
            backtest_result = await self.engine.run_backtest(
                data=data,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                mode=mode,
                config=config,
                progress_callback=update_progress
            )
            
            # Analyze performance
            performance_result = await self.performance.analyze(
                backtest_result=backtest_result,
                metrics=config.get('metrics', None)
            )
            
            # Combine results
            result = {
                'task_id': task_id,
                'config': config,
                'summary': backtest_result['summary'],
                'performance': performance_result,
                'trades': backtest_result['trades']
            }
            
            # Store result in cache
            self.results_cache[task_id] = (time.time(), result)
            
            # Remove from active tasks
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'completed'
                del self.active_tasks[task_id]
            
            logger.info(f"Completed backtest {task_id}")
            
            return result
            
        except asyncio.CancelledError:
            logger.info(f"Backtest {task_id} was canceled")
            raise
        except Exception as e:
            logger.error(f"Error in backtest {task_id}: {e}")
            logger.error(traceback.format_exc())
            
            # Store error in cache
            self.results_cache[task_id] = (time.time(), e)
            
            # Remove from active tasks
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'failed'
                del self.active_tasks[task_id]
            
            raise
    
    async def _execute_optimization(self, task_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a parameter optimization task.
        
        Args:
            task_id: Task ID
            config: Optimization configuration
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info(f"Executing optimization {task_id}")
            
            # Update progress
            def update_progress(progress: float) -> None:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]['progress'] = progress
            
            # Extract configuration
            symbol = config['symbol']
            timeframe = config['timeframe']
            start_date = config['start_date']
            end_date = config['end_date']
            strategy_name = config['strategy']
            param_space = config['parameter_space']
            
            # Get optimization method and metrics
            method = config.get('method', 'grid')
            objective = config.get('objective', 'sharpe_ratio')
            n_trials = config.get('n_trials', 100)
            
            # Load historical data
            data = await self.data_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if not data or len(data) == 0:
                raise BacktestDataError(f"No data available for {symbol} {timeframe} from {start_date} to {end_date}")
            
            # Run optimization
            optimization_result = await self.optimization.optimize(
                data=data,
                strategy_name=strategy_name,
                param_space=param_space,
                method=method,
                objective=objective,
                n_trials=n_trials,
                engine=self.engine,
                performance=self.performance,
                progress_callback=update_progress
            )
            
            # Combine results
            result = {
                'task_id': task_id,
                'config': config,
                'best_params': optimization_result['best_params'],
                'best_value': optimization_result['best_value'],
                'all_trials': optimization_result['all_trials']
            }
            
            # Store result in cache
            self.results_cache[task_id] = (time.time(), result)
            
            # Remove from active tasks
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'completed'
                del self.active_tasks[task_id]
            
            logger.info(f"Completed optimization {task_id}")
            
            return result
            
        except asyncio.CancelledError:
            logger.info(f"Optimization {task_id} was canceled")
            raise
        except Exception as e:
            logger.error(f"Error in optimization {task_id}: {e}")
            logger.error(traceback.format_exc())
            
            # Store error in cache
            self.results_cache[task_id] = (time.time(), e)
            
            # Remove from active tasks
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'failed'
                del self.active_tasks[task_id]
            
            raise
    
    async def _execute_scenario(self, task_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a scenario test task.
        
        Args:
            task_id: Task ID
            config: Scenario configuration
            
        Returns:
            dict: Scenario test results
        """
        try:
            logger.info(f"Executing scenario test {task_id}")
            
            # Update progress
            def update_progress(progress: float) -> None:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]['progress'] = progress
            
            # Extract configuration
            symbol = config['symbol']
            timeframe = config['timeframe']
            start_date = config.get('start_date')
            end_date = config.get('end_date')
            strategy_name = config['strategy']
            strategy_params = config.get('parameters', {})
            scenario_type = config['scenario_type']
            scenario_params = config.get('scenario_params', {})
            
            # Run scenario test
            scenario_result = await self.scenario.run_scenario(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                scenario_type=scenario_type,
                scenario_params=scenario_params,
                data_provider=self.data_provider,
                engine=self.engine,
                performance=self.performance,
                progress_callback=update_progress
            )
            
            # Combine results
            result = {
                'task_id': task_id,
                'config': config,
                'scenario_type': scenario_type,
                'summary': scenario_result['summary'],
                'results': scenario_result['results']
            }
            
            # Store result in cache
            self.results_cache[task_id] = (time.time(), result)
            
            # Remove from active tasks
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'completed'
                del self.active_tasks[task_id]
            
            logger.info(f"Completed scenario test {task_id}")
            
            return result
            
        except asyncio.CancelledError:
            logger.info(f"Scenario test {task_id} was canceled")
            raise
        except Exception as e:
            logger.error(f"Error in scenario test {task_id}: {e}")
            logger.error(traceback.format_exc())
            
            # Store error in cache
            self.results_cache[task_id] = (time.time(), e)
            
            # Remove from active tasks
            if task_id in self.active_tasks:
                self.active_tasks[task_id]['status'] = 'failed'
                del self.active_tasks[task_id]
            
            raise
    
    def _validate_backtest_config(self, config: Dict[str, Any]) -> None:
        """
        Validate backtest configuration.
        
        Args:
            config: Backtest configuration
            
        Raises:
            BacktestConfigError: If configuration is invalid
        """
        required_fields = ['symbol', 'timeframe', 'start_date', 'end_date', 'strategy']
        
        for field in required_fields:
            if field not in config:
                raise BacktestConfigError(f"Missing required field: {field}")
        
        # Check mode
        if 'mode' in config and config['mode'] not in ['standard', 'tick', 'monte_carlo', 'walk_forward']:
            raise BacktestConfigError(f"Invalid backtest mode: {config['mode']}")
        
        # Check timeframe format
        try:
            parse_timeframe(config['timeframe'])
        except ValueError:
            raise BacktestConfigError(f"Invalid timeframe format: {config['timeframe']}")
        
        # Check date format
        date_format = "%Y-%m-%d"
        try:
            datetime.strptime(config['start_date'], date_format)
            datetime.strptime(config['end_date'], date_format)
        except ValueError:
            raise BacktestConfigError("Invalid date format. Use YYYY-MM-DD format.")
    
    def _validate_optimization_config(self, config: Dict[str, Any]) -> None:
        """
        Validate optimization configuration.
        
        Args:
            config: Optimization configuration
            
        Raises:
            BacktestConfigError: If configuration is invalid
        """
        required_fields = ['symbol', 'timeframe', 'start_date', 'end_date', 'strategy', 'parameter_space']
        
        for field in required_fields:
            if field not in config:
                raise BacktestConfigError(f"Missing required field: {field}")
        
        # Check parameter space format
        if not isinstance(config['parameter_space'], dict):
            raise BacktestConfigError("parameter_space must be a dictionary")
        
        for param, space in config['parameter_space'].items():
            if not isinstance(space, (list, dict)):
                raise BacktestConfigError(f"Invalid parameter space for {param}")
        
        # Check optimization method
        if 'method' in config and config['method'] not in ['grid', 'random', 'bayesian', 'genetic']:
            raise BacktestConfigError(f"Invalid optimization method: {config['method']}")
        
        # Check objective
        if 'objective' in config and config['objective'] not in [
            'sharpe_ratio', 'sortino_ratio', 'return', 'max_drawdown', 'win_rate',
            'profit_factor', 'expectancy', 'calmar_ratio'
        ]:
            raise BacktestConfigError(f"Invalid optimization objective: {config['objective']}")
    
    def _validate_scenario_config(self, config: Dict[str, Any]) -> None:
        """
        Validate scenario test configuration.
        
        Args:
            config: Scenario test configuration
            
        Raises:
            BacktestConfigError: If configuration is invalid
        """
        required_fields = ['symbol', 'timeframe', 'strategy', 'scenario_type']
        
        for field in required_fields:
            if field not in config:
                raise BacktestConfigError(f"Missing required field: {field}")
        
        # Check scenario type
        valid_scenario_types = [
            'market_crash', 'bull_market', 'sideways_market',
            'high_volatility', 'low_volatility', 'liquidity_crisis',
            'flash_crash', 'custom'
        ]
        
        if config['scenario_type'] not in valid_scenario_types:
            raise BacktestConfigError(f"Invalid scenario type: {config['scenario_type']}")
        
        # Check scenario parameters for custom scenario
        if config['scenario_type'] == 'custom' and 'scenario_params' not in config:
            raise BacktestConfigError("Missing scenario_params for custom scenario")


class BacktesterServiceClient:
    """
    Client for the backtester service.
    
    This client provides methods to interact with the backtester service
    for running backtests, optimizations, and scenario tests.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backtester service client.
        
        Args:
            config: Client configuration dictionary (optional)
        """
        self.config = config or {}
        self.service = None
    
    async def connect(self) -> None:
        """Connect to the backtester service."""
        # In this implementation, we create a local service instance
        # In a distributed architecture, this would connect to a remote service
        self.service = BacktesterService(self.config)
        await self.service.start()
    
    async def disconnect(self) -> None:
        """Disconnect from the backtester service."""
        if self.service:
            await self.service.stop()
            self.service = None
    
    async def run_backtest(self, config: Dict[str, Any]) -> str:
        """
        Run a backtest with the given configuration.
        
        Args:
            config: Backtest configuration
            
        Returns:
            str: Task ID for the backtest
        """
        if not self.service:
            raise RuntimeError("Not connected to backtester service")
        
        return await self.service.run_backtest(config)
    
    async def get_backtest_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a backtest task.
        
        Args:
            task_id: Task ID from run_backtest
            
        Returns:
            dict: Task status information
        """
        if not self.service:
            raise RuntimeError("Not connected to backtester service")
        
        return await self.service.get_backtest_status(task_id)
    
    async def get_backtest_result(self, task_id: str) -> Dict[str, Any]:
        """
        Get the result of a completed backtest.
        
        Args:
            task_id: Task ID from run_backtest
            
        Returns:
            dict: Backtest results
        """
        if not self.service:
            raise RuntimeError("Not connected to backtester service")
        
        return await self.service.get_backtest_result(task_id)
    
    async def cancel_backtest(self, task_id: str) -> None:
        """
        Cancel a running backtest.
        
        Args:
            task_id: Task ID from run_backtest
        """
        if not self.service:
            raise RuntimeError("Not connected to backtester service")
        
        await self.service.cancel_backtest(task_id)
    
    async def run_optimization(self, config: Dict[str, Any]) -> str:
        """
        Run a parameter optimization for a strategy.
        
        Args:
            config: Optimization configuration
            
        Returns:
            str: Task ID for the optimization
        """
        if not self.service:
            raise RuntimeError("Not connected to backtester service")
        
        return await self.service.run_optimization(config)
    
    async def run_scenario(self, config: Dict[str, Any]) -> str:
        """
        Run a scenario test with customized market conditions.
        
        Args:
            config: Scenario configuration
            
        Returns:
            str: Task ID for the scenario test
        """
        if not self.service:
            raise RuntimeError("Not connected to backtester service")
        
        return await self.service.run_scenario(config)
    
    async def generate_report(self, result_ids: List[str], 
                             report_config: Dict[str, Any]) -> str:
        """
        Generate a report from backtest results.
        
        Args:
            result_ids: List of backtest task IDs to include in the report
            report_config: Report configuration
            
        Returns:
            str: Report ID
        """
        if not self.service:
            raise RuntimeError("Not connected to backtester service")
        
        return await self.service.generate_report(result_ids, report_config)
    
    async def get_report(self, report_id: str) -> str:
        """
        Get the path to a generated report.
        
        Args:
            report_id: Report ID from generate_report
            
        Returns:
            str: Path to the report file
        """
        if not self.service:
            raise RuntimeError("Not connected to backtester service")
        
        return await self.service.get_report(report_id)


# Application setup for standalone operation
async def run_service():
    """Run the backtester service."""
    service = BacktesterService()
    
    try:
        await service.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await service.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run_service())
    except KeyboardInterrupt:
        logger.info("Service shutdown complete")
