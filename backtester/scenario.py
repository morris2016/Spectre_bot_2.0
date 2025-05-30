
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Backtester Scenario Module

This module provides advanced scenario testing capabilities for evaluating strategies
under various market conditions, stress tests, and what-if analyses.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from backtester.engine import BacktestEngine
from backtester.data_provider import DataProvider
from backtester.performance import PerformanceAnalyzer
from common.exceptions import BacktestScenarioError, DataNotFoundError
from common.constants import TIMEFRAMES, MARKET_REGIMES, VOLATILITY_LEVELS, SCENARIO_TYPES
from common.utils import create_unique_id, load_json_file, save_json_file
from data_storage.models.market_data import MarketRegime
from dataclasses import dataclass


@dataclass
class Scenario:
    """Minimal scenario definition for unit tests."""

    scenario_type: str
    parameters: dict

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """
    Generates different market scenarios for backtesting strategy robustness
    across various market conditions and extreme events.
    """

    def __init__(self, data_provider: DataProvider):
        """Initialize the scenario generator with a data provider."""
        self.data_provider = data_provider
        self.scenario_templates = self._load_scenario_templates()
        self.custom_scenarios = {}

    def _load_scenario_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined scenario templates from config files."""
        try:
            template_path = os.path.join(os.path.dirname(__file__), 'config', 'scenario_templates.json')
            templates = load_json_file(template_path)
            logger.info(f"Loaded {len(templates)} scenario templates")
            return templates
        except Exception as e:
            logger.warning(f"Failed to load scenario templates: {e}. Using default templates.")
            return self._get_default_templates()

    def _get_default_templates(self) -> Dict[str, Dict[str, Any]]:
        """Provide default scenario templates if loading from file fails."""
        return {
            "bull_market": {
                "description": "Strong uptrend with low volatility",
                "trend_direction": "up",
                "volatility": "low",
                "volume_profile": "increasing",
                "market_regime": "bull",
                "duration": 60,  # days
                "modifications": {
                    "trend_strength": 0.8,
                    "volatility_factor": 0.7,
                    "gap_probability": 0.1,
                    "gap_size_factor": 0.5
                }
            },
            "bear_market": {
                "description": "Strong downtrend with high volatility",
                "trend_direction": "down",
                "volatility": "high",
                "volume_profile": "increasing",
                "market_regime": "bear",
                "duration": 45,  # days
                "modifications": {
                    "trend_strength": 0.9,
                    "volatility_factor": 1.5,
                    "gap_probability": 0.2,
                    "gap_size_factor": 1.2
                }
            },
            "sideways_market": {
                "description": "Range-bound market with moderate volatility",
                "trend_direction": "sideways",
                "volatility": "medium",
                "volume_profile": "decreasing",
                "market_regime": "sideways",
                "duration": 30,  # days
                "modifications": {
                    "trend_strength": 0.1,
                    "volatility_factor": 1.0,
                    "gap_probability": 0.05,
                    "gap_size_factor": 0.8
                }
            },
            "volatility_spike": {
                "description": "Sudden increase in volatility with no clear trend",
                "trend_direction": "mixed",
                "volatility": "very_high",
                "volume_profile": "spiking",
                "market_regime": "volatile",
                "duration": 15,  # days
                "modifications": {
                    "trend_strength": 0.3,
                    "volatility_factor": 3.0,
                    "gap_probability": 0.4,
                    "gap_size_factor": 2.0
                }
            },
            "flash_crash": {
                "description": "Rapid price decline followed by recovery",
                "trend_direction": "crash_recovery",
                "volatility": "extreme",
                "volume_profile": "panic",
                "market_regime": "crash",
                "duration": 3,  # days
                "modifications": {
                    "trend_strength": 0.95,
                    "volatility_factor": 5.0,
                    "gap_probability": 0.7,
                    "gap_size_factor": 3.5,
                    "crash_depth": 0.15,  # 15% drop
                    "recovery_strength": 0.6  # 60% recovery of the drop
                }
            },
            "trending_breakout": {
                "description": "Breakout from range into strong trend",
                "trend_direction": "breakout_up",
                "volatility": "increasing",
                "volume_profile": "breakout",
                "market_regime": "breakout",
                "duration": 20,  # days
                "modifications": {
                    "trend_strength": 0.7,
                    "volatility_factor": 1.3,
                    "gap_probability": 0.15,
                    "gap_size_factor": 1.0,
                    "consolidation_duration": 10,  # days of sideways before breakout
                    "breakout_size": 0.05  # 5% initial breakout move
                }
            },
            "liquidity_crisis": {
                "description": "Sharp drop with high volatility and low liquidity",
                "trend_direction": "down",
                "volatility": "extreme",
                "volume_profile": "thin",
                "market_regime": "liquidity_crisis",
                "duration": 10,  # days
                "modifications": {
                    "trend_strength": 0.85,
                    "volatility_factor": 4.0,
                    "gap_probability": 0.5,
                    "gap_size_factor": 2.5,
                    "slippage_factor": 3.0,
                    "spread_widening": 2.5
                }
            }
        }

    def create_custom_scenario(self, name: str, description: str, 
                              parameters: Dict[str, Any]) -> str:
        """
        Create a custom scenario with specific parameters.
        
        Args:
            name: Unique name for the scenario
            description: Description of the scenario
            parameters: Dict containing scenario parameters
            
        Returns:
            str: Scenario ID
        """
        scenario_id = create_unique_id()
        self.custom_scenarios[scenario_id] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "created_at": datetime.now().isoformat()
        }
        logger.info(f"Created custom scenario: {name} with ID {scenario_id}")
        return scenario_id

    def get_scenario_data(self, 
                         scenario_type: str, 
                         asset: str,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         timeframe: str = "1h",
                         custom_scenario_id: Optional[str] = None) -> pd.DataFrame:
        """
        Generate or retrieve data for a specific scenario type.
        
        Args:
            scenario_type: Type of scenario from predefined templates or 'custom'
            asset: Asset symbol to generate scenario for
            start_date: Start date for the scenario
            end_date: End date for the scenario
            timeframe: Data timeframe
            custom_scenario_id: ID of custom scenario if scenario_type is 'custom'
            
        Returns:
            pd.DataFrame: OHLCV data modified for the scenario
        """
        # Set default dates if not provided
        if not start_date:
            start_date = datetime.now() - timedelta(days=90)
        if not end_date:
            end_date = datetime.now()
            
        # Get base historical data
        try:
            base_data = self.data_provider.get_historical_data(
                asset=asset,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            if base_data.empty:
                raise DataNotFoundError(f"No data found for {asset} from {start_date} to {end_date}")
                
            logger.info(f"Retrieved base data for {asset}: {len(base_data)} candles")
            
            # Apply scenario modifications
            if scenario_type == 'custom' and custom_scenario_id:
                if custom_scenario_id not in self.custom_scenarios:
                    raise BacktestScenarioError(f"Custom scenario ID {custom_scenario_id} not found")
                scenario_params = self.custom_scenarios[custom_scenario_id]['parameters']
                return self._apply_scenario_modifications(base_data, scenario_params)
            
            elif scenario_type in self.scenario_templates:
                scenario_params = self.scenario_templates[scenario_type]
                return self._apply_scenario_modifications(base_data, scenario_params)
            
            else:
                raise BacktestScenarioError(f"Scenario type {scenario_type} not recognized")
                
        except Exception as e:
            logger.error(f"Error generating scenario data: {e}")
            raise

    def _apply_scenario_modifications(self, 
                                     data: pd.DataFrame, 
                                     scenario_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply scenario-specific modifications to price data.
        
        Args:
            data: Original OHLCV data
            scenario_params: Parameters defining the scenario modifications
            
        Returns:
            pd.DataFrame: Modified OHLCV data
        """
        # Create a copy to avoid modifying the original data
        modified_data = data.copy()
        
        # Extract modification parameters
        mods = scenario_params.get('modifications', {})
        trend_strength = mods.get('trend_strength', 0.5)
        volatility_factor = mods.get('volatility_factor', 1.0)
        gap_probability = mods.get('gap_probability', 0.1)
        gap_size_factor = mods.get('gap_size_factor', 1.0)
        
        # Get trend direction
        trend_direction = scenario_params.get('trend_direction', 'sideways')
        
        # Apply trend modification
        if trend_direction == 'up':
            trend_modifier = np.linspace(0, trend_strength * 0.1, len(modified_data))
            modified_data['close'] = modified_data['close'] * (1 + trend_modifier)
        elif trend_direction == 'down':
            trend_modifier = np.linspace(0, trend_strength * 0.1, len(modified_data))
            modified_data['close'] = modified_data['close'] * (1 - trend_modifier)
        elif trend_direction == 'crash_recovery':
            # Handle specific crash and recovery scenario
            crash_depth = mods.get('crash_depth', 0.15)
            recovery_strength = mods.get('recovery_strength', 0.6)
            
            # Determine crash point (roughly 1/3 into the data)
            crash_idx = len(modified_data) // 3
            crash_duration = min(len(modified_data) // 10, 10)  # Max 10 periods or 10% of data
            
            # Create crash effect
            crash_modifier = np.ones(len(modified_data))
            for i in range(crash_duration):
                crash_modifier[crash_idx + i] = 1 - (crash_depth * (crash_duration - i) / crash_duration)
            
            # Create recovery effect
            recovery_start = crash_idx + crash_duration
            recovery_duration = min(len(modified_data) // 5, 20)  # Max 20 periods or 20% of data
            
            for i in range(recovery_duration):
                recovery_progress = i / recovery_duration
                recovery_amount = crash_depth * recovery_strength * recovery_progress
                if recovery_start + i < len(crash_modifier):
                    crash_modifier[recovery_start + i] = 1 - crash_depth + recovery_amount
            
            # Apply the crash and recovery modifications
            modified_data['close'] = modified_data['close'] * crash_modifier
        elif trend_direction == 'breakout_up' or trend_direction == 'breakout_down':
            # Handle breakout scenarios
            consolidation_duration = mods.get('consolidation_duration', 10)
            breakout_size = mods.get('breakout_size', 0.05)
            breakout_point = len(modified_data) // 2
            
            # Create consolidation (reduce volatility before breakout)
            for i in range(consolidation_duration):
                if breakout_point - consolidation_duration + i >= 0:
                    idx = breakout_point - consolidation_duration + i
                    modified_data.loc[modified_data.index[idx], 'high'] = (
                        modified_data.loc[modified_data.index[idx], 'close'] * 1.002)
                    modified_data.loc[modified_data.index[idx], 'low'] = (
                        modified_data.loc[modified_data.index[idx], 'close'] * 0.998)
            
            # Create breakout effect
            direction = 1 if trend_direction == 'breakout_up' else -1
            for i in range(len(modified_data) - breakout_point):
                breakout_effect = min(breakout_size * (1 + i/10), breakout_size * 3)
                modified_data.loc[modified_data.index[breakout_point + i], 'close'] = (
                    modified_data.loc[modified_data.index[breakout_point + i], 'close'] * 
                    (1 + direction * breakout_effect))
        
        # Apply volatility modifications
        if volatility_factor != 1.0:
            for i in range(len(modified_data)):
                center_price = modified_data.loc[modified_data.index[i], 'close']
                range_size = (modified_data.loc[modified_data.index[i], 'high'] - 
                             modified_data.loc[modified_data.index[i], 'low'])
                
                # Adjust high and low based on volatility factor
                modified_data.loc[modified_data.index[i], 'high'] = (
                    center_price + (range_size * volatility_factor / 2))
                modified_data.loc[modified_data.index[i], 'low'] = (
                    center_price - (range_size * volatility_factor / 2))
        
        # Apply gaps where appropriate
        if gap_probability > 0:
            for i in range(1, len(modified_data)):
                if np.random.random() < gap_probability:
                    gap_direction = 1 if np.random.random() > 0.5 else -1
                    gap_size = np.random.random() * 0.03 * gap_size_factor * gap_direction
                    
                    modified_data.loc[modified_data.index[i], 'open'] = (
                        modified_data.loc[modified_data.index[i-1], 'close'] * (1 + gap_size))
                    modified_data.loc[modified_data.index[i], 'high'] = max(
                        modified_data.loc[modified_data.index[i], 'high'],
                        modified_data.loc[modified_data.index[i], 'open'])
                    modified_data.loc[modified_data.index[i], 'low'] = min(
                        modified_data.loc[modified_data.index[i], 'low'],
                        modified_data.loc[modified_data.index[i], 'open'])
        
        # Handle volume profile
        volume_profile = scenario_params.get('volume_profile', 'normal')
        
        if volume_profile == 'increasing':
            vol_modifier = np.linspace(1, 2, len(modified_data))
            modified_data['volume'] = modified_data['volume'] * vol_modifier
        elif volume_profile == 'decreasing':
            vol_modifier = np.linspace(1, 0.5, len(modified_data))
            modified_data['volume'] = modified_data['volume'] * vol_modifier
        elif volume_profile == 'spiking':
            # Random volume spikes
            for i in range(len(modified_data)):
                if np.random.random() < 0.15:  # 15% chance of spike
                    spike_factor = 1 + np.random.random() * 4  # 1-5x volume spike
                    modified_data.loc[modified_data.index[i], 'volume'] *= spike_factor
        elif volume_profile == 'panic':
            # Volume increases dramatically during crash periods
            if 'crash_depth' in mods:
                crash_idx = len(modified_data) // 3
                crash_duration = min(len(modified_data) // 10, 10)
                for i in range(crash_duration):
                    panic_factor = 3 + (crash_duration - i) * 0.7  # Highest at crash start
                    if crash_idx + i < len(modified_data):
                        modified_data.loc[modified_data.index[crash_idx + i], 'volume'] *= panic_factor
        elif volume_profile == 'breakout':
            # Volume increases at breakout point
            if 'breakout_size' in mods:
                breakout_point = len(modified_data) // 2
                for i in range(5):  # 5 periods of high volume
                    if breakout_point + i < len(modified_data):
                        modified_data.loc[modified_data.index[breakout_point + i], 'volume'] *= (3 - i * 0.4)
        
        # Ensure OHLC relationship integrity
        for i in range(len(modified_data)):
            high = modified_data.loc[modified_data.index[i], 'high']
            low = modified_data.loc[modified_data.index[i], 'low']
            open_price = modified_data.loc[modified_data.index[i], 'open']
            close = modified_data.loc[modified_data.index[i], 'close']
            
            # Make sure high is highest and low is lowest
            modified_data.loc[modified_data.index[i], 'high'] = max(high, open_price, close)
            modified_data.loc[modified_data.index[i], 'low'] = min(low, open_price, close)
        
        # Apply any special scenario-specific modifications
        if 'slippage_factor' in mods:
            # This would be used in execution simulation rather than data modification
            pass
        
        if 'spread_widening' in mods:
            # This would be used in execution simulation rather than data modification
            pass
        
        logger.info(f"Applied scenario modifications: {scenario_params.get('description', 'Unknown scenario')}")
        
        return modified_data


class ScenarioTester:
    """
    Runs backtests across different scenarios to evaluate strategy robustness.
    """
    
    def __init__(self, backtest_engine: BacktestEngine, 
                scenario_generator: ScenarioGenerator,
                performance_analyzer: PerformanceAnalyzer):
        """Initialize the scenario tester with required components."""
        self.backtest_engine = backtest_engine
        self.scenario_generator = scenario_generator
        self.performance_analyzer = performance_analyzer
        self.scenario_results = {}
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    
    async def run_scenario_suite(self, 
                               strategy_id: str,
                               asset: str,
                               scenario_types: List[str],
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None,
                               timeframe: str = "1h") -> Dict[str, Any]:
        """
        Run a suite of scenarios to test strategy robustness.
        
        Args:
            strategy_id: ID of the strategy to test
            asset: Asset to test on
            scenario_types: List of scenario types to test
            start_date: Start date for backtests
            end_date: End date for backtests
            timeframe: Data timeframe
            
        Returns:
            Dict containing results for all scenarios
        """
        logger.info(f"Running scenario suite for strategy {strategy_id} on {asset}")
        
        # Create tasks for each scenario
        tasks = []
        for scenario_type in scenario_types:
            task = asyncio.create_task(
                self.run_scenario_test(
                    strategy_id=strategy_id,
                    asset=asset,
                    scenario_type=scenario_type,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe
                )
            )
            tasks.append(task)
        
        # Run all scenarios concurrently
        results = await asyncio.gather(*tasks)
        
        # Compile results
        suite_results = {
            "strategy_id": strategy_id,
            "asset": asset,
            "timeframe": timeframe,
            "scenarios_tested": len(scenario_types),
            "date_range": f"{start_date} to {end_date}",
            "scenario_results": {result["scenario_type"]: result for result in results},
            "summary": self._generate_scenario_suite_summary(results)
        }
        
        logger.info(f"Completed scenario suite with {len(results)} scenarios")
        return suite_results
        
    async def run_scenario_test(self,
                              strategy_id: str,
                              asset: str,
                              scenario_type: str,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None,
                              timeframe: str = "1h",
                              custom_scenario_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a backtest for a specific scenario.
        
        Args:
            strategy_id: ID of the strategy to test
            asset: Asset to test on
            scenario_type: Type of scenario to test
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Data timeframe
            custom_scenario_id: ID of custom scenario if scenario_type is 'custom'
            
        Returns:
            Dict containing scenario test results
        """
        try:
            # Generate scenario data
            scenario_data = self.scenario_generator.get_scenario_data(
                scenario_type=scenario_type,
                asset=asset,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                custom_scenario_id=custom_scenario_id
            )
            
            scenario_description = (
                self.scenario_generator.scenario_templates.get(scenario_type, {}).get('description', 'Custom scenario')
                if scenario_type != 'custom' else 
                self.scenario_generator.custom_scenarios.get(custom_scenario_id, {}).get('description', 'Custom scenario')
            )
            
            logger.info(f"Running {scenario_type} scenario test: {scenario_description}")
            
            # Run backtest with scenario data
            backtest_result = await self.backtest_engine.run_backtest_with_data(
                strategy_id=strategy_id,
                data=scenario_data,
                asset=asset,
                timeframe=timeframe
            )
            
            # Analyze performance
            performance_metrics = self.performance_analyzer.analyze_backtest(backtest_result)
            
            # Store and return results
            result = {
                "scenario_type": scenario_type,
                "description": scenario_description,
                "strategy_id": strategy_id,
                "asset": asset,
                "timeframe": timeframe,
                "start_date": scenario_data.index[0].strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": scenario_data.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                "performance_metrics": performance_metrics,
                "trades_count": len(backtest_result.get('trades', [])),
                "final_equity": backtest_result.get('final_equity', 0),
                "max_drawdown": performance_metrics.get('max_drawdown', 0),
                "win_rate": performance_metrics.get('win_rate', 0),
                "sharpe_ratio": performance_metrics.get('sharpe_ratio', 0),
                "profit_factor": performance_metrics.get('profit_factor', 0)
            }
            
            # Store result for later comparison
            scenario_id = f"{scenario_type}_{asset}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.scenario_results[scenario_id] = result
            
            logger.info(f"Completed {scenario_type} scenario test with win rate: {result['win_rate']:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error in scenario test {scenario_type}: {e}")
            return {
                "scenario_type": scenario_type,
                "strategy_id": strategy_id,
                "asset": asset,
                "error": str(e),
                "status": "failed"
            }
    
    def _generate_scenario_suite_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of scenario test results for comparison."""
        if not results:
            return {"error": "No scenario results available"}
        
        # Extract key metrics for comparison
        win_rates = [r.get('win_rate', 0) for r in results if 'win_rate' in r]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results if 'sharpe_ratio' in r]
        profit_factors = [r.get('profit_factor', 0) for r in results if 'profit_factor' in r]
        max_drawdowns = [r.get('max_drawdown', 0) for r in results if 'max_drawdown' in r]
        
        # Find best and worst scenarios
        best_scenario = max(results, key=lambda x: x.get('sharpe_ratio', -999) if 'sharpe_ratio' in x else -999)
        worst_scenario = min(results, key=lambda x: x.get('sharpe_ratio', 999) if 'sharpe_ratio' in x else 999)
        
        # Calculate consistency scores
        win_rate_std = np.std(win_rates) if win_rates else 0
        sharpe_ratio_std = np.std(sharpe_ratios) if sharpe_ratios else 0
        
        # Create scenario performance ranking
        scenario_ranking = sorted(
            [(r.get('scenario_type'), r.get('sharpe_ratio', 0)) for r in results if 'sharpe_ratio' in r],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Generate recommendations based on results
        recommendations = []
        if win_rate_std > 10:
            recommendations.append("Strategy performance varies significantly across different market conditions")
        
        if any(dd > 25 for dd in max_drawdowns):
            recommendations.append("Strategy has high drawdown risk in some scenarios")
        
        if best_scenario.get('scenario_type') == 'sideways_market':
            recommendations.append("Strategy performs best in range-bound markets")
        elif best_scenario.get('scenario_type') == 'bull_market':
            recommendations.append("Strategy performs best in bull markets")
        elif best_scenario.get('scenario_type') == 'bear_market':
            recommendations.append("Strategy performs best in bear markets")
        
        if worst_scenario.get('scenario_type') in ['flash_crash', 'volatility_spike']:
            recommendations.append("Strategy is vulnerable to extreme market conditions")
        
        # Create summary
        summary = {
            "average_win_rate": np.mean(win_rates) if win_rates else 0,
            "win_rate_range": f"{min(win_rates) if win_rates else 0:.2f}% - {max(win_rates) if win_rates else 0:.2f}%",
            "win_rate_consistency": 100 - win_rate_std if win_rates else 0,  # Higher is more consistent
            "average_sharpe": np.mean(sharpe_ratios) if sharpe_ratios else 0,
            "sharpe_consistency": 100 - (sharpe_ratio_std * 100) if sharpe_ratios else 0,  # Higher is more consistent
            "average_profit_factor": np.mean(profit_factors) if profit_factors else 0,
            "max_drawdown_worst_case": max(max_drawdowns) if max_drawdowns else 0,
            "best_performing_scenario": best_scenario.get('scenario_type'),
            "worst_performing_scenario": worst_scenario.get('scenario_type'),
            "scenario_ranking": scenario_ranking,
            "recommendations": recommendations,
            "robustness_score": self._calculate_robustness_score(results)
        }
        
        return summary
    
    def _calculate_robustness_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate an overall robustness score (0-100) based on performance across scenarios.
        Higher score means the strategy is more robust to different market conditions.
        """
        if not results:
            return 0
        
        valid_results = [r for r in results if 'sharpe_ratio' in r and 'win_rate' in r and 'max_drawdown' in r]
        if not valid_results:
            return 0
        
        # Extract metrics
        win_rates = [r.get('win_rate', 0) for r in valid_results]
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in valid_results]
        max_drawdowns = [r.get('max_drawdown', 0) for r in valid_results]
        profit_factors = [r.get('profit_factor', 0) for r in valid_results]
        
        # Calculate consistency (lower standard deviation is better)
        win_rate_std = np.std(win_rates)
        sharpe_std = np.std(sharpe_ratios)
        
        # Calculate average performance
        avg_win_rate = np.mean(win_rates)
        avg_sharpe = np.mean(sharpe_ratios)
        avg_drawdown = np.mean(max_drawdowns)
        avg_profit_factor = np.mean(profit_factors)
        
        # Score components (0-100 each)
        performance_score = min(avg_win_rate, 100)
        
        sharpe_score = min(avg_sharpe * 20, 100)  # Scale Sharpe to 0-100
        
        consistency_score = max(0, 100 - (win_rate_std * 2) - (sharpe_std * 20))
        
        drawdown_score = max(0, 100 - (avg_drawdown * 2))
        
        profit_factor_score = min(avg_profit_factor * 20, 100)
        
        # Extreme condition handling (extra penalty for poor performance in stress tests)
        stress_scenarios = ['flash_crash', 'volatility_spike', 'liquidity_crisis']
        stress_results = [r for r in valid_results if r.get('scenario_type') in stress_scenarios]
        
        stress_penalty = 0
        if stress_results:
            stress_win_rates = [r.get('win_rate', 0) for r in stress_results]
            stress_drawdowns = [r.get('max_drawdown', 0) for r in stress_results]
            
            avg_stress_win_rate = np.mean(stress_win_rates)
            avg_stress_drawdown = np.mean(stress_drawdowns)
            
            # Apply penalty if performance deteriorates significantly in stress tests
            if avg_stress_win_rate < avg_win_rate * 0.7:
                stress_penalty += 15
            
            if avg_stress_drawdown > avg_drawdown * 1.5:
                stress_penalty += 15
        
        # Calculate final score with weights
        robustness_score = (
            (performance_score * 0.25) +  # 25% weight to win rate
            (sharpe_score * 0.20) +       # 20% weight to risk-adjusted returns
            (consistency_score * 0.25) +  # 25% weight to consistency
            (drawdown_score * 0.15) +     # 15% weight to drawdown protection
            (profit_factor_score * 0.15)  # 15% weight to profit factor
        )
        
        # Apply stress test penalty (if any)
        robustness_score = max(0, robustness_score - stress_penalty)
        
        return round(robustness_score, 2)

    def compare_strategies(self, 
                         strategy_ids: List[str],
                         asset: str,
                         scenario_types: List[str],
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         timeframe: str = "1h") -> Dict[str, Any]:
        """
        Compare multiple strategies across different market scenarios.
        
        Args:
            strategy_ids: List of strategy IDs to compare
            asset: Asset to test on
            scenario_types: List of scenario types to test
            start_date: Start date for backtests
            end_date: End date for backtests
            timeframe: Data timeframe
            
        Returns:
            Dict containing comparison results
        """
        async def run_all_comparisons():
            # Create tasks for each strategy's scenario suite
            tasks = []
            for strategy_id in strategy_ids:
                task = asyncio.create_task(
                    self.run_scenario_suite(
                        strategy_id=strategy_id,
                        asset=asset,
                        scenario_types=scenario_types,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe=timeframe
                    )
                )
                tasks.append(task)
            
            # Run all scenario suites concurrently
            return await asyncio.gather(*tasks)
        
        # Run the comparison
        loop = asyncio.get_event_loop()
        suite_results = loop.run_until_complete(run_all_comparisons())
        
        # Create comparison metrics
        comparison = {
            "asset": asset,
            "timeframe": timeframe,
            "scenarios_tested": scenario_types,
            "strategies_compared": strategy_ids,
            "date_range": f"{start_date} to {end_date}",
            "strategy_results": {result["strategy_id"]: result for result in suite_results},
            "comparison_metrics": self._generate_strategy_comparison(suite_results)
        }
        
        return comparison
    
    def _generate_strategy_comparison(self, suite_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison metrics between different strategies."""
        if not suite_results:
            return {"error": "No results available for comparison"}
        
        # Extract key metrics by strategy and scenario
        strategy_metrics = {}
        for result in suite_results:
            strategy_id = result["strategy_id"]
            summary = result["summary"]
            scenario_results = result["scenario_results"]
            
            strategy_metrics[strategy_id] = {
                "average_win_rate": summary["average_win_rate"],
                "robustness_score": summary["robustness_score"],
                "average_sharpe": summary["average_sharpe"],
                "max_drawdown_worst_case": summary["max_drawdown_worst_case"],
                "best_scenario": summary["best_performing_scenario"],
                "worst_scenario": summary["worst_performing_scenario"],
                "by_scenario": {
                    s_type: {
                        "win_rate": s_result.get("win_rate", 0),
                        "sharpe": s_result.get("sharpe_ratio", 0),
                        "max_drawdown": s_result.get("max_drawdown", 0),
                        "profit_factor": s_result.get("profit_factor", 0)
                    }
                    for s_type, s_result in scenario_results.items()
                    if "win_rate" in s_result
                }
            }
        
        # Create rankings for each scenario and overall
        rankings = {
            "overall_robustness": sorted(
                [(s_id, metrics["robustness_score"]) for s_id, metrics in strategy_metrics.items()],
                key=lambda x: x[1],
                reverse=True
            ),
            "best_win_rate": sorted(
                [(s_id, metrics["average_win_rate"]) for s_id, metrics in strategy_metrics.items()],
                key=lambda x: x[1],
                reverse=True
            ),
            "best_risk_adjusted": sorted(
                [(s_id, metrics["average_sharpe"]) for s_id, metrics in strategy_metrics.items()],
                key=lambda x: x[1],
                reverse=True
            ),
            "lowest_drawdown": sorted(
                [(s_id, metrics["max_drawdown_worst_case"]) for s_id, metrics in strategy_metrics.items()],
                key=lambda x: x[1]
            ),
            "by_scenario": {}
        }
        
        # Create rankings for each individual scenario
        all_scenarios = set()
        for s_metrics in strategy_metrics.values():
            all_scenarios.update(s_metrics["by_scenario"].keys())
        
        for scenario in all_scenarios:
            scenario_performances = []
            for strategy_id, metrics in strategy_metrics.items():
                if scenario in metrics["by_scenario"]:
                    # Use Sharpe ratio for ranking
                    scenario_performances.append(
                        (strategy_id, metrics["by_scenario"][scenario]["sharpe"])
                    )
            
            if scenario_performances:
                rankings["by_scenario"][scenario] = sorted(
                    scenario_performances,
                    key=lambda x: x[1],
                    reverse=True
                )
        
        # Determine optimal strategy per market condition
        optimal_strategies = {
            scenario: ranking[0][0] if ranking else None
            for scenario, ranking in rankings["by_scenario"].items()
        }
        
        # Create recommendations
        recommendations = []
        
        if len(strategy_metrics) > 1:
            most_robust = rankings["overall_robustness"][0][0] if rankings["overall_robustness"] else None
            
            if most_robust:
                recommendations.append(f"Strategy {most_robust} shows the best overall robustness across scenarios")
            
            # Identify specialists
            specialists = {}
            for scenario, top_strategy in optimal_strategies.items():
                if top_strategy not in specialists:
                    specialists[top_strategy] = []
                specialists[top_strategy].append(scenario)
            
            for strategy, best_scenarios in specialists.items():
                if len(best_scenarios) == 1:
                    recommendations.append(f"Strategy {strategy} specializes in {best_scenarios[0]} conditions")
                elif len(best_scenarios) > 1 and len(best_scenarios) < len(all_scenarios) / 2:
                    scenario_list = ", ".join(best_scenarios)
                    recommendations.append(f"Strategy {strategy} excels in: {scenario_list}")
            
            # Check if a strategy rotation might be beneficial
            if len(optimal_strategies) > 1 and len(set(optimal_strategies.values())) > 1:
                recommendations.append("Consider implementing strategy rotation based on detected market regime")
        
        comparison_metrics = {
            "rankings": rankings,
            "optimal_strategy_by_scenario": optimal_strategies,
            "recommendations": recommendations
        }
        
        return comparison_metrics
