#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Brain Council - Weighting System

This module implements the dynamic strategy weighting system that adapts based on
performance metrics, market conditions, and trading success. It provides mechanisms
to automatically adjust the influence of different strategy brains within the council
based on their historical and recent performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import json
import math
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from common.logger import get_logger
from common.constants import (
    TIMEFRAMES, MARKET_REGIMES, ASSETS, PLATFORMS,
    STRATEGY_CATEGORIES, WEIGHTING_METHODS
)
from common.metrics import performance_metrics, sharpe_ratio, sortino_ratio, calmar_ratio, win_rate
from common.exceptions import WeightingSystemError, InvalidStrategyError
from common.utils import exponential_decay

logger = get_logger("brain_council.weighting")


class WeightingSystem:
    """
    Dynamic strategy weighting system that adapts weights based on strategy performance,
    market conditions, and various performance metrics.
    """
    
    def __init__(self, 
                 council_name: str,
                 config: dict = None,
                 performance_db_connector = None):
        """
        Initialize the weighting system.
        
        Args:
            council_name: Name of the brain council this weighting system belongs to
            config: Configuration for the weighting system
            performance_db_connector: Database connector for retrieving strategy performance
        """
        self.council_name = council_name
        self.config = config or {}
        self._initialize_config()
        self.performance_db = performance_db_connector
        
        # Strategy weights storage
        self.weights = {}
        self.base_weights = {}  # Initial weights before adaptation
        self.weight_history = defaultdict(list)  # Track weight changes over time
        
        # Performance metrics cache
        self.performance_cache = {}
        self.last_performance_update = {}
        
        self.scaler = MinMaxScaler()
        
        logger.info(f"Initializing WeightingSystem for council: {council_name}")
        
    def _initialize_config(self):
        """Initialize configuration with defaults if not provided."""
        self.config.setdefault('weighting_method', 'performance_weighted')
        self.config.setdefault('lookback_periods', 5)
        self.config.setdefault('update_frequency', 3600)  # seconds
        self.config.setdefault('min_trades', 10)  # Minimum trades to consider
        self.config.setdefault('alpha', 0.75)  # Weighting between recent and historical
        self.config.setdefault('metrics', {
            'win_rate': 0.4,
            'sharpe_ratio': 0.2,
            'sortino_ratio': 0.2,
            'calmar_ratio': 0.1,
            'profit_factor': 0.1,
        })
        self.config.setdefault('regime_adaptation', True)
        self.config.setdefault('min_weight', 0.01)  # Minimum weight to assign
        self.config.setdefault('max_weight', 0.5)   # Maximum weight to assign
        self.config.setdefault('volatility_adjustment', True)
        self.config.setdefault('reversion_factor', 0.1)  # Speed of reversion to base weights
        
    async def initialize_weights(self, strategies: List[str], 
                                base_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the weights for a list of strategies.
        
        Args:
            strategies: List of strategy identifiers
            base_weights: Optional initial weights to assign
        """
        if not strategies:
            raise WeightingSystemError("No strategies provided for weight initialization")
            
        # If base weights not provided, assign equal weights
        if not base_weights:
            weight = 1.0 / len(strategies)
            base_weights = {strategy: weight for strategy in strategies}
        else:
            # Normalize provided weights
            total = sum(base_weights.values())
            base_weights = {k: v/total for k, v in base_weights.items()}
            
        self.base_weights = base_weights
        self.weights = base_weights.copy()
        
        # Record initial weights in history
        timestamp = time.time()
        for strategy, weight in self.weights.items():
            self.weight_history[strategy].append((timestamp, weight))
            
        logger.info(f"Initialized weights for {len(strategies)} strategies in council {self.council_name}")
        return self.weights
        
    async def update_weights(self, 
                           current_market_regime: str = None, 
                           force_update: bool = False) -> Dict[str, float]:
        """
        Update strategy weights based on performance metrics and current market regime.
        
        Args:
            current_market_regime: Current market regime classification
            force_update: Force weight update regardless of update frequency
            
        Returns:
            Updated strategy weights
        """
        now = time.time()
        
        # Check if update is needed based on frequency
        last_update = max(self.last_performance_update.values()) if self.last_performance_update else 0
        if not force_update and now - last_update < self.config['update_frequency']:
            logger.debug(f"Skipping weight update: last update was {now - last_update}s ago")
            return self.weights
            
        # Fetch latest performance metrics
        await self._update_performance_metrics()
        
        # Calculate new weights based on selected method
        if self.config['weighting_method'] == 'performance_weighted':
            new_weights = await self._calculate_performance_weighted()
        elif self.config['weighting_method'] == 'adaptive_ensemble':
            new_weights = await self._calculate_adaptive_ensemble(current_market_regime)
        elif self.config['weighting_method'] == 'bayesian_optimization':
            new_weights = await self._calculate_bayesian_weights()
        else:
            new_weights = await self._calculate_default_weights()
            
        # Apply weight constraints
        new_weights = self._apply_weight_constraints(new_weights)
        
        # Reversion to base weights (prevents extreme specialization)
        if self.config['reversion_factor'] > 0:
            for strategy in new_weights:
                reversion = self.config['reversion_factor']
                new_weights[strategy] = (1 - reversion) * new_weights[strategy] + reversion * self.base_weights[strategy]
                
        # Record weights in history
        for strategy, weight in new_weights.items():
            self.weight_history[strategy].append((now, weight))
            
        self.weights = new_weights
        logger.info(f"Updated weights for {len(new_weights)} strategies in council {self.council_name}")
        return self.weights
    
    async def _update_performance_metrics(self):
        """Fetch latest performance metrics for all strategies from the database."""
        if not self.performance_db:
            logger.warning("No performance database connector available")
            return
            
        strategies = list(self.base_weights.keys())
        lookback = self.config['lookback_periods']
        
        for strategy in strategies:
            try:
                # Fetch recent performance data
                performance = await self.performance_db.get_strategy_performance(
                    strategy_id=strategy,
                    lookback_periods=lookback,
                    min_trades=self.config['min_trades']
                )
                
                if performance:
                    self.performance_cache[strategy] = performance
                    self.last_performance_update[strategy] = time.time()
                
            except Exception as e:
                logger.error(f"Error fetching performance for strategy {strategy}: {str(e)}")
                
    async def _calculate_performance_weighted(self) -> Dict[str, float]:
        """
        Calculate weights based on weighted performance metrics.
        
        Returns:
            Dictionary of strategy weights
        """
        if not self.performance_cache:
            logger.warning("No performance data available for weight calculation")
            return self.weights.copy()
            
        metrics = self.config['metrics']
        scores = {}
        
        # Calculate score for each strategy
        for strategy, perf in self.performance_cache.items():
            if not perf or perf.get('trade_count', 0) < self.config['min_trades']:
                # Not enough data, use base weight
                scores[strategy] = self.base_weights[strategy]
                continue
                
            # Weighted sum of normalized metrics
            strategy_score = 0
            for metric_name, weight in metrics.items():
                if metric_name in perf:
                    # Handle negative values appropriately
                    metric_value = max(0, perf[metric_name])
                    strategy_score += metric_value * weight
            
            scores[strategy] = strategy_score
            
        # Normalize scores to weights
        if sum(scores.values()) <= 0:
            logger.warning("Total performance score is zero, reverting to base weights")
            return self.base_weights.copy()
            
        total_score = sum(scores.values())
        weights = {s: score/total_score for s, score in scores.items()}
        
        # For strategies without scores, use minimum weight
        for strategy in self.base_weights:
            if strategy not in weights:
                weights[strategy] = self.config['min_weight']
                
        return weights
    
    async def _calculate_adaptive_ensemble(self, market_regime: str = None) -> Dict[str, float]:
        """
        Calculate weights using adaptive ensemble approach considering market regime.
        
        Args:
            market_regime: Current market regime classification
            
        Returns:
            Dictionary of strategy weights
        """
        # Start with performance-weighted calculation
        weights = await self._calculate_performance_weighted()
        
        # If regime adaptation is enabled and regime is provided
        if self.config['regime_adaptation'] and market_regime:
            regime_adjustments = {}
            
            for strategy, perf in self.performance_cache.items():
                if not perf:
                    continue
                    
                # Get regime-specific performance
                regime_performance = perf.get('regime_performance', {}).get(market_regime, {})
                if not regime_performance:
                    continue
                    
                # Calculate regime adjustment factor based on win rate in this regime
                regime_win_rate = regime_performance.get('win_rate', 0.5)
                base_win_rate = perf.get('win_rate', 0.5)
                
                # Boost weights for strategies that perform well in current regime
                if regime_win_rate > base_win_rate:
                    # Adjustment factor grows with how much better strategy performs in this regime
                    adjustment = 1 + (regime_win_rate - base_win_rate) * 2  # Up to 2x boost
                    regime_adjustments[strategy] = adjustment
                else:
                    # Reduce weights for strategies that perform worse in current regime
                    adjustment = max(0.5, regime_win_rate / base_win_rate)  # Down to 0.5x reduction
                    regime_adjustments[strategy] = adjustment
            
            # Apply regime adjustments to weights
            for strategy, adjustment in regime_adjustments.items():
                if strategy in weights:
                    weights[strategy] *= adjustment
                    
            # Renormalize weights
            total = sum(weights.values())
            if total > 0:
                weights = {s: w/total for s, w in weights.items()}
                
        return weights
        
    async def _calculate_bayesian_weights(self) -> Dict[str, float]:
        """
        Calculate weights using Bayesian optimization approach.
        
        Returns:
            Dictionary of strategy weights
        """
        # This is a simplified version; a full Bayesian optimization would require
        # more complex modeling and optimization logic
        
        # Start with performance-weighted calculation
        weights = await self._calculate_performance_weighted()
        
        # Apply exponential boost based on recent improvement trends
        for strategy, perf in self.performance_cache.items():
            if 'improvement_trend' in perf and strategy in weights:
                trend = perf['improvement_trend']  # -1 to 1 value indicating trend
                # Boost weights for strategies showing improvement
                if trend > 0:
                    boost_factor = 1 + (trend * 0.5)  # Up to 1.5x boost
                    weights[strategy] *= boost_factor
        
        # Renormalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {s: w/total for s, w in weights.items()}
            
        return weights
    
    async def _calculate_default_weights(self) -> Dict[str, float]:
        """
        Calculate weights using default equal weighting.
        
        Returns:
            Dictionary of strategy weights
        """
        weight = 1.0 / len(self.base_weights)
        return {s: weight for s in self.base_weights}
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply minimum and maximum weight constraints.
        
        Args:
            weights: Strategy weights
            
        Returns:
            Constrained strategy weights
        """
        min_weight = self.config['min_weight']
        max_weight = self.config['max_weight']
        
        # Apply min/max constraints
        constrained = {}
        for strategy, weight in weights.items():
            constrained[strategy] = max(min_weight, min(max_weight, weight))
            
        # Renormalize
        total = sum(constrained.values())
        if total > 0:
            normalized = {s: w/total for s, w in constrained.items()}
            return normalized
            
        # Fallback to base weights if normalization fails
        return self.base_weights.copy()
    
    def get_strategy_weight(self, strategy_id: str) -> float:
        """
        Get the current weight for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Current weight or 0.0 if strategy not found
        """
        return self.weights.get(strategy_id, 0.0)
    
    def get_strategy_weight_history(self, 
                                  strategy_id: str, 
                                  lookback_days: int = 7) -> List[Tuple[float, float]]:
        """
        Get the weight history for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            lookback_days: Number of days to look back
            
        Returns:
            List of (timestamp, weight) tuples
        """
        if strategy_id not in self.weight_history:
            return []
            
        now = time.time()
        cutoff = now - (lookback_days * 86400)  # Convert days to seconds
        
        # Filter history by cutoff
        history = [(ts, w) for ts, w in self.weight_history[strategy_id] if ts >= cutoff]
        return history
    
    def get_all_weights(self) -> Dict[str, float]:
        """
        Get all current strategy weights.
        
        Returns:
            Dictionary of all strategy weights
        """
        return self.weights.copy()
    
    def get_performance_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get cached performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dictionary of performance metrics or empty dict if not found
        """
        return self.performance_cache.get(strategy_id, {})
    
    async def analyze_weight_effectiveness(self, 
                                        lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze how effective the weight adjustments have been over time.
        
        Args:
            lookback_days: Number of days to analyze
            
        Returns:
            Analysis results
        """
        results = {
            'correlation': {},
            'impact': {},
            'recommendation': {}
        }
        
        cutoff = time.time() - (lookback_days * 86400)
        
        for strategy, history in self.weight_history.items():
            # Filter by cutoff time
            filtered_history = [(ts, w) for ts, w in history if ts >= cutoff]
            if len(filtered_history) < 5:  # Need enough data points
                continue
                
            # Extract time series
            timestamps, weights = zip(*filtered_history)
            
            # Get performance timeline if available
            performance_timeline = []
            if self.performance_db:
                try:
                    perf_data = await self.performance_db.get_strategy_performance_timeline(
                        strategy_id=strategy,
                        start_time=cutoff
                    )
                    if perf_data:
                        performance_timeline = perf_data
                except Exception as e:
                    logger.error(f"Error fetching performance timeline for {strategy}: {str(e)}")
            
            # If we have performance data, analyze correlation between weight changes and performance
            if performance_timeline:
                # Align timestamps of weights and performance
                aligned_data = self._align_weight_and_performance(
                    weights=filtered_history,
                    performance=performance_timeline
                )
                
                if aligned_data and len(aligned_data) >= 5:
                    weight_series = [w for _, w, _ in aligned_data]
                    perf_series = [p for _, _, p in aligned_data]
                    
                    # Calculate correlation
                    correlation, p_value = stats.pearsonr(weight_series, perf_series)
                    
                    results['correlation'][strategy] = correlation
                    
                    # Estimate impact of weight adjustments
                    impact = self._estimate_weight_adjustment_impact(aligned_data)
                    results['impact'][strategy] = impact
                    
                    # Generate recommendation
                    if correlation > 0.3 and p_value < 0.05:
                        # Positive correlation - weight adjustments working well
                        results['recommendation'][strategy] = "maintain"
                    elif correlation < -0.3 and p_value < 0.05:
                        # Negative correlation - weight adjustments counter-productive
                        results['recommendation'][strategy] = "reverse"
                    else:
                        # No clear correlation
                        results['recommendation'][strategy] = "reassess"
        
        return results
    
    def _align_weight_and_performance(self, 
                                    weights: List[Tuple[float, float]], 
                                    performance: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        """
        Align weight and performance time series.
        
        Args:
            weights: List of (timestamp, weight) tuples
            performance: List of (timestamp, performance_metric) tuples
            
        Returns:
            List of (timestamp, weight, performance) tuples with aligned timestamps
        """
        # Convert to dictionaries for easier lookup
        weight_dict = dict(weights)
        perf_dict = dict(performance)
        
        # Get all unique timestamps
        all_timestamps = sorted(set(weight_dict.keys()).union(set(perf_dict.keys())))
        
        # Fill forward for missing values
        result = []
        last_weight = None
        last_perf = None
        
        for ts in all_timestamps:
            # Get or fill forward weight
            if ts in weight_dict:
                last_weight = weight_dict[ts]
            
            # Get or fill forward performance
            if ts in perf_dict:
                last_perf = perf_dict[ts]
            
            # Only add if both values are available
            if last_weight is not None and last_perf is not None:
                result.append((ts, last_weight, last_perf))
                
        return result
    
    def _estimate_weight_adjustment_impact(self, 
                                        aligned_data: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        Estimate the impact of weight adjustments on performance.
        
        Args:
            aligned_data: List of (timestamp, weight, performance) tuples
            
        Returns:
            Impact metrics
        """
        if len(aligned_data) < 2:
            return {}
            
        # Calculate weight changes and subsequent performance changes
        changes = []
        for i in range(1, len(aligned_data)):
            prev_ts, prev_weight, prev_perf = aligned_data[i-1]
            curr_ts, curr_weight, curr_perf = aligned_data[i]
            
            weight_change = curr_weight - prev_weight
            perf_change = curr_perf - prev_perf
            
            # Only consider significant weight changes
            if abs(weight_change) > 0.01:
                changes.append((weight_change, perf_change))
                
        if not changes:
            return {}
            
        # Separate positive and negative weight changes
        pos_changes = [(w, p) for w, p in changes if w > 0]
        neg_changes = [(w, p) for w, p in changes if w < 0]
        
        impact = {}
        
        # Impact of weight increases
        if pos_changes:
            pos_w, pos_p = zip(*pos_changes)
            avg_pos_impact = sum(pos_p) / len(pos_p)
            impact['increase_impact'] = avg_pos_impact
            
        # Impact of weight decreases
        if neg_changes:
            neg_w, neg_p = zip(*neg_changes)
            avg_neg_impact = sum(neg_p) / len(neg_p)
            impact['decrease_impact'] = avg_neg_impact
            
        # Overall effectiveness
        if pos_changes and neg_changes:
            # Are weight increases followed by better performance on average?
            if impact.get('increase_impact', 0) > 0:
                impact['increase_effective'] = True
            else:
                impact['increase_effective'] = False
                
            # Are weight decreases followed by worse performance on average?
            if impact.get('decrease_impact', 0) < 0:
                impact['decrease_effective'] = True
            else:
                impact['decrease_effective'] = False
                
        return impact
    
    async def export_weights_to_json(self, file_path: str):
        """
        Export current weights to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        try:
            data = {
                'council_name': self.council_name,
                'timestamp': time.time(),
                'weights': self.weights,
                'base_weights': self.base_weights,
                'config': self.config
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Exported weights to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export weights: {str(e)}")
            return False
    
    @classmethod
    async def import_weights_from_json(cls, file_path: str, 
                                     performance_db_connector = None):
        """
        Import weights from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            performance_db_connector: Database connector for performance data
            
        Returns:
            New WeightingSystem instance with imported data
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            council_name = data.get('council_name', 'imported_council')
            config = data.get('config', {})
            
            weighting_system = cls(
                council_name=council_name,
                config=config,
                performance_db_connector=performance_db_connector
            )
            
            # Set imported weights
            weighting_system.weights = data.get('weights', {})
            weighting_system.base_weights = data.get('base_weights', {})
            
            # Initialize weight history with current weights
            timestamp = time.time()
            for strategy, weight in weighting_system.weights.items():
                weighting_system.weight_history[strategy].append((timestamp, weight))
                
            logger.info(f"Imported weights for council {council_name}")
            return weighting_system
        except Exception as e:
            logger.error(f"Failed to import weights: {str(e)}")
            raise WeightingSystemError(f"Failed to import weights: {str(e)}")

