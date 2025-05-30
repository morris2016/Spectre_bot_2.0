#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Asset-Specific Brain Council

This module implements specialized brain councils for individual assets,
allowing for more focused and asset-specific trading decisions.
"""

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set

from common.logger import get_logger
from common.metrics import MetricsCollector
from common.exceptions import (
    ConfigurationError, ServiceStartupError, ServiceShutdownError,
    StrategyError, SignalGenerationError
)
from common.constants import SIGNAL_TYPES, POSITION_DIRECTION, MARKET_REGIMES
from .voting_system import VotingSystem, VotingResult

class AssetCouncil:
    """
    Asset-specific council that specializes in a single trading asset.
    
    This council aggregates signals from various strategy brains and ML models
    specifically for one asset, providing more focused and specialized decisions.
    """
    
    def __init__(self, 
                 asset_id: str, 
                 config: Dict[str, Any],
                 parent_council=None,
                 redis_client=None, 
                 db_client=None):
        """
        Initialize an asset-specific council.
        
        Args:
            asset_id: The asset identifier (e.g., "BTC/USD")
            config: Configuration dictionary
            parent_council: Reference to the parent council
            redis_client: Redis client for communication
            db_client: Database client
        """
        self.asset_id = asset_id
        self.config = config
        self.parent_council = parent_council
        self.redis_client = redis_client
        self.db_client = db_client
        self.logger = get_logger(f"AssetCouncil.{asset_id}")
        self.metrics = MetricsCollector(f"asset_council.{asset_id}")
        
        # Initialize voting system
        voting_config = config.get("asset_council", {}).get("voting_system", {})
        self.voting_system = VotingSystem({"voting_system": voting_config})
        
        # Track active brains for this asset
        self.active_brains = set()
        
        # Track ML models for this asset
        self.ml_models = {}
        
        # Track current market regime for this asset
        self.current_regime = MARKET_REGIMES["RANGING"]
        
        # Performance tracking
        self.performance_metrics = {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expected_value": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "overall_score": 0.5  # Default neutral score
        }
        
        # Signal history
        self.signal_history = []
        self.max_signal_history = config.get("asset_council", {}).get("max_signal_history", 100)
        
        self.logger.info(f"Asset Council initialized for {asset_id}")
    
    async def process_signals(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Process signals specific to this asset and generate a decision.
        
        Args:
            signals: List of signals from various brains and ML models
            
        Returns:
            Decision dictionary or None if no decision could be made
        """
        if not signals:
            return None
            
        self.logger.debug(f"Processing {len(signals)} signals for {self.asset_id}")
        
        # Group signals by timeframe
        timeframe_signals = self._group_by_timeframe(signals)
        
        # Process each timeframe group
        timeframe_decisions = {}
        for timeframe, tf_signals in timeframe_signals.items():
            # Skip timeframes with insufficient signals
            if len(tf_signals) < 2:
                continue
                
            # Prepare votes for voting system
            votes = {}
            weights = {}
            
            for i, signal in enumerate(tf_signals):
                source = signal.get("source", f"unknown_{i}")
                votes[source] = {
                    "direction": signal.get("direction", "hold"),
                    "confidence": signal.get("confidence", 0.5),
                    "timestamp": signal.get("timestamp", time.time()),
                    "signals": signal.get("signals", {}),
                    "metadata": signal.get("metadata", {})
                }
                
                # Determine weight based on source type
                if source.startswith("ml_"):
                    # Higher weight for ML models
                    weights[source] = self.config.get("asset_council", {}).get("ml_weight", 1.2)
                else:
                    # Standard weight for strategy brains
                    weights[source] = self.config.get("asset_council", {}).get("brain_weight", 1.0)
            
            # Generate decision for this timeframe
            if votes:
                # Add contextual information
                context = {
                    "asset_id": self.asset_id,
                    "timeframe": timeframe,
                    "regime": self.current_regime,
                    "timestamp": time.time()
                }
                
                # Get decision from voting system
                decision = await self.voting_system.generate_decision(votes, weights, context)
                timeframe_decisions[timeframe] = decision
        
        # Combine timeframe decisions into a final asset decision
        if timeframe_decisions:
            final_decision = self._combine_timeframe_decisions(timeframe_decisions)
            
            # Store in signal history
            self._update_signal_history(final_decision)
            
            return final_decision
        
        return None
    
    def _group_by_timeframe(self, signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group signals by timeframe.
        
        Args:
            signals: List of signals
            
        Returns:
            Dictionary mapping timeframes to signal lists
        """
        result = {}
        
        for signal in signals:
            timeframe = signal.get("timeframe", "unknown")
            if timeframe not in result:
                result[timeframe] = []
            result[timeframe].append(signal)
        
        return result
    
    def _combine_timeframe_decisions(self, timeframe_decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine decisions from multiple timeframes into a single decision.
        
        Args:
            timeframe_decisions: Dictionary mapping timeframes to decisions
            
        Returns:
            Combined decision dictionary
        """
        # Get timeframe weights from config
        timeframe_weights = self.config.get("asset_council", {}).get("timeframe_weights", {
            "1m": 0.5,
            "5m": 0.6,
            "15m": 0.7,
            "30m": 0.8,
            "1h": 1.0,
            "4h": 1.2,
            "1d": 1.5
        })
        
        # Default weights if not in config
        default_weight = 1.0
        
        # Count votes for each direction
        direction_votes = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        total_weight = 0.0
        
        # Weighted voting based on timeframe and confidence
        for timeframe, decision in timeframe_decisions.items():
            direction = decision.get("direction", "hold")
            confidence = decision.get("confidence", 0.5)
            weight = timeframe_weights.get(timeframe, default_weight)
            
            # Weight by both timeframe importance and decision confidence
            effective_weight = weight * confidence
            direction_votes[direction] += effective_weight
            total_weight += effective_weight
        
        # Determine winning direction
        if total_weight > 0:
            winning_direction = max(direction_votes.items(), key=lambda x: x[1])[0]
            confidence = direction_votes[winning_direction] / total_weight
        else:
            winning_direction = "hold"
            confidence = 0.0
        
        # Create final decision
        return {
            "asset_id": self.asset_id,
            "direction": winning_direction,
            "confidence": confidence,
            "timestamp": time.time(),
            "timeframe_decisions": timeframe_decisions,
            "regime": self.current_regime
        }
    
    def _update_signal_history(self, decision: Dict[str, Any]):
        """
        Update the signal history with a new decision.
        
        Args:
            decision: Decision dictionary
        """
        self.signal_history.append(decision)
        
        # Trim history if needed
        if len(self.signal_history) > self.max_signal_history:
            self.signal_history = self.signal_history[-self.max_signal_history:]
    
    async def update_regime(self, new_regime: str):
        """
        Update the current market regime for this asset.
        
        Args:
            new_regime: New market regime
        """
        if new_regime != self.current_regime:
            self.logger.info(f"Market regime changed for {self.asset_id}: {self.current_regime} -> {new_regime}")
            self.current_regime = new_regime
    
    async def register_brain(self, brain_name: str):
        """
        Register a strategy brain with this asset council.
        
        Args:
            brain_name: Name of the strategy brain
        """
        self.active_brains.add(brain_name)
        self.logger.debug(f"Registered brain {brain_name} with {self.asset_id} council")
    
    async def register_ml_model(self, model_name: str, model_type: str):
        """
        Register an ML model with this asset council.
        
        Args:
            model_name: Name of the ML model
            model_type: Type of the ML model
        """
        self.ml_models[model_name] = {
            "type": model_type,
            "registered_at": time.time()
        }
        self.logger.debug(f"Registered ML model {model_name} ({model_type}) with {self.asset_id} council")
    
    async def update_performance(self, metrics: Dict[str, float]):
        """
        Update performance metrics for this asset council.
        
        Args:
            metrics: Performance metrics dictionary
        """
        self.performance_metrics.update(metrics)
        self.logger.debug(f"Updated performance metrics for {self.asset_id} council")
