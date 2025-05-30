

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Brain Council - Base Council Class

This module provides the base class for all brain councils in the system.
Brain councils coordinate multiple strategy brains to generate coherent trading signals.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from common.logger import get_logger
from common.utils import generate_id, Singleton, TimeFrame, TradingMode
from common.metrics import MetricsCollector
from common.db_client import DatabaseClient
from common.redis_client import RedisClient
from common.exceptions import (
    CouncilInitializationError, BrainNotFoundError, 
    InsufficientDataError, InvalidSignalError
)

class BaseCouncil(ABC):
    """
    Base Council class for all brain councils.
    
    Brain councils are responsible for coordinating multiple strategy brains
    and generating coherent trading signals by aggregating and weighting
    the outputs from various strategy brains.
    """
    
    def __init__(
        self, 
        council_id: str,
        council_name: str,
        asset_id: str,
        platform: str,
        timeframe: TimeFrame,
        brain_ids: List[str] = None,
        config: Dict[str, Any] = None,
        db_client: DatabaseClient = None,
        redis_client: RedisClient = None,
        metrics_collector: MetricsCollector = None
    ):
        """
        Initialize the base council with configuration and connections.
        
        Args:
            council_id: Unique identifier for this council
            council_name: Human-readable name for this council
            asset_id: Asset identifier this council is responsible for
            platform: Trading platform (e.g., 'binance', 'deriv')
            timeframe: Time frame this council operates on
            brain_ids: List of brain IDs to include in this council
            config: Additional configuration parameters
            db_client: Database client for persistence
            redis_client: Redis client for real-time data
            metrics_collector: Metrics collection for performance monitoring
        """
        self.council_id = council_id
        self.council_name = council_name
        self.asset_id = asset_id
        self.platform = platform.lower()
        self.timeframe = timeframe
        self.brain_ids = brain_ids or []
        self.config = config or {}
        
        # Set up clients
        self.db_client = db_client
        self.redis_client = redis_client
        self.metrics_collector = metrics_collector
        
        # Set up logger
        self.logger = get_logger(
            f"council.{self.council_name}.{self.asset_id}.{self.timeframe.name}"
        )
        
        # Brain registry and weights
        self.brains = {}
        self.brain_weights = {}
        self.brain_performance = {}
        self.initial_weights = self.config.get('initial_weights', {})
        
        # Signal history and current state
        self.signal_history = []
        self.last_signal = None
        self.last_signal_time = None
        self.confidence_threshold = self.config.get('confidence_threshold', 0.65)
        self.voting_method = self.config.get('voting_method', 'weighted_average')
        
        # State tracking
        self.is_initialized = False
        self.is_running = False
        self.last_update_time = None
        
        # Performance metrics
        self.total_signals = 0
        self.correct_signals = 0
        self.incorrect_signals = 0
        self.win_rate = 0.0
        
        # For tracking conflicting signals and their resolution
        self.signal_conflicts = 0
        self.conflict_resolutions = {}
        
        # Asset-specific parameters
        self.asset_volatility = None
        self.asset_trend_strength = None
        self.asset_regime = None
        
        # Market context
        self.market_regime = None
        self.market_volatility = None
        self.market_sentiment = None
        
        # Advanced features
        self.confidence_decay_rate = self.config.get('confidence_decay_rate', 0.95)
        self.minimum_brain_count = self.config.get('minimum_brain_count', 3)
        self.dynamic_weight_adjustment = self.config.get('dynamic_weight_adjustment', True)
        self.performance_window = self.config.get('performance_window', 100)
        
        # For recording decision reasoning
        self.decision_reasons = []
        
        self.logger.info(f"Initialized {self.council_name} for {self.asset_id} on {self.platform}")
        
    async def initialize(self) -> bool:
        """
        Initialize the council by loading brains, configuration, and historical data.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            self.logger.info(f"Initializing council {self.council_id}")
            
            # Load brain registry from database
            await self._load_brain_registry()
            
            # Load historical performance data
            await self._load_performance_history()
            
            # Update brain weights based on historical performance
            await self._update_brain_weights()
            
            # Load market context
            await self._load_market_context()
            
            # Verify we have enough brains
            if len(self.brains) < self.minimum_brain_count:
                self.logger.warning(
                    f"Council has fewer brains ({len(self.brains)}) than minimum required ({self.minimum_brain_count})"
                )
            
            # Mark as initialized
            self.is_initialized = True
            self.logger.info(f"Council {self.council_id} initialization complete, {len(self.brains)} brains registered")
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_gauge(
                    f"council.{self.council_id}.brain_count", 
                    len(self.brains)
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize council {self.council_id}: {str(e)}")
            self.is_initialized = False
            raise CouncilInitializationError(f"Council initialization failed: {str(e)}") from e
    
    async def _load_brain_registry(self) -> None:
        """
        Load registered brains from the database.
        """
        if not self.db_client:
            self.logger.warning("No database client available, skipping brain registry loading")
            return
            
        try:
            # Query database for brains assigned to this council
            query = {
                "council_id": self.council_id,
                "asset_id": self.asset_id,
                "platform": self.platform,
                "timeframe": self.timeframe.name,
                "status": "active"
            }
            
            brain_records = await self.db_client.find("strategy_brains", query)
            
            if not brain_records:
                self.logger.warning(f"No active brains found for council {self.council_id}")
                return
                
            # Register each brain
            for brain_record in brain_records:
                brain_id = brain_record["brain_id"]
                brain_type = brain_record["brain_type"]
                
                # Register brain
                self.brains[brain_id] = {
                    "brain_id": brain_id,
                    "brain_type": brain_type,
                    "config": brain_record.get("config", {}),
                    "status": brain_record.get("status", "active"),
                    "metadata": brain_record.get("metadata", {})
                }
                
                # Set initial weight from config or default
                if brain_id in self.initial_weights:
                    self.brain_weights[brain_id] = self.initial_weights[brain_id]
                else:
                    # Default weight based on brain type
                    default_weight = self._get_default_weight(brain_type)
                    self.brain_weights[brain_id] = default_weight
                    
                self.logger.debug(f"Registered brain {brain_id} of type {brain_type} with weight {self.brain_weights[brain_id]}")
                
            # Normalize weights to sum to 1.0
            self._normalize_weights()
                
        except Exception as e:
            self.logger.error(f"Error loading brain registry: {str(e)}")
            raise
    
    def _get_default_weight(self, brain_type: str) -> float:
        """
        Get default weight based on brain type.
        
        Args:
            brain_type: Type of the brain
            
        Returns:
            float: Default weight for this brain type
        """
        # Different brain types might have different default weights
        # based on their historical performance or complexity
        default_weights = {
            "momentum": 0.8,
            "mean_reversion": 0.7,
            "breakout": 0.75,
            "volatility": 0.65,
            "pattern": 0.85,
            "sentiment": 0.6,
            "order_flow": 0.8,
            "market_structure": 0.85,
            "statistical": 0.7,
            "ml": 0.75,
            "reinforcement": 0.7,
            "onchain": 0.6,
            "regime": 0.65,
            "adaptive": 0.8,
            "trend": 0.75,
            "swing": 0.7,
            "scalping": 0.65,
            "arbitrage": 0.6,
            "correlation": 0.7,
            "divergence": 0.8,
            "ensemble": 0.9
        }
        
        return default_weights.get(brain_type.lower(), 0.5)
    
    def _normalize_weights(self) -> None:
        """
        Normalize brain weights to sum to 1.0.
        """
        if not self.brain_weights:
            return
            
        total_weight = sum(self.brain_weights.values())
        
        if total_weight == 0:
            # If all weights are zero, set equal weights
            equal_weight = 1.0 / len(self.brain_weights)
            for brain_id in self.brain_weights:
                self.brain_weights[brain_id] = equal_weight
        else:
            # Normalize weights
            for brain_id in self.brain_weights:
                self.brain_weights[brain_id] /= total_weight
    
    async def _load_performance_history(self) -> None:
        """
        Load historical performance data for all brains.
        """
        if not self.db_client:
            self.logger.warning("No database client available, skipping performance history loading")
            return
            
        try:
            # For each brain, load its recent performance history
            for brain_id in self.brains:
                query = {
                    "brain_id": brain_id,
                    "asset_id": self.asset_id,
                    "platform": self.platform,
                    "timeframe": self.timeframe.name
                }
                
                # Get the most recent performance records
                performance_records = await self.db_client.find(
                    "brain_performance",
                    query,
                    sort=[("timestamp", -1)],
                    limit=self.performance_window
                )
                
                if not performance_records:
                    self.logger.warning(f"No performance history found for brain {brain_id}")
                    self.brain_performance[brain_id] = {
                        "win_rate": 0.5,  # Default win rate
                        "total_signals": 0,
                        "correct_signals": 0,
                        "profit_factor": 1.0,
                        "average_return": 0.0,
                        "sharpe_ratio": 0.0,
                        "recent_signals": []
                    }
                else:
                    # Calculate performance metrics
                    total_signals = len(performance_records)
                    correct_signals = sum(1 for rec in performance_records if rec.get("is_correct", False))
                    
                    win_rate = correct_signals / total_signals if total_signals > 0 else 0.5
                    
                    # Calculate profit factor and other metrics if available
                    total_profit = sum(rec.get("profit", 0) for rec in performance_records if rec.get("profit", 0) > 0)
                    total_loss = abs(sum(rec.get("profit", 0) for rec in performance_records if rec.get("profit", 0) < 0))
                    profit_factor = total_profit / total_loss if total_loss > 0 else 1.0
                    
                    # Calculate average return
                    returns = [rec.get("profit_pct", 0) for rec in performance_records if "profit_pct" in rec]
                    average_return = sum(returns) / len(returns) if returns else 0.0
                    
                    # Calculate Sharpe ratio if we have return data
                    if returns and len(returns) > 1:
                        returns_std = np.std(returns)
                        sharpe_ratio = (average_return / returns_std) if returns_std > 0 else 0.0
                    else:
                        sharpe_ratio = 0.0
                    
                    # Store performance metrics
                    self.brain_performance[brain_id] = {
                        "win_rate": win_rate,
                        "total_signals": total_signals,
                        "correct_signals": correct_signals,
                        "profit_factor": profit_factor,
                        "average_return": average_return,
                        "sharpe_ratio": sharpe_ratio,
                        "recent_signals": [
                            {
                                "timestamp": rec.get("timestamp"),
                                "signal": rec.get("signal"),
                                "is_correct": rec.get("is_correct", False),
                                "profit": rec.get("profit", 0),
                                "profit_pct": rec.get("profit_pct", 0)
                            }
                            for rec in performance_records[:10]  # Keep last 10 signals
                        ]
                    }
                    
                    self.logger.debug(f"Loaded performance history for brain {brain_id}: win_rate={win_rate:.2f}, signals={total_signals}")
        
        except Exception as e:
            self.logger.error(f"Error loading performance history: {str(e)}")
            raise
    
    async def _update_brain_weights(self) -> None:
        """
        Update brain weights based on historical performance.
        """
        if not self.dynamic_weight_adjustment:
            self.logger.info("Dynamic weight adjustment disabled, using initial weights")
            return
            
        try:
            # Calculate performance-based weights
            performance_weights = {}
            
            for brain_id, perf in self.brain_performance.items():
                # Base weight from win rate
                win_rate = perf.get("win_rate", 0.5)
                
                # Incorporate profit factor and Sharpe ratio if available
                profit_factor = perf.get("profit_factor", 1.0)
                sharpe_ratio = perf.get("sharpe_ratio", 0.0)
                
                # Calculate a composite performance score
                # We give more weight to win rate but also consider profit factor and Sharpe ratio
                performance_score = (
                    0.6 * win_rate + 
                    0.3 * min(profit_factor / 3.0, 1.0) +  # Cap profit factor contribution at 1.0
                    0.1 * min(max(sharpe_ratio, 0) / 2.0, 1.0)  # Cap Sharpe ratio contribution at 1.0
                )
                
                # Adjust for signal count reliability
                signal_count = perf.get("total_signals", 0)
                signal_confidence = min(signal_count / 30.0, 1.0)  # Full confidence at 30+ signals
                
                # Combine performance score with signal confidence
                adjusted_score = 0.5 + (performance_score - 0.5) * signal_confidence
                
                # Store adjusted weight
                performance_weights[brain_id] = max(0.1, adjusted_score)  # Minimum weight of 0.1
                
                self.logger.debug(
                    f"Brain {brain_id} weight calculation: win_rate={win_rate:.2f}, "
                    f"profit_factor={profit_factor:.2f}, sharpe={sharpe_ratio:.2f}, "
                    f"signals={signal_count}, adjusted_score={adjusted_score:.2f}"
                )
            
            # Blend with existing weights for smooth transition (if any)
            if self.brain_weights:
                weight_blend_factor = 0.3  # 30% old weights, 70% new weights
                
                for brain_id in self.brains:
                    current_weight = self.brain_weights.get(brain_id, 0.0)
                    new_weight = performance_weights.get(brain_id, 0.5)
                    
                    # Blend weights
                    self.brain_weights[brain_id] = (
                        weight_blend_factor * current_weight +
                        (1 - weight_blend_factor) * new_weight
                    )
            else:
                # Use calculated weights directly
                self.brain_weights = performance_weights
            
            # Normalize weights
            self._normalize_weights()
            
            # Log updated weights
            for brain_id, weight in self.brain_weights.items():
                brain_type = self.brains[brain_id]["brain_type"] if brain_id in self.brains else "unknown"
                self.logger.info(f"Updated weight for brain {brain_id} ({brain_type}): {weight:.4f}")
                
        except Exception as e:
            self.logger.error(f"Error updating brain weights: {str(e)}")
            # Fall back to initial weights
            self.logger.warning("Falling back to initial weights due to update error")
    
    async def _load_market_context(self) -> None:
        """
        Load current market context for informed decision making.
        """
        if not self.redis_client:
            self.logger.warning("No Redis client available, skipping market context loading")
            return
            
        try:
            # Load market regime information
            market_regime_key = f"market:regime:{self.platform}:{self.asset_id}"
            market_regime = await self.redis_client.get(market_regime_key)
            
            if market_regime:
                self.market_regime = json.loads(market_regime)
                self.logger.debug(f"Loaded market regime: {self.market_regime}")
            
            # Load asset volatility
            volatility_key = f"market:volatility:{self.platform}:{self.asset_id}"
            volatility = await self.redis_client.get(volatility_key)
            
            if volatility:
                self.asset_volatility = float(volatility)
                self.logger.debug(f"Loaded asset volatility: {self.asset_volatility}")
            
            # Load asset trend strength
            trend_key = f"market:trend:{self.platform}:{self.asset_id}"
            trend = await self.redis_client.get(trend_key)
            
            if trend:
                self.asset_trend_strength = float(trend)
                self.logger.debug(f"Loaded asset trend strength: {self.asset_trend_strength}")
            
            # Load market sentiment
            sentiment_key = f"market:sentiment:{self.platform}:{self.asset_id}"
            sentiment = await self.redis_client.get(sentiment_key)
            
            if sentiment:
                self.market_sentiment = json.loads(sentiment)
                self.logger.debug(f"Loaded market sentiment: {self.market_sentiment}")
                
        except Exception as e:
            self.logger.error(f"Error loading market context: {str(e)}")
            # Not critical, can continue without context
    
    async def start(self) -> bool:
        """
        Start the council's operation.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.is_initialized:
            self.logger.error("Cannot start council that is not initialized")
            return False
            
        try:
            self.logger.info(f"Starting council {self.council_id}")
            self.is_running = True
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    f"council.{self.council_id}.start", 
                    1
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start council {self.council_id}: {str(e)}")
            self.is_running = False
            return False
    
    async def stop(self) -> bool:
        """
        Stop the council's operation.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        try:
            self.logger.info(f"Stopping council {self.council_id}")
            self.is_running = False
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_counter(
                    f"council.{self.council_id}.stop", 
                    1
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop council {self.council_id}: {str(e)}")
            return False
    
    @abstractmethod
    async def process_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming signals from brains and generate a consolidated signal.
        
        This method must be implemented by specific council types.
        
        Args:
            signals: Dictionary of signals from different brains
            
        Returns:
            Dict[str, Any]: Consolidated signal with confidence and metadata
        """
        pass
    
    async def get_weighted_vote(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate weighted vote from brain signals.
        
        Args:
            signals: Dictionary mapping brain IDs to their signal dictionaries
            
        Returns:
            Dict[str, Any]: Consolidated signal with confidence and metadata
        """
        if not signals:
            self.logger.warning("No signals provided for weighted vote")
            return {
                "signal": "NEUTRAL",
                "direction": 0,
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "reason": "No signals available",
                    "brain_count": 0,
                    "voting_method": "weighted_average"
                }
            }
        
        # Extract directional values and confidence
        weighted_sum = 0.0
        total_weight = 0.0
        participating_brains = []
        confidence_values = []
        
        for brain_id, signal_data in signals.items():
            if brain_id not in self.brain_weights:
                self.logger.warning(f"Signal received from unknown brain {brain_id}, ignoring")
                continue
                
            # Get the direction value (-1 for SELL, 0 for NEUTRAL, 1 for BUY)
            direction = signal_data.get("direction", 0)
            confidence = signal_data.get("confidence", 0.5)
            
            # Skip signals with very low confidence
            if confidence < 0.2:
                self.logger.debug(f"Skipping low confidence signal from brain {brain_id}: {confidence:.2f}")
                continue
                
            # Get brain weight
            weight = self.brain_weights.get(brain_id, 0.1)
            
            # Apply confidence-weighted direction
            weighted_sum += direction * weight * confidence
            total_weight += weight
            
            participating_brains.append({
                "brain_id": brain_id,
                "brain_type": self.brains.get(brain_id, {}).get("brain_type", "unknown"),
                "direction": direction,
                "confidence": confidence,
                "weight": weight,
                "contribution": direction * weight * confidence
            })
            
            confidence_values.append(confidence)
        
        if total_weight == 0:
            self.logger.warning("No valid signals with sufficient confidence")
            return {
                "signal": "NEUTRAL",
                "direction": 0,
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "reason": "No valid signals with sufficient confidence",
                    "brain_count": 0,
                    "voting_method": "weighted_average"
                }
            }
        
        # Calculate normalized weighted sum
        normalized_weighted_sum = weighted_sum / total_weight
        
        # Determine signal direction
        if normalized_weighted_sum > 0.3:
            signal = "BUY"
            direction = 1
        elif normalized_weighted_sum < -0.3:
            signal = "SELL"
            direction = -1
        else:
            signal = "NEUTRAL"
            direction = 0
        
        # Calculate overall confidence
        # This is a composite of the weighted sum magnitude and the average confidence
        weighted_sum_magnitude = abs(normalized_weighted_sum)
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.5
        
        # Blend the two measures for final confidence
        # Higher weight to weighted sum magnitude as it represents consensus
        confidence = (0.7 * weighted_sum_magnitude + 0.3 * avg_confidence)
        
        # Calculate agreement percentage
        agrees_with_result = sum(
            1 for brain in participating_brains 
            if (brain["direction"] > 0 and direction > 0) or 
               (brain["direction"] < 0 and direction < 0) or
               (brain["direction"] == 0 and direction == 0)
        )
        
        agreement_pct = agrees_with_result / len(participating_brains) if participating_brains else 0
        
        # Prepare result
        result = {
            "signal": signal,
            "direction": direction,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "weighted_sum": normalized_weighted_sum,
                "brain_count": len(participating_brains),
                "agreement_pct": agreement_pct,
                "voting_method": "weighted_average",
                "participating_brains": participating_brains
            }
        }
        
        # Record decision reasoning
        reason_components = []
        
        # Add reasons for the decision
        if direction > 0:
            buy_brains = [b for b in participating_brains if b["direction"] > 0]
            buy_brains.sort(key=lambda x: x["contribution"], reverse=True)
            
            top_contributors = buy_brains[:3]
            reason_components.append(f"BUY signal based on:")
            for brain in top_contributors:
                reason_components.append(
                    f"- {brain['brain_type']} brain with {brain['confidence']:.2f} confidence"
                )
        elif direction < 0:
            sell_brains = [b for b in participating_brains if b["direction"] < 0]
            sell_brains.sort(key=lambda x: x["contribution"])
            
            top_contributors = sell_brains[:3]
            reason_components.append(f"SELL signal based on:")
            for brain in top_contributors:
                reason_components.append(
                    f"- {brain['brain_type']} brain with {brain['confidence']:.2f} confidence"
                )
        else:
            reason_components.append(f"NEUTRAL signal due to conflicting or low-confidence inputs")
        
        result["metadata"]["reason"] = "\n".join(reason_components)
        self.decision_reasons.append(result["metadata"]["reason"])
        
        return result
    
    async def record_signal(self, consolidated_signal: Dict[str, Any]) -> None:
        """
        Record the consolidated signal for history and reporting.
        
        Args:
            consolidated_signal: The signal to record
        """
        if not self.db_client:
            self.logger.warning("No database client available, skipping signal recording")
            return
            
        try:
            # Add additional metadata
            signal_record = {
                "council_id": self.council_id,
                "council_type": self.__class__.__name__,
                "asset_id": self.asset_id,
                "platform": self.platform,
                "timeframe": self.timeframe.name,
                "signal": consolidated_signal.get("signal"),
                "direction": consolidated_signal.get("direction", 0),
                "confidence": consolidated_signal.get("confidence", 0.0),
                "timestamp": datetime.now(),
                "metadata": consolidated_signal.get("metadata", {}),
                "result": None,  # To be updated later
                "profit": None,  # To be updated later
                "is_correct": None  # To be updated later
            }
            
            # Store in database
            await self.db_client.insert("council_signals", signal_record)
            
            # Update signal history
            self.signal_history.append(signal_record)
            
            # Keep history to a reasonable size
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
                
            # Update last signal tracking
            self.last_signal = consolidated_signal
            self.last_signal_time = datetime.now()
            
            # Record metrics
            if self.metrics_collector:
                signal_type = consolidated_signal.get("signal", "NEUTRAL")
                self.metrics_collector.record_counter(
                    f"council.{self.council_id}.signal.{signal_type.lower()}", 
                    1
                )
                self.metrics_collector.record_gauge(
                    f"council.{self.council_id}.confidence", 
                    consolidated_signal.get("confidence", 0.0)
                )
                
        except Exception as e:
            self.logger.error(f"Error recording signal: {str(e)}")
            # Not critical, can continue
    
    async def register_brain(self, brain_id: str, brain_type: str, config: Dict[str, Any] = None) -> bool:
        """
        Register a new brain with this council.
        
        Args:
            brain_id: Unique ID of the brain
            brain_type: Type of the brain
            config: Configuration for the brain
            
        Returns:
            bool: True if registration was successful, False otherwise
        """
        try:
            if brain_id in self.brains:
                self.logger.warning(f"Brain {brain_id} already registered, updating configuration")
            
            # Register brain
            self.brains[brain_id] = {
                "brain_id": brain_id,
                "brain_type": brain_type,
                "config": config or {},
                "status": "active",
                "metadata": {}
            }
            
            # Set initial weight
            if brain_id in self.initial_weights:
                self.brain_weights[brain_id] = self.initial_weights[brain_id]
            else:
                # Default weight based on brain type
                default_weight = self._get_default_weight(brain_type)
                self.brain_weights[brain_id] = default_weight
                
            # Normalize weights
            self._normalize_weights()
            
            self.logger.info(f"Registered brain {brain_id} of type {brain_type} with weight {self.brain_weights[brain_id]}")
            
            # Persist to database if available
            if self.db_client:
                brain_record = {
                    "brain_id": brain_id,
                    "brain_type": brain_type,
                    "council_id": self.council_id,
                    "asset_id": self.asset_id,
                    "platform": self.platform,
                    "timeframe": self.timeframe.name,
                    "config": config or {},
                    "status": "active",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                
                await self.db_client.insert("strategy_brains", brain_record)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering brain {brain_id}: {str(e)}")
            return False
    
    async def unregister_brain(self, brain_id: str) -> bool:
        """
        Unregister a brain from this council.
        
        Args:
            brain_id: Unique ID of the brain to unregister
            
        Returns:
            bool: True if unregistration was successful, False otherwise
        """
        try:
            if brain_id not in self.brains:
                self.logger.warning(f"Brain {brain_id} not registered with this council")
                return False
            
            # Remove from registry
            del self.brains[brain_id]
            
            # Remove weight
            if brain_id in self.brain_weights:
                del self.brain_weights[brain_id]
                
            # Normalize remaining weights
            self._normalize_weights()
            
            self.logger.info(f"Unregistered brain {brain_id}")
            
            # Update database if available
            if self.db_client:
                query = {
                    "brain_id": brain_id,
                    "council_id": self.council_id
                }
                
                update = {
                    "$set": {
                        "status": "inactive",
                        "updated_at": datetime.now()
                    }
                }
                
                await self.db_client.update("strategy_brains", query, update)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error unregistering brain {brain_id}: {str(e)}")
            return False
    
    async def update_brain_performance(self, brain_id: str, performance_data: Dict[str, Any]) -> bool:
        """
        Update performance metrics for a brain.
        
        Args:
            brain_id: Unique ID of the brain
            performance_data: Performance metrics to update
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            if brain_id not in self.brains:
                self.logger.warning(f"Cannot update performance for unknown brain {brain_id}")
                return False
            
            # Update performance data
            if brain_id not in self.brain_performance:
                self.brain_performance[brain_id] = {
                    "win_rate": 0.5,
                    "total_signals": 0,
                    "correct_signals": 0,
                    "profit_factor": 1.0,
                    "average_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "recent_signals": []
                }
            
            # Update win rate if provided
            if "win_rate" in performance_data:
                self.brain_performance[brain_id]["win_rate"] = performance_data["win_rate"]
            
            # Update signal counts if provided
            if "total_signals" in performance_data:
                self.brain_performance[brain_id]["total_signals"] = performance_data["total_signals"]
                
            if "correct_signals" in performance_data:
                self.brain_performance[brain_id]["correct_signals"] = performance_data["correct_signals"]
                
            # Update other metrics if provided
            for metric in ["profit_factor", "average_return", "sharpe_ratio"]:
                if metric in performance_data:
                    self.brain_performance[brain_id][metric] = performance_data[metric]
            
            # Add recent signal if provided
            if "signal" in performance_data:
                recent_signal = {
                    "timestamp": performance_data.get("timestamp", datetime.now()),
                    "signal": performance_data.get("signal"),
                    "is_correct": performance_data.get("is_correct", None),
                    "profit": performance_data.get("profit", None),
                    "profit_pct": performance_data.get("profit_pct", None)
                }
                
                self.brain_performance[brain_id]["recent_signals"].insert(0, recent_signal)
                
                # Keep only last 10 signals
                if len(self.brain_performance[brain_id]["recent_signals"]) > 10:
                    self.brain_performance[brain_id]["recent_signals"] = self.brain_performance[brain_id]["recent_signals"][:10]
            
            self.logger.debug(f"Updated performance for brain {brain_id}")
            
            # If dynamic weight adjustment is enabled, update weights
            if self.dynamic_weight_adjustment:
                await self._update_brain_weights()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating brain performance: {str(e)}")
            return False
    
    async def update_consolidated_result(self, signal_timestamp: datetime, result: str, profit: float, is_correct: bool) -> bool:
        """
        Update the result of a previously generated consolidated signal.
        
        Args:
            signal_timestamp: Timestamp of the signal to update
            result: Result of the signal (e.g., "WIN", "LOSS")
            profit: Profit/loss amount
            is_correct: Whether the signal was correct
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        if not self.db_client:
            self.logger.warning("No database client available, skipping result update")
            return False
            
        try:
            # Find the signal in the database
            query = {
                "council_id": self.council_id,
                "timestamp": {
                    "$gte": signal_timestamp - timedelta(minutes=1),
                    "$lte": signal_timestamp + timedelta(minutes=1)
                }
            }
            
            update = {
                "$set": {
                    "result": result,
                    "profit": profit,
                    "is_correct": is_correct,
                    "updated_at": datetime.now()
                }
            }
            
            # Update the record
            result = await self.db_client.update("council_signals", query, update)
            
            if result:
                self.logger.info(f"Updated signal result: {result}, profit: {profit}, is_correct: {is_correct}")
                
                # Update performance metrics
                self.total_signals += 1
                if is_correct:
                    self.correct_signals += 1
                else:
                    self.incorrect_signals += 1
                    
                # Update win rate
                if self.total_signals > 0:
                    self.win_rate = self.correct_signals / self.total_signals
                    
                # Record metrics
                if self.metrics_collector:
                    self.metrics_collector.record_counter(
                        f"council.{self.council_id}.result.{result.lower()}", 
                        1
                    )
                    self.metrics_collector.record_gauge(
                        f"council.{self.council_id}.win_rate", 
                        self.win_rate
                    )
                
                return True
            else:
                self.logger.warning(f"No signal found matching timestamp {signal_timestamp}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating signal result: {str(e)}")
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this council.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = {
            "council_id": self.council_id,
            "council_name": self.council_name,
            "asset_id": self.asset_id,
            "platform": self.platform,
            "timeframe": self.timeframe.name,
            "total_signals": self.total_signals,
            "correct_signals": self.correct_signals,
            "incorrect_signals": self.incorrect_signals,
            "win_rate": self.win_rate,
            "brain_count": len(self.brains),
            "active_brain_count": sum(1 for b in self.brains.values() if b.get("status") == "active"),
            "signal_conflicts": self.signal_conflicts,
            "last_update_time": self.last_update_time,
            "brain_performance": self.brain_performance,
            "market_context": {
                "regime": self.market_regime,
                "volatility": self.asset_volatility,
                "trend_strength": self.asset_trend_strength,
                "sentiment": self.market_sentiment
            }
        }
        
        return metrics
