#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Brain Council Service

This module implements the Brain Council Service, an ensemble decision mechanism
that combines signals from multiple strategy brains and timeframes to generate
final trading decisions.
"""

import os
import time
import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set

from common.logger import get_logger
from common.metrics import MetricsCollector
from common.exceptions import (
    ConfigurationError, ServiceStartupError, ServiceShutdownError,
    StrategyError, SignalGenerationError
)
from common.constants import SIGNAL_TYPES, POSITION_DIRECTION, MARKET_REGIMES

# Import new council system
from .council_manager import CouncilManager
from .asset_council import AssetCouncil
from .ml_council import MLCouncil


class BrainCouncilService:
    """
    Service for combining signals from multiple strategy brains.
    
    Acts as an ensemble decision mechanism to improve overall system performance
    and reduce false signals.
    """
    
    def __init__(self, config, loop=None, redis_client=None, db_client=None):
        """
        Initialize the Brain Council Service.
        
        Args:
            config: System configuration
            loop: Event loop
            redis_client: Redis client for communication
            db_client: Database client
        """
        self.config = config
        self.loop = loop or asyncio.get_event_loop()
        self.redis_client = redis_client
        self.db_client = db_client
        self.logger = get_logger("BrainCouncilService")
        self.metrics = MetricsCollector("brain_council")
        
        self.running = False
        self.tasks = []
        
        # Signal queue
        self.pending_signals = []
        self.processed_signals = set()  # Track processed signal IDs
        
        # Initialize the new council system
        self.council_manager = CouncilManager(
            config=config,
            redis_client=redis_client,
            db_client=db_client
        )
        
        # Keep legacy council types for backward compatibility
        self.council_types = {
            "timeframe": self.config.get("brain_council.council_types.timeframe", True),
            "asset": self.config.get("brain_council.council_types.asset", True),
            "regime": self.config.get("brain_council.council_types.regime", True),
            "master": self.config.get("brain_council.council_types.master", True)
        }
        
        # Debug logging for council structure
        self.logger.info(f"Brain Council initialized with enhanced council system")
        self.logger.info(f"New asset-specific councils and ML council integration enabled")
        
        # Legacy voting method (for backward compatibility)
        self.voting_method = self.config.get("brain_council.voting_method", "weighted")
        
        # Consensus and confidence thresholds
        self.min_consensus = self.config.get("brain_council.signal_generation.min_consensus", 0.5)
        self.min_confidence = self.config.get("brain_council.signal_generation.min_confidence", 0.6)
        self.strength_threshold = self.config.get("brain_council.signal_generation.strength_threshold", 0.7)
        
        # Active positions (to track exits)
        self.active_positions = {}  # symbol -> position info
        
        # Brain weights (legacy, now handled by council manager)
        self.brain_weights = {}
        self.timeframe_weights = {}
        
    async def start(self):
        """Start the Brain Council Service."""
        self.logger.info("Starting Brain Council Service")
        
        # Initialize the council manager
        await self.council_manager.initialize()
        
        # Initialize legacy brain weights for backward compatibility
        await self._initialize_weights()
        
        # Start signal processing task
        self.tasks.append(asyncio.create_task(self._signal_processor()))
        
        # Subscribe to signal channel
        signal_channel = self.config.get("strategy_brains.signal_channel", "signals")
        await self._subscribe_to_signals(signal_channel)
        
        # Start weight adjustment task
        if self.config.get("brain_council.weighting.auto_adjust", True):
            adjustment_interval = self.config.get("brain_council.weighting.adjustment_interval", 10)
            self.tasks.append(asyncio.create_task(
                self._weight_adjustment_loop(adjustment_interval)
            ))
            
        # Subscribe to position updates
        await self._subscribe_to_position_updates()
        
        # Subscribe to ML model registrations
        await self._subscribe_to_ml_model_registrations()
        
        self.running = True
        self.logger.info("Brain Council Service started successfully")
        
    async def stop(self):
        """Stop the Brain Council Service."""
        self.logger.info("Stopping Brain Council Service")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        self.logger.info("Brain Council Service stopped successfully")
        
    async def health_check(self) -> bool:
        """
        Perform a health check on the Brain Council Service.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        if not self.running:
            return False
            
        # Check if redis client is healthy
        if self.redis_client and not await self.redis_client.ping():
            self.logger.warning("Redis connection is not healthy")
            return False
            
        # Check if db client is healthy
        if self.db_client and not await self.db_client.ping():
            self.logger.warning("Database connection is not healthy")
            return False
            
        return True
        
    async def _initialize_weights(self):
        """Initialize brain and timeframe weights."""
        self.logger.info("Initializing brain and timeframe weights")
        
        # Load brain weights
        initial_method = self.config.get("brain_council.weighting.initial", "equal")
        
        try:
            if initial_method == "equal":
                # Initialize with equal weights
                await self._initialize_equal_weights()
            elif initial_method == "performance":
                # Initialize weights based on historical performance
                await self._initialize_performance_weights()
            elif initial_method == "custom":
                # Initialize with custom weights from configuration
                await self._initialize_custom_weights()
            else:
                self.logger.warning(f"Unknown weight initialization method: {initial_method}")
                await self._initialize_equal_weights()
                
            self.logger.info("Brain and timeframe weights initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing weights: {str(e)}")
            # Fall back to equal weights
            await self._initialize_equal_weights()
            
    async def _initialize_equal_weights(self):
        """Initialize all weights equally."""
        # Get list of brains
        brains = await self._get_active_brains()
        
        # Equal weight for each brain
        if brains:
            weight = 1.0 / len(brains)
            for brain in brains:
                self.brain_weights[brain] = weight
                
        # Initialize timeframe weights
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        weight = 1.0 / len(timeframes)
        for tf in timeframes:
            self.timeframe_weights[tf] = weight
            
        self.logger.info("Initialized equal weights")
        
    async def _initialize_performance_weights(self):
        """Initialize weights based on historical performance."""
        try:
            # Get brain performance data
            if not self.db_client:
                raise Exception("Database client not available")
                
            query = """
            SELECT brain_name, overall_score FROM strategy_performance
            WHERE timestamp > $1
            ORDER BY timestamp DESC
            """
            
            # Get data from last 30 days
            cutoff_time = time.time() - (30 * 24 * 60 * 60)
            rows = await self.db_client.fetch_all(query, cutoff_time)
            
            # Calculate weights based on performance scores
            if rows:
                brain_scores = {}
                
                # Get most recent score for each brain
                for row in rows:
                    brain_name = row["brain_name"]
                    if brain_name not in brain_scores:
                        brain_scores[brain_name] = row["overall_score"]
                        
                # Calculate weights
                total_score = sum(brain_scores.values())
                if total_score > 0:
                    for brain, score in brain_scores.items():
                        # Calculate weight proportional to score
                        weight = score / total_score
                        
                        # Apply min/max constraints
                        min_weight = self.config.get("brain_council.weighting.min_weight", 0.05)
                        max_weight = self.config.get("brain_council.weighting.max_weight", 0.5)
                        weight = max(min_weight, min(max_weight, weight))
                        
                        self.brain_weights[brain] = weight
                        
                    self.logger.info(f"Initialized performance-based weights for {len(brain_scores)} brains")
                else:
                    # Fall back to equal weights
                    await self._initialize_equal_weights()
            else:
                # Fall back to equal weights
                await self._initialize_equal_weights()
                
            # Initialize timeframe weights (could be extended to use performance data)
            timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            weight = 1.0 / len(timeframes)
            for tf in timeframes:
                self.timeframe_weights[tf] = weight
                
        except Exception as e:
            self.logger.error(f"Error initializing performance weights: {str(e)}")
            # Fall back to equal weights
            await self._initialize_equal_weights()
            
    async def _initialize_custom_weights(self):
        """Initialize weights from custom configuration."""
        # Get custom brain weights
        custom_weights = self.config.get("brain_council.weighting.custom_brain_weights", {})
        if custom_weights:
            self.brain_weights = custom_weights.copy()
            self.logger.info(f"Initialized custom brain weights for {len(custom_weights)} brains")
        else:
            # Fall back to equal weights for brains
            await self._initialize_equal_weights()
            
        # Get custom timeframe weights
        custom_tf_weights = self.config.get("brain_council.weighting.custom_timeframe_weights", {})
        if custom_tf_weights:
            self.timeframe_weights = custom_tf_weights.copy()
            self.logger.info(f"Initialized custom timeframe weights for {len(custom_tf_weights)} timeframes")
        else:
            # Initialize timeframe weights
            timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            weight = 1.0 / len(timeframes)
            for tf in timeframes:
                self.timeframe_weights[tf] = weight
                
    async def _get_active_brains(self) -> List[str]:
        """
        Get a list of active strategy brains.
        
        Returns:
            List of brain names
        """
        try:
            # Request active brains from strategy brain service
            channel = "strategy_brains.active_brains.request"
            response_channel = f"strategy_brains.active_brains.response.{int(time.time())}"
            
            request = {
                "service": "brain_council",
                "request_id": response_channel,
                "response_channel": response_channel,
                "timestamp": time.time()
            }
            
            # Publish request
            await self.redis_client.publish(channel, request)
            
            # Wait for response with timeout
            timeout = 10  # seconds
            response = await self.redis_client.subscribe_and_wait(
                response_channel, timeout=timeout
            )
            
            if response and "active_brains" in response:
                return response["active_brains"]
                
            self.logger.warning("Failed to get active brains, using default")
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting active brains: {str(e)}")
            return []
            
    async def _subscribe_to_signals(self, channel: str):
        """
        Subscribe to the signals channel.
        
        Args:
            channel: Channel name to subscribe to
        """
        try:
            async def signal_callback(message):
                if self.running and message:
                    # Add to pending signals queue
                    self.pending_signals.append(message)
                    
            await self.redis_client.subscribe(channel, signal_callback)
            self.logger.info(f"Subscribed to signals on channel: {channel}")
            
        except Exception as e:
            self.logger.error(f"Error subscribing to signals: {str(e)}")
            raise ServiceStartupError(f"Failed to subscribe to signals: {str(e)}")
            
    async def _subscribe_to_position_updates(self):
        """Subscribe to position update channel."""
        try:
            channel = "execution.position_update"
            
            async def position_callback(message):
                if self.running and message:
                    await self._handle_position_update(message)
                    
            await self.redis_client.subscribe(channel, position_callback)
            self.logger.info(f"Subscribed to position updates on channel: {channel}")
            
        except Exception as e:
            self.logger.error(f"Error subscribing to position updates: {str(e)}")
            
    async def _handle_position_update(self, position: Dict[str, Any]):
        """
        Handle position update message.
        
        Args:
            position: Position update data
        """
        try:
            if "symbol" not in position or "exchange" not in position:
                return
                
            key = f"{position['exchange']}:{position['symbol']}"
            
            if position.get("status") == "closed":
                # Position closed, remove from active positions
                if key in self.active_positions:
                    del self.active_positions[key]
                    self.logger.debug(f"Removed closed position: {key}")
            else:
                # Update or add position
                self.active_positions[key] = position
                self.logger.debug(f"Updated active position: {key}")
                
        except Exception as e:
            self.logger.error(f"Error handling position update: {str(e)}")
            
    async def _signal_processor(self):
        """Process incoming signals and generate ensemble decisions."""
        self.logger.info("Starting signal processor")
        
        while self.running:
            try:
                # Process signals in batches
                if self.pending_signals:
                    # Get a batch of signals (up to 100)
                    batch = self.pending_signals[:100]
                    self.pending_signals = self.pending_signals[100:]
                    
                    # Group signals by symbol and timeframe
                    signal_groups = self._group_signals(batch)
                    
                    # Process each group
                    for key, signals in signal_groups.items():
                        await self._process_signal_group(key, signals)
                        
                # Short sleep before next processing cycle
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                self.logger.info("Signal processor cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in signal processor: {str(e)}")
                await asyncio.sleep(1)
                
    def _group_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group signals by symbol, exchange and timeframe.
        
        Args:
            signals: List of signals
            
        Returns:
            Grouped signals dictionary
        """
        groups = {}
        
        for signal in signals:
            if "symbol" not in signal or "exchange" not in signal:
                continue
                
            # Create a key based on symbol, exchange and timeframe
            timeframe = signal.get("timeframe", "unknown")
            key = f"{signal['exchange']}:{signal['symbol']}:{timeframe}"
            
            if key not in groups:
                groups[key] = []
                
            # Only add if not already processed
            signal_id = signal.get("id")
            if signal_id and signal_id in self.processed_signals:
                continue
                
            groups[key].append(signal)
            
            # Mark as processed
            if signal_id:
                self.processed_signals.add(signal_id)
                
        return groups
        
    async def _process_signal_group(self, group_key: str, signals: List[Dict[str, Any]]):
        """
        Process a group of signals for the same symbol/timeframe.
        
        Args:
            group_key: Group key (exchange:symbol:timeframe)
            signals: List of signals in the group
        """
        if not signals:
            return
            
        self.logger.debug(f"Processing {len(signals)} signals for {group_key}")
        
        # Extract asset information from group key
        parts = group_key.split(":")
        if len(parts) != 3:
            self.logger.warning(f"Invalid group key: {group_key}")
            return
            
        exchange, symbol, timeframe = parts
        
        # Process using the new council system
        try:
            # Add asset_id to signals if not present
            for signal in signals:
                if "asset_id" not in signal:
                    signal["asset_id"] = symbol
            
            # Process through council manager
            asset_decisions = await self.council_manager.process_signals(signals)
            
            # If we have a decision for this asset, publish it
            if symbol in asset_decisions:
                decision = asset_decisions[symbol]
                
                # Create final signal
                final_signal = self._create_final_signal(
                    decision, exchange, symbol, timeframe, signals
                )
                
                # Publish final signal
                await self._publish_final_signal(final_signal)
                return
        except Exception as e:
            self.logger.error(f"Error processing with new council system: {str(e)}")
            self.logger.info("Falling back to legacy processing method")
        
        # Legacy processing (fallback)
        try:
            # Check for conflicts (e.g., both long and short signals)
            has_conflicts = self._check_for_conflicts(signals)
            
            # Calculate ensemble decision
            if self.voting_method == "simple":
                decision = self._simple_voting(signals)
            elif self.voting_method == "weighted":
                decision = self._weighted_voting(signals)
            elif self.voting_method == "confidence":
                decision = self._confidence_voting(signals)
            else:
                self.logger.warning(f"Unknown voting method: {self.voting_method}")
                decision = self._weighted_voting(signals)  # Default to weighted
                
            # Check if we should generate a final signal
            if decision and self._should_generate_signal(decision, exchange, symbol):
                # Create final signal
                final_signal = self._create_final_signal(decision, exchange, symbol, timeframe, signals)
                
                # Publish final signal
                await self._publish_final_signal(final_signal)
        except Exception as e:
            self.logger.error(f"Error in legacy signal processing: {str(e)}")
        
        try:
            # Get signal details
            parts = group_key.split(":")
            if len(parts) != 3:
                self.logger.warning(f"Invalid group key: {group_key}")
                return
                
            exchange, symbol, timeframe = parts
            
            # Check for conflicts (e.g., both long and short signals)
            has_conflicts = self._check_for_conflicts(signals)
            
            # Calculate ensemble decision
            if self.voting_method == "simple":
                decision = self._simple_voting(signals)
            elif self.voting_method == "weighted":
                decision = self._weighted_voting(signals)
            elif self.voting_method == "confidence":
                decision = self._confidence_voting(signals)
            else:
                self.logger.warning(f"Unknown voting method: {self.voting_method}")
                decision = self._weighted_voting(signals)  # Default to weighted
                
            # Check if we should generate a final signal
            if decision and self._should_generate_signal(decision, exchange, symbol):
                # Create final signal
                final_signal = self._create_final_signal(decision, exchange, symbol, timeframe, signals)
                
                # Publish final signal
                await self._publish_final_signal(final_signal)
                
        except Exception as e:
            self.logger.error(f"Error processing signal group {group_key}: {str(e)}")
            
    def _check_for_conflicts(self, signals: List[Dict[str, Any]]) -> bool:
        """
        Check if signals contain conflicting directions.
        
        Args:
            signals: List of signals
            
        Returns:
            bool: True if conflicts exist, False otherwise
        """
        directions = set()
        for signal in signals:
            if "direction" in signal:
                directions.add(signal["direction"])
                
        # If both long and short signals exist, it's a conflict
        conflicting_directions = {
            (POSITION_DIRECTION["LONG"], POSITION_DIRECTION["SHORT"]),
            (POSITION_DIRECTION["SHORT"], POSITION_DIRECTION["LONG"])
        }
        
        for d1, d2 in conflicting_directions:
            if d1 in directions and d2 in directions:
                return True
                
        return False
        
    def _simple_voting(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Perform simple majority voting on signals.
        
        Args:
            signals: List of signals
            
        Returns:
            Decision dictionary or None if no decision
        """
        # Count signals by direction
        direction_counts = {}
        for signal in signals:
            direction = signal.get("direction")
            if not direction:
                continue
                
            if direction not in direction_counts:
                direction_counts[direction] = 0
                
            direction_counts[direction] += 1
            
        if not direction_counts:
            return None
            
        # Find majority direction
        max_count = 0
        majority_direction = None
        
        for direction, count in direction_counts.items():
            if count > max_count:
                max_count = count
                majority_direction = direction
                
        # Calculate consensus level
        total_votes = sum(direction_counts.values())
        consensus = max_count / total_votes if total_votes > 0 else 0
        
        # Check if consensus meets threshold
        if consensus < self.min_consensus:
            return None
            
        # Calculate average confidence
        total_confidence = 0
        confidence_count = 0
        
        for signal in signals:
            if signal.get("direction") == majority_direction and "confidence" in signal:
                total_confidence += signal["confidence"]
                confidence_count += 1
                
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
        
        # Return decision
        return {
            "direction": majority_direction,
            "confidence": avg_confidence,
            "strength": consensus,
            "consensus": consensus,
            "signal_count": max_count,
            "total_signals": total_votes
        }
        
    def _weighted_voting(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Perform weighted voting based on brain and timeframe weights.
        
        Args:
            signals: List of signals
            
        Returns:
            Decision dictionary or None if no decision
        """
        # Weighted votes by direction
        direction_votes = {}
        signal_counts = {}
        
        for signal in signals:
            direction = signal.get("direction")
            brain = signal.get("brain", "unknown")
            timeframe = signal.get("timeframe", "unknown")
            
            if not direction:
                continue
                
            # Get weights
            brain_weight = self.brain_weights.get(brain, 0.1)  # Default weight
            timeframe_weight = self.timeframe_weights.get(timeframe, 0.1)  # Default weight
            
            # Combined weight (brain weight has more influence)
            weight = 0.7 * brain_weight + 0.3 * timeframe_weight
            
            # Apply confidence as a factor
            if "confidence" in signal:
                weight *= signal["confidence"]
                
            # Add to direction votes
            if direction not in direction_votes:
                direction_votes[direction] = 0
                signal_counts[direction] = 0
                
            direction_votes[direction] += weight
            signal_counts[direction] += 1
            
        if not direction_votes:
            return None
            
        # Find direction with highest vote
        max_votes = 0
        selected_direction = None
        
        for direction, votes in direction_votes.items():
            if votes > max_votes:
                max_votes = votes
                selected_direction = direction
                
        if not selected_direction:
            return None
            
        # Calculate total votes and consensus
        total_votes = sum(direction_votes.values())
        consensus = direction_votes[selected_direction] / total_votes if total_votes > 0 else 0
        
        # Check if consensus meets threshold
        if consensus < self.min_consensus:
            return None
            
        # Calculate average confidence for selected direction
        total_confidence = 0
        confidence_count = 0
        
        for signal in signals:
            if signal.get("direction") == selected_direction and "confidence" in signal:
                total_confidence += signal["confidence"]
                confidence_count += 1
                
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
        
        # If confidence is too low, reject the decision
        if avg_confidence < self.min_confidence:
            return None
            
        # Calculate strength score (combination of consensus and confidence)
        strength = 0.6 * consensus + 0.4 * avg_confidence
        
        # Return decision
        return {
            "direction": selected_direction,
            "confidence": avg_confidence,
            "strength": strength,
            "consensus": consensus,
            "signal_count": signal_counts[selected_direction],
            "total_signals": sum(signal_counts.values())
        }
        
    def _confidence_voting(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Perform voting weighted primarily by signal confidence.
        
        Args:
            signals: List of signals
            
        Returns:
            Decision dictionary or None if no decision
        """
        # Group signals by direction
        direction_groups = {}
        
        for signal in signals:
            direction = signal.get("direction")
            if not direction:
                continue
                
            if direction not in direction_groups:
                direction_groups[direction] = []
                
            direction_groups[direction].append(signal)
            
        if not direction_groups:
            return None
            
        # Calculate confidence-weighted score for each direction
        direction_scores = {}
        
        for direction, group in direction_groups.items():
            # Calculate weighted score based on confidence
            score = 0
            for signal in group:
                confidence = signal.get("confidence", 0.5)  # Default confidence
                score += confidence
                
            direction_scores[direction] = score
            
        # Find direction with highest score
        max_score = 0
        selected_direction = None
        
        for direction, score in direction_scores.items():
            if score > max_score:
                max_score = score
                selected_direction = direction
                
        if not selected_direction:
            return None
            
        # Get signals for selected direction
        selected_signals = direction_groups[selected_direction]
        
        # Calculate average confidence
        total_confidence = sum(signal.get("confidence", 0.5) for signal in selected_signals)
        avg_confidence = total_confidence / len(selected_signals) if selected_signals else 0
        
        # Calculate consensus level
        total_score = sum(direction_scores.values())
        consensus = direction_scores[selected_direction] / total_score if total_score > 0 else 0
        
        # If confidence is too low, reject the decision
        if avg_confidence < self.min_confidence:
            return None
            
        # If consensus is too low, reject the decision
        if consensus < self.min_consensus:
            return None
            
        # Calculate strength score
        strength = 0.3 * consensus + 0.7 * avg_confidence  # More weight on confidence
        
        # Return decision
        return {
            "direction": selected_direction,
            "confidence": avg_confidence,
            "strength": strength,
            "consensus": consensus,
            "signal_count": len(selected_signals),
            "total_signals": len(signals)
        }
        
    def _should_generate_signal(self, decision: Dict[str, Any], exchange: str, symbol: str) -> bool:
        """
        Determine if a final signal should be generated.
        
        Args:
            decision: Decision data
            exchange: Exchange name
            symbol: Symbol name
            
        Returns:
            bool: True if signal should be generated, False otherwise
        """
        # Check strength threshold
        if decision["strength"] < self.strength_threshold:
            self.logger.debug(f"Decision strength {decision['strength']} below threshold {self.strength_threshold}")
            return False
            
        # Check active positions for exit signals
        position_key = f"{exchange}:{symbol}"
        
        if decision["direction"] == POSITION_DIRECTION["FLAT"]:
            # Exit signal - only generate if we have an active position
            if position_key not in self.active_positions:
                self.logger.debug(f"Exit signal ignored - no active position for {position_key}")
                return False
                
        elif decision["direction"] in [POSITION_DIRECTION["LONG"], POSITION_DIRECTION["SHORT"]]:
            # Entry signal
            if position_key in self.active_positions:
                active_pos = self.active_positions[position_key]
                active_direction = active_pos.get("direction")
                
                # If signal matches current position, don't generate
                if active_direction == decision["direction"]:
                    self.logger.debug(f"Entry signal ignored - already in {active_direction} position for {position_key}")
                    return False
                    
        return True
        
    def _create_final_signal(self, decision: Dict[str, Any], exchange: str, 
                            symbol: str, timeframe: str, 
                            source_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create final signal from decision.
        
        Args:
            decision: Decision data
            exchange: Exchange name
            symbol: Symbol name
            timeframe: Timeframe
            source_signals: Source signals that led to this decision
            
        Returns:
            Final signal dictionary
        """
        # Determine signal type
        if decision["direction"] == POSITION_DIRECTION["FLAT"]:
            signal_type = SIGNAL_TYPES["EXIT"]
        else:
            signal_type = SIGNAL_TYPES["ENTRY"]
            
        # Generate a unique ID
        signal_id = f"council_{exchange}_{symbol}_{int(time.time())}"
        
        # Create signal
        final_signal = {
            "id": signal_id,
            "timestamp": time.time(),
            "exchange": exchange,
            "symbol": symbol,
            "timeframe": timeframe,
            "type": signal_type,
            "direction": decision["direction"],
            "confidence": decision["confidence"],
            "strength": decision["strength"],
            "consensus": decision["consensus"],
            "signal_count": decision["signal_count"],
            "total_signals": decision["total_signals"],
            "service": "brain_council",
            "source_signals": [signal.get("id") for signal in source_signals if "id" in signal],
            "metadata": {
                "voting_method": self.voting_method,
                "source_brains": list(set(signal.get("brain", "unknown") for signal in source_signals)),
                "council_generated": True
            }
        }
        
        return final_signal
        
    async def _publish_final_signal(self, signal: Dict[str, Any]):
        """
        Publish final signal to execution channel.
        
        Args:
            signal: Final signal dictionary
        """
        try:
            # Log signal
            self.logger.info(
                f"Publishing {signal['type']} signal for {signal['exchange']}:{signal['symbol']} "
                f"direction={signal['direction']} confidence={signal['confidence']:.2f}"
            )
            
            # Increment metrics
            self.metrics.increment("signals.generated")
            self.metrics.increment(f"signals.type.{signal['type']}")
            self.metrics.increment(f"signals.direction.{signal['direction']}")
            
            # Publish to execution channel
            channel = "execution.signals"
            await self.redis_client.publish(channel, signal)
            
            # Store in database if available
            if self.db_client:
                await self._store_signal_in_db(signal)
                
        except Exception as e:
            self.logger.error(f"Error publishing final signal: {str(e)}")
            self.metrics.increment("signals.publish_error")
            
    async def _store_signal_in_db(self, signal: Dict[str, Any]):
        """
        Store signal in database for tracking and analysis.
        
        Args:
            signal: Signal dictionary
        """
        try:
            # Convert metadata to JSON
            metadata_json = json.dumps(signal.get("metadata", {}))
            source_signals_json = json.dumps(signal.get("source_signals", []))
            
            # Insert into database
            query = """
            INSERT INTO council_signals (
                signal_id, timestamp, exchange, symbol, timeframe, signal_type,
                direction, confidence, strength, consensus, signal_count,
                total_signals, service, source_signals, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            )
            """
            
            await self.db_client.execute(
                query,
                signal["id"],
                signal["timestamp"],
                signal["exchange"],
                signal["symbol"],
                signal["timeframe"],
                signal["type"],
                signal["direction"],
                signal["confidence"],
                signal["strength"],
                signal["consensus"],
                signal["signal_count"],
                signal["total_signals"],
                signal["service"],
                source_signals_json,
                metadata_json
            )
            
        except Exception as e:
            self.logger.error(f"Error storing signal in database: {str(e)}")
            
    async def _weight_adjustment_loop(self, interval: int):
        """
        Periodically adjust brain weights based on performance.
        
        Args:
            interval: Number of signals/trades between adjustments
        """
        self.logger.info(f"Starting weight adjustment loop (interval: {interval} trades)")
        
        signal_count = 0
        
        while self.running:
            try:
                # Check if enough signals have been processed
                current_count = self.metrics.get("signals.generated", 0)
                if current_count - signal_count >= interval:
                    signal_count = current_count
                    
                    # Adjust weights based on performance
                    await self._adjust_weights_by_performance()
                    
                # Sleep before next check
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                self.logger.info("Weight adjustment loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in weight adjustment loop: {str(e)}")
                await asyncio.sleep(30)
                
    async def _adjust_weights_by_performance(self):
        """Adjust brain weights based on recent performance."""
        if not self.db_client:
            return
            
        try:
            # Query recent signal performance
            query = """
            SELECT
                s.source_brain,
                COUNT(*) as total_signals,
                SUM(CASE WHEN r.result = 'success' THEN 1 ELSE 0 END) as successful,
                AVG(r.profit) as avg_profit
            FROM
                council_signals s
            JOIN
                signal_results r ON s.signal_id = r.signal_id
            WHERE
                s.timestamp > $1
            GROUP BY
                s.source_brain
            """
            
            # Get signals from the last 7 days
            cutoff_time = time.time() - (7 * 24 * 60 * 60)
            rows = await self.db_client.fetch_all(query, cutoff_time)
            
            if not rows:
                self.logger.info("No recent performance data available for weight adjustment")
                return
                
            # Calculate performance score for each brain
            brain_scores = {}
            for row in rows:
                brain = row["source_brain"]
                total = row["total_signals"]
                
                if total < 5:  # Require minimum number of signals
                    continue
                    
                success_rate = row["successful"] / total if total > 0 else 0
                avg_profit = row["avg_profit"] or 0
                
                # Calculate score (combination of success rate and profit)
                score = 0.6 * success_rate + 0.4 * (max(0, avg_profit) * 100)  # Scale profit
                brain_scores[brain] = score
                
            if not brain_scores:
                return
                
            # Adjust weights
            min_weight = self.config.get("brain_council.weighting.min_weight", 0.05)
            max_weight = self.config.get("brain_council.weighting.max_weight", 0.5)
            
            # Normalize scores
            total_score = sum(brain_scores.values())
            if total_score <= 0:
                return
                
            # Calculate new weights
            new_weights = {}
            for brain, score in brain_scores.items():
                # Base weight proportional to score
                weight = score / total_score
                
                # Apply min/max constraints
                weight = max(min_weight, min(max_weight, weight))
                new_weights[brain] = weight
                
            # Normalize weights to sum to 1.0
            total_new_weight = sum(new_weights.values())
            if total_new_weight > 0:
                for brain in new_weights:
                    new_weights[brain] /= total_new_weight
                    
            # Update weights
            for brain, weight in new_weights.items():
                old_weight = self.brain_weights.get(brain, 0)
                if abs(weight - old_weight) > 0.05:  # Only log significant changes
                    self.logger.info(f"Adjusted weight for brain {brain}: {old_weight:.2f} -> {weight:.2f}")
                    
                self.brain_weights[brain] = weight
                
            self.logger.info(f"Adjusted weights for {len(new_weights)} brains based on performance")
            
        except Exception as e:
            self.logger.error(f"Error adjusting weights by performance: {str(e)}")


    async def _subscribe_to_ml_model_registrations(self):
        """Subscribe to ML model registration channel."""
        try:
            channel = "ml_models.registration"
            
            async def registration_callback(message):
                if self.running and message:
                    await self._handle_ml_model_registration(message)
                    
            await self.redis_client.subscribe(channel, registration_callback)
            self.logger.info(f"Subscribed to ML model registrations on channel: {channel}")
            
        except Exception as e:
            self.logger.error(f"Error subscribing to ML model registrations: {str(e)}")
    
    async def _handle_ml_model_registration(self, registration: Dict[str, Any]):
        """
        Handle ML model registration message.
        
        Args:
            registration: Model registration data
        """
        try:
            model_name = registration.get("model_name")
            model_type = registration.get("model_type")
            asset_ids = registration.get("asset_ids")
            
            if not model_name or not model_type:
                return
                
            # Register with council manager
            await self.council_manager.register_ml_model(model_name, model_type, asset_ids)
            
            self.logger.info(f"Registered ML model {model_name} ({model_type}) with council system")
                
        except Exception as e:
            self.logger.error(f"Error handling ML model registration: {str(e)}")


async def create_app(config: Dict[str, Any]) -> BrainCouncilService:
    """
    Create and configure a Brain Council Service instance.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured BrainCouncilService instance
    """
    from common.redis_client import RedisClient
    from common.db_client import get_db_client
    
    # Initialize Redis client
    redis_client = RedisClient(
        host=config.get("redis.host", "localhost"),
        port=config.get("redis.port", 6379),
        db=config.get("redis.db", 0),
        password=config.get("redis.password", None)
    )
    
    # Initialize database client
    db_client = await get_db_client(
        db_type=config.get("database.type", "postgresql"),
        host=config.get("database.host", "localhost"),
        port=config.get("database.port", 5432),
        username=config.get("database.username", "postgres"),
        password=config.get("database.password", ""),
        database=config.get("database.database", "quantumspectre")
    )
    
    # Create and return service
    return BrainCouncilService(
        config=config,
        redis_client=redis_client,
        db_client=db_client
    )
