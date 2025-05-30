

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Brain Council - Timeframe Council

This module implements a timeframe-specific brain council that specializes in
coordinating strategy brains operating on a specific timeframe.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import json
import numpy as np

from common.logger import get_logger
from common.utils import TimeFrame, generate_id
from common.metrics import MetricsCollector
from common.db_client import DatabaseClient
from common.redis_client import RedisClient

from brain_council.base_council import BaseCouncil

class TimeframeCouncil(BaseCouncil):
    """
    TimeframeCouncil specializes in coordinating strategy brains that operate
    on a specific timeframe. It applies timeframe-specific logic to the signal
    aggregation process.
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
        Initialize a timeframe-specific brain council.
        
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
        super().__init__(
            council_id=council_id,
            council_name=council_name,
            asset_id=asset_id,
            platform=platform,
            timeframe=timeframe,
            brain_ids=brain_ids,
            config=config,
            db_client=db_client,
            redis_client=redis_client,
            metrics_collector=metrics_collector
        )
        
        # Timeframe-specific confidence adjustments
        self.timeframe_confidence_multiplier = self._get_timeframe_confidence_multiplier()
        
        # Trend alignment boost - higher timeframes should align with trend
        self.trend_alignment_boost = self.config.get('trend_alignment_boost', 0.15)
        
        # Signal validity windows - how long signals remain valid based on timeframe
        self.signal_validity_window = self._get_signal_validity_window()
        
        # Signal frequency control - how often signals can be generated
        self.min_signal_interval = self._get_min_signal_interval()
        self.last_signal_timestamp = None
        
        # For multi-timeframe context awareness
        self.higher_timeframe_context = None
        self.lower_timeframe_context = None
        
        # Noise filtering thresholds adjusted for timeframe
        self.noise_threshold = self._get_noise_threshold()
        
        # For pattern completion monitoring
        self.pattern_completion_states = {}
        
        # For trade management tracking
        self.active_trade_data = None
        
        self.logger.info(
            f"Initialized TimeframeCouncil for {self.asset_id} on {self.platform} "
            f"with timeframe {self.timeframe.name}"
        )
    
    def _get_timeframe_confidence_multiplier(self) -> float:
        """
        Get confidence multiplier based on timeframe.
        
        Different timeframes may have different reliability characteristics.
        For instance, signals on higher timeframes might be more reliable.
        
        Returns:
            float: Confidence multiplier for this timeframe
        """
        # Default multipliers by timeframe
        default_multipliers = {
            TimeFrame.M1: 0.7,    # 1-minute (more noise)
            TimeFrame.M5: 0.8,    # 5-minute
            TimeFrame.M15: 0.9,   # 15-minute
            TimeFrame.M30: 0.95,  # 30-minute
            TimeFrame.H1: 1.0,    # 1-hour (reference)
            TimeFrame.H4: 1.05,   # 4-hour
            TimeFrame.D1: 1.1,    # 1-day
            TimeFrame.W1: 1.15,   # 1-week
            TimeFrame.MN1: 1.2    # 1-month (less noise)
        }
        
        # Get configured value or use default
        multiplier = self.config.get(
            'timeframe_confidence_multiplier', 
            default_multipliers.get(self.timeframe, 1.0)
        )
        
        return multiplier
    
    def _get_signal_validity_window(self) -> timedelta:
        """
        Get how long signals remain valid based on timeframe.
        
        Returns:
            timedelta: Time period for which a signal remains valid
        """
        # Default validity windows by timeframe
        default_windows = {
            TimeFrame.M1: timedelta(minutes=3),      # 1-minute
            TimeFrame.M5: timedelta(minutes=15),     # 5-minute
            TimeFrame.M15: timedelta(minutes=45),    # 15-minute
            TimeFrame.M30: timedelta(hours=1.5),     # 30-minute
            TimeFrame.H1: timedelta(hours=3),        # 1-hour
            TimeFrame.H4: timedelta(hours=12),       # 4-hour
            TimeFrame.D1: timedelta(days=3),         # 1-day
            TimeFrame.W1: timedelta(weeks=2),        # 1-week
            TimeFrame.MN1: timedelta(days=60)        # 1-month
        }
        
        # Get configured value or use default
        window_minutes = self.config.get('signal_validity_window_minutes', None)
        if window_minutes is not None:
            return timedelta(minutes=window_minutes)
        
        return default_windows.get(self.timeframe, timedelta(hours=3))
    
    def _get_min_signal_interval(self) -> timedelta:
        """
        Get minimum interval between signals based on timeframe.
        
        Returns:
            timedelta: Minimum time between signals
        """
        # Default minimum intervals by timeframe
        default_intervals = {
            TimeFrame.M1: timedelta(seconds=30),     # 1-minute
            TimeFrame.M5: timedelta(minutes=2),      # 5-minute
            TimeFrame.M15: timedelta(minutes=5),     # 15-minute
            TimeFrame.M30: timedelta(minutes=10),    # 30-minute
            TimeFrame.H1: timedelta(minutes=20),     # 1-hour
            TimeFrame.H4: timedelta(hours=1),        # 4-hour
            TimeFrame.D1: timedelta(hours=6),        # 1-day
            TimeFrame.W1: timedelta(days=1),         # 1-week
            TimeFrame.MN1: timedelta(days=7)         # 1-month
        }
        
        # Get configured value or use default
        interval_minutes = self.config.get('min_signal_interval_minutes', None)
        if interval_minutes is not None:
            return timedelta(minutes=interval_minutes)
        
        return default_intervals.get(self.timeframe, timedelta(minutes=10))
    
    def _get_noise_threshold(self) -> float:
        """
        Get noise filtering threshold adjusted for timeframe.
        
        Returns:
            float: Noise threshold value
        """
        # Default thresholds by timeframe
        default_thresholds = {
            TimeFrame.M1: 0.4,    # 1-minute (more noise)
            TimeFrame.M5: 0.35,   # 5-minute
            TimeFrame.M15: 0.3,   # 15-minute
            TimeFrame.M30: 0.25,  # 30-minute
            TimeFrame.H1: 0.2,    # 1-hour
            TimeFrame.H4: 0.18,   # 4-hour
            TimeFrame.D1: 0.15,   # 1-day
            TimeFrame.W1: 0.12,   # 1-week
            TimeFrame.MN1: 0.1    # 1-month (less noise)
        }
        
        # Get configured value or use default
        threshold = self.config.get(
            'noise_threshold', 
            default_thresholds.get(self.timeframe, 0.2)
        )
        
        return threshold
    
    async def initialize(self) -> bool:
        """
        Initialize the timeframe council with additional context.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        # First, initialize base council
        result = await super().initialize()
        
        if not result:
            return False
            
        try:
            # Load multi-timeframe context
            await self._load_multi_timeframe_context()
            
            # Load active trade data if any
            await self._load_active_trade_data()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in TimeframeCouncil initialization: {str(e)}")
            return False
    
    async def _load_multi_timeframe_context(self) -> None:
        """
        Load context from higher and lower timeframes for better decision making.
        """
        if not self.redis_client:
            self.logger.warning("No Redis client available, skipping multi-timeframe context loading")
            return
            
        try:
            # Determine higher and lower timeframes
            higher_tf = self._get_higher_timeframe()
            lower_tf = self._get_lower_timeframe()
            
            # Load higher timeframe context if available
            if higher_tf:
                higher_tf_key = f"market:context:{self.platform}:{self.asset_id}:{higher_tf.name}"
                higher_tf_data = await self.redis_client.get(higher_tf_key)
                
                if higher_tf_data:
                    self.higher_timeframe_context = json.loads(higher_tf_data)
                    self.logger.debug(f"Loaded higher timeframe ({higher_tf.name}) context")
            
            # Load lower timeframe context if available
            if lower_tf:
                lower_tf_key = f"market:context:{self.platform}:{self.asset_id}:{lower_tf.name}"
                lower_tf_data = await self.redis_client.get(lower_tf_key)
                
                if lower_tf_data:
                    self.lower_timeframe_context = json.loads(lower_tf_data)
                    self.logger.debug(f"Loaded lower timeframe ({lower_tf.name}) context")
                    
        except Exception as e:
            self.logger.error(f"Error loading multi-timeframe context: {str(e)}")
            # Not critical, can continue without context
    
    def _get_higher_timeframe(self) -> Optional[TimeFrame]:
        """
        Get the next higher timeframe relative to the current one.
        
        Returns:
            Optional[TimeFrame]: Higher timeframe or None if already at highest
        """
        timeframe_order = [
            TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.M30,
            TimeFrame.H1, TimeFrame.H4, TimeFrame.D1, TimeFrame.W1, TimeFrame.MN1
        ]
        
        try:
            current_index = timeframe_order.index(self.timeframe)
            if current_index < len(timeframe_order) - 1:
                return timeframe_order[current_index + 1]
        except ValueError:
            self.logger.warning(f"Unknown timeframe {self.timeframe} in hierarchy")
            
        return None
    
    def _get_lower_timeframe(self) -> Optional[TimeFrame]:
        """
        Get the next lower timeframe relative to the current one.
        
        Returns:
            Optional[TimeFrame]: Lower timeframe or None if already at lowest
        """
        timeframe_order = [
            TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.M30,
            TimeFrame.H1, TimeFrame.H4, TimeFrame.D1, TimeFrame.W1, TimeFrame.MN1
        ]
        
        try:
            current_index = timeframe_order.index(self.timeframe)
            if current_index > 0:
                return timeframe_order[current_index - 1]
        except ValueError:
            self.logger.warning(f"Unknown timeframe {self.timeframe} in hierarchy")
            
        return None
    
    async def _load_active_trade_data(self) -> None:
        """
        Load data about any active trades for this asset/timeframe.
        """
        if not self.redis_client:
            return
            
        try:
            # Check if there's an active trade
            active_trade_key = f"trade:active:{self.platform}:{self.asset_id}:{self.timeframe.name}"
            active_trade_data = await self.redis_client.get(active_trade_key)
            
            if active_trade_data:
                self.active_trade_data = json.loads(active_trade_data)
                self.logger.info(
                    f"Loaded active trade data: direction={self.active_trade_data.get('direction')}, "
                    f"entry_price={self.active_trade_data.get('entry_price')}"
                )
                
        except Exception as e:
            self.logger.error(f"Error loading active trade data: {str(e)}")
    
    async def process_signals(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process incoming signals from brains and generate a consolidated signal.
        
        Applies timeframe-specific logic to the signal processing.
        
        Args:
            signals: Dictionary mapping brain IDs to their signal dictionaries
            
        Returns:
            Dict[str, Any]: Consolidated signal with confidence and metadata
        """
        if not signals:
            self.logger.warning("No signals provided to process")
            return {
                "signal": "NEUTRAL",
                "direction": 0,
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "timeframe": self.timeframe.name,
                "metadata": {
                    "reason": "No signals available",
                    "brain_count": 0
                }
            }
        
        # Check signal frequency control
        current_time = datetime.now()
        if (self.last_signal_timestamp and 
            current_time - self.last_signal_timestamp < self.min_signal_interval):
            self.logger.debug(
                f"Signal frequency control: Last signal was {current_time - self.last_signal_timestamp} ago, "
                f"minimum interval is {self.min_signal_interval}"
            )
            
            # Too soon for a new signal, return neutral
            return {
                "signal": "NEUTRAL",
                "direction": 0,
                "confidence": 0.0,
                "timestamp": current_time.isoformat(),
                "timeframe": self.timeframe.name,
                "metadata": {
                    "reason": "Signal frequency control - minimum interval not reached",
                    "brain_count": len(signals),
                    "min_interval": self.min_signal_interval.total_seconds(),
                    "time_since_last": (current_time - self.last_signal_timestamp).total_seconds() if self.last_signal_timestamp else None
                }
            }
        
        # Apply noise filtering
        filtered_signals = self._filter_noise(signals)
        
        if not filtered_signals:
            self.logger.debug("All signals filtered out as noise")
            return {
                "signal": "NEUTRAL",
                "direction": 0,
                "confidence": 0.0,
                "timestamp": current_time.isoformat(),
                "timeframe": self.timeframe.name,
                "metadata": {
                    "reason": "All signals filtered out as noise",
                    "brain_count": len(signals),
                    "noise_threshold": self.noise_threshold
                }
            }
        
        # Apply multi-timeframe context
        adjusted_signals = await self._apply_timeframe_context(filtered_signals)
        
        # Generate weighted vote
        consolidated_signal = await self.get_weighted_vote(adjusted_signals)
        
        # Apply timeframe-specific confidence adjustment
        confidence = consolidated_signal.get("confidence", 0.0)
        adjusted_confidence = min(confidence * self.timeframe_confidence_multiplier, 1.0)
        consolidated_signal["confidence"] = adjusted_confidence
        
        # Add timeframe to metadata
        consolidated_signal["timeframe"] = self.timeframe.name
        consolidated_signal["metadata"]["timeframe"] = self.timeframe.name
        consolidated_signal["metadata"]["confidence_multiplier"] = self.timeframe_confidence_multiplier
        consolidated_signal["metadata"]["original_confidence"] = confidence
        
        # Apply active trade context if applicable
        if self.active_trade_data:
            consolidated_signal = self._apply_active_trade_context(consolidated_signal)
        
        # Update last signal timestamp
        self.last_signal_timestamp = current_time
        
        # Record the signal
        await self.record_signal(consolidated_signal)
        
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_gauge(
                f"council.{self.council_id}.signal.confidence", 
                consolidated_signal.get("confidence", 0.0)
            )
            
            direction = consolidated_signal.get("direction", 0)
            if direction > 0:
                signal_type = "buy"
            elif direction < 0:
                signal_type = "sell"
            else:
                signal_type = "neutral"
                
            self.metrics_collector.record_counter(
                f"council.{self.council_id}.signal.{signal_type}", 
                1
            )
        
        return consolidated_signal
    
    def _filter_noise(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Filter out signals that are likely noise.
        
        Args:
            signals: Dictionary mapping brain IDs to their signal dictionaries
            
        Returns:
            Dict[str, Dict[str, Any]]: Filtered signals
        """
        filtered_signals = {}
        
        for brain_id, signal_data in signals.items():
            # Get confidence and check against threshold
            confidence = signal_data.get("confidence", 0.0)
            
            if confidence >= self.noise_threshold:
                filtered_signals[brain_id] = signal_data
            else:
                self.logger.debug(
                    f"Filtered out signal from brain {brain_id} with confidence {confidence} "
                    f"(below threshold {self.noise_threshold})"
                )
        
        return filtered_signals
    
    async def _apply_timeframe_context(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Apply multi-timeframe context to adjust signal confidence.
        
        Args:
            signals: Dictionary mapping brain IDs to their signal dictionaries
            
        Returns:
            Dict[str, Dict[str, Any]]: Adjusted signals
        """
        adjusted_signals = {}
        
        # If we don't have multi-timeframe context, return signals unchanged
        if not self.higher_timeframe_context and not self.lower_timeframe_context:
            return signals
        
        for brain_id, signal_data in signals.items():
            # Create a copy of the signal data for adjustment
            adjusted_signal = signal_data.copy()
            
            # Get original direction and confidence
            direction = signal_data.get("direction", 0)
            confidence = signal_data.get("confidence", 0.5)
            
            # Skip neutral signals
            if direction == 0:
                adjusted_signals[brain_id] = adjusted_signal
                continue
            
            # Apply higher timeframe context if available
            if self.higher_timeframe_context:
                higher_tf_trend = self.higher_timeframe_context.get("trend_direction", 0)
                
                # If higher timeframe trend aligns with signal, boost confidence
                if (higher_tf_trend > 0 and direction > 0) or (higher_tf_trend < 0 and direction < 0):
                    adjusted_confidence = min(confidence + self.trend_alignment_boost, 1.0)
                    adjusted_signal["confidence"] = adjusted_confidence
                    adjusted_signal["metadata"] = adjusted_signal.get("metadata", {})
                    adjusted_signal["metadata"]["higher_tf_alignment"] = True
                    adjusted_signal["metadata"]["confidence_boost"] = self.trend_alignment_boost
                    
                    self.logger.debug(
                        f"Boosted confidence for brain {brain_id} from {confidence:.2f} to {adjusted_confidence:.2f} "
                        f"due to higher timeframe alignment"
                    )
                # If higher timeframe trend contradicts signal, reduce confidence
                elif (higher_tf_trend > 0 and direction < 0) or (higher_tf_trend < 0 and direction > 0):
                    adjusted_confidence = max(confidence - self.trend_alignment_boost, 0.1)
                    adjusted_signal["confidence"] = adjusted_confidence
                    adjusted_signal["metadata"] = adjusted_signal.get("metadata", {})
                    adjusted_signal["metadata"]["higher_tf_alignment"] = False
                    adjusted_signal["metadata"]["confidence_reduction"] = self.trend_alignment_boost
                    
                    self.logger.debug(
                        f"Reduced confidence for brain {brain_id} from {confidence:.2f} to {adjusted_confidence:.2f} "
                        f"due to higher timeframe contradiction"
                    )
            
            # Apply lower timeframe context if available
            # Lower timeframe confirmation can fine-tune entry timing
            if self.lower_timeframe_context:
                lower_tf_momentum = self.lower_timeframe_context.get("momentum", 0)
                
                # If lower timeframe momentum is strong in the same direction, slight boost
                if (lower_tf_momentum > 0.5 and direction > 0) or (lower_tf_momentum < -0.5 and direction < 0):
                    current_confidence = adjusted_signal.get("confidence", confidence)
                    minor_boost = 0.05  # Smaller boost for lower timeframe
                    adjusted_confidence = min(current_confidence + minor_boost, 1.0)
                    adjusted_signal["confidence"] = adjusted_confidence
                    adjusted_signal["metadata"] = adjusted_signal.get("metadata", {})
                    adjusted_signal["metadata"]["lower_tf_momentum"] = True
                    
                    self.logger.debug(
                        f"Minor confidence boost for brain {brain_id} to {adjusted_confidence:.2f} "
                        f"due to lower timeframe momentum"
                    )
            
            adjusted_signals[brain_id] = adjusted_signal
        
        return adjusted_signals
    
    def _apply_active_trade_context(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust signal based on any active trade context.
        
        Args:
            signal: The consolidated signal
            
        Returns:
            Dict[str, Any]: Adjusted signal
        """
        # Create a copy of the signal for adjustment
        adjusted_signal = signal.copy()
        
        # Get active trade direction
        active_direction = self.active_trade_data.get("direction", 0)
        
        # If no active trade or neutral signal, return unchanged
        if active_direction == 0 or signal.get("direction", 0) == 0:
            return adjusted_signal
        
        # If signal direction matches active trade direction, it might be for adding to position
        # or it might be irrelevant since we're already in position
        if signal.get("direction", 0) == active_direction:
            # Check if signal is specifically for adding to position
            if signal.get("metadata", {}).get("trade_action") == "add_to_position":
                # This is an explicit add-to-position signal, leave it as is
                adjusted_signal["metadata"]["active_trade_context"] = "add_to_position"
            else:
                # Default behavior: If we're already in this direction, reduce signal confidence
                # as it's less relevant (we already have a position)
                confidence = signal.get("confidence", 0.5)
                adjusted_confidence = confidence * 0.7  # Reduce confidence
                adjusted_signal["confidence"] = adjusted_confidence
                adjusted_signal["metadata"]["active_trade_context"] = "already_in_position"
                adjusted_signal["metadata"]["confidence_reduction"] = "position_exists"
                
                self.logger.debug(
                    f"Reduced signal confidence from {confidence:.2f} to {adjusted_confidence:.2f} "
                    f"as we already have an active position in this direction"
                )
        
        # If signal direction is opposite to active trade direction, it might be an exit signal
        elif signal.get("direction", 0) == -active_direction:
            # Treat this as a potential exit signal
            # Check if signal confidence is high enough for exit
            confidence = signal.get("confidence", 0.5)
            
            if confidence > 0.7:  # Higher threshold for exiting positions
                adjusted_signal["metadata"]["trade_action"] = "exit_position"
                adjusted_signal["metadata"]["active_trade_context"] = "exit_signal"
                self.logger.info(
                    f"Strong counter-signal detected with confidence {confidence:.2f}, "
                    f"marking as exit signal"
                )
            else:
                # Not strong enough to exit, but note it in metadata
                adjusted_signal["metadata"]["active_trade_context"] = "potential_exit"
                adjusted_signal["metadata"]["exit_threshold"] = 0.7
                self.logger.debug(
                    f"Counter-signal detected with confidence {confidence:.2f}, "
                    f"but below exit threshold of 0.7"
                )
        
        return adjusted_signal
    
    async def track_pattern_completion(self, pattern_id: str, completion_pct: float, expected_direction: int = 0) -> None:
        """
        Track the completion status of a pattern to generate signals as patterns complete.
        
        Args:
            pattern_id: Unique identifier for the pattern
            completion_pct: Percentage of pattern completion (0-100)
            expected_direction: Expected direction when pattern completes (1 for buy, -1 for sell)
        """
        # Update pattern completion state
        self.pattern_completion_states[pattern_id] = {
            "completion_pct": completion_pct,
            "expected_direction": expected_direction,
            "last_update": datetime.now()
        }
        
        # Clean up old patterns
        current_time = datetime.now()
        old_patterns = [
            pid for pid, data in self.pattern_completion_states.items()
            if current_time - data["last_update"] > timedelta(hours=24)
        ]
        
        for pid in old_patterns:
            del self.pattern_completion_states[pid]
            
        # Record pattern completion in Redis for visualization
        if self.redis_client:
            pattern_key = f"pattern:completion:{self.platform}:{self.asset_id}:{self.timeframe.name}:{pattern_id}"
            
            pattern_data = {
                "pattern_id": pattern_id,
                "completion_pct": completion_pct,
                "expected_direction": expected_direction,
                "timestamp": current_time.isoformat()
            }
            
            await self.redis_client.set(
                pattern_key, 
                json.dumps(pattern_data),
                expire=86400  # 24 hours
            )
    
    async def get_context_for_other_timeframes(self) -> Dict[str, Any]:
        """
        Provide context information for other timeframes.
        
        Returns:
            Dict[str, Any]: Context information
        """
        # Calculate and provide useful context for other timeframes
        context = {
            "timeframe": self.timeframe.name,
            "trend_direction": 0,  # Default neutral
            "momentum": 0,         # Default neutral
            "volatility": 0,       # Default low
            "key_levels": [],      # Support/resistance levels
            "patterns": [],        # Active patterns
            "timestamp": datetime.now().isoformat()
        }
        
        # If we have latest signals, use them to determine trend direction
        if self.signal_history:
            # Look at last 5 signals
            recent_signals = self.signal_history[-5:]
            
            # Calculate average direction weighted by confidence
            weighted_sum = sum(
                signal.get("direction", 0) * signal.get("confidence", 0.5)
                for signal in recent_signals
            )
            
            # Normalize by count
            if recent_signals:
                trend_direction = weighted_sum / len(recent_signals)
                context["trend_direction"] = trend_direction
                
                # Set momentum based on signal strength and consistency
                signal_directions = [signal.get("direction", 0) for signal in recent_signals]
                if all(d > 0 for d in signal_directions if d != 0):
                    context["momentum"] = 0.8  # Strong consistent upward momentum
                elif all(d < 0 for d in signal_directions if d != 0):
                    context["momentum"] = -0.8  # Strong consistent downward momentum
                else:
                    # Mixed signals, calculate net momentum
                    context["momentum"] = sum(signal_directions) / len(signal_directions) if signal_directions else 0
        
        # If we have market context, include volatility
        if self.asset_volatility is not None:
            context["volatility"] = self.asset_volatility
        
        # Include active patterns being tracked
        for pattern_id, data in self.pattern_completion_states.items():
            if data["completion_pct"] > 50:  # Only include patterns more than 50% complete
                context["patterns"].append({
                    "pattern_id": pattern_id,
                    "completion_pct": data["completion_pct"],
                    "expected_direction": data["expected_direction"]
                })
        
        # Add any key levels we're tracking
        # This would come from support/resistance analysis
        # For now, just a placeholder
        
        return context
