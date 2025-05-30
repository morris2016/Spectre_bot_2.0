#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Market Structure Brain - Strategy Brain Module

This module implements a sophisticated market structure analysis strategy that
identifies swing highs/lows, support/resistance, market phases, and overall
structure to generate high-probability trading signals.

The MarketStructureBrain excels at identifying:
- Market phases (accumulation, markup, distribution, markdown)
- Structural shifts and breaks
- Higher highs/higher lows (uptrend) vs. lower highs/lower lows (downtrend)
- Range boundaries and breakouts
- Areas of value vs. areas of rejection
- Smart money manipulation tactics
"""

import asyncio
import json
import logging
import math
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from common.logger import get_logger
from common.utils import circular_buffer, find_peaks, timeit
from config import Config
from data_storage.market_data import MarketDataStorage
from feature_service.features.market_structure import MarketStructureFeatures, SwingPointFeatures, ValueAreaFeatures
from strategy_brains.base_brain import StrategyBrain


class MarketStructureBrain(StrategyBrain):
    """
    Market Structure Brain analyzes the overall market structure to identify significant
    structural pivot points, trend changes, and market phases for generating high-probability
    trading signals.
    """

    BRAIN_TYPE = "MarketStructureBrain"
    VERSION = "2.3.0"
    DEFAULT_PARAMS = {
        # Swing point detection
        "swing_lookback": 5,  # Lookback periods for swing identification
        "swing_strength_threshold": 0.4,  # Minimum swing strength
        "swing_memory_length": 50,  # Number of swings to keep in memory
        # Trend detection
        "trend_smoothing_periods": 20,  # EMA periods for trend smoothing
        "trend_threshold": 0.05,  # Minimum slope for trend detection
        "trend_noise_filter": 0.3,  # Filter for noise reduction
        "adaptive_trend_detection": True,  # Use adaptive period-based on volatility
        # Support/resistance detection
        "sr_significance_threshold": 0.6,  # Minimum significance for S/R levels
        "sr_proximity_ratio": 0.005,  # Price proximity ratio for clusters
        "sr_touch_threshold": 3,  # Minimum touches for valid level
        "sr_recency_weight": 0.8,  # Weight for more recent touches
        # Market phase detection
        "phase_detection_periods": 50,  # Periods to consider for phase detection
        "phase_volatility_threshold": 0.5,  # Threshold for volatility change detection
        "phase_volume_threshold": 0.6,  # Threshold for volume change detection
        # Value area detection
        "value_area_percentage": 0.7,  # Percentage of volume for value area
        "value_area_lookback": 20,  # Periods for value area calculation
        # Signal generation
        "signal_min_strength": 0.6,  # Minimum signal strength
        "signal_confirmation_count": 2,  # Number of confirmations needed
        "pullback_ratio": 0.382,  # Default pullback ratio (Fibonacci)
        "breakout_confirmation_periods": 3,  # Periods to confirm breakout
        # Adaptivity and Learning
        "learning_rate": 0.05,  # Rate of parameter adaptation
        "adaptivity_window": 100,  # Periods to consider for adaptivity
        "memory_decay_factor": 0.98,  # Decay factor for historical patterns
        # Advanced Configuration
        "detect_manipulation": True,  # Detect and adapt to manipulation tactics
        "multi_timeframe_confirmation": True,  # Use multi-timeframe confirmation
        "range_vs_trend_detection": True,  # Differentiate between range and trend
        "volatility_adjusted_signals": True,  # Adjust signals based on volatility
    }

    def __init__(self, config: Config, asset_id: str, timeframe: str, **kwargs):
        """
        Initialize the Market Structure Brain strategy.

        Args:
            config: System configuration
            asset_id: Asset identifier (e.g., 'BTCUSDT', 'EURUSD', etc.)
            timeframe: Strategy timeframe (e.g., '1m', '5m', '1h', etc.)
            **kwargs: Additional parameters to override defaults
        """
        super().__init__(config, asset_id, timeframe, **kwargs)

        self.logger = get_logger("MarketStructureBrain")
        self.logger.info(f"Initializing MarketStructureBrain v{self.VERSION} for {asset_id} on {timeframe} timeframe")

        # Initialize parameters with defaults and overrides
        self.params = {**self.DEFAULT_PARAMS, **kwargs}

        # Configure feature providers
        self.market_structure_features = MarketStructureFeatures(config, asset_id)
        self.swing_point_features = SwingPointFeatures(config, asset_id)
        self.value_area_features = ValueAreaFeatures(config, asset_id)

        # Data storage
        self.market_data = MarketDataStorage(config)

        # Memory structures for market structure analysis
        self.candle_history = []
        self.swing_highs = deque(maxlen=self.params["swing_memory_length"])
        self.swing_lows = deque(maxlen=self.params["swing_memory_length"])
        self.support_levels = []
        self.resistance_levels = []
        self.trend_history = deque(maxlen=self.params["adaptivity_window"])
        self.market_phase_history = deque(maxlen=self.params["adaptivity_window"])
        self.value_areas = {}

        # Current state tracking
        self.current_trend = "neutral"  # 'uptrend', 'downtrend', 'neutral'
        self.current_market_phase = "undefined"  # 'accumulation', 'markup', 'distribution', 'markdown'
        self.current_structure = {
            "pattern": None,  # Current pattern (if any)
            "pattern_completion": 0.0,  # Completion percentage of pattern
            "key_levels": {},  # Key structural levels
            "market_context": {},  # Additional context information
        }
        self.current_volatility = 0.0
        self.current_range = {
            "upper": None,  # Upper range boundary
            "lower": None,  # Lower range boundary
            "midpoint": None,  # Range midpoint
            "strength": 0.0,  # Range strength/probability
        }

        # Performance tracking
        self.signal_performance = {
            "tp_hit": 0,  # Take profit hits
            "sl_hit": 0,  # Stop loss hits
            "timeout": 0,  # Signal timeout
            "abandoned": 0,  # Signal abandoned due to invalidation
            "total": 0,  # Total signals generated
        }

        # State variables
        self.initialized = False
        self.last_update_time = None
        self.is_ready = False
        self.data_points_processed = 0

        # Performance metrics
        self.performance_metrics = {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_win_size": 0.0,
            "avg_loss_size": 0.0,
            "expectancy": 0.0,
            "sharpe_ratio": 0.0,
        }

        self.logger.info(f"MarketStructureBrain initialization complete")

    async def initialize(self) -> bool:
        """
        Initialize the strategy brain with historical data and prepare it for use.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info(f"Starting MarketStructureBrain initialization")

        try:
            # Load historical data
            candles = await self.market_data.get_candles(self.asset_id, self.timeframe, limit=max(200, self.params["phase_detection_periods"] * 2))

            if len(candles) < 50:  # Need sufficient data for reliable structure analysis
                self.logger.warning(f"Insufficient historical data for initialization")
                return False

            # Store historical candles
            self.candle_history = candles

            # Process historical data to identify market structure
            await self._process_historical_candles()

            # Load any saved state if available
            await self._load_state()

            self.initialized = True
            self.is_ready = True
            self.last_update_time = datetime.now()

            self.logger.info(f"MarketStructureBrain initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during MarketStructureBrain initialization: {str(e)}")
            return False

    async def process_candle(self, candle: Dict[str, Any]) -> None:
        """
        Process a new candle and update the strategy state.

        Args:
            candle: Candle data dictionary with OHLCV values and timestamp
        """
        if not self.initialized:
            self.logger.warning("Cannot process candle: Brain not initialized")
            return

        self.last_update_time = datetime.now()

        # Add new candle to history
        self.candle_history.append(candle)

        # Limit history length to reasonable size
        max_history = max(200, self.params["phase_detection_periods"] * 2)
        if len(self.candle_history) > max_history:
            self.candle_history = self.candle_history[-max_history:]

        # Update market structure analysis
        await self._update_market_structure(candle)

        # Increment counters
        self.data_points_processed += 1

    async def generate_signal(self) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal based on current market structure analysis.

        Returns:
            Optional[Dict[str, Any]]: Trading signal if conditions are met, None otherwise
        """
        if not self.is_ready or len(self.candle_history) < 30:
            return None

        try:
            # Check if we have a valid market structure
            if not self._has_valid_structure():
                return None

            # Generate different types of signals based on current structure
            breakout_signal = await self._generate_breakout_signal()
            trend_signal = await self._generate_trend_signal()
            reversal_signal = await self._generate_reversal_signal()
            range_signal = await self._generate_range_signal()

            # Select the strongest signal
            signals = [s for s in [breakout_signal, trend_signal, reversal_signal, range_signal] if s]
            if not signals:
                return None

            # Sort by signal strength in descending order
            signals.sort(key=lambda s: s["strength"], reverse=True)

            # Select the strongest signal
            selected_signal = signals[0]

            # Apply additional filters based on market conditions
            if not self._validate_signal_in_context(selected_signal):
                self.logger.debug(f"Signal rejected after context validation")
                return None

            # Calculate more precise entry, target and stop levels
            entry_price = self._get_current_price()
            target_price, stop_price = self._calculate_target_and_stop(
                signal_type=selected_signal["type"],
                direction=selected_signal["direction"],
                entry_price=entry_price,
                context=selected_signal["context"],
            )

            # Final signal construction
            signal = {
                "strategy": self.BRAIN_TYPE,
                "asset_id": self.asset_id,
                "timestamp": datetime.now().timestamp() * 1000,
                "timeframe": self.timeframe,
                "type": selected_signal["type"],
                "direction": selected_signal["direction"],
                "strength": selected_signal["strength"],
                "entry_price": entry_price,
                "target_price": target_price,
                "stop_price": stop_price,
                "context": selected_signal["context"],
                "expiration": (datetime.now() + timedelta(minutes=self._get_expiration_minutes())).timestamp() * 1000,
                "id": f"MS_{self.asset_id}_{int(datetime.now().timestamp())}",
                "metadata": {
                    "version": self.VERSION,
                    "confidence": min(0.99, selected_signal["strength"] * 1.1),  # Cap at 0.99
                    "market_phase": self.current_market_phase,
                    "market_structure": {
                        "trend": self.current_trend,
                        "pattern": self.current_structure["pattern"],
                        "completion": self.current_structure["pattern_completion"],
                    },
                },
            }

            # Log the signal and update performance counters
            self.logger.info(
                f"Generated signal: {signal['id']} Type: {signal['type']} Direction: {'BUY' if signal['direction'] > 0 else 'SELL'}"
                f" Strength: {signal['strength']:.2f}"
            )
            self.signal_performance["total"] += 1

            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return None

    async def update_parameters(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update strategy parameters based on recent performance.

        Args:
            performance_metrics: Dictionary with performance metrics
        """
        self.logger.info(f"Updating MarketStructureBrain parameters based on performance metrics")

        # Store performance metrics
        self.performance_metrics = performance_metrics

        # Skip if we don't have enough data yet
        if performance_metrics.get("total_trades", 0) < 10:
            self.logger.info("Not enough trades for parameter optimization")
            return

        # Adaptive adjustments based on performance
        win_rate = performance_metrics.get("win_rate", 0)
        profit_factor = performance_metrics.get("profit_factor", 0)

        # Only update if we have meaningful metrics
        if win_rate <= 0 or profit_factor <= 0:
            return

        # Calculate adjustment direction
        perf_score = win_rate * 0.4 + min(3.0, profit_factor) / 3.0 * 0.6
        adjustment = 0

        # Determine adjustment direction
        if perf_score < 0.4:  # Poor performance - more conservative
            adjustment = -1
        elif perf_score > 0.7:  # Good performance - more aggressive
            adjustment = 1

        # Apply the adjustments with learning rate
        if adjustment != 0:
            lr = self.params["learning_rate"]

            # Adjust swing detection parameters
            self.params["swing_strength_threshold"] *= 1 - adjustment * lr * 0.1
            self.params["swing_strength_threshold"] = max(0.2, min(0.7, self.params["swing_strength_threshold"]))

            # Adjust signal generation parameters
            self.params["signal_min_strength"] *= 1 - adjustment * lr * 0.05
            self.params["signal_min_strength"] = max(0.4, min(0.8, self.params["signal_min_strength"]))

            # Adjust confirmation parameters
            new_confirm = self.params["signal_confirmation_count"]
            if adjustment > 0 and win_rate > 0.7:
                # High win rate - can reduce confirmations
                new_confirm = max(1, self.params["signal_confirmation_count"] - 1)
            elif adjustment < 0 and win_rate < 0.5:
                # Low win rate - need more confirmations
                new_confirm = min(3, self.params["signal_confirmation_count"] + 1)
            self.params["signal_confirmation_count"] = new_confirm

            self.logger.info(
                f"Parameters adjusted: swing_threshold={self.params['swing_strength_threshold']:.2f}, "
                + f"signal_min_strength={self.params['signal_min_strength']:.2f}, "
                + f"confirmations={self.params['signal_confirmation_count']}"
            )

    async def update_signal(self, signal_id: str, status: str, result: Dict[str, Any] = None) -> None:
        """
        Update signal status and use the feedback for future improvement.

        Args:
            signal_id: ID of the signal to update
            status: New status ('filled', 'tp_hit', 'sl_hit', 'expired', 'cancelled')
            result: Additional result information
        """
        self.logger.info(f"Updating signal {signal_id} with status: {status}")

        # Track signal outcomes for performance measurement
        if status == "tp_hit":
            self.signal_performance["tp_hit"] += 1
        elif status == "sl_hit":
            self.signal_performance["sl_hit"] += 1
        elif status == "expired":
            self.signal_performance["timeout"] += 1
        elif status == "cancelled":
            self.signal_performance["abandoned"] += 1

        # Use feedback to adjust strategy parameters
        if result and "execution_metrics" in result:
            metrics = result["execution_metrics"]

            # Extract signal type from ID if available
            signal_type = None
            if result.get("signal_data") and "type" in result["signal_data"]:
                signal_type = result["signal_data"]["type"]

            # Adjust parameters based on outcome
            if status == "tp_hit":
                # Successful signal - strengthen this type of pattern
                if signal_type == "breakout":
                    self.params["breakout_confirmation_periods"] = max(1, self.params["breakout_confirmation_periods"] - 0.2)
                elif signal_type == "reversal":
                    self.params["swing_strength_threshold"] = max(0.2, self.params["swing_strength_threshold"] * 0.95)

            elif status == "sl_hit":
                # Failed signal - be more conservative with this type
                if signal_type == "breakout":
                    self.params["breakout_confirmation_periods"] = min(5, self.params["breakout_confirmation_periods"] + 0.3)
                elif signal_type == "reversal":
                    self.params["swing_strength_threshold"] = min(0.7, self.params["swing_strength_threshold"] * 1.05)

    async def save_state(self) -> Dict[str, Any]:
        """
        Save the current state of the strategy brain for persistence.

        Returns:
            Dict[str, Any]: State dictionary for persistence
        """
        state = {
            "version": self.VERSION,
            "params": self.params,
            "performance": self.signal_performance,
            "metrics": self.performance_metrics,
            "current_trend": self.current_trend,
            "current_market_phase": self.current_market_phase,
            "current_structure": self.current_structure,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "data_points_processed": self.data_points_processed,
            "timestamp": datetime.now().timestamp(),
        }

        return state

    async def load_state(self, state: Dict[str, Any]) -> bool:
        """
        Load a previously saved state.

        Args:
            state: State dictionary

        Returns:
            bool: True if state was loaded successfully, False otherwise
        """
        try:
            # Verify state compatibility
            if "version" not in state or not state["version"].startswith("2."):
                self.logger.warning(f"Incompatible state version: {state.get('version')}")
                return False

            # Load parameters with safeguards
            if "params" in state:
                # Only load compatible parameters
                for key, value in state["params"].items():
                    if key in self.params:
                        self.params[key] = value

            # Load other state elements
            if "performance" in state:
                self.signal_performance = state["performance"]

            if "metrics" in state:
                self.performance_metrics = state["metrics"]

            if "current_trend" in state:
                self.current_trend = state["current_trend"]

            if "current_market_phase" in state:
                self.current_market_phase = state["current_market_phase"]

            if "current_structure" in state:
                self.current_structure = state["current_structure"]

            if "support_levels" in state:
                self.support_levels = state["support_levels"]

            if "resistance_levels" in state:
                self.resistance_levels = state["resistance_levels"]

            if "data_points_processed" in state:
                self.data_points_processed = state["data_points_processed"]

            self.logger.info(f"State loaded successfully from {datetime.fromtimestamp(state.get('timestamp', 0))}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            return False

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the strategy brain.

        Returns:
            Dict[str, Any]: Status information
        """
        win_rate = 0
        if self.signal_performance["total"] > 0:
            win_rate = self.signal_performance["tp_hit"] / self.signal_performance["total"]

        return {
            "brain_type": self.BRAIN_TYPE,
            "version": self.VERSION,
            "asset_id": self.asset_id,
            "timeframe": self.timeframe,
            "is_ready": self.is_ready,
            "initialized": self.initialized,
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "data_points_processed": self.data_points_processed,
            "performance": {
                "total_signals": self.signal_performance["total"],
                "win_rate": win_rate,
                "tp_hit": self.signal_performance["tp_hit"],
                "sl_hit": self.signal_performance["sl_hit"],
                "timeout": self.signal_performance["timeout"],
                "abandoned": self.signal_performance["abandoned"],
            },
            "current_structure": {
                "trend": self.current_trend,
                "market_phase": self.current_market_phase,
                "pattern": self.current_structure["pattern"],
                "pattern_completion": self.current_structure["pattern_completion"],
                "support_levels_count": len(self.support_levels),
                "resistance_levels_count": len(self.resistance_levels),
            },
        }

    async def shutdown(self) -> None:
        """
        Perform any cleanup operations before shutdown.
        """
        self.logger.info(f"MarketStructureBrain shutdown initiated")

        # Save final state metrics
        try:
            state = await self.save_state()
            self.logger.info(f"Final state saved during shutdown")
        except Exception as e:
            self.logger.error(f"Error saving state during shutdown: {str(e)}")

        self.logger.info(f"MarketStructureBrain shutdown complete")

    # ========== Private Methods ==========

    async def _process_historical_candles(self) -> None:
        """
        Process historical candles to establish initial market structure.
        """
        self.logger.info(f"Processing historical candles for initial market structure")

        if not self.candle_history:
            self.logger.warning("No historical candles to process")
            return

        # Identify swing points
        swing_points = await self.swing_point_features.identify_swing_points(
            self.candle_history, lookback=self.params["swing_lookback"], strength_threshold=self.params["swing_strength_threshold"]
        )

        # Separate swing highs and lows
        for point in swing_points:
            if point["type"] == "high":
                self.swing_highs.append(point)
            else:
                self.swing_lows.append(point)

        # Identify support and resistance levels
        self.support_levels = await self.market_structure_features.identify_support_levels(
            self.candle_history,
            swing_points=[p for p in swing_points if p["type"] == "low"],
            significance_threshold=self.params["sr_significance_threshold"],
            proximity_ratio=self.params["sr_proximity_ratio"],
        )

        self.resistance_levels = await self.market_structure_features.identify_resistance_levels(
            self.candle_history,
            swing_points=[p for p in swing_points if p["type"] == "high"],
            significance_threshold=self.params["sr_significance_threshold"],
            proximity_ratio=self.params["sr_proximity_ratio"],
        )

        # Determine current trend
        self.current_trend = await self._determine_trend()

        # Detect market phase
        self.current_market_phase = await self._detect_market_phase()

        # Identify current price range
        self.current_range = await self._identify_price_range()

        # Calculate current value areas
        value_areas = await self.value_area_features.calculate_value_areas(
            self.candle_history[-self.params["value_area_lookback"] :], value_area_percentage=self.params["value_area_percentage"]
        )

        self.value_areas = {va["timestamp"]: va for va in value_areas}

        # Identify current market structure patterns
        await self._identify_structure_patterns()

        # Calculate current volatility
        self.current_volatility = self._calculate_volatility(self.candle_history[-20:])

        self.logger.info(f"Historical candle processing complete: trend={self.current_trend}, phase={self.current_market_phase}")

    async def _update_market_structure(self, candle: Dict[str, Any]) -> None:
        """
        Update market structure with new candle data.

        Args:
            candle: New candle data
        """
        # Check for new swing points
        new_swing = await self._check_for_swing_point(candle)

        if new_swing:
            # Update swing points collections
            if new_swing["type"] == "high":
                self.swing_highs.append(new_swing)
            else:
                self.swing_lows.append(new_swing)

            # Update support/resistance if this is a significant swing
            if new_swing["strength"] >= self.params["sr_significance_threshold"]:
                await self._update_support_resistance(new_swing)

        # Update current trend periodically
        if self.data_points_processed % 5 == 0:
            self.current_trend = await self._determine_trend()
            self.trend_history.append({"timestamp": candle["timestamp"], "trend": self.current_trend})

        # Update market phase periodically
        if self.data_points_processed % 10 == 0:
            self.current_market_phase = await self._detect_market_phase()
            self.market_phase_history.append({"timestamp": candle["timestamp"], "phase": self.current_market_phase})

        # Update price range
        self.current_range = await self._identify_price_range()

        # Update value areas periodically
        if self.data_points_processed % 5 == 0:
            value_areas = await self.value_area_features.calculate_value_areas(
                self.candle_history[-self.params["value_area_lookback"] :], value_area_percentage=self.params["value_area_percentage"]
            )

            for va in value_areas:
                self.value_areas[va["timestamp"]] = va

            # Clean up old value areas
            now = candle["timestamp"]
            old_threshold = now - (self.params["value_area_lookback"] * 2 * self._get_interval_ms())

            self.value_areas = {ts: va for ts, va in self.value_areas.items() if ts > old_threshold}

        # Update structure patterns periodically
        if self.data_points_processed % 5 == 0:
            await self._identify_structure_patterns()

        # Update volatility
        self.current_volatility = self._calculate_volatility(self.candle_history[-20:])

    async def _check_for_swing_point(self, candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if the new candle forms a swing point.

        Args:
            candle: New candle data

        Returns:
            Optional[Dict[str, Any]]: Swing point data or None
        """
        if len(self.candle_history) < self.params["swing_lookback"] * 2 + 1:
            return None

        # Get recent candles including the new one
        recent_candles = self.candle_history[-(self.params["swing_lookback"] * 2 + 1) :]

        # Get the middle candle (potential swing point)
        middle_idx = self.params["swing_lookback"]
        middle_candle = recent_candles[middle_idx]

        # Check if this forms a swing high
        is_swing_high = True
        for i in range(1, self.params["swing_lookback"] + 1):
            # Check candles before
            if middle_candle["high"] <= recent_candles[middle_idx - i]["high"]:
                is_swing_high = False
                break

            # Check candles after
            if middle_candle["high"] <= recent_candles[middle_idx + i]["high"]:
                is_swing_high = False
                break

        if is_swing_high:
            # Calculate swing strength
            left_candles = recent_candles[:middle_idx]
            right_candles = recent_candles[middle_idx + 1 :]

            left_max_high = max(c["high"] for c in left_candles)
            right_max_high = max(c["high"] for c in right_candles)

            # Normalize strength between 0-1
            strength = min(1.0, (middle_candle["high"] - max(left_max_high, right_max_high)) / middle_candle["high"] * 20)

            if strength >= self.params["swing_strength_threshold"]:
                return {
                    "timestamp": middle_candle["timestamp"],
                    "price": middle_candle["high"],
                    "type": "high",
                    "strength": strength,
                    "candle_idx": len(self.candle_history) - self.params["swing_lookback"] - 1,
                }

        # Check if this forms a swing low
        is_swing_low = True
        for i in range(1, self.params["swing_lookback"] + 1):
            # Check candles before
            if middle_candle["low"] >= recent_candles[middle_idx - i]["low"]:
                is_swing_low = False
                break

            # Check candles after
            if middle_candle["low"] >= recent_candles[middle_idx + i]["low"]:
                is_swing_low = False
                break

        if is_swing_low:
            # Calculate swing strength
            left_candles = recent_candles[:middle_idx]
            right_candles = recent_candles[middle_idx + 1 :]

            left_min_low = min(c["low"] for c in left_candles)
            right_min_low = min(c["low"] for c in right_candles)

            # Normalize strength between 0-1
            strength = min(1.0, (min(left_min_low, right_min_low) - middle_candle["low"]) / middle_candle["low"] * 20)

            if strength >= self.params["swing_strength_threshold"]:
                return {
                    "timestamp": middle_candle["timestamp"],
                    "price": middle_candle["low"],
                    "type": "low",
                    "strength": strength,
                    "candle_idx": len(self.candle_history) - self.params["swing_lookback"] - 1,
                }

        return None

    async def _update_support_resistance(self, swing_point: Dict[str, Any]) -> None:
        """
        Update support and resistance levels based on new swing point.

        Args:
            swing_point: New swing point data
        """
        price = swing_point["price"]
        swing_type = swing_point["type"]
        strength = swing_point["strength"]
        timestamp = swing_point["timestamp"]

        # Check if this swing point forms/reinforces a support level
        if swing_type == "low":
            # Check if we have a similar level already
            found = False
            for i, level in enumerate(self.support_levels):
                # If price is within proximity ratio, update the level
                if abs(level["price"] - price) / price < self.params["sr_proximity_ratio"]:
                    # Update level with weighted average
                    weight = self.params["sr_recency_weight"]
                    new_price = (level["price"] * level["strength"] * (1 - weight) + price * strength * weight) / (
                        level["strength"] * (1 - weight) + strength * weight
                    )

                    # Update strength and touch count
                    new_strength = level["strength"] * self.params["memory_decay_factor"] + strength * (1 - self.params["memory_decay_factor"])

                    self.support_levels[i] = {
                        "price": new_price,
                        "strength": new_strength,
                        "touch_count": level["touch_count"] + 1,
                        "last_touch": timestamp,
                        "created": level["created"],
                    }
                    found = True
                    break

            # If not found, add new level
            if not found:
                self.support_levels.append({"price": price, "strength": strength, "touch_count": 1, "last_touch": timestamp, "created": timestamp})

        # Check if this swing point forms/reinforces a resistance level
        elif swing_type == "high":
            # Check if we have a similar level already
            found = False
            for i, level in enumerate(self.resistance_levels):
                # If price is within proximity ratio, update the level
                if abs(level["price"] - price) / price < self.params["sr_proximity_ratio"]:
                    # Update level with weighted average
                    weight = self.params["sr_recency_weight"]
                    new_price = (level["price"] * level["strength"] * (1 - weight) + price * strength * weight) / (
                        level["strength"] * (1 - weight) + strength * weight
                    )

                    # Update strength and touch count
                    new_strength = level["strength"] * self.params["memory_decay_factor"] + strength * (1 - self.params["memory_decay_factor"])

                    self.resistance_levels[i] = {
                        "price": new_price,
                        "strength": new_strength,
                        "touch_count": level["touch_count"] + 1,
                        "last_touch": timestamp,
                        "created": level["created"],
                    }
                    found = True
                    break

            # If not found, add new level
            if not found:
                self.resistance_levels.append({"price": price, "strength": strength, "touch_count": 1, "last_touch": timestamp, "created": timestamp})

        # Clean up and sort levels
        await self._cleanup_sr_levels()

    async def _cleanup_sr_levels(self) -> None:
        """
        Clean up support and resistance levels by removing weak/invalid levels.
        """
        # Get current time
        now = datetime.now().timestamp() * 1000 if not self.candle_history else self.candle_history[-1]["timestamp"]

        # Remove old and weak support levels
        max_age_ms = self._get_interval_ms() * 100  # Adjust based on needs

        self.support_levels = [
            level
            for level in self.support_levels
            if (
                level["touch_count"] >= self.params["sr_touch_threshold"]
                or (now - level["last_touch"] < max_age_ms and level["strength"] >= self.params["sr_significance_threshold"])
            )
        ]

        # Remove old and weak resistance levels
        self.resistance_levels = [
            level
            for level in self.resistance_levels
            if (
                level["touch_count"] >= self.params["sr_touch_threshold"]
                or (now - level["last_touch"] < max_age_ms and level["strength"] >= self.params["sr_significance_threshold"])
            )
        ]

        # Sort levels by price
        self.support_levels.sort(key=lambda x: x["price"])
        self.resistance_levels.sort(key=lambda x: x["price"])

        # Remove overlapping levels (keep stronger one)
        if len(self.support_levels) > 1:
            clean_supports = [self.support_levels[0]]
            for level in self.support_levels[1:]:
                last_level = clean_supports[-1]
                if abs(level["price"] - last_level["price"]) / level["price"] < self.params["sr_proximity_ratio"]:
                    # Levels overlap - keep the stronger one
                    if level["strength"] > last_level["strength"]:
                        clean_supports[-1] = level
                else:
                    clean_supports.append(level)
            self.support_levels = clean_supports

        if len(self.resistance_levels) > 1:
            clean_resistances = [self.resistance_levels[0]]
            for level in self.resistance_levels[1:]:
                last_level = clean_resistances[-1]
                if abs(level["price"] - last_level["price"]) / level["price"] < self.params["sr_proximity_ratio"]:
                    # Levels overlap - keep the stronger one
                    if level["strength"] > last_level["strength"]:
                        clean_resistances[-1] = level
                else:
                    clean_resistances.append(level)
            self.resistance_levels = clean_resistances

    async def _determine_trend(self) -> str:
        """
        Determine the current market trend using multiple methods.

        Returns:
            str: 'uptrend', 'downtrend', or 'neutral'
        """
        if len(self.candle_history) < 20:
            return "neutral"

        # Method 1: Simple moving average direction
        closes = [candle["close"] for candle in self.candle_history[-self.params["trend_smoothing_periods"] :]]

        if len(closes) < 20:
            return "neutral"

        # Short-term and long-term EMAs
        short_period = min(len(closes) // 4, 10)
        long_period = min(len(closes) // 2, 20)

        short_ema = self._calculate_ema(closes, short_period)
        long_ema = self._calculate_ema(closes, long_period)

        # Method 2: Higher highs/higher lows or lower highs/lower lows
        swing_trend = self._analyze_swing_trend()

        # Method 3: Linear regression slope
        slope = self._calculate_price_slope(closes)

        # Method 4: Recent price action momentum
        recent_momentum = self._calculate_momentum(closes)

        # Combine methods for final trend determination
        trend_score = 0

        # EMA relationship (40%)
        if short_ema > long_ema:
            trend_score += 0.4
        elif short_ema < long_ema:
            trend_score -= 0.4

        # Swing trend (30%)
        if swing_trend == "uptrend":
            trend_score += 0.3
        elif swing_trend == "downtrend":
            trend_score -= 0.3

        # Linear regression slope (20%)
        normalized_slope = min(1.0, max(-1.0, slope * 100))  # Normalize to [-1, 1]
        trend_score += normalized_slope * 0.2

        # Recent momentum (10%)
        normalized_momentum = min(1.0, max(-1.0, recent_momentum * 10))  # Normalize to [-1, 1]
        trend_score += normalized_momentum * 0.1

        # Classify trend based on score
        if trend_score > self.params["trend_threshold"]:
            return "uptrend"
        elif trend_score < -self.params["trend_threshold"]:
            return "downtrend"
        else:
            return "neutral"

    def _analyze_swing_trend(self) -> str:
        """
        Analyze trend using swing highs and swing lows.

        Returns:
            str: 'uptrend', 'downtrend', or 'neutral'
        """
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return "neutral"

        # Get recent swing highs and lows (sorted by timestamp)
        recent_highs = sorted(list(self.swing_highs), key=lambda x: x["timestamp"])[-3:]
        recent_lows = sorted(list(self.swing_lows), key=lambda x: x["timestamp"])[-3:]

        # Check for higher highs and higher lows (uptrend)
        higher_highs = all(recent_highs[i]["price"] > recent_highs[i - 1]["price"] for i in range(1, len(recent_highs)))
        higher_lows = all(recent_lows[i]["price"] > recent_lows[i - 1]["price"] for i in range(1, len(recent_lows)))

        # Check for lower highs and lower lows (downtrend)
        lower_highs = all(recent_highs[i]["price"] < recent_highs[i - 1]["price"] for i in range(1, len(recent_highs)))
        lower_lows = all(recent_lows[i]["price"] < recent_lows[i - 1]["price"] for i in range(1, len(recent_lows)))

        if higher_highs and higher_lows:
            return "uptrend"
        elif lower_highs and lower_lows:
            return "downtrend"
        elif higher_highs or higher_lows:
            return "uptrend"  # Partial uptrend
        elif lower_highs or lower_lows:
            return "downtrend"  # Partial downtrend
        else:
            return "neutral"

    async def _detect_market_phase(self) -> str:
        """
        Detect current market phase: accumulation, markup, distribution, markdown.

        Returns:
            str: Market phase
        """
        if len(self.candle_history) < self.params["phase_detection_periods"]:
            return "undefined"

        # Get relevant data
        period = self.params["phase_detection_periods"]
        candles = self.candle_history[-period:]

        # Calculate price statistics
        closes = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]

        # Split into two halves to compare
        half_period = period // 2
        first_half_closes = closes[:half_period]
        second_half_closes = closes[half_period:]
        first_half_volumes = volumes[:half_period]
        second_half_volumes = volumes[half_period:]

        # Calculate price change
        first_half_change = (first_half_closes[-1] - first_half_closes[0]) / first_half_closes[0]
        second_half_change = (second_half_closes[-1] - second_half_closes[0]) / second_half_closes[0]

        # Calculate volatility
        first_half_volatility = self._calculate_volatility(candles[:half_period])
        second_half_volatility = self._calculate_volatility(candles[half_period:])

        # Calculate volume change
        first_half_avg_volume = sum(first_half_volumes) / len(first_half_volumes) if first_half_volumes else 0
        second_half_avg_volume = sum(second_half_volumes) / len(second_half_volumes) if second_half_volumes else 0
        volume_change = (second_half_avg_volume - first_half_avg_volume) / first_half_avg_volume if first_half_avg_volume else 0

        # Current trend info
        current_trend = self.current_trend

        # Detect phase based on patterns
        # Accumulation: sideways after downtrend, decreasing volatility, increasing volume
        if (
            abs(second_half_change) < 0.02
            and self.current_trend in ["neutral", "downtrend"]
            and second_half_volatility < first_half_volatility * (1 - self.params["phase_volatility_threshold"])
            and volume_change > self.params["phase_volume_threshold"]
        ):
            return "accumulation"

        # Markup: prices rising, increased volume
        elif second_half_change > 0.03 and second_half_change > first_half_change and volume_change > 0:
            return "markup"

        # Distribution: sideways after uptrend, decreasing volatility, decreasing volume
        elif (
            abs(second_half_change) < 0.02
            and self.current_trend in ["neutral", "uptrend"]
            and second_half_volatility < first_half_volatility * (1 - self.params["phase_volatility_threshold"])
            and volume_change < -self.params["phase_volume_threshold"]
        ):
            return "distribution"

        # Markdown: prices falling, increased volume
        elif second_half_change < -0.03 and second_half_change < first_half_change and volume_change > 0:
            return "markdown"

        # If no clear phase detected
        return "undefined"

    async def _identify_price_range(self) -> Dict[str, Any]:
        """
        Identify if the market is in a range and determine range boundaries.

        Returns:
            Dict[str, Any]: Range information
        """
        if len(self.candle_history) < 20:
            return {"upper": None, "lower": None, "midpoint": None, "strength": 0.0}

        # Get recent candles
        recent_candles = self.candle_history[-30:]

        # Calculate volatility
        volatility = self._calculate_volatility(recent_candles)

        # Determine if we're in a range based on trend and volatility
        is_range = self.current_trend == "neutral" and volatility < 0.02

        if not is_range:
            return {"upper": None, "lower": None, "midpoint": None, "strength": 0.0}

        # Identify range boundaries using recent swing points
        swing_highs = list(self.swing_highs)
        swing_lows = list(self.swing_lows)

        if not swing_highs or not swing_lows:
            return {"upper": None, "lower": None, "midpoint": None, "strength": 0.0}

        # Sort by timestamp and get recent swings
        recent_swing_highs = sorted(swing_highs, key=lambda x: x["timestamp"])[-5:]
        recent_swing_lows = sorted(swing_lows, key=lambda x: x["timestamp"])[-5:]

        if not recent_swing_highs or not recent_swing_lows:
            return {"upper": None, "lower": None, "midpoint": None, "strength": 0.0}

        # Calculate upper boundary: average of recent swing highs
        upper_boundary = sum(h["price"] for h in recent_swing_highs) / len(recent_swing_highs)

        # Calculate lower boundary: average of recent swing lows
        lower_boundary = sum(l["price"] for l in recent_swing_lows) / len(recent_swing_lows)
