#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Order Flow Brain - Strategy Brain Module

This module implements a sophisticated order flow analysis strategy brain that
uses market microstructure, order book imbalances, and flow patterns to generate
trading signals with exceptional accuracy.

The OrderFlowBrain identifies significant order flow patterns including:
- Order book imbalances
- Large limit order placement and cancellations
- Hidden liquidity detection
- Smart money flow tracking
- Footprint chart patterns
- Delta divergences

This strategy excels in detecting institutional activity and short-term
price movements for both trading signals and execution optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import logging
from collections import deque
import asyncio
import json

from config import Config
from common.logger import get_logger
from common.utils import normalize_data, timeit, calculate_dynamic_threshold
from data_storage.market_data import MarketDataStorage
try:
    from feature_service.features.order_flow import (
        OrderFlowFeatures, VolumeProfileFeatures, OrderBookFeatures
    )
except Exception:  # pragma: no cover - optional dependency
    from feature_service.features.order_flow import OrderFlowFeatures
    VolumeProfileFeatures = None  # type: ignore
    OrderBookFeatures = None  # type: ignore
from strategy_brains.base_brain import StrategyBrain
from intelligence.loophole_detection.microstructure import MicrostructureAnalyzer


class OrderFlowBrain(StrategyBrain):
    """
    Order Flow Brain analyzes market microstructure and order flow patterns
    to generate highly accurate trading signals.

    This advanced brain identifies institutional footprints, significant order flow
    imbalances, and hidden liquidity zones to predict short-term price movements.
    """

    BRAIN_TYPE = "OrderFlowBrain"
    VERSION = "2.1.0"
    DEFAULT_PARAMS = {
        # Order book imbalance thresholds
        "ob_imbalance_threshold": 1.8,        # Minimum ratio for significant imbalance
        "ob_depth_levels": 20,                # Order book depth levels to analyze
        "dynamic_threshold_window": 100,      # Samples for dynamic threshold calculation
        "min_significant_size": 100000,       # Minimum size for significant orders (in base units)

        # Delta and footprint settings
        "delta_lookback_periods": 5,          # Periods to calculate delta
        "delta_divergence_threshold": 0.4,    # Minimum threshold for delta divergence
        "footprint_node_significance": 0.65,  # Significance threshold for footprint nodes

        # Time and sales analysis
        "txn_cluster_time_window": 15,        # Seconds to identify transaction clusters
        "txn_cluster_min_count": 5,           # Minimum transactions for a cluster
        "volume_spike_threshold": 2.5,        # Standard deviations for volume spike detection

        # Liquidity and market microstructure
        "liquidity_sweep_detection": True,    # Enable liquidity sweep detection
        "liquidity_fade_detection": True,     # Enable liquidity fade detection
        "hidden_liquidity_tracking": True,    # Enable hidden liquidity tracking
        "iceberg_detection": True,            # Enable iceberg order detection

        # Signal generation parameters
        "confirmation_required": True,        # Require additional confirmation signals
        "min_signal_strength": 0.7,           # Minimum signal strength to generate signals
        "max_adverse_excursion": 0.002,       # Maximum allowed adverse excursion (0.2%)

        # Memory and adaptation
        "learning_rate": 0.025,               # Learning rate for adaptive parameters
        "memory_length": 500,                 # Number of periods to keep in memory
        "session_reset": False,               # Reset accumulated data at session boundaries
        "volume_lookback": 20,                # Lookback periods for volume comparison
    }

    async def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate a list with a single order flow signal."""
        signal = await self.generate_signal()
        return [signal] if signal else []

    async def on_regime_change(self, new_regime: str) -> None:
        """Reset state when a new market regime is detected."""
        self.logger.info("Adapting to regime change: %s", new_regime)
        await self._reset_session_data()
    

    def __init__(self, config: Config, asset_id: str, timeframe: str, **kwargs):
        """
        Initialize the Order Flow Brain strategy.

        Args:
            config: System configuration
            asset_id: Asset identifier (e.g., 'BTCUSDT', 'EURUSD', etc.)
            timeframe: Strategy timeframe (e.g., '1m', '5m', '1h', etc.)
            **kwargs: Additional parameters to override defaults
        """
        super().__init__(config, asset_id, timeframe, **kwargs)

        self.logger = get_logger("OrderFlowBrain")
        self.logger.info(f"Initializing OrderFlowBrain v{self.VERSION} for {asset_id} on {timeframe} timeframe")

        # Initialize parameters with defaults and overrides
        self.params = {**self.DEFAULT_PARAMS, **kwargs}

        # Configure feature providers
        self.order_flow_features = OrderFlowFeatures(config, asset_id)
        self.volume_profile_features = VolumeProfileFeatures(config, asset_id)
        self.order_book_features = OrderBookFeatures(config, asset_id)

        # Microstructure analyzer for loophole detection
        self.microstructure = MicrostructureAnalyzer(config, asset_id)

        # Data storage
        self.market_data = MarketDataStorage(config)

        # Track current market regime for risk adjustments
        self.current_regime: str | None = None
        
        # Memory structures for tracking order flow patterns
        self.order_book_imbalances = deque(maxlen=self.params["memory_length"])
        self.delta_history = deque(maxlen=self.params["memory_length"])
        self.volume_nodes = {}
        self.significant_transactions = deque(maxlen=self.params["memory_length"])
        self.iceberg_detections = deque(maxlen=self.params["memory_length"])
        self.liquidity_events = deque(maxlen=self.params["memory_length"])
        self.recent_cluster_volumes = deque(maxlen=50)

        # Detected patterns and states
        self.current_ob_imbalance = 0.0
        self.current_delta = 0.0
        self.current_delta_divergence = 0.0
        self.liquidity_zones = {}
        self.supply_demand_levels = {}

        # Performance tracking
        self.signal_performance = {
            'tp_hit': 0,         # Take profit hits
            'sl_hit': 0,         # Stop loss hits
            'timeout': 0,        # Signal timeout
            'abandoned': 0,      # Signal abandoned due to invalidation
            'total': 0           # Total signals generated
        }

        # State variables
        self.initialized = False
        self.last_update_time = None
        self.is_ready = False

        # Counter for session data
        self.data_points_processed = 0
        self.last_session_date = None

        # Initialize performance metrics
        self.performance_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win_size': 0.0,
            'avg_loss_size': 0.0,
            'expectancy': 0.0,
            'sharpe_ratio': 0.0
        }

        self.logger.info(f"OrderFlowBrain initialization complete")

    async def initialize(self) -> bool:
        """
        Initialize the strategy brain with historical data and prepare it for use.

        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info(f"Starting OrderFlowBrain initialization")

        try:
            # Load historical data
            candles = await self.market_data.get_candles(
                self.asset_id,
                self.timeframe,
                limit=self.params["memory_length"]
            )

            if len(candles) < 10:  # Need some minimum data
                self.logger.warning("Insufficient historical data for initialization")
                return False

            # Initialize order book state
            orderbook = await self.market_data.get_order_book_snapshot(self.asset_id)
            if not orderbook:
                self.logger.warning("Could not retrieve order book snapshot")
                return False

            # Process historical data
            for candle in candles:
                await self._process_candle(candle)

            # Set up order book analysis
            await self._analyze_order_book(orderbook)

            # Initialize volume profile
            await self._initialize_volume_profile(candles)

            # Load any saved state if available
            await self._load_state()

            self.initialized = True
            self.is_ready = True
            self.last_update_time = datetime.now()

            self.logger.info(f"OrderFlowBrain initialization completed successfully")

            return True

        except Exception as e:
            self.logger.error(f"Error during OrderFlowBrain initialization: {str(e)}")
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

        # Check for session boundary
        candle_date = datetime.fromtimestamp(candle['timestamp'] / 1000).date()
        if self.params["session_reset"] and self.last_session_date and candle_date != self.last_session_date:
            self.logger.info("New trading session detected, resetting session-specific data")
            await self._reset_session_data()

        self.last_session_date = candle_date

        # Process this candle
        await self._process_candle(candle)

        # Increment counters
        self.data_points_processed += 1

    async def process_order_book(self, order_book: Dict[str, Any]) -> None:
        """
        Process order book update and look for significant imbalances and patterns.

        Args:
            order_book: Order book data with bids and asks
        """
        if not self.initialized:
            self.logger.warning("Cannot process order book: Brain not initialized")
            return

        self.last_update_time = datetime.now()

        # Analyze order book for significant imbalances and patterns
        await self._analyze_order_book(order_book)

    async def process_transactions(self, transactions: List[Dict[str, Any]]) -> None:
        """
        Process market transactions (time and sales) data to identify patterns.

        Args:
            transactions: List of recent market transactions
        """
        if not self.initialized:
            self.logger.warning("Cannot process transactions: Brain not initialized")
            return

        self.last_update_time = datetime.now()

        # Process transaction data for patterns
        await self._analyze_transactions(transactions)

    async def generate_signal(self) -> Optional[Dict[str, Any]]:
        """
        Generate trading signal based on current order flow analysis.

        Returns:
            Optional[Dict[str, Any]]: Trading signal if conditions are met, None otherwise
        """
        if not self.is_ready:
            return None

        try:
            # Calculate signal components
            ob_signal = await self._generate_ob_signal()
            flow_signal = await self._generate_flow_signal()
            vs_signal = await self._generate_volume_structure_signal()

            # Skip if any essential component is missing
            if ob_signal is None or flow_signal is None or vs_signal is None:
                return None

            # Combine signal components with weights
            signal_direction = 0
            signal_components = {}

            # Order book signal (40% weight)
            if abs(ob_signal['strength']) >= self.params["min_signal_strength"]:
                signal_direction += 0.4 * ob_signal['direction']
                signal_components['order_book'] = {
                    'direction': ob_signal['direction'],
                    'strength': ob_signal['strength'],
                    'details': ob_signal['details']
                }

            # Flow signal (35% weight)
            if abs(flow_signal['strength']) >= self.params["min_signal_strength"]:
                signal_direction += 0.35 * flow_signal['direction']
                signal_components['flow'] = {
                    'direction': flow_signal['direction'],
                    'strength': flow_signal['strength'],
                    'details': flow_signal['details']
                }

            # Volume structure signal (25% weight)
            if abs(vs_signal['strength']) >= self.params["min_signal_strength"]:
                signal_direction += 0.25 * vs_signal['direction']
                signal_components['volume_structure'] = {
                    'direction': vs_signal['direction'],
                    'strength': vs_signal['strength'],
                    'details': vs_signal['details']
                }

            # Determine final signal direction and strength
            final_direction = 1 if signal_direction > 0 else (-1 if signal_direction < 0 else 0)
            final_strength = abs(signal_direction)

            # Apply confirmation filter if enabled
            if self.params["confirmation_required"] and len(signal_components) < 2:
                self.logger.debug(f"Signal rejected: confirmation required but only {len(signal_components)} components")
                return None

            # Apply strength threshold
            if final_strength < self.params["min_signal_strength"]:
                self.logger.debug(f"Signal rejected: strength {final_strength:.2f} below threshold {self.params['min_signal_strength']}")
                return None

            # Calculate target and stop levels
            entry_price = self._get_current_price()
            target_distance, stop_distance = self._calculate_risk_levels(
                direction=final_direction,
                components=signal_components,
                current_price=entry_price
            )

            # Check if risk-reward meets minimum criteria (at least 1:1)
            if stop_distance > 0 and target_distance / stop_distance < 1.0:
                self.logger.debug(f"Signal rejected: poor risk-reward ratio {target_distance/stop_distance:.2f}")
                return None

            # Form the final signal
            signal = {
                'strategy': self.BRAIN_TYPE,
                'asset_id': self.asset_id,
                'timestamp': datetime.now().timestamp() * 1000,
                'timeframe': self.timeframe,
                'direction': final_direction,
                'strength': final_strength,
                'entry_price': entry_price,
                'target_price': entry_price * (1 + final_direction * target_distance),
                'stop_price': entry_price * (1 - final_direction * stop_distance),
                'components': signal_components,
                'expiration': (datetime.now() + timedelta(minutes=self._get_expiration_minutes())).timestamp() * 1000,
                'id': f"OF_{self.asset_id}_{int(datetime.now().timestamp())}",
                'metadata': {
                    'version': self.VERSION,
                    'confidence': min(0.99, final_strength * 1.15),  # Cap at 0.99
                    'expected_duration': self._estimate_signal_duration(final_direction, components=signal_components),
                    'market_context': await self._get_market_context()
                }
            }

            # Log the signal and update performance counters
            self.logger.info(f"Generated signal: {signal['id']} Direction: {'BUY' if final_direction > 0 else 'SELL'} Strength: {final_strength:.2f}")
            self.signal_performance['total'] += 1

            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return None

    async def generate_signals(self) -> List[Dict[str, Any]]:
        """Return a list of generated signals for compatibility."""

        signal = await self.generate_signal()
        return [signal] if signal else []

    async def on_regime_change(self, new_regime: str) -> None:
        """React to market regime changes by updating internal state."""
        self.logger.info(f"Market regime changed to {new_regime}")
        self.current_regime = new_regime

    async def update_parameters(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update strategy parameters based on recent performance.

        Args:
            performance_metrics: Dictionary with performance metrics
        """
        self.logger.info(f"Updating OrderFlowBrain parameters based on performance metrics")


        # Store performance metrics
        self.performance_metrics = performance_metrics

        # Skip if we don't have enough data yet
        if performance_metrics.get('total_trades', 0) < 10:
            self.logger.info("Not enough trades for parameter optimization")
            return

        # Adaptive adjustments based on performance
        win_rate = performance_metrics.get('win_rate', 0)
        profit_factor = performance_metrics.get('profit_factor', 0)

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

            # Adjust order book sensitivity
            self.params["ob_imbalance_threshold"] *= (1 - adjustment * lr * 0.1)
            self.params["min_significant_size"] *= (1 - adjustment * lr * 0.15)

            # Adjust signal generation parameters
            self.params["min_signal_strength"] *= (1 - adjustment * lr * 0.05)

            # Adjust other parameters
            self.params["delta_divergence_threshold"] *= (1 - adjustment * lr * 0.1)

            self.logger.info(f"Parameters adjusted: ob_imbalance_threshold={self.params['ob_imbalance_threshold']:.2f}, " +
                            f"min_signal_strength={self.params['min_signal_strength']:.2f}")

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
        if status == 'tp_hit':
            self.signal_performance['tp_hit'] += 1
        elif status == 'sl_hit':
            self.signal_performance['sl_hit'] += 1
        elif status == 'expired':
            self.signal_performance['timeout'] += 1
        elif status == 'cancelled':
            self.signal_performance['abandoned'] += 1

        # Use feedback to adjust strategy parameters
        if result and 'execution_metrics' in result:
            metrics = result['execution_metrics']

            # Calculate signal quality metrics
            max_adverse_excursion = metrics.get('max_adverse_excursion', 0)
            time_to_outcome = metrics.get('time_to_outcome', 0)

            # Adjust parameters based on outcome
            if status == 'tp_hit' and max_adverse_excursion < self.params['max_adverse_excursion'] * 0.5:
                # Clean win - could be more aggressive
                adjustment = 0.05
            elif status == 'sl_hit' and max_adverse_excursion > self.params['max_adverse_excursion'] * 0.9:
                # Clear loss - need to be more conservative
                adjustment = -0.1
            else:
                # No adjustment needed
                adjustment = 0

            if adjustment != 0:
                # Apply small adjustment to relevant parameters
                lr = self.params["learning_rate"] * 0.5  # Reduced learning rate for individual updates
                self.params["min_signal_strength"] = max(0.5, min(0.95,
                                                        self.params["min_signal_strength"] * (1 + adjustment * lr)))

                self.logger.info(f"Parameter adjusted after signal feedback: min_signal_strength={self.params['min_signal_strength']:.2f}")

    async def generate_signals(self) -> List[Dict[str, Any]]:
        """Return a list of generated signals."""
        signal = await self.generate_signal()
        return [signal] if signal else []

    async def on_regime_change(self, new_regime: str) -> None:
        """Handle market regime changes by adjusting sensitivity."""
        self.logger.info(f"Regime changed to {new_regime}")
        if new_regime == "volatile":
            self.params["min_signal_strength"] = min(0.95, self.params["min_signal_strength"] * 1.1)
        elif new_regime == "calm":
            self.params["min_signal_strength"] = max(0.5, self.params["min_signal_strength"] * 0.9)

    async def save_state(self) -> Dict[str, Any]:
        """
        Save the current state of the strategy brain for persistence.

        Returns:
            Dict[str, Any]: State dictionary for persistence
        """
        state = {
            'version': self.VERSION,
            'params': self.params,
            'performance': self.signal_performance,
            'metrics': self.performance_metrics,
            'liquidity_zones': self.liquidity_zones,
            'supply_demand_levels': self.supply_demand_levels,
            'data_points_processed': self.data_points_processed,
            'timestamp': datetime.now().timestamp()
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
            if 'version' not in state or not state['version'].startswith('2.'):
                self.logger.warning(f"Incompatible state version: {state.get('version')}")
                return False

            # Load parameters with safeguards
            if 'params' in state:
                # Only load compatible parameters
                for key, value in state['params'].items():
                    if key in self.params:
                        self.params[key] = value

            # Load other state elements
            if 'performance' in state:
                self.signal_performance = state['performance']

            if 'metrics' in state:
                self.performance_metrics = state['metrics']

            if 'liquidity_zones' in state:
                self.liquidity_zones = state['liquidity_zones']

            if 'supply_demand_levels' in state:
                self.supply_demand_levels = state['supply_demand_levels']

            if 'data_points_processed' in state:
                self.data_points_processed = state['data_points_processed']

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
        if self.signal_performance['total'] > 0:
            win_rate = self.signal_performance['tp_hit'] / self.signal_performance['total']

        return {
            'brain_type': self.BRAIN_TYPE,
            'version': self.VERSION,
            'asset_id': self.asset_id,
            'timeframe': self.timeframe,
            'is_ready': self.is_ready,
            'initialized': self.initialized,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'data_points_processed': self.data_points_processed,
            'performance': {
                'total_signals': self.signal_performance['total'],
                'win_rate': win_rate,
                'tp_hit': self.signal_performance['tp_hit'],
                'sl_hit': self.signal_performance['sl_hit'],
                'timeout': self.signal_performance['timeout'],
                'abandoned': self.signal_performance['abandoned']
            },
            'current_state': {
                'ob_imbalance': self.current_ob_imbalance,
                'delta': self.current_delta,
                'delta_divergence': self.current_delta_divergence,
                'liquidity_zones_count': len(self.liquidity_zones),
                'supply_demand_levels_count': len(self.supply_demand_levels)
            }
        }

    async def shutdown(self) -> None:
        """
        Perform any cleanup operations before shutdown.
        """
        self.logger.info(f"OrderFlowBrain shutdown initiated")

        # Save final state metrics
        try:
            await self.save_state()
            self.logger.info("Final state saved during shutdown")
        except Exception as e:
            self.logger.error(f"Error saving state during shutdown: {str(e)}")

        self.logger.info(f"OrderFlowBrain shutdown complete")

    async def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate trading signals based on current analysis."""
        signal = await self.generate_signal()
        return [signal] if signal else []

    async def on_regime_change(self, new_regime: str) -> None:
        """React to a detected market regime change."""
        self.logger.info(f"Regime changed to {new_regime}")
    

    # ========== Private Methods ==========

    async def _process_candle(self, candle: Dict[str, Any]) -> None:
        """
        Process a single candle for order flow analysis.

        Args:
            candle: Candle data
        """
        # Extract delta (difference between buy and sell volume)
        delta = await self.order_flow_features.calculate_delta(candle)
        self.delta_history.append({
            'timestamp': candle['timestamp'],
            'delta': delta,
            'close': candle['close'],
            'volume': candle.get('volume', 0)
        })

        # Update current delta
        self.current_delta = delta

        # Calculate delta divergence
        if len(self.delta_history) > 2:
            self.current_delta_divergence = await self._calculate_delta_divergence()

        # Update volume nodes
        await self._update_volume_nodes(candle)

        # Update supply/demand zones based on significant candles
        is_significant = self._is_significant_candle(candle)
        if is_significant:
            await self._update_supply_demand_levels(candle)

    async def _analyze_order_book(self, order_book: Dict[str, Any]) -> None:
        """
        Analyze the order book for imbalances and significant levels.

        Args:
            order_book: Order book data with bids and asks
        """
        # Calculate order book imbalance ratio
        imbalance = await self.order_book_features.calculate_imbalance(
            order_book,
            depth_levels=self.params["ob_depth_levels"]
        )

        self.order_book_imbalances.append({
            'timestamp': order_book.get('timestamp', datetime.now().timestamp() * 1000),
            'imbalance': imbalance
        })

        self.current_ob_imbalance = imbalance

        # Detect significant orders (large or iceberg)
        significant_levels = await self.order_book_features.detect_significant_levels(
            order_book,
            min_size=self.params["min_significant_size"]
        )

        for level in significant_levels:
            if level['type'] == 'iceberg' and self.params["iceberg_detection"]:
                self.iceberg_detections.append({
                    'timestamp': order_book.get('timestamp', datetime.now().timestamp() * 1000),
                    'price': level['price'],
                    'size': level['size'],
                    'side': level['side']
                })

            # Update liquidity zones
            price_key = f"{level['price']:.8f}"
            if price_key not in self.liquidity_zones:
                self.liquidity_zones[price_key] = {
                    'price': level['price'],
                    'side': level['side'],
                    'strength': 1,
                    'first_seen': datetime.now().timestamp() * 1000,
                    'last_seen': datetime.now().timestamp() * 1000
                }
            else:
                self.liquidity_zones[price_key]['strength'] += 1
                self.liquidity_zones[price_key]['last_seen'] = datetime.now().timestamp() * 1000

        # Cleanup old liquidity zones (older than 24 hours)
        current_time = datetime.now().timestamp() * 1000
        self.liquidity_zones = {
            k: v for k, v in self.liquidity_zones.items()
            if current_time - v['last_seen'] < 24 * 60 * 60 * 1000
        }

        # Detect liquidity sweeps and fades
        if self.params["liquidity_sweep_detection"]:
            await self._detect_liquidity_events(order_book)

    async def _analyze_transactions(self, transactions: List[Dict[str, Any]]) -> None:
        """
        Analyze time and sales data for transaction patterns.

        Args:
            transactions: List of market transactions
        """
        # Group transactions into time clusters
        clusters = []
        current_cluster = []

        sorted_txns = sorted(transactions, key=lambda x: x['timestamp'])

        for txn in sorted_txns:
            if not current_cluster:
                current_cluster = [txn]
            else:
                # Check if this transaction is within the time window of the first in cluster
                time_diff = (txn['timestamp'] - current_cluster[0]['timestamp']) / 1000  # seconds
                if time_diff <= self.params["txn_cluster_time_window"]:
                    current_cluster.append(txn)
                else:
                    # This transaction starts a new cluster
                    if len(current_cluster) >= self.params["txn_cluster_min_count"]:
                        clusters.append(current_cluster)
                    current_cluster = [txn]

        # Add the last cluster if it meets the minimum size
        if len(current_cluster) >= self.params["txn_cluster_min_count"]:
            clusters.append(current_cluster)

        # Analyze each significant cluster
        for cluster in clusters:
            buy_volume = sum(t['size'] for t in cluster if t.get('side') == 'buy')
            sell_volume = sum(t['size'] for t in cluster if t.get('side') == 'sell')

            # Calculate volume ratio and price movement
            volume_ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
            start_price = cluster[0]['price']
            end_price = cluster[-1]['price']
            price_change_pct = (end_price - start_price) / start_price

            # Detect if this is a significant transaction cluster
            is_significant = False
            significance_reason = []

            # Check volume ratio (heavy imbalance)
            if volume_ratio > 3.0 or volume_ratio < 0.33:
                is_significant = True
                significance_reason.append('volume_imbalance')

            # Check price impact
            if abs(price_change_pct) > 0.001:  # 0.1% move
                is_significant = True
                significance_reason.append('price_impact')

            # Check total volume versus recent average
            total_volume = buy_volume + sell_volume
            recent_avg = (
                np.mean(self.recent_cluster_volumes)
                if self.recent_cluster_volumes
                else 0
            )
            if self.recent_cluster_volumes and total_volume > recent_avg:
                is_significant = True
                significance_reason.append("volume_spike")

            # Update cluster volume history
            self.recent_cluster_volumes.append(total_volume)

            if is_significant:
                self.significant_transactions.append({
                    'timestamp': cluster[0]['timestamp'],
                    'duration_ms': cluster[-1]['timestamp'] - cluster[0]['timestamp'],
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'volume_ratio': volume_ratio,
                    'price_change_pct': price_change_pct,
                    'start_price': start_price,
                    'end_price': end_price,
                    'reasons': significance_reason
                })

    async def _initialize_volume_profile(self, candles: List[Dict[str, Any]]) -> None:
        """
        Initialize volume profile from historical candles.

        Args:
            candles: List of historical candles
        """
        # Calculate initial volume profile
        volume_profile = await self.volume_profile_features.calculate_volume_profile(candles)

        # Initialize volume nodes
        for node in volume_profile:
            price_key = f"{node['price']:.8f}"
            self.volume_nodes[price_key] = {
                'price': node['price'],
                'volume': node['volume'],
                'buys': node.get('buys', node['volume'] / 2),  # Estimate if not provided
                'sells': node.get('sells', node['volume'] / 2),  # Estimate if not provided
                'delta': node.get('delta', 0),
                'strength': node.get('strength', 1)
            }

    async def _update_volume_nodes(self, candle: Dict[str, Any]) -> None:
        """
        Update volume profile nodes with new candle data.

        Args:
            candle: New candle data
        """
        # Get price range for this candle
        high = candle['high']
        low = candle['low']

        # Estimate price points within the candle
        price_points = np.linspace(low, high, num=5)

        # Estimate volume distribution
        volume = candle['volume']
        buy_volume = volume * (candle['close'] > candle['open'])
        sell_volume = volume - buy_volume

        # Distribute volume across price points
        for price in price_points:
            price_key = f"{price:.8f}"

            if price_key in self.volume_nodes:
                # Update existing node
                self.volume_nodes[price_key]['volume'] += volume / len(price_points)
                self.volume_nodes[price_key]['buys'] += buy_volume / len(price_points)
                self.volume_nodes[price_key]['sells'] += sell_volume / len(price_points)
                self.volume_nodes[price_key]['delta'] = self.volume_nodes[price_key]['buys'] - self.volume_nodes[price_key]['sells']

                # Gradually increase strength based on continued activity
                self.volume_nodes[price_key]['strength'] = min(10, self.volume_nodes[price_key]['strength'] * 1.05)
            else:
                # Create new node
                self.volume_nodes[price_key] = {
                    'price': price,
                    'volume': volume / len(price_points),
                    'buys': buy_volume / len(price_points),
                    'sells': sell_volume / len(price_points),
                    'delta': (buy_volume - sell_volume) / len(price_points),
                    'strength': 1
                }

        # Decay old nodes
        for price_key in list(self.volume_nodes.keys()):
            self.volume_nodes[price_key]['strength'] *= 0.99

            # Remove very weak nodes
            if self.volume_nodes[price_key]['strength'] < 0.1:
                del self.volume_nodes[price_key]

    async def _update_supply_demand_levels(self, candle: Dict[str, Any]) -> None:
        """
        Update supply and demand levels based on significant candles.

        Args:
            candle: Candle data
        """
        # Determine if this candle forms a supply or demand level
        if candle['close'] > candle['open']:  # Bullish candle
            # Potential demand level at bottom
            level_price = candle['low']
            level_type = 'demand'
        else:  # Bearish candle
            # Potential supply level at top
            level_price = candle['high']
            level_type = 'supply'

        # Create a unique key for this price level
        price_key = f"{level_price:.8f}"

        # Check if this level already exists
        if price_key in self.supply_demand_levels:
            # Update existing level
            self.supply_demand_levels[price_key]['strength'] += 1
            self.supply_demand_levels[price_key]['last_seen'] = candle['timestamp']

            # If we see the opposite type at the same level, reduce strength
            if self.supply_demand_levels[price_key]['type'] != level_type:
                self.supply_demand_levels[price_key]['strength'] = max(0, self.supply_demand_levels[price_key]['strength'] - 2)

                # If strength is reduced to 0, flip the level type
                if self.supply_demand_levels[price_key]['strength'] == 0:
                    self.supply_demand_levels[price_key]['type'] = level_type
                    self.supply_demand_levels[price_key]['strength'] = 1
        else:
            # Create new level
            self.supply_demand_levels[price_key] = {
                'price': level_price,
                'type': level_type,
                'strength': 1,
                'first_seen': candle['timestamp'],
                'last_seen': candle['timestamp'],
                'volume': candle['volume']
            }

        # Clean up levels that are too old (older than 50 candles)
        current_time = candle['timestamp']
        # This logic depends on your candle interval
        candle_interval_ms = self._get_interval_ms()
        max_age_ms = candle_interval_ms * 50

        self.supply_demand_levels = {
            k: v for k, v in self.supply_demand_levels.items()
            if current_time - v['last_seen'] < max_age_ms
        }

    async def _detect_liquidity_events(self, order_book: Dict[str, Any]) -> None:
        """
        Detect liquidity sweeps and fades in the order book.

        Args:
            order_book: Current order book data
        """
        if not self.params["liquidity_sweep_detection"]:
            return

        # Get current prices and volumes
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        if not bids or not asks:
            return

        # Get best bid and ask
        best_bid_price = bids[0][0] if bids else 0
        best_ask_price = asks[0][0] if asks else float('inf')

        # Look for recent liquidity zones near the current price
        current_time = datetime.now().timestamp() * 1000
        recent_threshold = current_time - (5 * 60 * 1000)  # 5 minutes

        for price_key, zone in list(self.liquidity_zones.items()):
            zone_price = zone['price']

            # Skip if the zone isn't recent enough
            if zone['last_seen'] < recent_threshold:
                continue

            # Detect liquidity sweep at bid side
            if zone['side'] == 'bid' and best_bid_price < zone_price and zone['strength'] > 3:
                self.liquidity_events.append({
                    'timestamp': current_time,
                    'type': 'sweep',
                    'side': 'bid',
                    'price': zone_price,
                    'strength': zone['strength']
                })

                # Remove this liquidity zone as it was swept
                del self.liquidity_zones[price_key]

            # Detect liquidity sweep at ask side
            elif zone['side'] == 'ask' and best_ask_price > zone_price and zone['strength'] > 3:
                self.liquidity_events.append({
                    'timestamp': current_time,
                    'type': 'sweep',
                    'side': 'ask',
                    'price': zone_price,
                    'strength': zone['strength']
                })

                # Remove this liquidity zone as it was swept
                del self.liquidity_zones[price_key]

    async def _generate_ob_signal(self) -> Optional[Dict[str, Any]]:
        """
        Generate signal component based on order book analysis.

        Returns:
            Optional[Dict[str, Any]]: Signal component or None
        """
        if not self.order_book_imbalances:
            return None

        # Calculate average and current imbalance
        recent_imbalances = list(self.order_book_imbalances)[-20:]
        avg_imbalance = sum(item['imbalance'] for item in recent_imbalances) / len(recent_imbalances)

        # Threshold based on historical volatility
        dynamic_threshold = calculate_dynamic_threshold(
            [item['imbalance'] for item in recent_imbalances],
            base_threshold=self.params["ob_imbalance_threshold"],
            volatility_factor=0.5
        )

        # Check if current imbalance exceeds the threshold
        current_imbalance = self.current_ob_imbalance
        signal_strength = 0
        signal_direction = 0
        details = {}

        if abs(current_imbalance) > dynamic_threshold:
            signal_direction = 1 if current_imbalance > 0 else -1

            # Calculate signal strength based on imbalance magnitude
            normalized_imbalance = abs(current_imbalance) / dynamic_threshold
            signal_strength = min(0.95, 0.5 + normalized_imbalance * 0.3)

            details = {
                'current_imbalance': current_imbalance,
                'avg_imbalance': avg_imbalance,
                'threshold': dynamic_threshold,
                'normalized_value': normalized_imbalance
            }

        # Check for recent significant orders
        recent_icebergs = [ic for ic in self.iceberg_detections
                        if datetime.now().timestamp() * 1000 - ic['timestamp'] < 5 * 60 * 1000]

        significant_buy_volume = sum(ic['size'] for ic in recent_icebergs if ic['side'] == 'bid')
        significant_sell_volume = sum(ic['size'] for ic in recent_icebergs if ic['side'] == 'ask')

        if significant_buy_volume > 0 or significant_sell_volume > 0:
            iceberg_ratio = significant_buy_volume / max(1, significant_sell_volume)
            if iceberg_ratio > 3.0:
                # Strong buy-side imbalance from icebergs
                iceberg_direction = 1
                iceberg_strength = min(0.9, 0.6 + (iceberg_ratio / 10))
            elif iceberg_ratio < 0.33:
                # Strong sell-side imbalance from icebergs
                iceberg_direction = -1
                iceberg_strength = min(0.9, 0.6 + (1 / iceberg_ratio / 10))
            else:
                # No strong imbalance
                iceberg_direction = 0
                iceberg_strength = 0

            # Combine with existing signal or create new
            if iceberg_direction != 0 and iceberg_strength > 0:
                if signal_direction == 0:
                    signal_direction = iceberg_direction
                    signal_strength = iceberg_strength
                else:
                    # If both signals agree, strengthen; if disagree, weaken
                    if signal_direction == iceberg_direction:
                        signal_strength = min(0.99, signal_strength + 0.1)
                    else:
                        # Conflicting signals - go with the stronger one
                        if iceberg_strength > signal_strength:
                            signal_direction = iceberg_direction
                            signal_strength = iceberg_strength * 0.8  # Reduce due to conflict
                        else:
                            signal_strength *= 0.8  # Reduce due to conflict

                details.update({
                    'iceberg_buy_volume': significant_buy_volume,
                    'iceberg_sell_volume': significant_sell_volume,
                    'iceberg_ratio': iceberg_ratio,
                    'iceberg_strength': iceberg_strength
                })

        if signal_direction == 0 or signal_strength < self.params["min_signal_strength"]:
            return None

        return {
            'direction': signal_direction,
            'strength': signal_strength,
            'details': details
        }

    async def _generate_flow_signal(self) -> Optional[Dict[str, Any]]:
        """
        Generate signal component based on order flow analysis.

        Returns:
            Optional[Dict[str, Any]]: Signal component or None
        """
        if len(self.delta_history) < self.params["delta_lookback_periods"]:
            return None

        # Analyze delta divergence
        delta_divergence = self.current_delta_divergence

        signal_direction = 0
        signal_strength = 0
        details = {}

        if abs(delta_divergence) > self.params["delta_divergence_threshold"]:
            # Delta divergence generates a contrarian signal
            signal_direction = -1 if delta_divergence > 0 else 1

            # Calculate signal strength based on divergence magnitude
            normalized_divergence = abs(delta_divergence) / self.params["delta_divergence_threshold"]
            signal_strength = min(0.9, 0.5 + normalized_divergence * 0.3)

            details = {
                'delta_divergence': delta_divergence,
                'threshold': self.params["delta_divergence_threshold"],
                'normalized_value': normalized_divergence
            }

        # Check for recent significant transactions
        recent_txns = [tx for tx in self.significant_transactions
                    if datetime.now().timestamp() * 1000 - tx['timestamp'] < 10 * 60 * 1000]

        if recent_txns:
            # Aggregate transaction signals
            buy_pressure = sum(tx['buy_volume'] for tx in recent_txns) / len(recent_txns)
            sell_pressure = sum(tx['sell_volume'] for tx in recent_txns) / len(recent_txns)

            txn_ratio = buy_pressure / max(1, sell_pressure)

            if txn_ratio > 2.0:
                # Strong buy-side pressure
                txn_direction = 1
                txn_strength = min(0.9, 0.6 + (txn_ratio / 10))
            elif txn_ratio < 0.5:
                # Strong sell-side pressure
                txn_direction = -1
                txn_strength = min(0.9, 0.6 + (1 / txn_ratio / 5))
            else:
                # No strong imbalance
                txn_direction = 0
                txn_strength = 0

            # Combine with existing signal or create new
            if txn_direction != 0 and txn_strength > 0:
                if signal_direction == 0:
                    signal_direction = txn_direction
                    signal_strength = txn_strength
                else:
                    # If both signals agree, strengthen; if disagree, weaken
                    if signal_direction == txn_direction:
                        signal_strength = min(0.99, signal_strength + 0.1)
                    else:
                        # Conflicting signals - go with the stronger one
                        if txn_strength > signal_strength:
                            signal_direction = txn_direction
                            signal_strength = txn_strength * 0.8  # Reduce due to conflict
                        else:
                            signal_strength *= 0.8  # Reduce due to conflict

                details.update({
                    'transaction_buy_pressure': buy_pressure,
                    'transaction_sell_pressure': sell_pressure,
                    'transaction_ratio': txn_ratio,
                    'transaction_strength': txn_strength,
                    'transaction_count': len(recent_txns)
                })

        # Check for liquidity events
        recent_liquidity_events = [le for le in self.liquidity_events
                                if datetime.now().timestamp() * 1000 - le['timestamp'] < 5 * 60 * 1000]

        if recent_liquidity_events:
            # Analyze sweep events (sweeping bids is bearish, sweeping asks is bullish)
            bid_sweeps = [le for le in recent_liquidity_events if le['type'] == 'sweep' and le['side'] == 'bid']
            ask_sweeps = [le for le in recent_liquidity_events if le['type'] == 'sweep' and le['side'] == 'ask']

            if bid_sweeps and not ask_sweeps:
                sweep_direction = -1
                sweep_strength = min(0.95, 0.7 + len(bid_sweeps) * 0.05)
            elif ask_sweeps and not bid_sweeps:
                sweep_direction = 1
                sweep_strength = min(0.95, 0.7 + len(ask_sweeps) * 0.05)
            elif bid_sweeps and ask_sweeps:
                # Both sides swept - compare strength
                bid_sweep_strength = sum(le['strength'] for le in bid_sweeps)
                ask_sweep_strength = sum(le['strength'] for le in ask_sweeps)

                if bid_sweep_strength > ask_sweep_strength * 1.5:
                    sweep_direction = -1
                    sweep_strength = 0.7
                elif ask_sweep_strength > bid_sweep_strength * 1.5:
                    sweep_direction = 1
                    sweep_strength = 0.7
                else:
                    sweep_direction = 0
                    sweep_strength = 0
            else:
                sweep_direction = 0
                sweep_strength = 0

            if sweep_direction != 0 and sweep_strength > 0:
                if signal_direction == 0:
                    signal_direction = sweep_direction
                    signal_strength = sweep_strength
                else:
                    # If both signals agree, strengthen; if disagree, weaken
                    if signal_direction == sweep_direction:
                        signal_strength = min(0.99, signal_strength + 0.15)  # Sweeps are significant
                    else:
                        # Conflicting signals - go with the stronger one
                        if sweep_strength > signal_strength:
                            signal_direction = sweep_direction
                            signal_strength = sweep_strength * 0.9
                        else:
                            signal_strength *= 0.7  # Reduce more due to sweep conflict

                details.update({
                    'bid_sweeps': len(bid_sweeps),
                    'ask_sweeps': len(ask_sweeps),
                    'sweep_direction': sweep_direction,
                    'sweep_strength': sweep_strength
                })

        if signal_direction == 0 or signal_strength < self.params["min_signal_strength"]:
            return None

        return {
            'direction': signal_direction,
            'strength': signal_strength,
            'details': details
        }

    async def _generate_volume_structure_signal(self) -> Optional[Dict[str, Any]]:
        """
        Generate signal component based on volume structure analysis.

        Returns:
            Optional[Dict[str, Any]]: Signal component or None
        """
        if not self.volume_nodes or not self.supply_demand_levels:
            return None

        current_price = self._get_current_price()

        # Find significant volume nodes near current price
        significant_nodes = []

        for key, node in self.volume_nodes.items():
            # Check if node is significant
            if node['strength'] >= self.params["footprint_node_significance"]:
                # Check if node is near current price (within 1%)
                if abs(node['price'] - current_price) / current_price < 0.01:
                    significant_nodes.append(node)

        # Find active supply/demand levels near current price
        active_levels = []

        for key, level in self.supply_demand_levels.items():
            # Check if level is strong enough
            if level['strength'] >= 2:
                # Check if level is near current price (within 2%)
                if abs(level['price'] - current_price) / current_price < 0.02:
                    active_levels.append(level)

        signal_direction = 0
        signal_strength = 0
        details = {}

        # Analyze volume nodes for bullish/bearish bias
        if significant_nodes:
            total_delta = sum(node['delta'] for node in significant_nodes)
            total_volume = sum(node['volume'] for node in significant_nodes)

            if total_volume > 0:
                delta_ratio = total_delta / total_volume

                if delta_ratio > 0.2:  # Strong buy pressure
                    node_direction = 1
                    node_strength = min(0.85, 0.5 + delta_ratio)
                elif delta_ratio < -0.2:  # Strong sell pressure
                    node_direction = -1
                    node_strength = min(0.85, 0.5 - delta_ratio)
                else:
                    node_direction = 0
                    node_strength = 0

                if node_direction != 0:
                    signal_direction = node_direction
                    signal_strength = node_strength

                    details = {
                        'significant_nodes': len(significant_nodes),
                        'total_volume': total_volume,
                        'delta_ratio': delta_ratio,
                        'node_strength': node_strength
                    }

        # Analyze supply/demand levels
        if active_levels:
            # Count levels above and below current price
            supply_above = [lvl for lvl in active_levels if lvl['type'] == 'supply' and lvl['price'] > current_price]
            demand_below = [lvl for lvl in active_levels if lvl['type'] == 'demand' and lvl['price'] < current_price]

            # Also check for levels we're currently testing
            supply_at = [lvl for lvl in active_levels if lvl['type'] == 'supply' and
                        abs(lvl['price'] - current_price) / current_price < 0.002]
            demand_at = [lvl for lvl in active_levels if lvl['type'] == 'demand' and
                        abs(lvl['price'] - current_price) / current_price < 0.002]

            level_direction = 0
            level_strength = 0

            # We're at supply - bearish signal
            if supply_at and not demand_at:
                level_direction = -1
                level_strength = min(0.9, 0.6 + len(supply_at) * 0.1)

            # We're at demand - bullish signal
            elif demand_at and not supply_at:
                level_direction = 1
                level_strength = min(0.9, 0.6 + len(demand_at) * 0.1)

            # No current levels, check nearby ones
            elif not supply_at and not demand_at:
                # Check if we're sandwiched between levels
                if supply_above and demand_below:
                    nearest_supply = min(supply_above, key=lambda x: abs(x['price'] - current_price))
                    nearest_demand = min(demand_below, key=lambda x: abs(x['price'] - current_price))

                    # Calculate distances to nearest levels
                    distance_to_supply = abs(nearest_supply['price'] - current_price) / current_price
                    distance_to_demand = abs(nearest_demand['price'] - current_price) / current_price

                    # Check which level we're closer to
                    if distance_to_supply < distance_to_demand * 0.7:
                        # Closer to supply than demand - bearish
                        level_direction = -1
                        level_strength = 0.6
                    elif distance_to_demand < distance_to_supply * 0.7:
                        # Closer to demand than supply - bullish
                        level_direction = 1
                        level_strength = 0.6

                # Only supply above - potential bullish movement until supply
                elif supply_above and not demand_below:
                    level_direction = 1
                    level_strength = 0.5

                # Only demand below - potential bearish movement until demand
                elif demand_below and not supply_above:
                    level_direction = -1
                    level_strength = 0.5

            # Incorporate level analysis into signal
            if level_direction != 0 and level_strength > 0:
                if signal_direction == 0:
                    signal_direction = level_direction
                    signal_strength = level_strength
                else:
                    # If both signals agree, strengthen; if disagree, weaken
                    if signal_direction == level_direction:
                        signal_strength = min(0.99, signal_strength + 0.1)
                    else:
                        # Conflicting signals - go with the stronger one
                        if level_strength > signal_strength:
                            signal_direction = level_direction
                            signal_strength = level_strength * 0.8  # Reduce due to conflict
                        else:
                            signal_strength *= 0.8  # Reduce due to conflict

                details.update({
                    'supply_above': len(supply_above),
                    'demand_below': len(demand_below),
                    'supply_at': len(supply_at),
                    'demand_at': len(demand_at),
                    'level_direction': level_direction,
                    'level_strength': level_strength
                })

        if signal_direction == 0 or signal_strength < self.params["min_signal_strength"]:
            return None

        return {
            'direction': signal_direction,
            'strength': signal_strength,
            'details': details
        }

    async def _calculate_delta_divergence(self) -> float:
        """
        Calculate delta divergence (price movement vs volume delta relationship).

        Returns:
            float: Delta divergence value
        """
        if len(self.delta_history) < 3:
            return 0.0

        # Get recent delta history
        recent_delta = list(self.delta_history)[-self.params["delta_lookback_periods"]:]

        # Calculate delta sum
        delta_sum = sum(item['delta'] for item in recent_delta)

        # Calculate price change
        start_price = recent_delta[0]['close']
        end_price = recent_delta[-1]['close']
        price_change = (end_price - start_price) / start_price

        # Normalize delta sum
        avg_volume = sum(abs(item['delta']) for item in recent_delta) / len(recent_delta)
        if avg_volume == 0:
            return 0.0

        normalized_delta_sum = delta_sum / (avg_volume * len(recent_delta))

        # Calculate divergence: positive when price moves against delta
        # e.g., price goes up but delta is negative, or price goes down but delta is positive
        if abs(price_change) < 0.0001 or abs(normalized_delta_sum) < 0.01:
            return 0.0

        return normalized_delta_sum * (-1 if (price_change > 0) else 1)

    def _is_significant_candle(self, candle: Dict[str, Any]) -> bool:
        """
        Determine if a candle is significant for supply/demand analysis.

        Args:
            candle: Candle data

        Returns:
            bool: True if candle is significant, False otherwise
        """
        # Calculate candle size
        candle_range = candle['high'] - candle['low']
        body_size = abs(candle['close'] - candle['open'])

        # Check body to range ratio
        body_range_ratio = body_size / candle_range if candle_range > 0 else 0

        lookback = self.params.get("volume_lookback", 20)
        recent = list(self.delta_history)[-lookback:]
        avg_volume = (
            sum(item.get("volume", 0) for item in recent) / len(recent)
            if recent
            else 0
        )
        high_volume = candle.get("volume", 0) > avg_volume if avg_volume > 0 else False

        return body_range_ratio > 0.6 and high_volume

    def _get_current_price(self) -> float:
        """
        Get the current price estimate from recent data.

        Returns:
            float: Current price estimate
        """
        # Use most recent delta history item as price reference
        if self.delta_history:
            return self.delta_history[-1]['close']

        # Fallback
        return 0.0

    def _calculate_risk_levels(
        self,
        direction: int,
        components: Dict[str, Dict],
        current_price: float
    ) -> Tuple[float, float]:
        """
        Calculate target and stop distances based on signal components.

        Args:
            direction: Signal direction (1 for buy, -1 for sell)
            components: Signal components
            current_price: Current price

        Returns:
            Tuple[float, float]: Target distance and stop distance as ratios
        """
        # Default risk-reward values
        target_distance = 0.01  # 1% target
        stop_distance = 0.005   # 0.5% stop

        # Adjust based on order book signal
        if 'order_book' in components:
            ob_signal = components['order_book']

            # Stronger signal = wider target
            target_multiplier = 1.0 + (ob_signal['strength'] - 0.5) * 0.5
            target_distance *= target_multiplier

            # If there are icebergs, they can provide a natural stop level
            if 'iceberg_buy_volume' in ob_signal['details'] or 'iceberg_sell_volume' in ob_signal['details']:
                stop_multiplier = 0.8  # Tighter stop
                stop_distance *= stop_multiplier

        # Adjust based on flow signal
        if 'flow' in components:
            flow_signal = components['flow']

            # Sweep events provide strong levels
            if 'sweep_direction' in flow_signal['details'] and flow_signal['details']['sweep_direction'] != 0:
                # Sweeps often lead to stronger moves
                target_multiplier = 1.2
                stop_multiplier = 0.9

                target_distance *= target_multiplier
                stop_distance *= stop_multiplier

            # Delta divergence can indicate reversals
            if 'delta_divergence' in flow_signal['details']:
                div_strength = abs(flow_signal['details'].get('delta_divergence', 0))
                if div_strength > 0.5:
                    # Strong divergence might lead to stronger move
                    target_distance *= 1.1

        # Adjust based on volume structure
        if 'volume_structure' in components:
            vs_signal = components['volume_structure']

            # Supply/demand levels provide natural targets/stops
            if 'supply_above' in vs_signal['details'] or 'demand_below' in vs_signal['details']:
                # If direction is up and we have supply above, adjust target
                if direction > 0 and vs_signal['details'].get('supply_above', 0) > 0:
                    # Target should be near the supply level
                    # This is a simplification - ideally would use actual level price
                    target_distance = 0.008  # 0.8% target

                # If direction is down and we have demand below, adjust target
                elif direction < 0 and vs_signal['details'].get('demand_below', 0) > 0:
                    # Target should be near the demand level
                    target_distance = 0.008  # 0.8% target

        # Ensure minimum values
        target_distance = max(0.003, target_distance)  # At least 0.3%
        stop_distance = max(0.002, stop_distance)      # At least 0.2%

        # Ensure risk-reward ratio is reasonable
        if target_distance / stop_distance < 1.5:
            # Adjust to maintain at least 1.5:1 ratio
            target_distance = stop_distance * 1.5

        return target_distance, stop_distance

    def _get_interval_ms(self) -> int:
        """
        Get the interval in milliseconds for the current timeframe.

        Returns:
            int: Interval in milliseconds
        """
        # Map timeframe to milliseconds
        timeframe_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }

        return timeframe_map.get(self.timeframe, 60 * 1000)  # Default to 1m

    def _get_expiration_minutes(self) -> int:
        """
        Get the appropriate signal expiration time in minutes.

        Returns:
            int: Expiration time in minutes
        """
        # Map timeframe to expiration minutes
        timeframe_map = {
            '1m': 15,
            '3m': 30,
            '5m': 45,
            '15m': 90,
            '30m': 180,
            '1h': 360,
            '2h': 720,
            '4h': 1440,
            '6h': 2160,
            '12h': 4320,
            '1d': 8640
        }

        return timeframe_map.get(self.timeframe, 60)  # Default to 60 minutes

    def _estimate_signal_duration(self, direction: int, components: Dict[str, Dict]) -> int:
        """
        Estimate the expected duration of a signal in minutes.

        Args:
            direction: Signal direction
            components: Signal components

        Returns:
            int: Estimated duration in minutes
        """
        # Base duration on timeframe
        base_duration = self._get_expiration_minutes() / 3

        # Adjust based on signal strength and composition
        duration_multiplier = 1.0

        # Flow signals typically play out faster
        if 'flow' in components:
            flow_strength = components['flow']['strength']
            duration_multiplier *= 0.8

            # Sweep events can lead to quick moves
            if 'sweep_direction' in components['flow']['details']:
                duration_multiplier *= 0.9

        # Structure signals may take longer to play out
        if 'volume_structure' in components:
            vs_strength = components['volume_structure']['strength']
            duration_multiplier *= 1.2

        # Calculate final duration
        estimated_duration = int(base_duration * duration_multiplier)

        # Ensure minimum duration
        return max(5, estimated_duration)

    async def _get_market_context(self) -> Dict[str, Any]:
        """
        Get current market context information.

        Returns:
            Dict[str, Any]: Market context
        """
        return {
            'volatility': await self._estimate_current_volatility(),
            'recent_trend': await self._detect_recent_trend(),
            'current_session': self._determine_current_session(),
            'liquidity_state': self._assess_liquidity_state()
        }

    async def _estimate_current_volatility(self) -> float:
        """
        Estimate current market volatility.

        Returns:
            float: Estimated volatility
        """
        # Simple volatility estimate based on recent price ranges
        if len(self.delta_history) < 10:
            return 0.0

        recent_data = list(self.delta_history)[-20:]
        prices = [item['close'] for item in recent_data]

        if not prices:
            return 0.0

        # Calculate returns
        returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]

        # Calculate standard deviation of returns
        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

        return (variance ** 0.5) * 100  # Convert to percentage

    async def _detect_recent_trend(self) -> str:
        """
        Detect recent market trend.

        Returns:
            str: Trend direction ('up', 'down', 'sideways')
        """
        if len(self.delta_history) < 10:
            return 'unknown'

        recent_data = list(self.delta_history)[-20:]
        prices = [item['close'] for item in recent_data]

        if not prices:
            return 'unknown'

        # Simple trend detection based on start/end price
        start_price = prices[0]
        end_price = prices[-1]
        price_change = (end_price - start_price) / start_price * 100  # percentage

        if price_change > 0.5:
            return 'up'
        elif price_change < -0.5:
            return 'down'
        else:
            return 'sideways'

    def _determine_current_session(self) -> str:
        """
        Determine the current trading session.

        Returns:
            str: Current session ('asian', 'european', 'american', 'weekend')
        """
        # This is a simplified version - would need proper timezone handling
        current_time = datetime.now()
        current_hour = current_time.hour

        # Simplified session logic
        if current_time.weekday() >= 5:
            return 'weekend'
        elif 0 <= current_hour < 8:
            return 'asian'
        elif 8 <= current_hour < 16:
            return 'european'
        else:
            return 'american'

    def _assess_liquidity_state(self) -> str:
        """
        Assess current market liquidity state.

        Returns:
            str: Liquidity state ('high', 'normal', 'low')
        """
        # This would normally use order book depth and spread
        # Simplified implementation for now
        return 'normal'

    async def _reset_session_data(self) -> None:
        """
        Reset session-specific data at session boundaries.
        """
        # Clear volatile data structures
        self.order_book_imbalances.clear()
        self.significant_transactions.clear()
        self.iceberg_detections.clear()
        self.liquidity_events.clear()

        # Reset certain counters
        self.current_ob_imbalance = 0.0

        self.logger.info(f"Session data reset complete")

    async def _load_state(self) -> None:
        """
        Load saved state from storage if available.
        """
        try:
            # Attempt to load state from database
            state_data = await self.market_data.get_brain_state(
                self.asset_id,
                self.timeframe,
                self.BRAIN_TYPE
            )

            if state_data:
                await self.load_state(state_data)
                self.logger.info(f"Loaded saved state for {self.asset_id} on {self.timeframe}")
        except Exception as e:
            self.logger.warning(f"Could not load saved state: {str(e)}")

    async def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate basic momentum signals from recent close prices."""
        if not self.initialized:
            return []
        candles = await self.market_data.get_candles(self.asset_id, self.timeframe, limit=2)
        if len(candles) < 2:
            return []
        prev_close = candles[-2]["close"]
        last_close = candles[-1]["close"]
        timestamp = candles[-1]["timestamp"]
        if last_close > prev_close:
            signal = {"type": "buy", "timestamp": timestamp}
        elif last_close < prev_close:
            signal = {"type": "sell", "timestamp": timestamp}
        else:
            return []
        self.signal_performance["total"] += 1
        return [signal]

    async def on_regime_change(self, new_regime: str) -> None:
        """Respond to market regime shifts by resetting trackers."""
        self.logger.info("Regime changed to %s", new_regime)
        await self._reset_session_data()
        self.current_regime = new_regime


