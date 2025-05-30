#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Strategy Brains Service

This module implements the Strategy Brains Service which manages multiple trading
strategy brains, orchestrates signal generation, and manages adaptive tactics.
"""

import os
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple

from common.logger import get_logger
from common.metrics import MetricsCollector
from common.exceptions import (
    StrategyError, SignalGenerationError, ConfigurationError,
    ServiceStartupError, ServiceShutdownError
)
from common.constants import SIGNAL_TYPES, POSITION_DIRECTION, MARKET_REGIMES
from strategy_brains.base_brain import StrategyBrain
from strategy_brains.trend_brain import TrendBrain
from strategy_brains.mean_reversion_brain import MeanReversionBrain
from strategy_brains.breakout_brain import BreakoutBrain
from strategy_brains.pattern_brain import PatternBrain
from strategy_brains.ml_brain import MLBrain
from strategy_brains.adaptive_brain import AdaptiveBrain


class StrategyBrainService:
    """
    Service for managing trading strategy brains.
    Orchestrates signal generation and manages adaptive tactics.
    """
    
    def __init__(self, config, loop=None, redis_client=None, db_client=None):
        """
        Initialize the Strategy Brains Service.
        
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
        self.logger = get_logger("StrategyBrainService")
        self.metrics = MetricsCollector("strategy_brains")
        
        self.brains = {}
        self.active_brains = set()
        self.running = False
        self.tasks = []
        self.signal_subscribers = []
        
        self.market_regime = MARKET_REGIMES["RANGE_BOUND"]  # Default regime
        self.regime_updated_at = time.time()
        
    async def start(self):
        """Start the Strategy Brains Service."""
        self.logger.info("Starting Strategy Brains Service")
        
        # Initialize strategy brains
        await self._initialize_brains()
        
        # Start signal generation tasks
        signal_interval = self.config.get("strategy_brains.signal_interval", 60)
        self.tasks.append(asyncio.create_task(
            self._signal_generation_loop(signal_interval)
        ))
        
        # Start regime update monitoring
        regime_interval = self.config.get("strategy_brains.regime_update_interval", 300)
        self.tasks.append(asyncio.create_task(
            self._regime_monitoring_loop(regime_interval)
        ))
        
        # Start performance tracking
        performance_interval = self.config.get("strategy_brains.performance_interval", 3600)
        self.tasks.append(asyncio.create_task(
            self._performance_tracking_loop(performance_interval)
        ))
        
        # Subscribe to regime change notifications
        await self._subscribe_to_regime_changes()
        
        self.running = True
        self.logger.info("Strategy Brains Service started successfully")
        self.logger.info("Strategy Brains Service could benefit from asset-specific specialization and council integration")
        
    async def stop(self):
        """Stop the Strategy Brains Service."""
        self.logger.info("Stopping Strategy Brains Service")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        # Stop all brains
        for brain_name, brain in self.brains.items():
            try:
                await brain.stop()
                self.logger.info(f"Stopped brain: {brain_name}")
            except Exception as e:
                self.logger.error(f"Error stopping brain {brain_name}: {str(e)}")
                
        self.logger.info("Strategy Brains Service stopped successfully")
        
    async def health_check(self) -> bool:
        """
        Perform a health check on the Strategy Brains Service.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        if not self.running:
            return False
            
        # Check if all active brains are healthy
        for brain_name in self.active_brains:
            if brain_name in self.brains:
                brain = self.brains[brain_name]
                if not await brain.health_check():
                    self.logger.warning(f"Brain {brain_name} health check failed")
                    return False
                    
        return True
        
    async def _initialize_brains(self):
        """Initialize strategy brains based on configuration."""
        brain_configs = self.config.get("strategy_brains.brains", {})
        
        # Brain class mapping
        brain_classes = {
            "trend": TrendBrain,
            "trend_following": TrendBrain,
            "mean_reversion": MeanReversionBrain,
            "breakout": BreakoutBrain,
            "pattern": PatternBrain,
            "pattern_recognition": PatternBrain,
            "ml": MLBrain,
            "adaptive": AdaptiveBrain,
        }
        
        # Initialize each configured brain
        for brain_name, brain_config in brain_configs.items():
            if not brain_config.get("enabled", True):
                self.logger.info(f"Brain {brain_name} is disabled in configuration")
                continue
                
            brain_type = brain_config.get("type")
            if brain_type not in brain_classes:
                self.logger.error(f"Unknown brain type: {brain_type}")
                continue
                
            try:
                brain_class = brain_classes[brain_type]
                brain = brain_class(
                    config=brain_config,
                    name=brain_name,
                    redis_client=self.redis_client,
                    db_client=self.db_client,
                    loop=self.loop
                )
                
                await brain.initialize()
                self.brains[brain_name] = brain
                
                if brain_config.get("active", True):
                    self.active_brains.add(brain_name)
                    
                self.logger.info(f"Initialized brain: {brain_name} (type: {brain_type})")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize brain {brain_name}: {str(e)}")
                
        self.logger.info(f"Initialized {len(self.brains)} strategy brains")
        
    async def _signal_generation_loop(self, interval: int):
        """
        Main loop for generating signals from all active brains.
        
        Args:
            interval: Signal generation interval in seconds
        """
        self.logger.info(f"Starting signal generation loop (interval: {interval}s)")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Generate signals from all active brains
                signals = []
                for brain_name in self.active_brains:
                    if brain_name in self.brains:
                        brain = self.brains[brain_name]
                        try:
                            # Time the signal generation
                            with self.metrics.timer(f"signal_generation.{brain_name}"):
                                brain_signals = await brain.generate_signals()
                                
                            if brain_signals:
                                signals.extend(brain_signals)
                                self.metrics.increment(f"signals.generated.{brain_name}", len(brain_signals))
                                self.logger.debug(f"Brain {brain_name} generated {len(brain_signals)} signals")
                                
                        except Exception as e:
                            self.logger.error(f"Error generating signals from brain {brain_name}: {str(e)}")
                            self.metrics.increment(f"signals.error.{brain_name}")
                            
                # Process and publish signals
                if signals:
                    await self._process_and_publish_signals(signals)
                    
                # Calculate execution time and sleep for the remainder of the interval
                execution_time = time.time() - start_time
                self.metrics.observe("signal_generation.execution_time", execution_time)
                
                sleep_time = max(0.1, interval - execution_time)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                self.logger.info("Signal generation loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in signal generation loop: {str(e)}")
                await asyncio.sleep(5)  # Sleep briefly before retrying
                
    async def _process_and_publish_signals(self, signals: List[Dict[str, Any]]):
        """
        Process and publish generated signals.
        
        Args:
            signals: List of signal dictionaries
        """
        if not signals:
            return
            
        self.logger.info(f"Processing {len(signals)} signals")
        
        # Add metadata to signals
        timestamp = time.time()
        processed_signals = []
        
        for signal in signals:
            # Add common fields
            signal["timestamp"] = timestamp
            signal["processed_at"] = timestamp
            signal["service"] = "strategy_brains"
            signal["regime"] = self.market_regime
            
            # Validate signal
            if self._validate_signal(signal):
                processed_signals.append(signal)
            else:
                self.logger.warning(f"Skipping invalid signal: {signal}")
                self.metrics.increment("signals.invalid")
                
        # Log and publish signals
        if processed_signals:
            self.logger.info(f"Publishing {len(processed_signals)} valid signals")
            
            # Publish to Redis for other services
            try:
                channel = self.config.get("strategy_brains.signal_channel", "signals")
                for signal in processed_signals:
                    await self.redis_client.publish(channel, signal)
                    
                self.metrics.increment("signals.published", len(processed_signals))
                
            except Exception as e:
                self.logger.error(f"Error publishing signals: {str(e)}")
                self.metrics.increment("signals.publish_error")
                
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate a trading signal.
        
        Args:
            signal: Signal dictionary
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_fields = ["symbol", "exchange", "type", "direction", "confidence"]
        
        # Check for required fields
        for field in required_fields:
            if field not in signal:
                self.logger.warning(f"Signal missing required field: {field}")
                return False
                
        # Validate signal type
        if signal["type"] not in SIGNAL_TYPES.values():
            self.logger.warning(f"Invalid signal type: {signal['type']}")
            return False
            
        # Validate direction
        if signal["direction"] not in POSITION_DIRECTION.values():
            self.logger.warning(f"Invalid signal direction: {signal['direction']}")
            return False
            
        # Validate confidence
        if not isinstance(signal["confidence"], (int, float)) or not (0 <= signal["confidence"] <= 1):
            self.logger.warning(f"Invalid signal confidence: {signal['confidence']}")
            return False
            
        return True
        
    async def _regime_monitoring_loop(self, interval: int):
        """
        Monitor for market regime changes.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.logger.info(f"Starting regime monitoring loop (interval: {interval}s)")
        
        while self.running:
            try:
                # Check if we've recently received a regime update
                time_since_update = time.time() - self.regime_updated_at
                
                # Only query for regime if we haven't received an update recently
                if time_since_update > interval:
                    # Get current market regime from intelligence service
                    try:
                        regime_data = await self._get_current_regime()
                        if regime_data and "regime" in regime_data:
                            self._update_market_regime(regime_data["regime"])
                    except Exception as e:
                        self.logger.error(f"Error getting current market regime: {str(e)}")
                        
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                self.logger.info("Regime monitoring loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in regime monitoring loop: {str(e)}")
                await asyncio.sleep(5)
                
    async def _get_current_regime(self) -> Dict[str, Any]:
        """
        Get the current market regime from the intelligence service.
        
        Returns:
            Dict containing regime information
        """
        try:
            # Query intelligence service via Redis
            channel = "intelligence.market_regime.request"
            response_channel = f"intelligence.market_regime.response.{int(time.time())}"
            
            request = {
                "service": "strategy_brains",
                "request_id": response_channel,
                "response_channel": response_channel,
                "timestamp": time.time()
            }
            
            # Publish request
            await self.redis_client.publish(channel, request)
            
            # Wait for response with timeout
            timeout = self.config.get("strategy_brains.regime_request_timeout", 10)
            response = await self.redis_client.subscribe_and_wait(
                response_channel, timeout=timeout
            )
            
            if response:
                return response
            else:
                self.logger.warning("Timeout waiting for regime response")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting market regime: {str(e)}")
            return {}
            
    async def _subscribe_to_regime_changes(self):
        """Subscribe to market regime change notifications."""
        try:
            channel = "intelligence.market_regime.update"
            
            async def regime_callback(message):
                if message and "regime" in message:
                    self._update_market_regime(message["regime"])
                    
            await self.redis_client.subscribe(channel, regime_callback)
            self.logger.info(f"Subscribed to regime changes on channel: {channel}")
            
        except Exception as e:
            self.logger.error(f"Error subscribing to regime changes: {str(e)}")
            
    def _update_market_regime(self, regime: str):
        """
        Update the current market regime.
        
        Args:
            regime: New market regime
        """
        if regime not in MARKET_REGIMES.values():
            self.logger.warning(f"Ignoring invalid market regime: {regime}")
            return
            
        if regime != self.market_regime:
            self.logger.info(f"Market regime changed: {self.market_regime} -> {regime}")
            self.metrics.increment(f"regime_changes.{self.market_regime}_to_{regime}")
            
            # Update regime
            self.market_regime = regime
            self.regime_updated_at = time.time()
            
            # Notify all brains about regime change
            for brain_name, brain in self.brains.items():
                try:
                    asyncio.create_task(brain.on_regime_change(regime))
                except Exception as e:
                    self.logger.error(f"Error notifying brain {brain_name} of regime change: {str(e)}")
                    
    async def _performance_tracking_loop(self, interval: int):
        """
        Track and update brain performance metrics.
        
        Args:
            interval: Performance tracking interval in seconds
        """
        self.logger.info(f"Starting performance tracking loop (interval: {interval}s)")
        
        while self.running:
            try:
                # Update performance metrics for each brain
                for brain_name, brain in self.brains.items():
                    try:
                        performance = await brain.get_performance_metrics()
                        
                        # Update metrics
                        for metric_name, value in performance.items():
                            self.metrics.set(f"performance.{brain_name}.{metric_name}", value)
                            
                        self.logger.debug(f"Updated performance metrics for brain: {brain_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Error updating performance metrics for brain {brain_name}: {str(e)}")
                        
                # Adjust active brains based on performance if enabled
                if self.config.get("strategy_brains.auto_adjust_active", True):
                    await self._adjust_active_brains()
                    
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                self.logger.info("Performance tracking loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in performance tracking loop: {str(e)}")
                await asyncio.sleep(5)
                
    async def _adjust_active_brains(self):
        """Adjust active brains based on performance metrics."""
        try:
            # Get performance threshold from config
            threshold = self.config.get("strategy_brains.performance_threshold", 0.3)
            
            # Calculate performance for each brain
            performances = {}
            for brain_name, brain in self.brains.items():
                metrics = await brain.get_performance_metrics()
                if "overall_score" in metrics:
                    performances[brain_name] = metrics["overall_score"]
                    
            if not performances:
                return
                
            # Find the best and worst performing brains
            best_brain = max(performances.items(), key=lambda x: x[1])
            worst_brain = min(performances.items(), key=lambda x: x[1])
            
            self.logger.debug(f"Best performing brain: {best_brain[0]} (score: {best_brain[1]})")
            self.logger.debug(f"Worst performing brain: {worst_brain[0]} (score: {worst_brain[1]})")
            
            # If the worst brain is significantly underperforming, deactivate it
            if worst_brain[0] in self.active_brains and worst_brain[1] < threshold:
                self.logger.info(f"Deactivating underperforming brain: {worst_brain[0]} (score: {worst_brain[1]})")
                self.active_brains.remove(worst_brain[0])
                self.metrics.increment("brain_adjustments.deactivated")
                
            # If the best brain is not active, activate it
            if best_brain[0] not in self.active_brains:
                self.logger.info(f"Activating high-performing brain: {best_brain[0]} (score: {best_brain[1]})")
                self.active_brains.add(best_brain[0])
                self.metrics.increment("brain_adjustments.activated")
                
        except Exception as e:
            self.logger.error(f"Error adjusting active brains: {str(e)}")


async def create_app(config: Dict[str, Any]) -> StrategyBrainService:
    """
    Create and configure a Strategy Brains Service instance.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured StrategyBrainService instance
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
    return StrategyBrainService(
        config=config,
        redis_client=redis_client,
        db_client=db_client
    )
