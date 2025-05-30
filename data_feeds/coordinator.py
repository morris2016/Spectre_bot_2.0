#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Feed Coordinator

This module implements the Feed Coordinator that manages all data feeds,
coordinates data flow, handles redundancy, and ensures data quality.
"""

import os
import asyncio
import signal
import time
from typing import Dict, List, Set, Any, Optional, Union, Tuple
import threading
import logging
import uuid
import json
from datetime import datetime, timedelta
import heapq
from collections import defaultdict

# Internal imports
from config import Config
from common.logger import get_logger
from common.constants import (
    SERVICE_NAMES, EXCHANGE_NAMES, FEED_TYPES, FEED_STATUS,
    DATA_PRIORITY_LEVELS, TIMEFRAMES, ASSET_TYPES
)
from common.exceptions import (
    FeedCoordinationError, DataQualityError, 
    RedundancyFailureError, FeedPriorityError
)
from common.utils import retry_with_backoff_decorator, merge_deep
from common.metrics import MetricsCollector
from common.redis_client import RedisClient
from data_feeds.base_feed import BaseFeed

class FeedCoordinator:
    """
    Feed Coordinator for managing and coordinating multiple data feeds.
    
    This component is responsible for:
    1. Prioritizing data sources when multiple feeds provide the same data
    2. Handling data quality issues and applying corrections
    3. Managing feed redundancy and failover
    4. Coordinating data flow across the system
    5. Optimizing performance based on data usage patterns
    """
    
    def __init__(self, config: Config, logger: logging.Logger, 
                 metrics: MetricsCollector, redis: RedisClient):
        """
        Initialize the Feed Coordinator.
        
        Args:
            config: System configuration
            logger: Logger instance
            metrics: Metrics collector
            redis: Redis client
        """
        self.coordinator_id = f"feed_coordinator_{uuid.uuid4().hex[:8]}"
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.redis = redis
        
        # Coordinator state
        self.running = False
        self.shutdown_event = asyncio.Event()
        self.maintenance_mode = False
        
        # Feed registry
        self.feeds: Dict[str, BaseFeed] = {}
        self.feed_by_source: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self.feed_priority: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        
        # Data quality
        self.data_quality_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.anomaly_detection_thresholds = config.data_feeds.get('anomaly_thresholds', {})
        
        # Feed usage statistics
        self.feed_usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Data subscriber registry
        self.subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # Background tasks
        self.tasks = {}
        
        self.logger.info(f"Feed Coordinator initialized with ID: {self.coordinator_id}")
    
    async def start(self) -> bool:
        """
        Start the Feed Coordinator.
        
        Returns:
            bool: True if startup successful, False otherwise
        """
        if self.running:
            self.logger.warning("Coordinator already running, ignoring start request")
            return True
        
        try:
            self.logger.info("Starting Feed Coordinator")
            self.running = True
            
            # Initialize metrics
            self.metrics.register_gauge("active_feeds", "Number of active feeds managed by coordinator")
            self.metrics.register_gauge("data_quality", "Data quality score", ["feed_name", "data_type"])
            self.metrics.register_counter("feed_failovers", "Number of feed failovers", ["source", "destination"])
            self.metrics.register_counter("anomalies_detected", "Number of data anomalies detected", ["feed_name", "data_type"])
            self.metrics.register_histogram("coordination_latency", "Feed coordination latency in ms", ["operation"])
            
            # Start background tasks
            self._start_background_tasks()
            
            # Initialize feed registry from config if available
            await self._load_feed_registry()
            
            # Initialize data quality baselines
            await self._initialize_data_quality_baselines()
            
            self.logger.info("Feed Coordinator startup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Feed Coordinator: {str(e)}", exc_info=True)
            self.running = False
            return False
    
    async def stop(self) -> bool:
        """
        Stop the Feed Coordinator.
        
        Returns:
            bool: True if shutdown successful, False otherwise
        """
        if not self.running:
            self.logger.warning("Coordinator not running, ignoring stop request")
            return True
        
        self.logger.info("Stopping Feed Coordinator")
        self.running = False
        self.shutdown_event.set()
        
        try:
            # Stop all background tasks
            for task_name, task in self.tasks.items():
                if not task.done():
                    self.logger.info(f"Cancelling task: {task_name}")
                    task.cancel()
            
            # Wait for tasks to complete
            for task_name, task in self.tasks.items():
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Task {task_name} did not complete in time")
                except asyncio.CancelledError:
                    pass
                
            # Save feed registry and stats
            await self._save_feed_registry()
            
            self.logger.info("Feed Coordinator shutdown complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during Feed Coordinator shutdown: {str(e)}", exc_info=True)
            return False
    
    def register_feed(self, feed: BaseFeed) -> bool:
        """
        Register a feed with the coordinator.
        
        Args:
            feed: The feed instance to register
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            feed_name = feed.name
            feed_type = feed.feed_type
            
            if feed_name in self.feeds:
                self.logger.warning(f"Feed {feed_name} already registered, updating")
            
            self.feeds[feed_name] = feed
            
            # Register feed by its data sources
            for source in feed.data_sources:
                for data_type in feed.supported_data_types:
                    self.feed_by_source[source][data_type].add(feed_name)
                    
                    # Set default priority if not already set
                    if feed_name not in self.feed_priority[source][data_type]:
                        priority = feed.get_priority_for_data_type(data_type)
                        self.feed_priority[source][data_type][feed_name] = priority
            
            # Initialize usage stats
            self.feed_usage_stats[feed_name] = {
                'last_accessed': datetime.utcnow().isoformat(),
                'data_requested': 0,
                'data_provided': 0,
                'data_quality_issues': 0,
                'subscribers': 0
            }
            
            # Initialize data quality scores
            for data_type in feed.supported_data_types:
                if feed_name not in self.data_quality_scores[data_type]:
                    self.data_quality_scores[data_type][feed_name] = 1.0  # Start with perfect score
            
            self.metrics.set_gauge("active_feeds", len(self.feeds))
            self.logger.info(f"Registered feed {feed_name} of type {feed_type}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register feed {feed.name}: {str(e)}", exc_info=True)
            return False
    
    def unregister_feed(self, feed_name: str) -> bool:
        """
        Unregister a feed from the coordinator.
        
        Args:
            feed_name: Name of the feed to unregister
            
        Returns:
            bool: True if unregistration successful, False otherwise
        """
        if feed_name not in self.feeds:
            self.logger.warning(f"Feed {feed_name} not found, cannot unregister")
            return False
        
        try:
            feed = self.feeds[feed_name]
            
            # Remove from feed registry
            del self.feeds[feed_name]
            
            # Remove from source mappings
            for source in feed.data_sources:
                for data_type in feed.supported_data_types:
                    if feed_name in self.feed_by_source[source][data_type]:
                        self.feed_by_source[source][data_type].remove(feed_name)
                    if feed_name in self.feed_priority[source][data_type]:
                        del self.feed_priority[source][data_type][feed_name]
            
            # Clean up empty sets
            for source in list(self.feed_by_source.keys()):
                for data_type in list(self.feed_by_source[source].keys()):
                    if not self.feed_by_source[source][data_type]:
                        del self.feed_by_source[source][data_type]
                if not self.feed_by_source[source]:
                    del self.feed_by_source[source]
            
            # Remove from data quality scores
            for data_type in list(self.data_quality_scores.keys()):
                if feed_name in self.data_quality_scores[data_type]:
                    del self.data_quality_scores[data_type][feed_name]
            
            # Remove usage stats
            if feed_name in self.feed_usage_stats:
                del self.feed_usage_stats[feed_name]
            
            self.metrics.set_gauge("active_feeds", len(self.feeds))
            self.logger.info(f"Unregistered feed {feed_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister feed {feed_name}: {str(e)}", exc_info=True)
            return False
    
    async def get_feed_for_data(self, source: str, data_type: str, 
                               asset: Optional[str] = None) -> Optional[BaseFeed]:
        """
        Get the best feed to provide a specific type of data.
        
        Args:
            source: Data source (e.g., 'binance', 'news')
            data_type: Type of data needed (e.g., 'candles', 'orderbook')
            asset: Specific asset if applicable
            
        Returns:
            Optional[BaseFeed]: The best feed for the requested data or None if not available
        """
        start_time = time.time()
        
        try:
            # Check if any feeds are available for this data
            if source not in self.feed_by_source or data_type not in self.feed_by_source[source]:
                self.logger.warning(f"No feeds available for {source}/{data_type}")
                return None
            
            available_feeds = self.feed_by_source[source][data_type]
            if not available_feeds:
                return None
            
            # Filter feeds by asset if specified
            if asset:
                asset_feeds = []
                for feed_name in available_feeds:
                    feed = self.feeds[feed_name]
                    if asset in feed.supported_assets:
                        asset_feeds.append(feed_name)
                available_feeds = asset_feeds
            
            if not available_feeds:
                self.logger.warning(f"No feeds available for {source}/{data_type}/{asset}")
                return None
            
            # Get best feed based on priority and data quality
            best_feed_name = await self._select_best_feed(source, data_type, available_feeds)
            
            if best_feed_name and best_feed_name in self.feeds:
                # Update usage statistics
                if best_feed_name in self.feed_usage_stats:
                    self.feed_usage_stats[best_feed_name]['last_accessed'] = datetime.utcnow().isoformat()
                    self.feed_usage_stats[best_feed_name]['data_requested'] += 1
                
                return self.feeds[best_feed_name]
            
            return None
            
        finally:
            coordination_time = (time.time() - start_time) * 1000
            self.metrics.observe_histogram("coordination_latency", coordination_time, 
                                          labels={"operation": "get_feed_for_data"})
    
    async def subscribe_to_data(self, subscriber_id: str, source: str, 
                               data_type: str, asset: Optional[str] = None) -> bool:
        """
        Subscribe to a specific type of data.
        
        Args:
            subscriber_id: Unique ID of the subscriber
            source: Data source
            data_type: Type of data
            asset: Specific asset if applicable
            
        Returns:
            bool: True if subscription successful, False otherwise
        """
        subscription_key = f"{source}:{data_type}:{asset if asset else '*'}"
        
        try:
            # Register subscriber
            self.subscribers[subscription_key].add(subscriber_id)
            
            # Update feed usage stats for relevant feeds
            feeds_for_data = self.feed_by_source.get(source, {}).get(data_type, set())
            for feed_name in feeds_for_data:
                if feed_name in self.feed_usage_stats:
                    self.feed_usage_stats[feed_name]['subscribers'] += 1
            
            # Log and return
            self.logger.info(f"Subscriber {subscriber_id} subscribed to {subscription_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {subscription_key}: {str(e)}", exc_info=True)
            return False
    
    async def unsubscribe_from_data(self, subscriber_id: str, source: str, 
                                  data_type: str, asset: Optional[str] = None) -> bool:
        """
        Unsubscribe from a specific type of data.
        
        Args:
            subscriber_id: Unique ID of the subscriber
            source: Data source
            data_type: Type of data
            asset: Specific asset if applicable
            
        Returns:
            bool: True if unsubscription successful, False otherwise
        """
        subscription_key = f"{source}:{data_type}:{asset if asset else '*'}"
        
        try:
            # Unregister subscriber
            if subscription_key in self.subscribers and subscriber_id in self.subscribers[subscription_key]:
                self.subscribers[subscription_key].remove(subscriber_id)
                
                # Clean up empty sets
                if not self.subscribers[subscription_key]:
                    del self.subscribers[subscription_key]
                
                # Update feed usage stats
                feeds_for_data = self.feed_by_source.get(source, {}).get(data_type, set())
                for feed_name in feeds_for_data:
                    if feed_name in self.feed_usage_stats and self.feed_usage_stats[feed_name]['subscribers'] > 0:
                        self.feed_usage_stats[feed_name]['subscribers'] -= 1
                
                self.logger.info(f"Subscriber {subscriber_id} unsubscribed from {subscription_key}")
                return True
                
            self.logger.warning(f"Subscriber {subscriber_id} not found for {subscription_key}")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from {subscription_key}: {str(e)}", exc_info=True)
            return False
    
    async def update_data_quality(self, feed_name: str, data_type: str, quality_score: float) -> None:
        """
        Update the data quality score for a feed.
        
        Args:
            feed_name: Name of the feed
            data_type: Type of data
            quality_score: New quality score (0.0 to 1.0)
        """
        if feed_name not in self.feeds:
            self.logger.warning(f"Feed {feed_name} not found, cannot update quality score")
            return
        
        # Apply smoothing to prevent rapid fluctuations
        smoothing_factor = self.config.data_feeds.get('quality_smoothing_factor', 0.8)
        current_score = self.data_quality_scores.get(data_type, {}).get(feed_name, 1.0)
        new_score = (current_score * smoothing_factor) + (quality_score * (1 - smoothing_factor))
        
        # Update score
        self.data_quality_scores[data_type][feed_name] = new_score
        
        # Update metrics
        self.metrics.set_gauge("data_quality", new_score, labels={"feed_name": feed_name, "data_type": data_type})
        
        # Check if score is below critical threshold
        critical_threshold = self.config.data_feeds.get('quality_critical_threshold', 0.3)
        if new_score < critical_threshold:
            self.logger.warning(f"Feed {feed_name} data quality for {data_type} below critical threshold: {new_score:.2f}")
            # Mark for potential failover
            await self._handle_feed_quality_critical(feed_name, data_type)
    
    async def report_data_anomaly(self, feed_name: str, data_type: str, anomaly_details: Dict[str, Any]) -> None:
        """
        Report a data anomaly detected in a feed.
        
        Args:
            feed_name: Name of the feed
            data_type: Type of data
            anomaly_details: Details about the anomaly
        """
        if feed_name not in self.feeds:
            self.logger.warning(f"Feed {feed_name} not found, cannot report anomaly")
            return
        
        self.logger.warning(f"Data anomaly detected in {feed_name} for {data_type}: {anomaly_details}")
        
        # Increment anomaly counter
        self.metrics.increment_counter("anomalies_detected", labels={"feed_name": feed_name, "data_type": data_type})
        
        # Update feed usage stats
        if feed_name in self.feed_usage_stats:
            self.feed_usage_stats[feed_name]['data_quality_issues'] += 1
        
        # Publish anomaly notification
        await self.redis.publish(
            "notifications:anomalies",
            json.dumps({
                "feed": feed_name,
                "data_type": data_type,
                "details": anomaly_details,
                "timestamp": datetime.utcnow().isoformat()
            })
        )
        
        # Apply quality penalty
        penalty = self.config.data_feeds.get('anomaly_penalty_factor', 0.1)
        current_score = self.data_quality_scores.get(data_type, {}).get(feed_name, 1.0)
        new_score = max(0.0, current_score - penalty)
        
        # Update quality score
        await self.update_data_quality(feed_name, data_type, new_score)
    
    async def set_maintenance_mode(self, enabled: bool) -> None:
        """
        Set the coordinator maintenance mode.
        
        Args:
            enabled: True to enable maintenance mode, False to disable
        """
        if self.maintenance_mode == enabled:
            return
            
        self.maintenance_mode = enabled
        self.logger.info(f"Maintenance mode {'enabled' if enabled else 'disabled'}")
        
        # Adjust operations based on maintenance mode
        if enabled:
            # Reduce polling frequency, disable non-critical tasks
            for task_name, task in self.tasks.items():
                if task_name.startswith('optimization') and not task.done():
                    self.logger.info(f"Cancelling non-critical task during maintenance: {task_name}")
                    task.cancel()
        else:
            # Restore normal operations
            self._start_background_tasks()
    
    async def get_feed_stats(self, feed_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get usage statistics for feeds.
        
        Args:
            feed_name: Name of a specific feed or None for all feeds
            
        Returns:
            Dict containing feed statistics
        """
        if feed_name:
            return self.feed_usage_stats.get(feed_name, {})
        return self.feed_usage_stats
    
    async def _select_best_feed(self, source: str, data_type: str, 
                               available_feeds: Set[str]) -> Optional[str]:
        """
        Select the best feed based on priority and data quality.
        
        Args:
            source: Data source
            data_type: Type of data
            available_feeds: Set of available feed names
            
        Returns:
            Optional[str]: Name of the best feed or None if no suitable feed found
        """
        if not available_feeds:
            return None
        
        # Get feeds sorted by priority (highest first)
        priority_dict = self.feed_priority.get(source, {}).get(data_type, {})
        feeds_by_priority = sorted(
            [(feed, priority_dict.get(feed, 0)) for feed in available_feeds],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Consider data quality scores
        quality_weight = self.config.data_feeds.get('quality_priority_weight', 0.7)
        priority_weight = 1.0 - quality_weight
        
        # Calculate combined score (priority and quality)
        feed_scores = []
        max_priority = max([p for _, p in feeds_by_priority]) if feeds_by_priority else 1
        
        for feed_name, priority in feeds_by_priority:
            # Skip feeds that are not healthy
            if feed_name in self.feeds and not self.feeds[feed_name].is_healthy:
                continue
                
            # Normalize priority to 0-1 range
            norm_priority = priority / max_priority if max_priority > 0 else 0
            
            # Get quality score
            quality = self.data_quality_scores.get(data_type, {}).get(feed_name, 1.0)
            
            # Combined score
            combined_score = (norm_priority * priority_weight) + (quality * quality_weight)
            
            feed_scores.append((feed_name, combined_score))
        
        # Sort by combined score (highest first)
        feed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best feed
        return feed_scores[0][0] if feed_scores else None
    
    async def _handle_feed_quality_critical(self, feed_name: str, data_type: str) -> None:
        """
        Handle a feed with critically low data quality.
        
        Args:
            feed_name: Name of the feed
            data_type: Type of data with quality issues
        """
        feed = self.feeds.get(feed_name)
        if not feed:
            return
            
        source = None
        for src in feed.data_sources:
            if data_type in self.feed_by_source.get(src, {}) and feed_name in self.feed_by_source[src].get(data_type, set()):
                source = src
                break
        
        if not source:
            return
        
        # Find alternative feeds
        alternative_feeds = self.feed_by_source.get(source, {}).get(data_type, set()) - {feed_name}
        
        if not alternative_feeds:
            self.logger.warning(f"No alternative feeds available for {source}/{data_type}")
            return
        
        # Get best alternative
        alternative = await self._select_best_feed(source, data_type, alternative_feeds)
        
        if alternative:
            # Log and notify about failover
            self.logger.info(f"Data quality failover from {feed_name} to {alternative} for {source}/{data_type}")
            self.metrics.increment_counter("feed_failovers", labels={"source": feed_name, "destination": alternative})
            
            # Publish failover notification
            await self.redis.publish(
                "notifications:failovers",
                json.dumps({
                    "source_feed": feed_name,
                    "destination_feed": alternative,
                    "data_type": data_type,
                    "reason": "data_quality_critical",
                    "timestamp": datetime.utcnow().isoformat()
                })
            )
    
    def _start_background_tasks(self) -> None:
        """
        Start background tasks for feed coordination.
        """
        # Only start tasks if not already running
        for task_name, task in self.tasks.items():
            if not task.done():
                continue
        
        # Start quality monitoring task
        self.tasks['quality_monitoring'] = asyncio.create_task(
            self._quality_monitoring_task()
        )
        self.tasks['quality_monitoring'].set_name("feed_coordinator_quality_monitoring")
        
        # Start feed optimization task
        self.tasks['optimization'] = asyncio.create_task(
            self._feed_optimization_task()
        )
        self.tasks['optimization'].set_name("feed_coordinator_optimization")
        
        # Start registry persistence task
        self.tasks['persistence'] = asyncio.create_task(
            self._registry_persistence_task()
        )
        self.tasks['persistence'].set_name("feed_coordinator_persistence")
    
    async def _quality_monitoring_task(self) -> None:
        """
        Task to monitor data quality and detect anomalies.
        """
        quality_check_interval = self.config.data_feeds.get('quality_check_interval', 60)
        
        self.logger.info("Starting data quality monitoring task")
        
        while self.running:
            try:
                # Check data quality for all active feeds
                for feed_name, feed in self.feeds.items():
                    if not feed.is_healthy:
                        continue
                        
                    # Get quality measurements from feed
                    quality_metrics = await feed.get_data_quality_metrics()
                    
                    for data_type, metrics in quality_metrics.items():
                        # Check for anomalies
                        anomalies = await self._detect_anomalies(feed_name, data_type, metrics)
                        
                        if anomalies:
                            # Report each anomaly
                            for anomaly in anomalies:
                                await self.report_data_anomaly(feed_name, data_type, anomaly)
                        
                        # Update quality score
                        quality_score = metrics.get('quality_score', None)
                        if quality_score is not None:
                            await self.update_data_quality(feed_name, data_type, quality_score)
                
            except Exception as e:
                self.logger.error(f"Error in quality monitoring task: {str(e)}", exc_info=True)
                
            await asyncio.sleep(quality_check_interval)
    
    async def _detect_anomalies(self, feed_name: str, data_type: str, 
                               metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in feed data.
        
        Args:
            feed_name: Name of the feed
            data_type: Type of data
            metrics: Quality metrics
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Get thresholds from config
        thresholds = self.anomaly_detection_thresholds.get(data_type, {})
        
        # Check for missing data anomaly
        if 'data_gaps' in metrics and metrics['data_gaps'] > thresholds.get('max_data_gaps', 10):
            anomalies.append({
                'type': 'data_gaps',
                'value': metrics['data_gaps'],
                'threshold': thresholds.get('max_data_gaps', 10),
                'severity': 'high' if metrics['data_gaps'] > thresholds.get('max_data_gaps', 10) * 2 else 'medium'
            })
        
        # Check for latency anomaly
        if 'latency_ms' in metrics and metrics['latency_ms'] > thresholds.get('max_latency_ms', 5000):
            anomalies.append({
                'type': 'high_latency',
                'value': metrics['latency_ms'],
                'threshold': thresholds.get('max_latency_ms', 5000),
                'severity': 'high' if metrics['latency_ms'] > thresholds.get('max_latency_ms', 5000) * 2 else 'medium'
            })
        
        # Check for data inconsistency anomaly
        if 'data_inconsistency' in metrics and metrics['data_inconsistency'] > thresholds.get('max_inconsistency', 0.05):
            anomalies.append({
                'type': 'data_inconsistency',
                'value': metrics['data_inconsistency'],
                'threshold': thresholds.get('max_inconsistency', 0.05),
                'severity': 'high' if metrics['data_inconsistency'] > thresholds.get('max_inconsistency', 0.05) * 2 else 'medium'
            })
        
        # Check for price deviation anomaly (for price data)
        if data_type in ['candles', 'ticks', 'orderbook'] and 'price_deviation' in metrics:
            max_deviation = thresholds.get('max_price_deviation', 0.1)
            if metrics['price_deviation'] > max_deviation:
                anomalies.append({
                    'type': 'price_deviation',
                    'value': metrics['price_deviation'],
                    'threshold': max_deviation,
                    'severity': 'high' if metrics['price_deviation'] > max_deviation * 2 else 'medium'
                })
        
        return anomalies
    
    async def _feed_optimization_task(self) -> None:
        """
        Task to optimize feed priorities and usage based on performance.
        """
        optimization_interval = self.config.data_feeds.get('optimization_interval', 300)
        
        self.logger.info("Starting feed optimization task")
        
        while self.running:
            try:
                if self.maintenance_mode:
                    await asyncio.sleep(optimization_interval)
                    continue
                
                # Analyze feed performance
                feed_performance = {}
                for feed_name, feed in self.feeds.items():
                    # Get feed performance metrics
                    performance = await feed.get_performance_metrics()
                    feed_performance[feed_name] = performance
                
                # Optimize feed priorities
                await self._optimize_feed_priorities(feed_performance)
                
                # Optimize data retention based on usage patterns
                await self._optimize_data_retention()
                
            except Exception as e:
                self.logger.error(f"Error in feed optimization task: {str(e)}", exc_info=True)
                
            await asyncio.sleep(optimization_interval)
    
    async def _optimize_feed_priorities(self, feed_performance: Dict[str, Dict[str, Any]]) -> None:
        """
        Optimize feed priorities based on performance.
        
        Args:
            feed_performance: Dictionary of feed performance metrics
        """
        # For each feed type and data type, adjust priorities
        for source in self.feed_by_source:
            for data_type in self.feed_by_source[source]:
                feeds = list(self.feed_by_source[source][data_type])
                
                # Skip if fewer than 2 feeds (no optimization needed)
                if len(feeds) < 2:
                    continue
                
                # Calculate performance score for each feed
                feed_scores = []
                for feed_name in feeds:
                    if feed_name not in feed_performance:
                        continue
                        
                    # Calculate score based on latency, reliability, and quality
                    perf = feed_performance[feed_name]
                    
                    # Normalize latency (lower is better)
                    latency_score = 1.0 - min(1.0, perf.get('avg_latency_ms', 0) / 5000.0)
                    
                    # Reliability score
                    reliability_score = perf.get('uptime_percentage', 100) / 100.0
                    
                    # Quality score
                    quality_score = self.data_quality_scores.get(data_type, {}).get(feed_name, 1.0)
                    
                    # Combined score with weights
                    combined_score = (
                        latency_score * 0.3 + 
                        reliability_score * 0.4 + 
                        quality_score * 0.3
                    )
                    
                    feed_scores.append((feed_name, combined_score))
                
                # Sort feeds by score (highest first)
                feed_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Update priorities
                for i, (feed_name, score) in enumerate(feed_scores):
                    # Higher score = higher priority (invert index)
                    new_priority = len(feed_scores) - i
                    
                    # Only update if priority changed significantly
                    current_priority = self.feed_priority[source][data_type].get(feed_name, 0)
                    if abs(new_priority - current_priority) >= 1:
                        self.feed_priority[source][data_type][feed_name] = new_priority
                        self.logger.info(f"Updated priority for {feed_name} ({source}/{data_type}): {current_priority} -> {new_priority}")
    
    async def _optimize_data_retention(self) -> None:
        """
        Optimize data retention policies based on usage patterns.
        """
        # For each feed, adjust data retention based on usage
        for feed_name, feed in self.feeds.items():
            if not hasattr(feed, 'set_data_retention_policy') or not callable(feed.set_data_retention_policy):
                continue
                
            usage = self.feed_usage_stats.get(feed_name, {})
            
            # Skip feeds with no usage data
            if 'data_requested' not in usage:
                continue
                
            # Calculate usage intensity
            requests = usage.get('data_requested', 0)
            subscribers = usage.get('subscribers', 0)
            
            # Define retention tiers based on usage
            if subscribers > 5 or requests > 1000:
                # High usage - extended retention
                policy = 'extended'
            elif subscribers > 2 or requests > 100:
                # Medium usage - standard retention
                policy = 'standard'
            else:
                # Low usage - minimal retention
                policy = 'minimal'
                
            # Apply retention policy
            try:
                await feed.set_data_retention_policy(policy)
            except Exception as e:
                self.logger.error(f"Failed to set retention policy for {feed_name}: {str(e)}")
    
    async def _registry_persistence_task(self) -> None:
        """
        Task to periodically persist the feed registry and statistics.
        """
        persistence_interval = self.config.data_feeds.get('registry_persistence_interval', 300)
        
        self.logger.info("Starting registry persistence task")
        
        while self.running:
            try:
                # Save feed registry and stats
                await self._save_feed_registry()
                
            except Exception as e:
                self.logger.error(f"Error in registry persistence task: {str(e)}", exc_info=True)
                
            await asyncio.sleep(persistence_interval)
    
    async def _save_feed_registry(self) -> None:
        """
        Save the feed registry and statistics to Redis.
        """
        try:
            # Prepare registry data
            registry_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'feed_priority': self.feed_priority,
                'data_quality_scores': self.data_quality_scores,
                'feed_usage_stats': self.feed_usage_stats,
                'feed_count': len(self.feeds),
                'coordinator_id': self.coordinator_id
            }
            
            # Save to Redis
            await self.redis.set(
                'feed_coordinator:registry',
                json.dumps(registry_data),
                ex=86400  # 24 hours expiry
            )
            
            self.logger.debug("Feed registry saved to Redis")
            
        except Exception as e:
            self.logger.error(f"Failed to save feed registry: {str(e)}", exc_info=True)
    
    async def _load_feed_registry(self) -> None:
        """
        Load the feed registry and statistics from Redis.
        """
        try:
            # Get registry data from Redis
            registry_json = await self.redis.get('feed_coordinator:registry')
            
            if not registry_json:
                self.logger.info("No feed registry found in Redis, starting fresh")
                return
                
            registry_data = json.loads(registry_json)
            
            # Load registry data
            if 'feed_priority' in registry_data:
                # Convert string keys to appropriate types
                self.feed_priority = defaultdict(lambda: defaultdict(dict))
                for source, data_types in registry_data['feed_priority'].items():
                    for data_type, feeds in data_types.items():
                        for feed_name, priority in feeds.items():
                            self.feed_priority[source][data_type][feed_name] = priority
            
            if 'data_quality_scores' in registry_data:
                self.data_quality_scores = defaultdict(dict)
                for data_type, feeds in registry_data['data_quality_scores'].items():
                    for feed_name, score in feeds.items():
                        self.data_quality_scores[data_type][feed_name] = score
            
            if 'feed_usage_stats' in registry_data:
                self.feed_usage_stats = registry_data['feed_usage_stats']
                
            self.logger.info(f"Feed registry loaded from Redis, last updated: {registry_data.get('timestamp', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Failed to load feed registry: {str(e)}", exc_info=True)
    
    async def _initialize_data_quality_baselines(self) -> None:
        """
        Initialize data quality baselines for feeds.
        """
        self.logger.info("Initializing data quality baselines")
        
        try:
            # For each feed, initialize quality baselines if not already set
            for feed_name, feed in self.feeds.items():
                for data_type in feed.supported_data_types:
                    if data_type not in self.data_quality_scores or feed_name not in self.data_quality_scores[data_type]:
                        # Set initial quality score to 1.0 (perfect)
                        self.data_quality_scores[data_type][feed_name] = 1.0
                        
                        # Update metrics
                        self.metrics.set_gauge("data_quality", 1.0, 
                                             labels={"feed_name": feed_name, "data_type": data_type})
        
        except Exception as e:
            self.logger.error(f"Failed to initialize data quality baselines: {str(e)}", exc_info=True)

