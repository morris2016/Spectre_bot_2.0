#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Data Feeds Service

This module implements the main Data Feeds Service, responsible for managing
connections to various data sources and providing real-time market data.
"""

import os
import sys
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional

from common.logger import get_logger
from common.metrics import MetricsCollector
from common.event_bus import EventBus
from common.exceptions import (
    FeedError,
    FeedConnectionError,
    FeedAuthenticationError,
    ServiceStartupError,
    ServiceShutdownError,
)

from data_feeds.base_feed import BaseFeed
from data_feeds.binance_feed import BinanceFeed
from data_feeds.deriv_feed import DerivFeed, DerivCredentials, DerivFeedOptions


class DataFeedService:
    """Service for managing data feeds from various sources."""

    def __init__(self, config, loop=None, redis_client=None, db_client=None, event_bus: Optional[EventBus] = None):
        """
        Initialize the data feed service.
        
        Args:
            config: Configuration object
            loop: Optional asyncio event loop
            redis_client: Optional Redis client for data publishing
            db_client: Optional database client
        """
        self.config = config
        self.loop = loop or asyncio.get_event_loop()
        self.redis_client = redis_client
        self.db_client = db_client
        self.event_bus = event_bus or EventBus.get_instance()
        self.logger = get_logger("DataFeedService")
        
        # Feed state
        self.feeds = {}
        self.running = False
        self.shutting_down = False
        self.task = None
        
        # Metrics collector
        self.metrics = MetricsCollector("data_feeds")
    
    async def start(self):
        """Start the data feed service."""
        self.logger.info("Starting Data Feeds Service")
        
        # Initialize configured feeds
        await self._initialize_feeds()
        
        # Start all feeds
        await self._start_feeds()
        
        # Start monitoring task
        self.task = asyncio.create_task(self._monitor_feeds())
        
        self.running = True
        self.logger.info("Data Feeds Service started successfully")
    
    async def stop(self):
        """Stop the data feed service."""
        self.logger.info("Stopping Data Feeds Service")
        self.shutting_down = True
        
        # Cancel monitoring task
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        # Stop all feeds
        await self._stop_feeds()
        
        self.running = False
        self.logger.info("Data Feeds Service stopped successfully")
    
    async def health_check(self):
        """Perform health check on all feeds."""
        if not self.running:
            return False
        
        # Check all feeds
        all_healthy = True
        for feed_name, feed in self.feeds.items():
            try:
                feed_healthy = await feed.health_check()
                if not feed_healthy:
                    self.logger.warning(f"{feed_name} feed health check failed")
                    all_healthy = False
            except Exception as e:
                self.logger.error(f"Error checking health of {feed_name} feed: {str(e)}")
                all_healthy = False
        
        return all_healthy
    
    async def _initialize_feeds(self):
        """Initialize configured feeds."""
        feed_configs = self.config.data_feeds
        
        # Initialize Binance feed if enabled
        if feed_configs.get("binance", {}).get("enabled", False):
            self.logger.info("Initializing Binance feed")
            try:
                binance_feed = BinanceFeed(
                    config=feed_configs.get("binance", {}),
                    loop=self.loop,
                    redis_client=self.redis_client
                )
                self.feeds["binance"] = binance_feed
            except Exception as e:
                self.logger.error(f"Failed to initialize Binance feed: {str(e)}")
                self.metrics.increment("feed.init.error")
        
        # Initialize Deriv feed if enabled
        if feed_configs.get("deriv", {}).get("enabled", False):
            self.logger.info("Initializing Deriv feed")
            try:
                deriv_conf = feed_configs.get("deriv", {})
                credentials = DerivCredentials(
                    app_id=deriv_conf.get("app_id", ""),
                    api_token=deriv_conf.get("api_token"),
                    account_id=deriv_conf.get("account_id")
                )
                options = DerivFeedOptions(
                    ping_interval=deriv_conf.get("websocket", {}).get("ping_interval", 30),
                    subscription_timeout=deriv_conf.get("websocket", {}).get("ping_timeout", 10)
                )
                deriv_feed = DerivFeed(credentials=credentials, options=options)
                self.feeds["deriv"] = deriv_feed
            except Exception as e:
                self.logger.error(f"Failed to initialize Deriv feed: {str(e)}")
                self.metrics.increment("feed.init.error")
    
    async def _start_feeds(self):
        """Start all initialized feeds."""
        for feed_name, feed in list(self.feeds.items()):
            self.logger.info(f"Starting {feed_name} feed")
            try:
                await feed.start()
                self.logger.info(f"{feed_name} feed started successfully")
                self.metrics.increment("feed.start.success")
            except Exception as e:
                self.logger.error(f"Failed to start {feed_name} feed: {str(e)}")
                self.metrics.increment("feed.start.error")

                # Disable feed if network is unreachable to avoid endless restart attempts
                if isinstance(e, FeedConnectionError) and "Could not contact DNS servers" in str(e):
                    self.logger.error(
                        f"Disabling {feed_name} feed due to network unavailability"
                    )
                    self.feeds.pop(feed_name, None)
                    self.config.data_feeds.get(feed_name, {})["enabled"] = False
                    continue

                # Disable feed on authentication failures to prevent restart loops
                if isinstance(e, FeedAuthenticationError):
                    self.logger.error(
                        f"Disabling {feed_name} feed due to authentication error"

                    )
                    self.feeds.pop(feed_name, None)
                    self.config.data_feeds.get(feed_name, {})["enabled"] = False
                    continue

                # If this is a critical feed, raise an error
                if self.config.data_feeds.get(feed_name, {}).get("critical", False):
                    raise ServiceStartupError(f"Failed to start critical feed {feed_name}")
    
    async def _stop_feeds(self):
        """Stop all running feeds."""
        for feed_name, feed in self.feeds.items():
            self.logger.info(f"Stopping {feed_name} feed")
            try:
                await feed.stop()
                self.logger.info(f"{feed_name} feed stopped successfully")
            except Exception as e:
                self.logger.error(f"Error stopping {feed_name} feed: {str(e)}")
    
    async def _monitor_feeds(self):
        """Monitor feed status and restart failed feeds if needed."""
        self.logger.info("Starting feed monitoring")
        
        while not self.shutting_down:
            try:
                # Check and log feed statistics
                for feed_name, feed in self.feeds.items():
                    # Update metrics
                    self.metrics.set(f"feed.{feed_name}.connected", 1 if feed.is_connected else 0)
                    
                    # Log statistics
                    if hasattr(feed, 'data_stats'):
                        stats = feed.data_stats
                        self.logger.info(
                            f"{feed_name} feed stats: "
                            f"Received: {stats.get('received', 0)}, "
                            f"Published: {stats.get('published', 0)}, "
                            f"Errors: {stats.get('errors', 0)}"
                        )
                        
                        # Update metrics
                        self.metrics.set(f"feed.{feed_name}.received", stats.get('received', 0))
                        self.metrics.set(f"feed.{feed_name}.published", stats.get('published', 0))
                        self.metrics.set(f"feed.{feed_name}.errors", stats.get('errors', 0))
                    
                    # Check health and restart if needed
                    feed_config = self.config.data_feeds.get(feed_name, {})
                    if feed_config.get("auto_restart", True):
                        healthy = await feed.health_check()
                        if not healthy and not self.shutting_down:
                            self.logger.warning(f"{feed_name} feed health check failed, attempting restart")
                            try:
                                await feed.stop()
                                await asyncio.sleep(2)  # Short delay before restart
                                await feed.start()
                                self.logger.info(f"{feed_name} feed restarted successfully")
                                self.metrics.increment("feed.restart.success")
                            except Exception as e:
                                self.logger.error(f"Failed to restart {feed_name} feed: {str(e)}")
                                self.metrics.increment("feed.restart.error")
                
                # Wait for next monitoring interval
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                self.logger.info("Feed monitoring cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in feed monitoring: {str(e)}")
                await asyncio.sleep(10)  # Shorter interval on error


def create_app(config: Dict[str, Any]) -> DataFeedService:
    """
    Create and initialize the data feed service application.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized DataFeedService instance
    """
    # Create event loop if not exists
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Initialize service
    service = DataFeedService(config, loop=loop, event_bus=EventBus.get_instance())
    
    return service
