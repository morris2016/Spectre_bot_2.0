#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Base Feed Class

This module defines the base class for all data feeds in the system.
Data feeds connect to external data sources and provide real-time data.
"""

import time
import asyncio
import logging
from abc import ABC, abstractmethod

from common.logger import get_logger
from common.exceptions import FeedError, FeedConnectionError, FeedDisconnectedError
from common.metrics import MetricsCollector
from common.event_bus import EventBus
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, Any


class BaseFeed(ABC):
    """Base class for all data feeds."""
    
    def __init__(self, config, loop=None, redis_client=None, event_bus: Optional[EventBus] = None):
        """
        Initialize the data feed.
        
        Args:
            config: Feed configuration
            loop: Optional asyncio event loop
            redis_client: Optional Redis client for publishing data
        """
        self.config = config
        self.loop = loop or asyncio.get_event_loop()
        self.redis_client = redis_client
        self.event_bus = event_bus or EventBus.get_instance()
        self.logger = get_logger(self.__class__.__name__)
        
        # Feed state
        self.running = False
        self.shutting_down = False
        self.connected = False
        self.last_data_time = 0
        self.reconnect_attempt = 0
        self.is_connected = False
        
        # Data tracking
        self.subscriptions = set()
        self.channels = {}
        self.data_stats = {
            "received": 0,
            "published": 0,
            "errors": 0
        }
        
        # Create metrics collector
        self.metrics = MetricsCollector(f"feed.{self.__class__.__name__.lower()}")
    
    @abstractmethod
    async def start(self):
        """Start the data feed. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the data feed. Must be implemented by subclasses."""
        pass
    
    async def health_check(self) -> bool:
        """
        Check the health of the data feed.
        
        Returns:
            True if the feed is healthy, False otherwise
        """
        # Default implementation checks if the feed is running and connected
        if not self.running:
            self.logger.warning("Feed is not running")
            return False
        
        if not self.is_connected:
            self.logger.warning("Feed is not connected")
            return False
        
        # Check if we've received data recently
        if self.last_data_time > 0:
            time_since_last_data = time.time() - self.last_data_time
            max_data_age = self.config.get("max_data_age", 60)
            
            if time_since_last_data > max_data_age:
                self.logger.warning(f"No data received for {time_since_last_data:.1f} seconds")
                return False
        
        return True
    
    async def subscribe(self, channel, callback=None):
        """
        Subscribe to a data channel.
        
        Args:
            channel: Channel name or identifier
            callback: Optional callback function to process data
            
        Returns:
            Subscription ID or channel name
        """
        if channel in self.subscriptions:
            self.logger.debug(f"Already subscribed to channel: {channel}")
            return channel
        
        self.logger.info(f"Subscribing to channel: {channel}")
        self.subscriptions.add(channel)
        
        if callback:
            self.channels[channel] = callback
        
        return channel
    
    async def unsubscribe(self, channel):
        """
        Unsubscribe from a data channel.
        
        Args:
            channel: Channel name or identifier
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        if channel not in self.subscriptions:
            self.logger.debug(f"Not subscribed to channel: {channel}")
            return False
        
        self.logger.info(f"Unsubscribing from channel: {channel}")
        self.subscriptions.remove(channel)
        
        if channel in self.channels:
            del self.channels[channel]
        
        return True
    
    async def _publish_data(self, data, channel=None):
        """
        Publish data to Redis and process callbacks.
        
        Args:
            data: Data to publish
            channel: Optional channel name (defaults to data type)
        """
        try:
            # Update last data time
            self.last_data_time = time.time()
            
            # Determine channel if not provided
            if channel is None:
                data_type = data.get("type", "unknown")
                channel = f"data.{data_type}"
            
            # Process callbacks if registered
            if channel in self.channels and callable(self.channels[channel]):
                try:
                    await self.channels[channel](data)
                except Exception as e:
                    self.logger.error(f"Error in channel callback for {channel}: {str(e)}")
            
            # Publish to Redis if available
            if self.redis_client:
                await self.redis_client.publish(channel, data)
                self.metrics.increment("data.published")
                self.data_stats["published"] += 1

            # Publish to EventBus
            if self.event_bus:
                await self.event_bus.publish(channel, data)
            
            # Update metrics
            self.metrics.increment("data.received")
            self.data_stats["received"] += 1
            
        except Exception as e:
            self.logger.error(f"Error publishing data: {str(e)}")
            self.metrics.increment("data.error")
            self.data_stats["errors"] += 1
    
    def update_connection_state(self, connected):
        """Update the connection state and reset reconnect attempts if connected."""
        previous_state = self.is_connected
        self.is_connected = connected
        
        if connected and not previous_state:
            self.logger.info("Feed connected")
            self.reconnect_attempt = 0
            self.metrics.set("connected", 1)
        elif not connected and previous_state:
            self.logger.warning("Feed disconnected")
            self.metrics.set("connected", 0)
    
    async def _reconnect_with_backoff(self, max_attempts=5, initial_delay=1, max_delay=60):
        """
        Attempt to reconnect with exponential backoff.
        
        Args:
            max_attempts: Maximum number of reconnection attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            
        Returns:
            True if reconnected successfully, False otherwise
        """
        delay = initial_delay
        
        for attempt in range(1, max_attempts + 1):
            self.reconnect_attempt = attempt
            self.logger.info(f"Reconnection attempt {attempt}/{max_attempts} (delay: {delay}s)")
            
            try:
                # Attempt reconnection
                await self.start()
                self.logger.info("Reconnected successfully")
                return True
            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt} failed: {str(e)}")
                
                if attempt < max_attempts and not self.shutting_down:
                    # Wait with exponential backoff
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_delay)
                else:
                    self.logger.error("Maximum reconnection attempts reached")
                    return False
        
        return False

class BaseDataFeed(BaseFeed):
    """
    Base class for specialized data feeds in the trading system.
    
    This class extends the BaseFeed class with additional functionality
    specific to data feeds used for trading analysis and signal generation.
    """
    
    def __init__(self, name, config, event_bus: Optional[EventBus] = None):
        """
        Initialize the data feed.
        
        Args:
            name: Feed name
            config: System configuration
        """
        super().__init__(config, event_bus=event_bus)
        self.name = name
        self.feed_type = "data"
        
    async def process_data(self, data):
        """
        Process received data.
        
        Args:
            data: Received data
            
        Returns:
            Processed data
        """
        # Base implementation just passes data through
        # Subclasses should override this to implement custom processing
        return data
    
    async def validate_data(self, data):
        """
        Validate data before processing.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        # Base implementation assumes all data is valid
        # Subclasses should override this to implement custom validation
        return True
    
    async def get_feed_info(self):
        """
        Get information about this data feed.
        
        Returns:
            Feed information dictionary
        """
        return {
            "name": self.name,
            "type": self.feed_type,
            "running": self.running,
            "connected": self.is_connected,
            "stats": self.data_stats,
            "subscriptions": list(self.subscriptions)
        }

@dataclass
class FeedOptions:
    """Configuration options for a data feed."""
    
    # Connection options
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    connection_timeout: float = 30.0
    
    # Data processing options
    batch_size: int = 100
    process_interval: float = 0.1
    max_queue_size: int = 10000
    
    # Monitoring options
    healthcheck_interval: float = 60.0
    max_data_age: float = 300.0
    
    # Authentication options
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    passphrase: Optional[str] = None
    
    # Additional options
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketData:
    """Simple OHLCV data container."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OrderBookData:
    """Simplified order book snapshot used for microstructure analysis."""
    timestamp: float
    bids: List[Tuple[float, float]]
    asks: List[Tuple[float, float]]
    update_id: Optional[str] = None


@dataclass
class TradeData:
    """Trade information container used by execution components."""
    timestamp: float
    price: float
    quantity: float
    is_buyer_maker: bool
    trade_id: Optional[str] = None
    side: Optional[str] = None


class DataProcessor:
    """Base class for data processors."""
    
    def __init__(self, name, config=None):
        """
        Initialize the data processor.
        
        Args:
            name: Processor name
            config: Optional configuration
        """
        self.name = name
        self.config = config or {}
        self.logger = get_logger(f"processor.{name}")
        self.metrics = MetricsCollector(f"processor.{name}")
        self.processing_stats = {
            "processed": 0,
            "filtered": 0,
            "errors": 0
        }
    
    async def process(self, data):
        """
        Process a single data item.
        
        Args:
            data: Data to process
            
        Returns:
            Processed data or None if data should be filtered out
        """
        try:
            # Apply pre-processing
            processed_data = await self._preprocess(data)
            if processed_data is None:
                self.processing_stats["filtered"] += 1
                return None
            
            # Apply main processing
            processed_data = await self._process(processed_data)
            if processed_data is None:
                self.processing_stats["filtered"] += 1
                return None
            
            # Apply post-processing
            processed_data = await self._postprocess(processed_data)
            if processed_data is None:
                self.processing_stats["filtered"] += 1
                return None
            
            # Update stats
            self.processing_stats["processed"] += 1
            self.metrics.increment("data.processed")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            self.processing_stats["errors"] += 1
            self.metrics.increment("data.error")
            return None
    
    async def _preprocess(self, data):
        """
        Pre-process data before main processing.
        
        Args:
            data: Data to pre-process
            
        Returns:
            Pre-processed data
        """
        # Default implementation returns data unchanged
        return data
    
    async def _process(self, data):
        """
        Main processing of data.
        
        Args:
            data: Data to process
            
        Returns:
            Processed data
        """
        # Default implementation returns data unchanged
        return data
    
    async def _postprocess(self, data):
        """
        Post-process data after main processing.
        
        Args:
            data: Data to post-process
            
        Returns:
            Post-processed data
        """
        # Default implementation returns data unchanged
        return data
    
    async def get_stats(self):
        """
        Get processing statistics.
        
        Returns:
            Processing statistics
        """
        return self.processing_stats
    
    async def reset_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            "processed": 0,
            "filtered": 0,
            "errors": 0
        }
