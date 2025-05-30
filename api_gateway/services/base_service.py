#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Base Service Class

This module provides the base service class for all API Gateway services,
implementing common functionality for service communication, state management,
and lifecycle handling.
"""

import asyncio
import json
import time
import traceback
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
import logging

from common.logger import get_logger
from common.metrics import MetricsCollector
from common.redis_client import RedisClient
from common.db_client import DatabaseClient
from common.exceptions import ServiceUnavailableError

# Initialize logger
logger = get_logger(__name__)

class BaseService:
    """Base class for API Gateway services."""
    
    def __init__(self, name: str, redis_client: RedisClient, db_client: DatabaseClient):
        """
        Initialize the base service.
        
        Args:
            name: Service name
            redis_client: Redis client for real-time data
            db_client: Database client for persistent storage
        """
        self.name = name
        self.redis_client = redis_client
        self.db_client = db_client
        self.metrics = MetricsCollector(namespace=f"service_{name}")
        self.running = False
        self.tasks = []
        self.pubsub = None
        self.subscriptions = {}
    
    async def start(self):
        """Start the service."""
        logger.info(f"Starting service: {self.name}")
        self.running = True
        
        # Start Redis PubSub subscriber
        pubsub_task = asyncio.create_task(self._redis_subscriber())
        self.tasks.append(pubsub_task)
        
        # Start health check task
        health_task = asyncio.create_task(self._health_check())
        self.tasks.append(health_task)
        
        # Update service status
        await self._update_service_status("running")
        
        logger.info(f"Service started: {self.name}")
    
    async def stop(self):
        """Stop the service."""
        logger.info(f"Stopping service: {self.name}")
        self.running = False
        
        # Unsubscribe from Redis channels
        if self.pubsub:
            try:
                for pattern in self.subscriptions.keys():
                    await self.pubsub.punsubscribe(pattern)
                await self.pubsub.close()
            except Exception as e:
                logger.error(f"Error closing Redis PubSub: {str(e)}")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Update service status
        await self._update_service_status("stopped")
        
        logger.info(f"Service stopped: {self.name}")
    
    async def subscribe(self, pattern: str, callback: Callable):
        """
        Subscribe to Redis channel pattern.
        
        Args:
            pattern: Channel pattern to subscribe to
            callback: Callback function to handle messages
        """
        self.subscriptions[pattern] = callback
        if self.pubsub:
            await self.pubsub.psubscribe(pattern)
            logger.debug(f"Subscribed to Redis channel pattern: {pattern}")
    
    async def publish(self, channel: str, message: Dict[str, Any]):
        """
        Publish message to Redis channel.
        
        Args:
            channel: Channel to publish to
            message: Message to publish
        """
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add service name
        if "service" not in message:
            message["service"] = self.name
        
        json_message = json.dumps(message)
        await self.redis_client.publish(channel, json_message)
    
    async def _redis_subscriber(self):
        """Redis PubSub subscriber task."""
        try:
            logger.info(f"Starting Redis PubSub subscriber for {self.name}")
            
            # Create Redis PubSub connection
            self.pubsub = self.redis_client.pubsub()
            
            # Subscribe to all configured patterns
            for pattern in self.subscriptions.keys():
                await self.pubsub.psubscribe(pattern)
                logger.debug(f"Subscribed to Redis channel pattern: {pattern}")
            
            # Process messages
            while self.running:
                try:
                    # Get message with timeout
                    message = await asyncio.wait_for(self.pubsub.get_message(), timeout=1.0)
                    
                    if message and message['type'] == 'pmessage':
                        try:
                            # Process message
                            pattern = message['pattern']
                            channel = message['channel']
                            data = json.loads(message['data'])
                            
                            # Find callback for this pattern
                            callback = self.subscriptions.get(pattern)
                            if callback:
                                # Run callback
                                await callback(channel, data)
                            
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON in Redis message: {message['data']}")
                        except Exception as e:
                            logger.error(f"Error processing Redis message: {str(e)}")
                            logger.error(traceback.format_exc())
                except asyncio.TimeoutError:
                    # This is expected, just continue
                    continue
                except asyncio.CancelledError:
                    # Task is being canceled, exit loop
                    break
                except Exception as e:
                    logger.error(f"Redis subscriber error: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Reconnect on error
                    try:
                        await self.pubsub.close()
                    except Exception:
                        pass
                    
                    # Wait before reconnecting
                    await asyncio.sleep(5)
                    
                    # Reconnect
                    try:
                        self.pubsub = self.redis_client.pubsub()
                        for pattern in self.subscriptions.keys():
                            await self.pubsub.psubscribe(pattern)
                    except Exception as e2:
                        logger.error(f"Redis reconnection error: {str(e2)}")
        except Exception as e:
            logger.error(f"Redis subscriber task error: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"Redis PubSub subscriber stopped for {self.name}")
    
    async def _health_check(self):
        """Health check task."""
        try:
            logger.info(f"Starting health check task for {self.name}")
            
            while self.running:
                try:
                    # Update health status every 30 seconds
                    await self._update_health_status()
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    # Task is being canceled, exit loop
                    break
                except Exception as e:
                    logger.error(f"Health check error: {str(e)}")
                    await asyncio.sleep(60)  # Longer delay on error
        except Exception as e:
            logger.error(f"Health check task error: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"Health check task stopped for {self.name}")
    
    async def _update_service_status(self, status: str):
        """
        Update service status in Redis.
        
        Args:
            status: Service status
        """
        try:
            # Update status in Redis hash
            await self.redis_client.hset("service:status", self.name, status)
            
            # Also publish event
            await self.publish("service:status", {
                "service": self.name,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Update metrics
            self.metrics.gauge("service_status", 1 if status == "running" else 0)
        except Exception as e:
            logger.error(f"Error updating service status: {str(e)}")
    
    async def _update_health_status(self):
        """Update service health status in Redis."""
        try:
            # Get service info
            info = {
                "service": self.name,
                "status": "healthy" if self.running else "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime": time.time() - (getattr(self, "start_time", time.time()))
            }
            
            # Add service-specific health info
            additional_info = await self._get_health_info()
            if additional_info:
                info.update(additional_info)
            
            # Update health in Redis hash
            await self.redis_client.hset("service:health", self.name, json.dumps(info))
            
            # Update metrics
            self.metrics.gauge("service_health", 1 if info["status"] == "healthy" else 0)
            self.metrics.gauge("service_uptime", info["uptime"])
        except Exception as e:
            logger.error(f"Error updating health status: {str(e)}")
    
    async def _get_health_info(self) -> Dict[str, Any]:
        """
        Get service-specific health information.
        
        Returns:
            Dictionary with health information
        """
        # To be overridden by subclasses
        return {}
