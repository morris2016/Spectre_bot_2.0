#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
WebSocket Handler Module

This module implements real-time WebSocket communication for the QuantumSpectre
Elite Trading System, providing streaming market data, trading signals, and system
updates to the frontend user interface.
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, List, Set, Any, Optional, Callable
from datetime import datetime

import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect, Depends, status
from websockets.exceptions import ConnectionClosed

from common.logger import get_logger
from common.exceptions import (
    AuthenticationError, PermissionDeniedError, RateLimitExceededError,
    ValidationError, InternalSystemError
)
from common.redis_client import get_redis_pool
from common.metrics import MetricsCollector
from api_gateway.authentication import validate_token, get_user_from_token
from api_gateway.rate_limiter import RateLimiter

logger = get_logger(__name__)
metrics = MetricsCollector()

class WebSocketManager:
    """
    WebSocket Manager for handling client connections, subscriptions, and message routing.
    
    This manager handles:
    - Client connection management
    - Message routing based on topics
    - Subscription management
    - Authorization and rate limiting
    - Broadcasting to specific clients or groups
    """
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> set of connection_ids
        self.connection_subscriptions: Dict[str, Set[str]] = {}  # connection_id -> set of topics
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.connection_info: Dict[str, Dict[str, Any]] = {}  # connection_id -> connection metadata
        self.rate_limiter = RateLimiter(max_calls=100, time_frame=60)
        self.redis_pool = None
        self._background_tasks = set()
        self._shutdown_event = asyncio.Event()
        self._broadcast_queue = asyncio.Queue()
        self._subscription_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the WebSocket manager with required connections."""
        self.redis_pool = await get_redis_pool()
        redis_subscriber = redis.Redis(connection_pool=self.redis_pool)
        self.pubsub = redis_subscriber.pubsub()
        await self.pubsub.subscribe('websocket_broadcast')
        
        # Start background tasks
        broadcast_task = asyncio.create_task(self._broadcast_listener())
        redis_task = asyncio.create_task(self._redis_listener())
        self._background_tasks.add(broadcast_task)
        self._background_tasks.add(redis_task)
        broadcast_task.add_done_callback(self._background_tasks.discard)
        redis_task.add_done_callback(self._background_tasks.discard)
        
    async def shutdown(self):
        """Shutdown the WebSocket manager and close all connections."""
        self._shutdown_event.set()
        
        # Close all active connections
        close_tasks = []
        for connection_id, websocket in self.active_connections.items():
            try:
                close_tasks.append(websocket.close(code=1001, reason="Server shutdown"))
            except Exception as e:
                logger.error(f"Error closing WebSocket {connection_id}: {str(e)}")
                
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
            
        # Wait for background tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
        # Clean up Redis connections
        if self.pubsub:
            await self.pubsub.unsubscribe()
        
    async def connect(self, websocket: WebSocket, token: str):
        """
        Handle a new WebSocket connection with authentication.
        
        Args:
            websocket: The WebSocket connection
            token: Authentication token
            
        Returns:
            str: The connection ID
            
        Raises:
            AuthenticationError: If authentication fails
        """
        # Validate the token first
        try:
            user = await get_user_from_token(token)
            if not user:
                raise AuthenticationError("Invalid authentication token")
        except Exception as e:
            logger.warning(f"Authentication failed: {str(e)}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Authentication failed")
            raise AuthenticationError(f"Authentication failed: {str(e)}")
            
        # Accept the connection
        await websocket.accept()
        
        # Generate a unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Store connection in our registry
        self.active_connections[connection_id] = websocket
        
        # Initialize connection metadata
        self.connection_info[connection_id] = {
            'user_id': user['id'],
            'username': user['username'],
            'connected_at': datetime.now().isoformat(),
            'client_ip': websocket.client.host,
            'last_activity': time.time(),
            'message_count': 0
        }
        
        # Track user's connections
        if user['id'] not in self.user_connections:
            self.user_connections[user['id']] = set()
        self.user_connections[user['id']].add(connection_id)
        
        # Initialize connection's subscriptions
        self.connection_subscriptions[connection_id] = set()
        
        logger.info(f"New WebSocket connection established: {connection_id} for user {user['username']}")
        metrics.increment('websocket_connections_total')
        metrics.gauge('websocket_connections_active', len(self.active_connections))
        
        # Send welcome message
        await self.send_to_connection(connection_id, {
            'type': 'system',
            'event': 'connected',
            'connection_id': connection_id,
            'message': 'Connected to QuantumSpectre Elite Trading System'
        })
        
        return connection_id
        
    async def disconnect(self, connection_id: str):
        """
        Handle WebSocket disconnection.
        
        Args:
            connection_id: The connection ID
        """
        if connection_id in self.active_connections:
            # Clean up subscriptions
            if connection_id in self.connection_subscriptions:
                topics = list(self.connection_subscriptions[connection_id])
                for topic in topics:
                    await self.unsubscribe(connection_id, topic)
                del self.connection_subscriptions[connection_id]
                
            # Clean up user connection mapping
            user_id = self.connection_info[connection_id]['user_id']
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
                    
            # Remove connection info
            if connection_id in self.connection_info:
                del self.connection_info[connection_id]
                
            # Remove from active connections
            del self.active_connections[connection_id]
            
            logger.info(f"WebSocket connection closed: {connection_id}")
            metrics.gauge('websocket_connections_active', len(self.active_connections))
            
    async def subscribe(self, connection_id: str, topic: str):
        """
        Subscribe a connection to a specific topic.
        
        Args:
            connection_id: The connection ID
            topic: The topic to subscribe to
            
        Returns:
            bool: True if successful
        """
        async with self._subscription_lock:
            # Initialize topic set if needed
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
                
            # Add connection to topic subscribers
            self.subscriptions[topic].add(connection_id)
            
            # Add topic to connection's subscriptions
            if connection_id in self.connection_subscriptions:
                self.connection_subscriptions[connection_id].add(topic)
        
        logger.debug(f"Connection {connection_id} subscribed to topic: {topic}")
        metrics.increment(f'websocket_subscriptions_topic.{topic}')
        
        # Notify client about subscription success
        await self.send_to_connection(connection_id, {
            'type': 'subscription',
            'event': 'subscribed',
            'topic': topic,
            'message': f'Successfully subscribed to {topic}'
        })
        
        return True
        
    async def unsubscribe(self, connection_id: str, topic: str):
        """
        Unsubscribe a connection from a specific topic.
        
        Args:
            connection_id: The connection ID
            topic: The topic to unsubscribe from
            
        Returns:
            bool: True if successful
        """
        async with self._subscription_lock:
            # Remove connection from topic subscribers
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(connection_id)
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
                    
            # Remove topic from connection's subscriptions
            if connection_id in self.connection_subscriptions:
                self.connection_subscriptions[connection_id].discard(topic)
        
        logger.debug(f"Connection {connection_id} unsubscribed from topic: {topic}")
        
        # Notify client about unsubscription
        await self.send_to_connection(connection_id, {
            'type': 'subscription',
            'event': 'unsubscribed',
            'topic': topic,
            'message': f'Successfully unsubscribed from {topic}'
        })
        
        return True
        
    async def broadcast(self, topic: str, message: dict):
        """
        Broadcast a message to all connections subscribed to a topic.
        
        Args:
            topic: The topic to broadcast to
            message: The message to broadcast
        """
        if topic in self.subscriptions:
            # Add topic to the message
            enriched_message = {**message, 'topic': topic}
            
            # Get all connection IDs subscribed to this topic
            connection_ids = list(self.subscriptions[topic])
            
            send_tasks = []
            for connection_id in connection_ids:
                if connection_id in self.active_connections:
                    send_tasks.append(self.send_to_connection(connection_id, enriched_message))
            
            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)
                metrics.increment(f'websocket_messages_broadcast_topic.{topic}', len(send_tasks))
                
    async def broadcast_to_user(self, user_id: str, message: dict):
        """
        Broadcast a message to all connections belonging to a specific user.
        
        Args:
            user_id: The user ID
            message: The message to broadcast
        """
        if user_id in self.user_connections:
            connection_ids = list(self.user_connections[user_id])
            
            send_tasks = []
            for connection_id in connection_ids:
                if connection_id in self.active_connections:
                    send_tasks.append(self.send_to_connection(connection_id, message))
            
            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)
                metrics.increment('websocket_messages_user_broadcast', len(send_tasks))
                
    async def send_to_connection(self, connection_id: str, message: dict):
        """
        Send a message to a specific connection.
        
        Args:
            connection_id: The connection ID
            message: The message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        if connection_id not in self.active_connections:
            logger.warning(f"Attempted to send message to non-existent connection: {connection_id}")
            return False
            
        websocket = self.active_connections[connection_id]
        try:
            # Add timestamp if not present
            if 'timestamp' not in message:
                message['timestamp'] = datetime.now().isoformat()
                
            # Send the message
            await websocket.send_json(message)
            
            # Update connection info
            if connection_id in self.connection_info:
                self.connection_info[connection_id]['last_activity'] = time.time()
                self.connection_info[connection_id]['message_count'] += 1
                
            metrics.increment('websocket_messages_sent')
            return True
            
        except ConnectionClosed:
            logger.info(f"Connection closed while sending message: {connection_id}")
            await self.disconnect(connection_id)
            return False
        except Exception as e:
            logger.error(f"Error sending WebSocket message to {connection_id}: {str(e)}")
            metrics.increment('websocket_errors')
            return False
            
    async def receive_and_process_message(self, connection_id: str):
        """
        Receive and process a message from a client.
        
        Args:
            connection_id: The connection ID
            
        Returns:
            dict: The processed message or None if disconnected
        """
        if connection_id not in self.active_connections:
            logger.warning(f"Attempted to receive from non-existent connection: {connection_id}")
            return None
            
        websocket = self.active_connections[connection_id]
        
        try:
            # Apply rate limiting
            user_id = self.connection_info[connection_id]['user_id']
            if not self.rate_limiter.check_rate_limit(user_id):
                await self.send_to_connection(connection_id, {
                    'type': 'error',
                    'code': 'rate_limit_exceeded',
                    'message': 'You have exceeded the rate limit. Please slow down.',
                })
                metrics.increment('websocket_rate_limit_exceeded')
                return None
                
            # Receive message from client
            data = await websocket.receive_json()
            
            # Update connection info
            if connection_id in self.connection_info:
                self.connection_info[connection_id]['last_activity'] = time.time()
                
            metrics.increment('websocket_messages_received')
            
            # Process the message
            if not isinstance(data, dict):
                await self.send_to_connection(connection_id, {
                    'type': 'error',
                    'code': 'invalid_format',
                    'message': 'Message must be a JSON object'
                })
                return None
                
            # Handle different message types
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                topic = data.get('topic')
                if not topic:
                    await self.send_to_connection(connection_id, {
                        'type': 'error',
                        'code': 'missing_topic',
                        'message': 'Subscription requires a topic'
                    })
                else:
                    await self.subscribe(connection_id, topic)
                    
            elif message_type == 'unsubscribe':
                topic = data.get('topic')
                if not topic:
                    await self.send_to_connection(connection_id, {
                        'type': 'error',
                        'code': 'missing_topic',
                        'message': 'Unsubscription requires a topic'
                    })
                else:
                    await self.unsubscribe(connection_id, topic)
                    
            elif message_type == 'ping':
                await self.send_to_connection(connection_id, {
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                })
                
            else:
                # Return the processed message for further handling
                return data
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
            await self.disconnect(connection_id)
            return None
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received from connection {connection_id}")
            await self.send_to_connection(connection_id, {
                'type': 'error',
                'code': 'invalid_json',
                'message': 'Invalid JSON format'
            })
            return None
        except Exception as e:
            logger.error(f"Error processing WebSocket message from {connection_id}: {str(e)}")
            metrics.increment('websocket_errors')
            return None
        
        return None  # Default return if message was handled internally
            
    async def _broadcast_listener(self):
        """Background task to process broadcast messages from the queue."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Get next message with a timeout
                    message = await asyncio.wait_for(
                        self._broadcast_queue.get(), 
                        timeout=1.0
                    )
                    
                    topic = message.get('topic')
                    if not topic:
                        logger.warning("Broadcast message missing topic field")
                        continue
                        
                    await self.broadcast(topic, message['data'])
                    self._broadcast_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # This is expected, just check the shutdown event and continue
                    continue
                except Exception as e:
                    logger.error(f"Error in broadcast listener: {str(e)}")
                    metrics.increment('websocket_errors')
                    await asyncio.sleep(0.1)  # Avoid tight loop in case of persistent errors
                    
        except asyncio.CancelledError:
            logger.info("Broadcast listener task cancelled")
            
    async def _redis_listener(self):
        """Background task to listen for Redis pubsub messages."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Listen for a message with a timeout
                    message = await asyncio.wait_for(
                        self.pubsub.get_message(ignore_subscribe_messages=True), 
                        timeout=1.0
                    )
                    
                    if message and message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            if 'topic' in data and 'message' in data:
                                await self.broadcast(data['topic'], data['message'])
                        except json.JSONDecodeError:
                            logger.warning("Invalid JSON received from Redis pubsub")
                        except KeyError:
                            logger.warning("Missing required fields in Redis pubsub message")
                            
                except asyncio.TimeoutError:
                    # This is expected, just check the shutdown event and continue
                    continue
                except Exception as e:
                    logger.error(f"Error in Redis listener: {str(e)}")
                    metrics.increment('websocket_errors')
                    await asyncio.sleep(0.1)  # Avoid tight loop in case of persistent errors
                    
        except asyncio.CancelledError:
            logger.info("Redis listener task cancelled")
            
    async def start_connection_handler(self, websocket: WebSocket, token: str):
        """
        Main handler for WebSocket connections.
        
        Args:
            websocket: The WebSocket connection
            token: Authentication token
        """
        connection_id = None
        try:
            # Connect and authenticate
            connection_id = await self.connect(websocket, token)
            
            # Process messages until disconnection
            while True:
                message = await self.receive_and_process_message(connection_id)
                if message is None:
                    # None is returned for handled messages or disconnection
                    continue
                    
                # Otherwise, this is a message that needs further processing
                message_type = message.get('type')
                
                # Handle UI-specific message types
                if message_type == 'ui_action':
                    action = message.get('action')
                    if action == 'chart_update':
                        # Pass to chart manager
                        pass
                    elif action == 'settings_change':
                        # Handle settings change
                        pass
                    elif action == 'command':
                        # Handle UI command
                        pass
                    else:
                        await self.send_to_connection(connection_id, {
                            'type': 'error',
                            'code': 'unknown_action',
                            'message': f'Unknown UI action: {action}'
                        })
                elif message_type == 'trade_request':
                    # Handle trade request - will be processed by the trading controller
                    # This just acknowledges receipt
                    await self.send_to_connection(connection_id, {
                        'type': 'trade_received',
                        'request_id': message.get('request_id'),
                        'message': 'Trade request received and being processed'
                    })
                else:
                    # Unhandled message type
                    await self.send_to_connection(connection_id, {
                        'type': 'error',
                        'code': 'unknown_message_type',
                        'message': f'Unknown message type: {message_type}'
                    })
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket connection handler: {str(e)}")
            metrics.increment('websocket_errors')
        finally:
            if connection_id:
                await self.disconnect(connection_id)

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

async def initialize_websocket_manager():
    """Initialize the global WebSocket manager."""
    await websocket_manager.initialize()

async def shutdown_websocket_manager():
    """Shutdown the global WebSocket manager."""
    await websocket_manager.shutdown()
