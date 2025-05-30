#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Advanced WebSocket Client

This module provides a highly optimized WebSocket client for connecting to
various streaming APIs, with advanced features such as:
- Automatic reconnection with exponential backoff
- Connection health monitoring
- Heartbeat management
- Message buffering and batching
- Subscription management
- Comprehensive error handling
- Performance optimizations
- Metrics collection
"""

import json
import zlib
import time
import logging
import asyncio
import aiohttp
import backoff
import websockets
from enum import Enum
from typing import Dict, List, Set, Any, Optional, Union, Callable, Tuple, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlencode, urlparse
from functools import wraps
import random
import hmac
import hashlib
import base64
import uuid

# Internal imports
from config import Config
from common.logger import get_logger
from common.exceptions import (
    WebSocketError, ConnectionError, AuthenticationError, SubscriptionError,
    MessageError, DisconnectError
)
from common.metrics import MetricsCollector
from common.utils import generate_nonce, timeit
from common.constants import (
    WS_DEFAULT_PING_INTERVAL, WS_DEFAULT_PING_TIMEOUT,
    WS_DEFAULT_CLOSE_TIMEOUT, WS_DEFAULT_MAX_QUEUE_SIZE,
    WS_DEFAULT_RECONNECT_DELAY, WS_MAX_RECONNECT_DELAY,
    WS_RECONNECT_JITTER
)


class ConnectionState(Enum):
    """Enum representing the connection states of a WebSocket connection."""
    DISCONNECTED = 'disconnected'
    CONNECTING = 'connecting'
    CONNECTED = 'connected'
    RECONNECTING = 'reconnecting'
    CLOSING = 'closing'
    CLOSED = 'closed'
    FAILED = 'failed'


@dataclass
class Subscription:
    """Data class for representing a WebSocket subscription."""
    id: str
    channel: str
    params: Dict[str, Any] = field(default_factory=dict)
    active: bool = False
    last_message_time: Optional[float] = None
    message_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class Message:
    """Data class for representing a WebSocket message."""
    data: Any
    received_at: float = field(default_factory=time.time)
    subscription_id: Optional[str] = None
    raw_data: Optional[Union[str, bytes]] = None


class WebSocketClient:
    """
    Advanced WebSocket client for connecting to streaming APIs with sophisticated
    connection management, automatic reconnection, and message processing.
    """
    
    def __init__(
        self,
        url: str,
        config: Config,
        on_message: Optional[Callable[[Message], Coroutine]] = None,
        on_connect: Optional[Callable[[], Coroutine]] = None,
        on_disconnect: Optional[Callable[[bool, Optional[str]], Coroutine]] = None,
        on_error: Optional[Callable[[Exception], Coroutine]] = None,
        auto_reconnect: bool = True,
        ping_interval: float = WS_DEFAULT_PING_INTERVAL,
        ping_timeout: float = WS_DEFAULT_PING_TIMEOUT,
        close_timeout: float = WS_DEFAULT_CLOSE_TIMEOUT,
        max_queue_size: int = WS_DEFAULT_MAX_QUEUE_SIZE,
        name: Optional[str] = None,
        compression: bool = False,
        extra_headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the WebSocket client.
        
        Args:
            url: WebSocket URL to connect to
            config: Application configuration
            on_message: Callback function for received messages
            on_connect: Callback function when connection is established
            on_disconnect: Callback function when connection is lost
            on_error: Callback function for errors
            auto_reconnect: Whether to automatically reconnect on disconnection
            ping_interval: Interval between ping messages
            ping_timeout: Timeout for ping responses
            close_timeout: Timeout for clean connection close
            max_queue_size: Maximum size of the message queue
            name: Client name for identification
            compression: Whether to use compression
            extra_headers: Additional headers for the WebSocket connection
        """
        self.url = url
        self.config = config
        self.on_message_callback = on_message
        self.on_connect_callback = on_connect
        self.on_disconnect_callback = on_disconnect
        self.on_error_callback = on_error
        self.auto_reconnect = auto_reconnect
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.close_timeout = close_timeout
        self.max_queue_size = max_queue_size
        self.name = name or f"ws_{urlparse(url).netloc.split(':')[0]}"
        self.compression = compression
        self.extra_headers = extra_headers or {}
        
        # Add user agent header
        self.extra_headers['User-Agent'] = f"QuantumSpectre/{self.config.get('version', '1.0.0')}"
        
        # Connection state and objects
        self.state = ConnectionState.DISCONNECTED
        self.ws = None
        self.session = None
        
        # Tasks and events
        self.connect_task = None
        self.receive_task = None
        self.heartbeat_task = None
        self.processing_task = None
        self.reconnect_task = None
        self.stop_event = asyncio.Event()
        
        # Message handling
        self.message_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Subscription management
        self.subscriptions: Dict[str, Subscription] = {}
        
        # Reconnection settings
        self.reconnect_delay = WS_DEFAULT_RECONNECT_DELAY
        self.reconnect_attempt = 0
        
        # Last activity tracking
        self.last_message_received = time.time()
        self.last_ping_sent = 0
        self.last_pong_received = 0
        
        # Authentication credentials
        self.api_key = None
        self.secret_key = None
        
        # Metrics
        self.metrics = MetricsCollector.get_instance()
        
        # Logging
        self.logger = get_logger(f"ws_client.{self.name}")
        
    async def start(self):
        """Start the WebSocket client and connect."""
        self.stop_event.clear()
        
        # Set initial state
        self.state = ConnectionState.CONNECTING
        
        # Start connection
        self.connect_task = asyncio.create_task(self._connect())
        
    async def stop(self):
        """Stop the WebSocket client and close connection."""
        self.logger.info(f"Stopping WebSocket client for {self.name}")
        
        # Set the stop event to signal all tasks to exit
        self.stop_event.set()
        
        # Set state to closing
        self.state = ConnectionState.CLOSING
        
        # Wait for tasks to complete
        tasks = []
        if self.connect_task and not self.connect_task.done():
            tasks.append(self.connect_task)
        if self.receive_task and not self.receive_task.done():
            tasks.append(self.receive_task)
        if self.heartbeat_task and not self.heartbeat_task.done():
            tasks.append(self.heartbeat_task)
        if self.processing_task and not self.processing_task.done():
            tasks.append(self.processing_task)
        if self.reconnect_task and not self.reconnect_task.done():
            tasks.append(self.reconnect_task)
            
        # Cancel and wait for all tasks
        for task in tasks:
            task.cancel()
            
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        # Close the WebSocket connection if still open
        if self.ws and not self.ws.closed:
            self.logger.info(f"Closing WebSocket connection for {self.name}")
            try:
                await self.ws.close(code=1000, message=b'Client shutdown')
            except Exception as e:
                self.logger.warning(f"Error closing WebSocket connection: {str(e)}")
                
        # Close the aiohttp session if it exists
        if self.session:
            await self.session.close()
            
        # Set final state
        self.state = ConnectionState.CLOSED
        self.logger.info(f"WebSocket client for {self.name} stopped")
        
    async def _connect(self):
        """
        Establish a WebSocket connection with automatic reconnection.
        """
        while not self.stop_event.is_set():
            try:
                # Create a new session if needed
                if not self.session:
                    self.session = aiohttp.ClientSession()
                    
                self.logger.info(f"Connecting to WebSocket at {self.url}")
                self.state = ConnectionState.CONNECTING
                
                # Establish connection
                self.ws = await self.session.ws_connect(
                    self.url,
                    headers=self.extra_headers,
                    timeout=self.config.get('ws_client.connect_timeout', 30),
                    heartbeat=self.ping_interval,
                    compress=self.compression,
                    autoclose=False,
                    autoping=True
                )
                
                # Reset reconnection counters on successful connection
                self.reconnect_delay = WS_DEFAULT_RECONNECT_DELAY
                self.reconnect_attempt = 0
                
                # Update state and notify
                self.state = ConnectionState.CONNECTED
                self.logger.info(f"WebSocket connection established to {self.url}")
                
                # Start tasks
                self.receive_task = asyncio.create_task(self._receive_loop())
                self.processing_task = asyncio.create_task(self._process_messages())
                self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
                # Call the connection callback if set
                if self.on_connect_callback:
                    try:
                        await self.on_connect_callback()
                    except Exception as e:
                        self.logger.error(f"Error in on_connect callback: {str(e)}")
                        
                # Resubscribe to active channels
                await self._resubscribe()
                
                # Wait for receive loop to end (indicates disconnection)
                await self.receive_task
                
            except aiohttp.ClientError as e:
                self.state = ConnectionState.FAILED
                self.logger.error(f"WebSocket connection error: {str(e)}")
                
                # Metrics
                self.metrics.increment(f"ws_client.{self.name}.connection_errors")
                
                # Call error callback if set
                if self.on_error_callback:
                    try:
                        await self.on_error_callback(e)
                    except Exception as callback_error:
                        self.logger.error(f"Error in on_error callback: {str(callback_error)}")
                
                # Reconnect if configured
                if not await self._handle_reconnect():
                    break
                    
            except asyncio.CancelledError:
                # Normal cancellation, exit cleanly
                self.logger.info(f"WebSocket connect task cancelled for {self.name}")
                break
                
            except Exception as e:
                self.state = ConnectionState.FAILED
                self.logger.exception(f"Unexpected error in WebSocket connection: {str(e)}")
                
                # Metrics
                self.metrics.increment(f"ws_client.{self.name}.unexpected_errors")
                
                # Call error callback if set
                if self.on_error_callback:
                    try:
                        await self.on_error_callback(e)
                    except Exception as callback_error:
                        self.logger.error(f"Error in on_error callback: {str(callback_error)}")
                
                # Reconnect if configured
                if not await self._handle_reconnect():
                    break
        
        self.logger.info(f"WebSocket connect loop exited for {self.name}")
        
    async def _receive_loop(self):
        """
        Continuously receive and process WebSocket messages.
        """
        try:
            self.logger.info(f"Starting WebSocket receive loop for {self.name}")
            
            while not self.stop_event.is_set() and not self.ws.closed:
                try:
                    # Receive message with timeout
                    msg = await asyncio.wait_for(
                        self.ws.receive(),
                        timeout=self.ping_interval + self.ping_timeout
                    )
                    
                    # Update last message time
                    self.last_message_received = time.time()
                    
                    # Handle different message types
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._handle_text_message(msg.data)
                        
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        await self._handle_binary_message(msg.data)
                        
                    elif msg.type == aiohttp.WSMsgType.PING:
                        # Automatically handled by aiohttp, but we'll log it
                        self.logger.debug(f"Received WebSocket PING")
                        
                    elif msg.type == aiohttp.WSMsgType.PONG:
                        # Update last pong time
                        self.last_pong_received = time.time()
                        self.logger.debug(f"Received WebSocket PONG")
                        
                    elif msg.type == aiohttp.WSMsgType.CLOSE:
                        self.logger.info(f"Received WebSocket CLOSE message: {msg.data}, {msg.extra}")
                        # Close code and reason are in data and extra
                        await self._handle_disconnect(False, f"Server closed connection: {msg.data}, {msg.extra}")
                        break
                        
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        self.logger.info("WebSocket connection is already closed")
                        await self._handle_disconnect(False, "Connection already closed")
                        break
                        
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        self.logger.error(f"WebSocket connection error: {self.ws.exception()}")
                        await self._handle_disconnect(True, f"Connection error: {self.ws.exception()}")
                        break
                        
                except asyncio.TimeoutError:
                    # No message received within timeout period
                    elapsed = time.time() - self.last_message_received
                    self.logger.warning(f"No WebSocket message received in {elapsed:.2f}s")
                    
                    # Check if connection is still alive
                    if elapsed > self.ping_interval * 2 + self.ping_timeout:
                        self.logger.error(f"WebSocket connection appears dead, reconnecting")
                        
                        # Metrics
                        self.metrics.increment(f"ws_client.{self.name}.timeouts")
                        
                        # Close and reconnect
                        await self._handle_disconnect(True, "Connection timeout")
                        break
                        
        except asyncio.CancelledError:
            # Normal cancellation during shutdown
            self.logger.info(f"WebSocket receive loop cancelled for {self.name}")
            
        except Exception as e:
            # Unexpected error
            self.logger.exception(f"Unexpected error in WebSocket receive loop: {str(e)}")
            await self._handle_disconnect(True, f"Unexpected error: {str(e)}")
            
        finally:
            self.logger.info(f"WebSocket receive loop exited for {self.name}")
            
    async def _process_messages(self):
        """
        Process messages from the queue and deliver to callbacks.
        """
        try:
            self.logger.info(f"Starting message processing loop for {self.name}")
            
            while not self.stop_event.is_set():
                try:
                    # Get message from queue with timeout
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=1.0  # Check stop event every second
                    )
                    
                    # Process the message
                    try:
                        # Track metrics
                        processing_start = time.time()
                        
                        # Call the message callback if set
                        if self.on_message_callback:
                            await self.on_message_callback(message)
                            
                        # Update subscription stats if applicable
                        if message.subscription_id and message.subscription_id in self.subscriptions:
                            sub = self.subscriptions[message.subscription_id]
                            sub.last_message_time = message.received_at
                            sub.message_count += 1
                            
                        # Track processing time
                        processing_time = (time.time() - processing_start) * 1000  # ms
                        self.metrics.timing(f"ws_client.{self.name}.message_processing", processing_time)
                            
                    except Exception as e:
                        self.logger.error(f"Error processing WebSocket message: {str(e)}")
                        self.metrics.increment(f"ws_client.{self.name}.message_processing_errors")
                        
                        # Call error callback if set
                        if self.on_error_callback:
                            try:
                                await self.on_error_callback(e)
                            except Exception as callback_error:
                                self.logger.error(f"Error in on_error callback: {str(callback_error)}")
                    finally:
                        # Mark the task as done
                        self.message_queue.task_done()
                            
                except asyncio.TimeoutError:
                    # Queue get timed out, just continue to check stop event
                    continue
                    
        except asyncio.CancelledError:
            # Normal cancellation during shutdown
            self.logger.info(f"Message processing loop cancelled for {self.name}")
            
        except Exception as e:
            # Unexpected error
            self.logger.exception(f"Unexpected error in message processing loop: {str(e)}")
            
        finally:
            self.logger.info(f"Message processing loop exited for {self.name}")
            
    async def _heartbeat_loop(self):
        """
        Send periodic heartbeats to keep the connection alive.
        """
        try:
            self.logger.info(f"Starting heartbeat loop for {self.name}")
            
            while not self.stop_event.is_set() and not self.ws.closed:
                try:
                    # Sleep for the ping interval
                    await asyncio.sleep(self.ping_interval)
                    
                    # Check if we need to send a ping
                    # Note: aiohttp handles pings automatically if configured
                    # This is just additional monitoring
                    
                    # Check if we haven't received a message in a while
                    elapsed = time.time() - self.last_message_received
                    if elapsed > self.ping_interval:
                        self.logger.debug(f"No message received in {elapsed:.2f}s, checking connection")
                        
                        # Send a ping manually (even though aiohttp should be doing this)
                        if not self.ws.closed:
                            self.last_ping_sent = time.time()
                            await self.ws.ping()
                            self.logger.debug(f"Sent manual WebSocket PING")
                            
                            # Check if the last ping was acknowledged
                            if self.last_ping_sent > 0 and self.last_pong_received < self.last_ping_sent:
                                ping_elapsed = time.time() - self.last_ping_sent
                                if ping_elapsed > self.ping_timeout:
                                    self.logger.warning(f"PING not acknowledged in {ping_elapsed:.2f}s, connection may be dead")
                                    
                                    # If no pong received for too long, close and reconnect
                                    if ping_elapsed > self.ping_timeout * 2:
                                        self.logger.error(f"Connection appears dead, initiating reconnect")
                                        
                                        # Metrics
                                        self.metrics.increment(f"ws_client.{self.name}.ping_timeouts")
                                        
                                        # Close and reconnect
                                        try:
                                            await self.ws.close(code=1001, message=b'Ping timeout')
                                        except Exception as e:
                                            self.logger.warning(f"Error closing WebSocket after ping timeout: {str(e)}")
                                            
                                        # Signal disconnect handler
                                        await self._handle_disconnect(True, "Ping timeout")
                                        break
                    
                except asyncio.CancelledError:
                    raise
                    
                except Exception as e:
                    self.logger.error(f"Error in heartbeat loop: {str(e)}")
                    
        except asyncio.CancelledError:
            # Normal cancellation during shutdown
            self.logger.info(f"Heartbeat loop cancelled for {self.name}")
            
        except Exception as e:
            # Unexpected error
            self.logger.exception(f"Unexpected error in heartbeat loop: {str(e)}")
            
        finally:
            self.logger.info(f"Heartbeat loop exited for {self.name}")
            
    async def _handle_text_message(self, data: str):
        """
        Handle a text message received from the WebSocket.
        
        Args:
            data: Text message data
        """
        # Track metrics
        self.metrics.increment(f"ws_client.{self.name}.messages.text")
        
        try:
            # Parse JSON data
            json_data = json.loads(data)
            
            # Find matching subscription
            subscription_id = None
            for sub_id, sub in self.subscriptions.items():
                # Logic to match message to subscription depends on the exchange
                # This is a simplified example
                if 'channel' in json_data and json_data['channel'] == sub.channel:
                    subscription_id = sub_id
                    break
                elif 'topic' in json_data and json_data['topic'] == sub.channel:
                    subscription_id = sub_id
                    break
                    
            # Create message object
            message = Message(
                data=json_data,
                received_at=time.time(),
                subscription_id=subscription_id,
                raw_data=data
            )
            
            # Add to processing queue
            await self._queue_message(message)
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Received invalid JSON: {str(e)}")
            self.metrics.increment(f"ws_client.{self.name}.errors.json_decode")
            
        except Exception as e:
            self.logger.error(f"Error handling text message: {str(e)}")
            self.metrics.increment(f"ws_client.{self.name}.errors.message_handling")
            
    async def _handle_binary_message(self, data: bytes):
        """
        Handle a binary message received from the WebSocket.
        
        Args:
            data: Binary message data
        """
        # Track metrics
        self.metrics.increment(f"ws_client.{self.name}.messages.binary")
        
        try:
            # Try to decompress if it's compressed
            try:
                # Try zlib decompression
                decompressed_data = zlib.decompress(data)
                text_data = decompressed_data.decode('utf-8')
                self.logger.debug(f"Decompressed binary message: {len(data)} -> {len(text_data)} bytes")
            except Exception:
                # Not compressed or different compression, try direct decode
                text_data = data.decode('utf-8')
                
            # Parse JSON data
            json_data = json.loads(text_data)
            
            # Find matching subscription
            subscription_id = None
            for sub_id, sub in self.subscriptions.items():
                # Logic to match message to subscription depends on the exchange
                if 'channel' in json_data and json_data['channel'] == sub.channel:
                    subscription_id = sub_id
                    break
                elif 'topic' in json_data and json_data['topic'] == sub.channel:
                    subscription_id = sub_id
                    break
                    
            # Create message object
            message = Message(
                data=json_data,
                received_at=time.time(),
                subscription_id=subscription_id,
                raw_data=data
            )
            
            # Add to processing queue
            await self._queue_message(message)
            
        except Exception as e:
            self.logger.error(f"Error handling binary message: {str(e)}")
            self.metrics.increment(f"ws_client.{self.name}.errors.binary_handling")
            
    async def _queue_message(self, message: Message):
        """
        Queue a message for processing.
        
        Args:
            message: Message to queue
        """
        try:
            # Check if queue is full
            if self.message_queue.full():
                self.logger.warning(f"Message queue is full ({self.max_queue_size} messages), dropping oldest message")
                self.metrics.increment(f"ws_client.{self.name}.queue_drops")
                
                # Remove oldest message to make room
                try:
                    # This is not ideal but works as a fallback
                    _ = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                    self.message_queue.task_done()
                except asyncio.TimeoutError:
                    # Could not get a message, just continue
                    pass
                    
            # Add new message to queue
            await asyncio.wait_for(self.message_queue.put(message), timeout=0.5)
            
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout putting message in queue, message dropped")
            self.metrics.increment(f"ws_client.{self.name}.queue_timeouts")
            
        except Exception as e:
            self.logger.error(f"Error queuing message: {str(e)}")
            self.metrics.increment(f"ws_client.{self.name}.errors.queue")
            
    async def _handle_disconnect(self, was_error: bool, reason: Optional[str] = None):
        """
        Handle a WebSocket disconnection.
        
        Args:
            was_error: Whether the disconnection was due to an error
            reason: Reason for disconnection
        """
        # Update state
        previous_state = self.state
        self.state = ConnectionState.DISCONNECTED
        
        # Log disconnect
        if was_error:
            self.logger.error(f"WebSocket disconnected due to error: {reason}")
            self.metrics.increment(f"ws_client.{self.name}.error_disconnects")
        else:
            self.logger.info(f"WebSocket disconnected: {reason}")
            self.metrics.increment(f"ws_client.{self.name}.clean_disconnects")
            
        # Cancel receive task if it's running
        if self.receive_task and not self.receive_task.done() and previous_state == ConnectionState.CONNECTED:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
                
        # Cancel heartbeat task if it's running
        if self.heartbeat_task and not self.heartbeat_task.done() and previous_state == ConnectionState.CONNECTED:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
                
        # Call the disconnect callback if set
        if self.on_disconnect_callback:
            try:
                await self.on_disconnect_callback(was_error, reason)
            except Exception as e:
                self.logger.error(f"Error in on_disconnect callback: {str(e)}")
                
        # Reconnect if configured and not stopping
        if self.auto_reconnect and not self.stop_event.is_set():
            # Schedule reconnection
            await self._handle_reconnect()
            
    async def _handle_reconnect(self) -> bool:
        """
        Handle reconnection after disconnection.
        
        Returns:
            bool: Whether reconnection was initiated
        """
        # Check if we should reconnect
        if not self.auto_reconnect or self.stop_event.is_set():
            self.logger.info(f"Not reconnecting: auto_reconnect={self.auto_reconnect}, stopping={self.stop_event.is_set()}")
            return False
            
        # Calculate backoff delay
        self.reconnect_attempt += 1
        delay = min(
            WS_DEFAULT_RECONNECT_DELAY * (2 ** (self.reconnect_attempt - 1)),
            WS_MAX_RECONNECT_DELAY
        )
        
        # Add jitter to avoid thundering herd
        jitter = random.uniform(-WS_RECONNECT_JITTER, WS_RECONNECT_JITTER)
        delay = max(0.1, delay + (delay * jitter))
        
        self.logger.info(f"Reconnecting in {delay:.2f}s (attempt {self.reconnect_attempt})")
        
        # Update state
        self.state = ConnectionState.RECONNECTING
        
        # Wait for delay
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.logger.info(f"Reconnect cancelled during delay for {self.name}")
            return False
            
        # Check if we were cancelled during the delay
        if self.stop_event.is_set():
            self.logger.info(f"Reconnect cancelled during delay because stop event is set")
            return False
            
        # Start reconnection
        self.reconnect_task = asyncio.create_task(self._connect())
        return True
        
    async def _resubscribe(self):
        """Resubscribe to all active subscriptions after reconnection."""
        active_subs = [sub for sub in self.subscriptions.values() if sub.active]
        if not active_subs:
            self.logger.info(f"No active subscriptions to resubscribe to")
            return
            
        self.logger.info(f"Resubscribing to {len(active_subs)} channels")
        
        # Resubscribe to each channel
        for sub in active_subs:
            try:
                # Mark as inactive until subscription is confirmed
                sub.active = False
                
                # Resubscribe
                await self.subscribe(sub.channel, sub.params, sub.id)
                
            except Exception as e:
                self.logger.error(f"Error resubscribing to {sub.channel}: {str(e)}")
                
    async def send(self, data: Union[Dict[str, Any], str, bytes]):
        """
        Send a message to the WebSocket server.
        
        Args:
            data: Message data to send
            
        Raises:
            ConnectionError: If not connected
            WebSocketError: If send fails
        """
        if self.state != ConnectionState.CONNECTED or not self.ws or self.ws.closed:
            raise ConnectionError(f"Cannot send message, WebSocket not connected (state: {self.state})")
            
        try:
            # Convert dict to JSON string
            if isinstance(data, dict):
                data = json.dumps(data)
                
            # Send the message
            if isinstance(data, str):
                await self.ws.send_str(data)
                self.metrics.increment(f"ws_client.{self.name}.messages.sent.text")
            elif isinstance(data, bytes):
                await self.ws.send_bytes(data)
                self.metrics.increment(f"ws_client.{self.name}.messages.sent.binary")
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
                
            self.logger.debug(f"Sent WebSocket message: {data}")
            
        except Exception as e:
            self.logger.error(f"Error sending WebSocket message: {str(e)}")
            self.metrics.increment(f"ws_client.{self.name}.errors.send")
            raise WebSocketError(f"Failed to send WebSocket message: {str(e)}")
            
    async def authenticate(self, api_key: str, secret_key: str):
        """
        Authenticate with the WebSocket server.
        
        Args:
            api_key: API key for authentication
            secret_key: Secret key for authentication
            
        Raises:
            ConnectionError: If not connected
            AuthenticationError: If authentication fails
        """
        if self.state != ConnectionState.CONNECTED:
            raise ConnectionError(f"Cannot authenticate, WebSocket not connected (state: {self.state})")
            
        self.api_key = api_key
        self.secret_key = secret_key
        
        # Authentication logic depends on the exchange
        # This example shows Binance-style authentication
        if 'binance.com' in self.url:
            await self._authenticate_binance()
        elif 'deriv.com' in self.url or 'binary.com' in self.url:
            await self._authenticate_deriv()
        else:
            self.logger.warning(f"Authentication not implemented for URL: {self.url}")
            raise AuthenticationError(f"Authentication not implemented for this WebSocket service")
            
    async def _authenticate_binance(self):
        """Authenticate with Binance WebSocket."""
        try:
            self.logger.info(f"Authenticating with Binance WebSocket")
            
            # Generate timestamp
            timestamp = int(time.time() * 1000)
            
            # Generate signature
            signature_payload = f"timestamp={timestamp}"
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                signature_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Create authentication message
            auth_message = {
                "method": "AUTH",
                "params": {
                    "apiKey": self.api_key,
                    "timestamp": timestamp,
                    "signature": signature
                },
                "id": generate_nonce()
            }
            
            # Send authentication message
            await self.send(auth_message)
            
            # Note: Response will be handled in the message processing loop
            # We should implement a waiting mechanism for the auth response
            
            self.logger.info(f"Binance authentication message sent")
            
        except Exception as e:
            self.logger.error(f"Binance authentication error: {str(e)}")
            raise AuthenticationError(f"Binance authentication failed: {str(e)}")
            
    async def _authenticate_deriv(self):
        """Authenticate with Deriv WebSocket."""
        try:
            self.logger.info(f"Authenticating with Deriv WebSocket")
            
            # Deriv uses direct token authentication
            auth_message = {
                "authorize": self.api_key
            }
            
            # Send authentication message
            await self.send(auth_message)
            
            # Note: Response will be handled in the message processing loop
            # We should implement a waiting mechanism for the auth response
            
            self.logger.info(f"Deriv authentication message sent")
            
        except Exception as e:
            self.logger.error(f"Deriv authentication error: {str(e)}")
            raise AuthenticationError(f"Deriv authentication failed: {str(e)}")
            
    async def subscribe(
        self,
        channel: str,
        params: Optional[Dict[str, Any]] = None,
        subscription_id: Optional[str] = None
    ) -> str:
        """
        Subscribe to a WebSocket channel.
        
        Args:
            channel: Channel name to subscribe to
            params: Subscription parameters
            subscription_id: Optional ID for the subscription
            
        Returns:
            str: Subscription ID
            
        Raises:
            ConnectionError: If not connected
            SubscriptionError: If subscription fails
        """
        if self.state != ConnectionState.CONNECTED:
            raise ConnectionError(f"Cannot subscribe, WebSocket not connected (state: {self.state})")
            
        # Generate subscription ID if not provided
        sub_id = subscription_id or f"sub_{generate_nonce()}"
        
        # Create subscription object
        subscription = Subscription(
            id=sub_id,
            channel=channel,
            params=params or {},
            active=False,
            created_at=time.time()
        )
        
        try:
            self.logger.info(f"Subscribing to channel {channel} with params {params}")
            
            # Create subscription message based on exchange
            subscription_message = None
            
            if 'binance.com' in self.url:
                subscription_message = self._create_binance_subscription(channel, params, sub_id)
            elif 'deriv.com' in self.url or 'binary.com' in self.url:
                subscription_message = self._create_deriv_subscription(channel, params, sub_id)
            else:
                # Generic subscription format
                subscription_message = {
                    "method": "SUBSCRIBE",
                    "params": [channel] if params is None else [channel, params],
                    "id": sub_id
                }
                
            # Send subscription message
            await self.send(subscription_message)
            
            # Store subscription (will be marked active when confirmation is received)
            self.subscriptions[sub_id] = subscription
            
            # Note: Confirmation should be handled in the message processing
            # For now, we'll assume it succeeds immediately (not ideal)
            # In a production system, we would wait for confirmation
            subscription.active = True
            
            self.logger.info(f"Subscription message sent for {channel}")
            self.metrics.increment(f"ws_client.{self.name}.subscriptions")
            
            return sub_id
            
        except Exception as e:
            self.logger.error(f"Subscription error for channel {channel}: {str(e)}")
            self.metrics.increment(f"ws_client.{self.name}.errors.subscription")
            raise SubscriptionError(f"Failed to subscribe to {channel}: {str(e)}")
            
    def _create_binance_subscription(
        self,
        channel: str,
        params: Optional[Dict[str, Any]],
        sub_id: str
    ) -> Dict[str, Any]:
        """
        Create a Binance-specific subscription message.
        
        Args:
            channel: Channel name
            params: Subscription parameters
            sub_id: Subscription ID
            
        Returns:
            Dict[str, Any]: Subscription message
        """
        # Binance WebSocket API v2 format
        stream_names = []
        
        # Handle different channel formats
        if channel == "markPrice":
            # Mark price channel
            symbol = params.get("symbol", "").lower()
            frequency = params.get("frequency", "1s")
            stream_names.append(f"{symbol}@markPrice@{frequency}")
            
        elif channel == "kline":
            # Kline/candlestick channel
            symbol = params.get("symbol", "").lower()
            interval = params.get("interval", "1m")
            stream_names.append(f"{symbol}@kline_{interval}")
            
        elif channel == "trade":
            # Trade channel
            symbol = params.get("symbol", "").lower()
            stream_names.append(f"{symbol}@trade")
            
        elif channel == "depth":
            # Order book channel
            symbol = params.get("symbol", "").lower()
            level = params.get("level", "20")
            update_speed = params.get("speed", "100ms")
            stream_names.append(f"{symbol}@depth{level}@{update_speed}")
            
        else:
            # Generic channel format
            if params and "symbol" in params:
                symbol = params.get("symbol", "").lower()
                stream_names.append(f"{symbol}@{channel}")
            else:
                stream_names.append(channel)
                
        # Create subscription message
        return {
            "method": "SUBSCRIBE",
            "params": stream_names,
            "id": sub_id
        }
        
    def _create_deriv_subscription(
        self,
        channel: str,
        params: Optional[Dict[str, Any]],
        sub_id: str
    ) -> Dict[str, Any]:
        """
        Create a Deriv-specific subscription message.
        
        Args:
            channel: Channel name
            params: Subscription parameters
            sub_id: Subscription ID
            
        Returns:
            Dict[str, Any]: Subscription message
        """
        # Deriv uses a different format with command as the root key
        subscribe_message = {
            # Use channel name as the command
            channel: 1,
            # Add subscription_id for tracking
            "req_id": sub_id
        }
        
        # Add all parameters
        if params:
            subscribe_message.update(params)
            
        return subscribe_message
        
    async def unsubscribe(self, subscription_id: str):
        """
        Unsubscribe from a WebSocket channel.
        
        Args:
            subscription_id: Subscription ID to unsubscribe
            
        Raises:
            ConnectionError: If not connected
            SubscriptionError: If unsubscription fails
        """
        if self.state != ConnectionState.CONNECTED:
            raise ConnectionError(f"Cannot unsubscribe, WebSocket not connected (state: {self.state})")
            
        # Check if subscription exists
        if subscription_id not in self.subscriptions:
            self.logger.warning(f"Cannot unsubscribe from unknown subscription: {subscription_id}")
            return
            
        subscription = self.subscriptions[subscription_id]
        
        try:
            self.logger.info(f"Unsubscribing from channel {subscription.channel}")
            
            # Create unsubscription message based on exchange
            unsubscription_message = None
            
            if 'binance.com' in self.url:
                unsubscription_message = self._create_binance_unsubscription(subscription)
            elif 'deriv.com' in self.url or 'binary.com' in self.url:
                unsubscription_message = self._create_deriv_unsubscription(subscription)
            else:
                # Generic unsubscription format
                unsubscription_message = {
                    "method": "UNSUBSCRIBE",
                    "params": [subscription.channel],
                    "id": generate_nonce()
                }
                
            # Send unsubscription message
            await self.send(unsubscription_message)
            
            # Mark subscription as inactive
            subscription.active = False
            
            self.logger.info(f"Unsubscription message sent for {subscription.channel}")
            self.metrics.increment(f"ws_client.{self.name}.unsubscriptions")
            
        except Exception as e:
            self.logger.error(f"Unsubscription error for {subscription.channel}: {str(e)}")
            self.metrics.increment(f"ws_client.{self.name}.errors.unsubscription")
            raise SubscriptionError(f"Failed to unsubscribe from {subscription.channel}: {str(e)}")
            
    def _create_binance_unsubscription(self, subscription: Subscription) -> Dict[str, Any]:
        """
        Create a Binance-specific unsubscription message.
        
        Args:
            subscription: Subscription to unsubscribe
            
        Returns:
            Dict[str, Any]: Unsubscription message
        """
        # Binance WebSocket API format
        stream_names = []
        
        # Handle different channel formats (similar to subscription)
        if subscription.channel == "markPrice":
            symbol = subscription.params.get("symbol", "").lower()
            frequency = subscription.params.get("frequency", "1s")
            stream_names.append(f"{symbol}@markPrice@{frequency}")
            
        elif subscription.channel == "kline":
            symbol = subscription.params.get("symbol", "").lower()
            interval = subscription.params.get("interval", "1m")
            stream_names.append(f"{symbol}@kline_{interval}")
            
        elif subscription.channel == "trade":
            symbol = subscription.params.get("symbol", "").lower()
            stream_names.append(f"{symbol}@trade")
            
        elif subscription.channel == "depth":
            symbol = subscription.params.get("symbol", "").lower()
            level = subscription.params.get("level", "20")
            update_speed = subscription.params.get("speed", "100ms")
            stream_names.append(f"{symbol}@depth{level}@{update_speed}")
            
        else:
            if subscription.params and "symbol" in subscription.params:
                symbol = subscription.params.get("symbol", "").lower()
                stream_names.append(f"{symbol}@{subscription.channel}")
            else:
                stream_names.append(subscription.channel)
                
        # Create unsubscription message
        return {
            "method": "UNSUBSCRIBE",
            "params": stream_names,
            "id": generate_nonce()
        }
        
    def _create_deriv_unsubscription(self, subscription: Subscription) -> Dict[str, Any]:
        """
        Create a Deriv-specific unsubscription message.
        
        Args:
            subscription: Subscription to unsubscribe
            
        Returns:
            Dict[str, Any]: Unsubscription message
        """
        # Deriv uses a different format with "forget" command
        forget_message = {
            "forget": subscription.params.get("subscribe", "")
        }
        
        return forget_message
        
    def is_connected(self) -> bool:
        """
        Check if the WebSocket is connected.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.state == ConnectionState.CONNECTED and self.ws and not self.ws.closed
        
    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """
        Get a subscription by ID.
        
        Args:
            subscription_id: Subscription ID
            
        Returns:
            Optional[Subscription]: Subscription if found, None otherwise
        """
        return self.subscriptions.get(subscription_id)
        
    def get_subscriptions_by_channel(self, channel: str) -> List[Subscription]:
        """
        Get all subscriptions for a channel.
        
        Args:
            channel: Channel name
            
        Returns:
            List[Subscription]: List of subscriptions
        """
        return [sub for sub in self.subscriptions.values() if sub.channel == channel]
        
    def get_active_subscriptions(self) -> List[Subscription]:
        """
        Get all active subscriptions.
        
        Returns:
            List[Subscription]: List of active subscriptions
        """
        return [sub for sub in self.subscriptions.values() if sub.active]
