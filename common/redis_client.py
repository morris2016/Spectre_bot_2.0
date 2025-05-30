#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Redis Client Module

This module provides a Redis client for pub/sub, caching, and temporary storage.
"""

import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable

try:
    import redis.asyncio as redis  # type: ignore
    REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    redis = None  # type: ignore
    REDIS_AVAILABLE = False

from common.logger import get_logger
from common.exceptions import RedisError, RedisConnectionError

_redis_client: Optional['RedisClient'] = None
_redis_pool = None


async def get_redis_pool(host: str = "localhost", port: int = 6379, db: int = 0,
                        password: Optional[str] = None, ssl: bool = False,
                        timeout: int = 10, max_connections: int = 50):
    """Get a shared Redis connection pool.

    Creates a :class:`RedisClient` on first use and reuses its connection pool
    on subsequent calls.
    """
    global _redis_client, _redis_pool
    if _redis_pool is None:
        _redis_client = RedisClient(
            host=host,
            port=port,
            db=db,
            password=password,
            ssl=ssl,
            timeout=timeout,
            max_connections=max_connections,
        )
        await _redis_client.initialize()
        _redis_pool = _redis_client.client.connection_pool
    return _redis_pool

class RedisClient:
    """Client for Redis operations."""
    
    def __init__(self, host="localhost", port=6379, db=0, password=None, 
                ssl=False, timeout=10, max_connections=50):
        """
        Initialize Redis client.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database
            password: Redis password
            ssl: Whether to use SSL
            timeout: Connection timeout in seconds
            max_connections: Maximum number of connections
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ssl = ssl
        self.timeout = timeout
        self.max_connections = max_connections
        self.client = None
        self.pubsub = None
        self.subscriber_tasks = {}
        self.logger = get_logger("RedisClient")
        
    async def initialize(self):
        """
        Initialize the Redis connection pool.
        
        Raises:
            RedisConnectionError: If connection fails
        """
        try:
            # Create connection pool
            connection_kwargs = {
                'host': self.host,
                'port': self.port,
                'db': self.db,
                'password': self.password,
                'ssl': self.ssl,
                'socket_timeout': self.timeout,
                'socket_connect_timeout': self.timeout,
                'socket_keepalive': True,
                'retry_on_timeout': True,
                'max_connections': self.max_connections
            }
            
            self.client = redis.Redis(**connection_kwargs)
            self.pubsub = self.client.pubsub()
            
            # Test connection
            await self.client.ping()
            
            self.logger.info(f"Connected to Redis at {self.host}:{self.port} (db: {self.db})")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {str(e)}")
            raise RedisConnectionError(f"Failed to connect to Redis: {str(e)}")
            
    async def close(self):
        """Close the Redis connection."""
        if self.client:
            # Cancel all subscriber tasks
            for task in self.subscriber_tasks.values():
                if not task.done():
                    task.cancel()
                    
            # Close pubsub
            if self.pubsub:
                await self.pubsub.close()
                
            # Close client
            await self.client.close()
            
            self.logger.info("Redis connection closed")
            
    async def ping(self):
        """
        Ping the Redis server.
        
        Returns:
            True if successful
            
        Raises:
            RedisError: If ping fails
        """
        try:
            return await self.client.ping()
        except Exception as e:
            self.logger.error(f"Redis ping failed: {str(e)}")
            raise RedisError(f"Redis ping failed: {str(e)}")
            
    async def get(self, key):
        """
        Get a value from Redis.
        
        Args:
            key: Key to get
            
        Returns:
            Value or None if not found
            
        Raises:
            RedisError: If get fails
        """
        try:
            result = await self.client.get(key)
            if result is not None:
                try:
                    return json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    return result.decode('utf-8') if isinstance(result, bytes) else result
            return None
        except Exception as e:
            self.logger.error(f"Redis get failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis get failed: {str(e)}")
            
    async def set(self, key, value, expire=None):
        """
        Set a value in Redis.
        
        Args:
            key: Key to set
            value: Value to set
            expire: Optional expiration time in seconds
            
        Returns:
            True if successful
            
        Raises:
            RedisError: If set fails
        """
        try:
            # Convert value to JSON if it's not a string or bytes
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
                
            return await self.client.set(key, value, ex=expire)
        except Exception as e:
            self.logger.error(f"Redis set failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis set failed: {str(e)}")
            
    async def delete(self, key):
        """
        Delete a key from Redis.
        
        Args:
            key: Key to delete
            
        Returns:
            Number of keys deleted
            
        Raises:
            RedisError: If delete fails
        """
        try:
            return await self.client.delete(key)
        except Exception as e:
            self.logger.error(f"Redis delete failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis delete failed: {str(e)}")
            
    async def exists(self, key):
        """
        Check if a key exists in Redis.
        
        Args:
            key: Key to check
            
        Returns:
            True if the key exists
            
        Raises:
            RedisError: If check fails
        """
        try:
            return await self.client.exists(key)
        except Exception as e:
            self.logger.error(f"Redis exists failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis exists failed: {str(e)}")
            
    async def publish(self, channel, message):
        """
        Publish a message to a Redis channel.
        
        Args:
            channel: Channel to publish to
            message: Message to publish
            
        Returns:
            Number of clients that received the message
            
        Raises:
            RedisError: If publish fails
        """
        try:
            # Convert message to JSON if it's not a string or bytes
            if not isinstance(message, (str, bytes)):
                message = json.dumps(message)
                
            return await self.client.publish(channel, message)
        except Exception as e:
            self.logger.error(f"Redis publish failed for channel '{channel}': {str(e)}")
            raise RedisError(f"Redis publish failed: {str(e)}")
            
    async def subscribe(self, channels, callback):
        """
        Subscribe to Redis channels.
        
        Args:
            channels: Channel or list of channels to subscribe to
            callback: Callback function to call when a message is received
            
        Raises:
            RedisError: If subscribe fails
        """
        try:
            if isinstance(channels, str):
                channels = [channels]
                
            # Subscribe to channels
            for channel in channels:
                await self.pubsub.subscribe(**{channel: callback})
                
            # Start listening task if not already running
            task_key = ','.join(channels)
            if task_key not in self.subscriber_tasks:
                self.subscriber_tasks[task_key] = asyncio.create_task(self._listener(channels))
                
            self.logger.info(f"Subscribed to Redis channels: {channels}")
            
        except Exception as e:
            self.logger.error(f"Redis subscribe failed for channels '{channels}': {str(e)}")
            raise RedisError(f"Redis subscribe failed: {str(e)}")
            
    async def _listener(self, channels):
        """
        Listen for messages on subscribed channels.
        
        Args:
            channels: Channels being listened to
        """
        self.logger.debug(f"Starting Redis listener for channels: {channels}")
        try:
            while True:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is not None:
                    self.logger.debug(f"Received Redis message on channel {message['channel']}: {message['data']}")
                await asyncio.sleep(0.01)  # Prevent CPU hogging
        except asyncio.CancelledError:
            self.logger.debug(f"Redis listener cancelled for channels: {channels}")
        except Exception as e:
            self.logger.error(f"Error in Redis listener for channels {channels}: {str(e)}")
            
    async def unsubscribe(self, channels):
        """
        Unsubscribe from Redis channels.
        
        Args:
            channels: Channel or list of channels to unsubscribe from
            
        Raises:
            RedisError: If unsubscribe fails
        """
        try:
            if isinstance(channels, str):
                channels = [channels]
                
            # Unsubscribe from channels
            await self.pubsub.unsubscribe(*channels)
            
            # Cancel listener task
            task_key = ','.join(channels)
            if task_key in self.subscriber_tasks:
                if not self.subscriber_tasks[task_key].done():
                    self.subscriber_tasks[task_key].cancel()
                del self.subscriber_tasks[task_key]
                
            self.logger.info(f"Unsubscribed from Redis channels: {channels}")
            
        except Exception as e:
            self.logger.error(f"Redis unsubscribe failed for channels '{channels}': {str(e)}")
            raise RedisError(f"Redis unsubscribe failed: {str(e)}")
            
    async def incr(self, key):
        """
        Increment a key in Redis.
        
        Args:
            key: Key to increment
            
        Returns:
            New value
            
        Raises:
            RedisError: If increment fails
        """
        try:
            return await self.client.incr(key)
        except Exception as e:
            self.logger.error(f"Redis incr failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis incr failed: {str(e)}")
            
    async def decr(self, key):
        """
        Decrement a key in Redis.
        
        Args:
            key: Key to decrement
            
        Returns:
            New value
            
        Raises:
            RedisError: If decrement fails
        """
        try:
            return await self.client.decr(key)
        except Exception as e:
            self.logger.error(f"Redis decr failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis decr failed: {str(e)}")
            
    async def expire(self, key, seconds):
        """
        Set expiration on a key.
        
        Args:
            key: Key to set expiration on
            seconds: Expiration time in seconds
            
        Returns:
            True if successful
            
        Raises:
            RedisError: If expire fails
        """
        try:
            return await self.client.expire(key, seconds)
        except Exception as e:
            self.logger.error(f"Redis expire failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis expire failed: {str(e)}")
            
    async def ttl(self, key):
        """
        Get the time to live for a key.
        
        Args:
            key: Key to get TTL for
            
        Returns:
            TTL in seconds, -1 if no expiry, -2 if key doesn't exist
            
        Raises:
            RedisError: If TTL fails
        """
        try:
            return await self.client.ttl(key)
        except Exception as e:
            self.logger.error(f"Redis ttl failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis ttl failed: {str(e)}")
            
    async def hget(self, key, field):
        """
        Get a field from a hash in Redis.
        
        Args:
            key: Hash key
            field: Field to get
            
        Returns:
            Field value or None if not found
            
        Raises:
            RedisError: If hget fails
        """
        try:
            result = await self.client.hget(key, field)
            if result is not None:
                try:
                    return json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    return result.decode('utf-8') if isinstance(result, bytes) else result
            return None
        except Exception as e:
            self.logger.error(f"Redis hget failed for key '{key}', field '{field}': {str(e)}")
            raise RedisError(f"Redis hget failed: {str(e)}")
            
    async def hset(self, key, field, value):
        """
        Set a field in a hash in Redis.
        
        Args:
            key: Hash key
            field: Field to set
            value: Value to set
            
        Returns:
            1 if field is new, 0 if field already exists
            
        Raises:
            RedisError: If hset fails
        """
        try:
            # Convert value to JSON if it's not a string or bytes
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
                
            return await self.client.hset(key, field, value)
        except Exception as e:
            self.logger.error(f"Redis hset failed for key '{key}', field '{field}': {str(e)}")
            raise RedisError(f"Redis hset failed: {str(e)}")
            
    async def hdel(self, key, field):
        """
        Delete a field from a hash in Redis.
        
        Args:
            key: Hash key
            field: Field to delete
            
        Returns:
            1 if field was deleted, 0 if field doesn't exist
            
        Raises:
            RedisError: If hdel fails
        """
        try:
            return await self.client.hdel(key, field)
        except Exception as e:
            self.logger.error(f"Redis hdel failed for key '{key}', field '{field}': {str(e)}")
            raise RedisError(f"Redis hdel failed: {str(e)}")
            
    async def hgetall(self, key):
        """
        Get all fields and values from a hash in Redis.
        
        Args:
            key: Hash key
            
        Returns:
            Dictionary of fields and values
            
        Raises:
            RedisError: If hgetall fails
        """
        try:
            result = await self.client.hgetall(key)
            if not result:
                return {}
                
            # Parse values
            parsed = {}
            for field, value in result.items():
                field_str = field.decode('utf-8') if isinstance(field, bytes) else field
                try:
                    parsed[field_str] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    parsed[field_str] = value.decode('utf-8') if isinstance(value, bytes) else value
                    
            return parsed
        except Exception as e:
            self.logger.error(f"Redis hgetall failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis hgetall failed: {str(e)}")
            
    async def rpush(self, key, value):
        """
        Push a value to the end of a list in Redis.
        
        Args:
            key: List key
            value: Value to push
            
        Returns:
            Length of the list after the push
            
        Raises:
            RedisError: If rpush fails
        """
        try:
            # Convert value to JSON if it's not a string or bytes
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
                
            return await self.client.rpush(key, value)
        except Exception as e:
            self.logger.error(f"Redis rpush failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis rpush failed: {str(e)}")
            
    async def lpush(self, key, value):
        """
        Push a value to the start of a list in Redis.
        
        Args:
            key: List key
            value: Value to push
            
        Returns:
            Length of the list after the push
            
        Raises:
            RedisError: If lpush fails
        """
        try:
            # Convert value to JSON if it's not a string or bytes
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
                
            return await self.client.lpush(key, value)
        except Exception as e:
            self.logger.error(f"Redis lpush failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis lpush failed: {str(e)}")
            
    async def lpop(self, key):
        """
        Pop a value from the start of a list in Redis.
        
        Args:
            key: List key
            
        Returns:
            Popped value or None if list is empty
            
        Raises:
            RedisError: If lpop fails
        """
        try:
            result = await self.client.lpop(key)
            if result is not None:
                try:
                    return json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    return result.decode('utf-8') if isinstance(result, bytes) else result
            return None
        except Exception as e:
            self.logger.error(f"Redis lpop failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis lpop failed: {str(e)}")
            
    async def rpop(self, key):
        """
        Pop a value from the end of a list in Redis.
        
        Args:
            key: List key
            
        Returns:
            Popped value or None if list is empty
            
        Raises:
            RedisError: If rpop fails
        """
        try:
            result = await self.client.rpop(key)
            if result is not None:
                try:
                    return json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    return result.decode('utf-8') if isinstance(result, bytes) else result
            return None
        except Exception as e:
            self.logger.error(f"Redis rpop failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis rpop failed: {str(e)}")
            
    async def lrange(self, key, start, end):
        """
        Get a range of values from a list in Redis.
        
        Args:
            key: List key
            start: Start index
            end: End index
            
        Returns:
            List of values
            
        Raises:
            RedisError: If lrange fails
        """
        try:
            result = await self.client.lrange(key, start, end)
            if not result:
                return []
                
            # Parse values
            parsed = []
            for value in result:
                try:
                    parsed.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    parsed.append(value.decode('utf-8') if isinstance(value, bytes) else value)
                    
            return parsed
        except Exception as e:
            self.logger.error(f"Redis lrange failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis lrange failed: {str(e)}")
            
    async def sadd(self, key, value):
        """
        Add a value to a set in Redis.
        
        Args:
            key: Set key
            value: Value to add
            
        Returns:
            Number of elements added (0 if already exists)
            
        Raises:
            RedisError: If sadd fails
        """
        try:
            # Convert value to JSON if it's not a string or bytes
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
                
            return await self.client.sadd(key, value)
        except Exception as e:
            self.logger.error(f"Redis sadd failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis sadd failed: {str(e)}")
            
    async def srem(self, key, value):
        """
        Remove a value from a set in Redis.
        
        Args:
            key: Set key
            value: Value to remove
            
        Returns:
            Number of elements removed (0 if not a member)
            
        Raises:
            RedisError: If srem fails
        """
        try:
            # Convert value to JSON if it's not a string or bytes
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
                
            return await self.client.srem(key, value)
        except Exception as e:
            self.logger.error(f"Redis srem failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis srem failed: {str(e)}")
            
    async def smembers(self, key):
        """
        Get all members of a set in Redis.
        
        Args:
            key: Set key
            
        Returns:
            Set of members
            
        Raises:
            RedisError: If smembers fails
        """
        try:
            result = await self.client.smembers(key)
            if not result:
                return set()
                
            # Parse values
            parsed = set()
            for value in result:
                try:
                    parsed.add(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    parsed.add(value.decode('utf-8') if isinstance(value, bytes) else value)
                    
            return parsed
        except Exception as e:
            self.logger.error(f"Redis smembers failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis smembers failed: {str(e)}")
            
    async def sismember(self, key, value):
        """
        Check if a value is a member of a set in Redis.
        
        Args:
            key: Set key
            value: Value to check
            
        Returns:
            True if member, False otherwise
            
        Raises:
            RedisError: If sismember fails
        """
        try:
            # Convert value to JSON if it's not a string or bytes
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
                
            return bool(await self.client.sismember(key, value))
        except Exception as e:
            self.logger.error(f"Redis sismember failed for key '{key}': {str(e)}")
            raise RedisError(f"Redis sismember failed: {str(e)}")
