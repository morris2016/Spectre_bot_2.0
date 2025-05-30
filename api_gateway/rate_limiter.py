#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Rate Limiter Module

This module implements rate limiting for API requests and WebSocket messages
to prevent abuse and ensure fair system usage.
"""

import time
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

import redis.asyncio as redis
from fastapi import Request, HTTPException, status, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint

from common.logger import get_logger
from common.redis_client import get_redis_pool
from common.metrics import MetricsCollector

logger = get_logger(__name__)
metrics = MetricsCollector()

class RateLimiter:
    """
    Rate limiter implementation using a sliding window algorithm.
    
    This class can be used both for API rate limiting and WebSocket message limiting.
    It supports both in-memory and Redis-backed implementations.
    """
    
    def __init__(self, max_calls: int, time_frame: int, use_redis: bool = False):
        """
        Initialize the rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed within the time frame
            time_frame: Time frame in seconds
            use_redis: Whether to use Redis for distributed rate limiting
        """
        self.max_calls = max_calls
        self.time_frame = time_frame
        self.use_redis = use_redis
        self.requests: Dict[str, List[float]] = {}  # key -> list of timestamps
        self.redis_pool = None
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the rate limiter if using Redis."""
        if self.use_redis:
            self.redis_pool = await get_redis_pool()
        
    async def check_rate_limit(self, key: str) -> bool:
        """
        Check if a request is allowed under the rate limit.
        
        Args:
            key: Identifier for the client (e.g., IP address or user ID)
            
        Returns:
            bool: True if request is allowed, False if rate limit exceeded
        """
        if self.use_redis and self.redis_pool:
            return await self._check_rate_limit_redis(key)
        else:
            return await self._check_rate_limit_memory(key)
    
    async def _check_rate_limit_memory(self, key: str) -> bool:
        """In-memory implementation of rate limiting."""
        async with self._lock:
            current_time = time.time()
            
            # Initialize if this is the first request from this key
            if key not in self.requests:
                self.requests[key] = []
                
            # Remove timestamps outside the current time window
            self.requests[key] = [ts for ts in self.requests[key] if current_time - ts <= self.time_frame]
                
            # Check if adding a new request would exceed the limit
            if len(self.requests[key]) >= self.max_calls:
                metrics.increment('rate_limit_exceeded')
                return False
                
            # Add the current request timestamp
            self.requests[key].append(current_time)
            return True
        
    async def _check_rate_limit_redis(self, key: str) -> bool:
        """Redis-based implementation of rate limiting for distributed systems."""
        try:
            redis_client = redis.Redis(connection_pool=self.redis_pool)
            current_time = time.time()
            redis_key = f"rate_limit:{key}"
            
            # Using Redis pipeline for atomic operations
            pipe = redis_client.pipeline()
            
            # Add current timestamp to the sorted set
            pipe.zadd(redis_key, {str(current_time): current_time})
            
            # Remove timestamps outside the current time window
            cutoff_time = current_time - self.time_frame
            pipe.zremrangebyscore(redis_key, 0, cutoff_time)
            
            # Count timestamps in window
            pipe.zcard(redis_key)
            
            # Set expiry on the key to clean up automatically
            pipe.expire(redis_key, self.time_frame * 2)
            
            # Execute the pipeline
            results = await pipe.execute()
            request_count = results[2]  # The result of ZCARD operation
            
            # Check if we're over the limit
            if request_count > self.max_calls:
                metrics.increment('rate_limit_exceeded')
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Redis rate limiter error: {str(e)}")
            # Fail open if Redis is unavailable
            return True
    
    async def get_remaining_requests(self, key: str) -> int:
        """
        Get the number of remaining allowed requests in the current time window.
        
        Args:
            key: Identifier for the client
            
        Returns:
            int: Number of remaining allowed requests
        """
        if self.use_redis and self.redis_pool:
            return await self._get_remaining_redis(key)
        else:
            return await self._get_remaining_memory(key)
    
    async def _get_remaining_memory(self, key: str) -> int:
        """In-memory implementation for remaining request calculation."""
        async with self._lock:
            current_time = time.time()
            
            # Key doesn't exist yet
            if key not in self.requests:
                return self.max_calls
                
            # Clean up old timestamps
            self.requests[key] = [ts for ts in self.requests[key] if current_time - ts <= self.time_frame]
            
            return max(0, self.max_calls - len(self.requests[key]))
            
    async def _get_remaining_redis(self, key: str) -> int:
        """Redis implementation for remaining request calculation."""
        try:
            redis_client = redis.Redis(connection_pool=self.redis_pool)
            current_time = time.time()
            redis_key = f"rate_limit:{key}"
            
            # Using Redis pipeline for atomic operations
            pipe = redis_client.pipeline()
            
            # Remove timestamps outside the current time window
            cutoff_time = current_time - self.time_frame
            pipe.zremrangebyscore(redis_key, 0, cutoff_time)
            
            # Count timestamps in window
            pipe.zcard(redis_key)
            
            # Execute the pipeline
            results = await pipe.execute()
            request_count = results[1]  # The result of ZCARD operation
            
            return max(0, self.max_calls - request_count)
            
        except Exception as e:
            logger.error(f"Redis rate limiter error: {str(e)}")
            # Return a conservative estimate if Redis is unavailable
            return 1
    
    async def reset_rate_limit(self, key: str) -> bool:
        """
        Reset the rate limit for a specific key.
        
        Args:
            key: Identifier for the client
            
        Returns:
            bool: True if successful
        """
        if self.use_redis and self.redis_pool:
            try:
                redis_client = redis.Redis(connection_pool=self.redis_pool)
                await redis_client.delete(f"rate_limit:{key}")
                return True
            except Exception as e:
                logger.error(f"Failed to reset rate limit in Redis: {str(e)}")
                return False
        else:
            async with self._lock:
                if key in self.requests:
                    del self.requests[key]
                return True

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for API rate limiting.
    
    This middleware applies rate limiting to all API requests based on
    client IP address or authenticated user ID.
    """
    
    def __init__(self, app, *, 
                 general_rate_limiter: RateLimiter, 
                 endpoint_rate_limiters: Dict[str, RateLimiter] = None,
                 user_rate_limiter: RateLimiter = None):
        """
        Initialize the rate limit middleware.
        
        Args:
            app: The FastAPI application
            general_rate_limiter: Default rate limiter for all endpoints
            endpoint_rate_limiters: Optional specific rate limiters for certain endpoints
            user_rate_limiter: Optional rate limiter for authenticated users
        """
        super().__init__(app)
        self.general_rate_limiter = general_rate_limiter
        self.endpoint_rate_limiters = endpoint_rate_limiters or {}
        self.user_rate_limiter = user_rate_limiter
        
    async def initialize_middleware(self):
        """Initialize the middleware's rate limiters."""
        await self.general_rate_limiter.initialize()
        
        for limiter in self.endpoint_rate_limiters.values():
            await limiter.initialize()
            
        if self.user_rate_limiter:
            await self.user_rate_limiter.initialize()
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        """
        Process the request and apply rate limiting.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler
            
        Returns:
            Response: The HTTP response
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        # Skip WebSocket connections (handled separately)
        if request.scope["type"] == "websocket":
            return await call_next(request)
            
        # Skip certain paths like health checks
        path = request.url.path
        if path in ["/health", "/metrics", "/favicon.ico"]:
            return await call_next(request)
        
        client_ip = request.client.host
        user_id = None
        
        # Try to get authenticated user ID if available
        try:
            # This assumes that user details are set by an auth middleware
            if hasattr(request.state, "user") and request.state.user:
                user_id = request.state.user.get("id")
        except Exception:
            pass
            
        # Determine which rate limiter to use
        rate_limiter = self.general_rate_limiter
        rate_limit_key = client_ip
        
        # Use endpoint-specific rate limiter if exists
        for endpoint_path, endpoint_limiter in self.endpoint_rate_limiters.items():
            if path.startswith(endpoint_path):
                rate_limiter = endpoint_limiter
                break
                
        # Use authenticated user rate limiter if user is authenticated
        if user_id and self.user_rate_limiter:
            rate_limiter = self.user_rate_limiter
            rate_limit_key = f"user:{user_id}"
            
        # Check rate limit
        allowed = await rate_limiter.check_rate_limit(rate_limit_key)
        if not allowed:
            metrics.increment("api_rate_limit_exceeded")
            remaining = await rate_limiter.get_remaining_requests(rate_limit_key)
            
            logger.warning(f"Rate limit exceeded: {rate_limit_key} on path {path}")
            
            # Return a proper rate limit exceeded response
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": str(rate_limiter.time_frame),
                    "X-RateLimit-Limit": str(rate_limiter.max_calls),
                    "X-RateLimit-Remaining": str(remaining),
                    "X-RateLimit-Reset": str(int(time.time() + rate_limiter.time_frame))
                }
            )
        
        # Get remaining request count for headers
        remaining = await rate_limiter.get_remaining_requests(rate_limit_key)
        
        # Process the request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + rate_limiter.time_frame))
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

async def get_rate_limiter():
    """
    Dependency to get a rate limiter instance.
    
    Returns:
        RateLimiter: Rate limiter instance
    """
    rate_limiter = RateLimiter(max_calls=100, time_frame=60)
    await rate_limiter.initialize()
    return rate_limiter

# Create default rate limiters
api_general_limiter = RateLimiter(max_calls=120, time_frame=60, use_redis=True)
api_auth_limiter = RateLimiter(max_calls=20, time_frame=60, use_redis=True)
api_trading_limiter = RateLimiter(max_calls=60, time_frame=60, use_redis=True)
ws_message_limiter = RateLimiter(max_calls=100, time_frame=60, use_redis=True)
user_limiter = RateLimiter(max_calls=600, time_frame=60, use_redis=True)

# Define endpoint-specific rate limiters
endpoint_limiters = {
    "/api/v1/auth": api_auth_limiter,
    "/api/v1/trading": api_trading_limiter
}

async def initialize_rate_limiters():
    """Initialize all rate limiters."""
    await api_general_limiter.initialize()
    await api_auth_limiter.initialize()
    await api_trading_limiter.initialize()
    await ws_message_limiter.initialize()
    await user_limiter.initialize()
