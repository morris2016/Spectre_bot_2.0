#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
API Middleware

This module provides middleware components for the API Gateway, handling
cross-cutting concerns like logging, security, error handling, and request/response
processing.
"""

import time
import json
import asyncio
from datetime import datetime, timezone
import logging
import traceback
from typing import Dict, List, Callable, Any, Optional

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp
import jwt

from config import Config
from common.logger import get_logger
from common.metrics import MetricsCollector
from common.exceptions import (
    AuthenticationError, PermissionDeniedError, NotFoundError,
    ValidationError, ServiceUnavailableError
)

# Initialize logger and metrics collector
logger = get_logger(__name__)
metrics = MetricsCollector(namespace="api_gateway")

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and metrics collection."""
    
    def __init__(self, app: ASGIApp, config: Config):
        """Initialize with app and config."""
        super().__init__(app)
        self.config = config
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request and response for logging and metrics."""
        # Generate request ID if not present
        request_id = request.headers.get("X-Request-ID", f"req_{time.time_ns()}")
        
        # Start timing
        start_time = time.time()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Prepare path for metrics (remove IDs to avoid high cardinality)
        path = request.url.path
        metric_path = path
        
        # Clean metric path - replace IDs with placeholders
        parts = path.split('/')
        metric_parts = []
        for part in parts:
            # If part looks like an ID (UUID, int), replace with placeholder
            if part.isdigit() or (len(part) > 30 and '-' in part):
                metric_parts.append("{id}")
            else:
                metric_parts.append(part)
        
        metric_path = "/".join(metric_parts)
        if not metric_path:
            metric_path = "/"
        
        # Log request
        logger.info(
            f"Request: id={request_id} method={request.method} "
            f"path={path} client={request.client.host}"
        )
        
        # Track metrics
        metrics.increment("http_requests_total", tags={
            "method": request.method,
            "path": metric_path
        })
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Add headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = str(round(response_time * 1000))
            
            # Log response
            logger.info(
                f"Response: id={request_id} method={request.method} "
                f"path={path} status={response.status_code} "
                f"time={response_time:.4f}s"
            )
            
            # Track response metrics
            metrics.histogram("http_request_duration_seconds", response_time, tags={
                "method": request.method,
                "path": metric_path,
                "status": str(response.status_code)
            })
            
            metrics.increment("http_responses_total", tags={
                "method": request.method,
                "path": metric_path,
                "status": str(response.status_code)
            })
            
            return response
            
        except Exception as e:
            # Calculate response time even for errors
            response_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Error: id={request_id} method={request.method} "
                f"path={path} error={str(e)} time={response_time:.4f}s"
            )
            logger.error(traceback.format_exc())
            
            # Track error metrics
            metrics.increment("http_errors_total", tags={
                "method": request.method,
                "path": metric_path,
                "error": type(e).__name__
            })
            
            # Convert exception to HTTP response
            if isinstance(e, HTTPException):
                # Pass through FastAPI HTTPExceptions
                error_response = JSONResponse(
                    status_code=e.status_code,
                    content={
                        "error": e.detail,
                        "status": "error",
                        "request_id": request_id
                    }
                )
            elif isinstance(e, AuthenticationError):
                error_response = JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": str(e),
                        "status": "error",
                        "request_id": request_id
                    }
                )
            elif isinstance(e, PermissionDeniedError):
                error_response = JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": str(e),
                        "status": "error",
                        "request_id": request_id
                    }
                )
            elif isinstance(e, NotFoundError):
                error_response = JSONResponse(
                    status_code=status.HTTP_404_NOT_FOUND,
                    content={
                        "error": str(e),
                        "status": "error",
                        "request_id": request_id
                    }
                )
            elif isinstance(e, ValidationError):
                error_response = JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": str(e),
                        "status": "error",
                        "request_id": request_id,
                        "validation_errors": getattr(e, "errors", None)
                    }
                )
            elif isinstance(e, ServiceUnavailableError):
                error_response = JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "error": str(e),
                        "status": "error",
                        "request_id": request_id
                    }
                )
            else:
                # Generic server error for unhandled exceptions
                error_response = JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "error": "Internal server error",
                        "status": "error",
                        "request_id": request_id
                    }
                )
            
            # Add headers to error response
            error_response.headers["X-Request-ID"] = request_id
            error_response.headers["X-Response-Time"] = str(round(response_time * 1000))
            
            return error_response

class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and protections."""
    
    def __init__(self, app: ASGIApp, config: Config):
        """Initialize with app and config."""
        super().__init__(app)
        self.config = config
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add security headers and protections."""
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add HSTS header if using HTTPS
        if self.config.get("api.use_https", False):
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Add Content Security Policy
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
            "img-src 'self' data: https: blob:",
            "font-src 'self' https://cdn.jsdelivr.net",
            "connect-src 'self' wss:",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""
    
    def __init__(self, app: ASGIApp, config: Config, redis_client):
        """Initialize with app, config, and Redis client."""
        super().__init__(app)
        self.config = config
        self.redis_client = redis_client
        self.default_rate_limit = config.get("api.rate_limit", 100)  # requests per minute
        self.rate_limit_window = config.get("api.rate_limit_window", 60)  # seconds
        
        # Path-specific rate limits
        self.path_rate_limits = config.get("api.path_rate_limits", {
            "/api/v1/auth/login": 10,  # 10 login attempts per minute
            "/api/v1/auth/register": 5  # 5 registration attempts per minute
        })
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply rate limiting."""
        # Skip rate limiting for certain paths
        path = request.url.path
        if path.startswith("/api/v1/health") or path.startswith("/api/v1/metrics"):
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host
        
        # Get user ID from request state if available (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        
        # Determine rate limit key and limit
        if user_id:
            # Use user ID for authenticated requests
            rate_limit_key = f"rate_limit:user:{user_id}"
            rate_limit = self.default_rate_limit * 2  # Higher limit for authenticated users
        else:
            # Use IP for unauthenticated requests
            rate_limit_key = f"rate_limit:ip:{client_ip}"
            rate_limit = self.default_rate_limit
        
        # Check path-specific rate limits
        for pattern, limit in self.path_rate_limits.items():
            if path.startswith(pattern):
                # Path-specific key and limit
                path_key = pattern.replace("/", "_")
                if user_id:
                    rate_limit_key = f"rate_limit:path:{path_key}:user:{user_id}"
                else:
                    rate_limit_key = f"rate_limit:path:{path_key}:ip:{client_ip}"
                rate_limit = limit
                break
        
        # Check rate limit
        try:
            # Increment counter
            current = await self.redis_client.incr(rate_limit_key)
            
            # Set expiration if new key
            if current == 1:
                await self.redis_client.expire(rate_limit_key, self.rate_limit_window)
            
            # Check if limit exceeded
            if current > rate_limit:
                # Track rate limit hits
                metrics.increment("rate_limit_exceeded_total", tags={
                    "path": path,
                    "user_id": str(user_id) if user_id else "anonymous"
                })
                
                # Return rate limit error
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "status": "error",
                        "request_id": getattr(request.state, "request_id", "unknown")
                    }
                )
            
            # Add rate limit headers
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(rate_limit)
            response.headers["X-RateLimit-Remaining"] = str(max(0, rate_limit - current))
            response.headers["X-RateLimit-Reset"] = str(self.rate_limit_window)
            
            return response
            
        except Exception as e:
            # Log error but don't block request on rate limit failure
            logger.error(f"Rate limit error: {str(e)}")
            return await call_next(request)

def configure_middleware(app, config: Config, redis_client):
    """Configure middleware for the FastAPI application."""
    # Add CORS middleware first
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("api.cors.origins", ["*"]),
        allow_credentials=config.get("api.cors.allow_credentials", True),
        allow_methods=config.get("api.cors.methods", ["*"]),
        allow_headers=config.get("api.cors.headers", ["*"]),
        expose_headers=config.get("api.cors.expose_headers", [
            "X-Request-ID", "X-Response-Time", 
            "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"
        ])
    )
    
    # Add custom middlewares
    app.add_middleware(LoggingMiddleware, config=config)
    app.add_middleware(SecurityMiddleware, config=config)
    
    # Add rate limiting if Redis is available
    if redis_client:
        app.add_middleware(RateLimitMiddleware, config=config, redis_client=redis_client)
