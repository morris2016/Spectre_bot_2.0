#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Advanced REST Client

This module provides a highly optimized REST client for fetching data from
various APIs, with advanced features such as:
- Adaptive retry with exponential backoff
- Circuit breaker pattern implementation
- Comprehensive error handling
- Connection pooling and performance optimization
- Rate limiting protection
- Request signing for exchange APIs
- Transparent compression/decompression
- Metrics collection
"""

import time
import json
import hmac
import hashlib
import base64
import zlib
import logging
import random
import asyncio
import aiohttp
import backoff
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlencode
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

# Internal imports
from config import Config
from common.logger import get_logger
from common.exceptions import (
    RESTClientError, RequestError, AuthenticationError, RateLimitError,
    ServerError, TimeoutError, CircuitBreakerError
)
from common.metrics import MetricsCollector
from common.utils import generate_nonce, timeit, RetryState
from common.constants import (
    HTTP_SUCCESS_CODES, HTTP_RETRY_CODES, HTTP_FATAL_CODES,
    DEFAULT_TIMEOUT, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY,
    MAX_RETRY_DELAY, JITTER_FACTOR
)


class CircuitBreaker:
    """
    Implements the Circuit Breaker pattern to prevent cascading failures
    when a service is degraded or unavailable.
    """
    
    # States
    CLOSED = 'closed'      # Normal operation, requests are allowed
    OPEN = 'open'          # Service considered down, requests are blocked
    HALF_OPEN = 'half_open'  # Testing if service recovered
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 3
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.logger = get_logger('circuit_breaker')
        
    def record_success(self):
        """Record a successful call."""
        if self.state == self.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = self.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0
                self.logger.info("Circuit closed after successful recovery")
        elif self.state == self.CLOSED:
            # Reset any accumulated failures
            self.failure_count = 0
            
    def record_failure(self):
        """Record a failed call."""
        self.last_failure_time = time.time()
        
        if self.state == self.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.state = self.OPEN
                self.logger.warning(f"Circuit opened after {self.failure_count} consecutive failures")
        elif self.state == self.HALF_OPEN:
            self.state = self.OPEN
            self.logger.warning("Circuit re-opened after failure in half-open state")
    
    def allow_request(self) -> bool:
        """
        Determine if a request should be allowed.
        
        Returns:
            bool: True if request is allowed, False otherwise
        """
        if self.state == self.CLOSED:
            return True
        
        if self.state == self.OPEN:
            recovery_time = self.last_failure_time + self.recovery_timeout
            if time.time() > recovery_time:
                self.state = self.HALF_OPEN
                self.half_open_calls = 0
                self.logger.info("Circuit half-opened to test service recovery")
                return True
            return False
        
        # In HALF_OPEN state
        return self.half_open_calls < self.half_open_max_calls


@dataclass
class Request:
    """Data class for representing REST requests with all parameters."""
    method: str
    url: str
    params: Optional[Dict[str, Any]] = None
    data: Optional[Union[Dict[str, Any], str, bytes]] = None
    headers: Optional[Dict[str, str]] = None
    auth: Optional[Tuple[str, str]] = None
    timeout: float = DEFAULT_TIMEOUT
    require_auth: bool = False
    rate_limit_key: Optional[str] = None
    circuit_breaker_key: Optional[str] = None
    compress: bool = False


@dataclass
class Response:
    """Data class for representing REST responses with all metadata."""
    status_code: int
    data: Any
    headers: Dict[str, str]
    url: str
    request_time: float
    raw_response: Optional[bytes] = None
    
    @property
    def is_success(self) -> bool:
        """Check if the response has a success status code."""
        return self.status_code in HTTP_SUCCESS_CODES


class RateLimiter:
    """
    Implements rate limiting to respect API limits and prevent bans.
    Uses a token bucket algorithm for flexibility.
    """
    
    def __init__(self, rate_limit: int, time_period: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            rate_limit: Maximum number of requests allowed
            time_period: Time period in seconds for the rate limit
        """
        self.rate_limit = rate_limit
        self.time_period = time_period
        self.token_refresh_rate = rate_limit / time_period
        self.tokens = rate_limit
        self.last_refresh_time = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self) -> bool:
        """
        Acquire a token if available.
        
        Returns:
            bool: True if a token was acquired, False otherwise
        """
        async with self.lock:
            self._refresh_tokens()
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False
            
    def _refresh_tokens(self):
        """Refresh tokens based on elapsed time."""
        current_time = time.time()
        time_elapsed = current_time - self.last_refresh_time
        
        # Add new tokens based on elapsed time
        new_tokens = time_elapsed * self.token_refresh_rate
        self.tokens = min(self.rate_limit, self.tokens + new_tokens)
        self.last_refresh_time = current_time
        
    def get_wait_time(self) -> float:
        """
        Calculate the time to wait before a request can be made.
        
        Returns:
            float: Time to wait in seconds
        """
        self._refresh_tokens()
        if self.tokens >= 1:
            return 0
        
        # Calculate time needed for at least one token
        return (1 - self.tokens) / self.token_refresh_rate


class RESTClient:
    """
    Advanced REST client for making HTTP requests to APIs with sophisticated
    retry logic, error handling, and performance optimization.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the REST client.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger('rest_client')
        self.metrics = MetricsCollector.get_instance()
        
        # Connection pooling
        self.session = None
        self.connector = None
        
        # Circuit breakers by endpoint/host
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Rate limiters by endpoint/category
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
        # Authentication credentials
        self.api_keys: Dict[str, Dict[str, str]] = {}
        
        # Initialize additional components
        self._load_api_keys()
        
    async def start(self):
        """Initialize the client session and resources."""
        # Configure connection pooling
        self.connector = aiohttp.TCPConnector(
            limit=self.config.get('rest_client.connection_limit', 100),
            ttl_dns_cache=self.config.get('rest_client.dns_cache_ttl', 300),
            ssl=self.config.get('rest_client.verify_ssl', True),
            use_dns_cache=True,
            enable_cleanup_closed=True
        )
        
        # Create client session
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(
                total=self.config.get('rest_client.default_timeout', DEFAULT_TIMEOUT)
            ),
            headers={
                'User-Agent': f"QuantumSpectre/{self.config.get('version', '1.0.0')}",
                'Accept': 'application/json'
            }
        )
        
        # Initialize circuit breakers
        circuit_breaker_configs = self.config.get('rest_client.circuit_breakers', {})
        for key, cfg in circuit_breaker_configs.items():
            self.circuit_breakers[key] = CircuitBreaker(
                failure_threshold=cfg.get('failure_threshold', 5),
                recovery_timeout=cfg.get('recovery_timeout', 30),
                half_open_max_calls=cfg.get('half_open_max_calls', 3)
            )
            
        # Initialize rate limiters
        rate_limit_configs = self.config.get('rest_client.rate_limits', {})
        for key, cfg in rate_limit_configs.items():
            self.rate_limiters[key] = RateLimiter(
                rate_limit=cfg.get('limit', 60),
                time_period=cfg.get('period', 60)
            )
            
        self.logger.info("REST client initialized and ready")
            
    async def stop(self):
        """Close the client session and release resources."""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
        self.logger.info("REST client resources released")
            
    def _load_api_keys(self):
        """Load API keys from configuration."""
        api_keys = self.config.get('api_keys', {})
        for exchange, keys in api_keys.items():
            # Add additional security checks here
            if 'api_key' in keys and 'secret_key' in keys:
                self.api_keys[exchange] = {
                    'api_key': keys['api_key'],
                    'secret_key': keys['secret_key']
                }
                if 'passphrase' in keys:  # For exchanges like Coinbase that need it
                    self.api_keys[exchange]['passphrase'] = keys['passphrase']
                    
    def _get_circuit_breaker(self, key: str) -> CircuitBreaker:
        """
        Get or create a circuit breaker for the given key.
        
        Args:
            key: Circuit breaker identifier
            
        Returns:
            CircuitBreaker: The circuit breaker instance
        """
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = CircuitBreaker()
        return self.circuit_breakers[key]
        
    def _get_rate_limiter(self, key: str) -> RateLimiter:
        """
        Get or create a rate limiter for the given key.
        
        Args:
            key: Rate limiter identifier
            
        Returns:
            RateLimiter: The rate limiter instance
        """
        if key not in self.rate_limiters:
            default_limit = self.config.get('rest_client.default_rate_limit', 60)
            default_period = self.config.get('rest_client.default_rate_period', 60)
            self.rate_limiters[key] = RateLimiter(default_limit, default_period)
        return self.rate_limiters[key]
    
    async def _prepare_request(self, req: Request) -> Request:
        """
        Prepare the request by adding authentication, headers, etc.
        
        Args:
            req: The request to prepare
            
        Returns:
            Request: The prepared request
        """
        # Create a copy to avoid modifying the original
        prepared = Request(
            method=req.method,
            url=req.url,
            params=req.params.copy() if req.params else None,
            data=req.data.copy() if isinstance(req.data, dict) else req.data,
            headers=req.headers.copy() if req.headers else {},
            auth=req.auth,
            timeout=req.timeout,
            require_auth=req.require_auth,
            rate_limit_key=req.rate_limit_key,
            circuit_breaker_key=req.circuit_breaker_key,
            compress=req.compress
        )
        
        # Extract exchange name from URL for authentication
        exchange = None
        if 'binance.com' in req.url:
            exchange = 'binance'
        elif 'deriv.com' in req.url or 'binary.com' in req.url:
            exchange = 'deriv'
        # Add more exchange detections as needed
        
        # Add authentication if required
        if req.require_auth and exchange and exchange in self.api_keys:
            auth_headers = self._sign_request(exchange, prepared)
            if prepared.headers:
                prepared.headers.update(auth_headers)
            else:
                prepared.headers = auth_headers
                
        # Add common headers
        if not prepared.headers:
            prepared.headers = {}
        
        prepared.headers['Accept'] = 'application/json'
        
        # Handle compression if requested
        if req.compress and req.data:
            if isinstance(req.data, dict):
                prepared.data = json.dumps(req.data).encode('utf-8')
            elif isinstance(req.data, str):
                prepared.data = req.data.encode('utf-8')
                
            prepared.data = zlib.compress(prepared.data)
            prepared.headers['Content-Encoding'] = 'gzip'
            prepared.headers['Content-Type'] = 'application/json'
            
        return prepared
    
    def _sign_request(self, exchange: str, req: Request) -> Dict[str, str]:
        """
        Sign a request for authentication with the exchange.
        
        Args:
            exchange: The exchange name
            req: The request to sign
            
        Returns:
            Dict[str, str]: Authentication headers
        """
        api_key = self.api_keys[exchange]['api_key']
        secret_key = self.api_keys[exchange]['secret_key']
        
        if exchange == 'binance':
            return self._sign_binance_request(api_key, secret_key, req)
        elif exchange == 'deriv':
            return self._sign_deriv_request(api_key, secret_key, req)
        else:
            self.logger.warning(f"Signing not implemented for exchange: {exchange}")
            return {}
            
    def _sign_binance_request(self, api_key: str, secret_key: str, req: Request) -> Dict[str, str]:
        """
        Sign a request for Binance API authentication.
        
        Args:
            api_key: The API key
            secret_key: The secret key
            req: The request to sign
            
        Returns:
            Dict[str, str]: Authentication headers
        """
        headers = {'X-MBX-APIKEY': api_key}
        
        # For GET requests, parameters go in the query string
        if req.method.upper() == 'GET' and req.params:
            # Add timestamp if not present
            if 'timestamp' not in req.params:
                req.params['timestamp'] = str(int(time.time() * 1000))
                
            # Create the signature
            query_string = urlencode(req.params)
            signature = hmac.new(
                secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Add signature to parameters
            req.params['signature'] = signature
            
        # For POST requests, data goes in the body
        elif req.method.upper() == 'POST' and isinstance(req.data, dict):
            # Add timestamp if not present
            if 'timestamp' not in req.data:
                req.data['timestamp'] = str(int(time.time() * 1000))
                
            # Create the signature
            body_string = urlencode(req.data)
            signature = hmac.new(
                secret_key.encode('utf-8'),
                body_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Add signature to data
            req.data['signature'] = signature
            
        return headers
    
    def _sign_deriv_request(self, api_key: str, secret_key: str, req: Request) -> Dict[str, str]:
        """
        Sign a request for Deriv API authentication.
        
        Args:
            api_key: The API key (app_id for Deriv)
            secret_key: The secret key
            req: The request to sign
            
        Returns:
            Dict[str, str]: Authentication headers
        """
        # Deriv uses app_id and API tokens rather than signing
        headers = {}
        
        # Add app_id to parameters
        if req.params is None:
            req.params = {}
        req.params['app_id'] = api_key
        
        # Add API token to parameters if using JSON RPC
        if isinstance(req.data, dict) and 'token' not in req.data:
            req.data['token'] = secret_key
            
        return headers
            
    async def request(
        self, 
        method: str, 
        url: str, 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict[str, Any], str, bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Tuple[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        require_auth: bool = False,
        rate_limit_key: Optional[str] = None,
        circuit_breaker_key: Optional[str] = None,
        compress: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY
    ) -> Response:
        """
        Send an HTTP request with advanced retry logic and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            params: Query parameters
            data: Request body data
            headers: Request headers
            auth: Basic auth credentials (username, password)
            timeout: Request timeout
            require_auth: Whether to sign the request for exchange auth
            rate_limit_key: Key for identifying rate limit category
            circuit_breaker_key: Key for identifying circuit breaker
            compress: Whether to compress the request body
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            
        Returns:
            Response: The API response
            
        Raises:
            RESTClientError: Base class for all REST client errors
            RequestError: Error in request format or parameters
            AuthenticationError: Authentication failed
            RateLimitError: Rate limit exceeded
            ServerError: Server returned an error
            TimeoutError: Request timed out
            CircuitBreakerError: Circuit breaker is open
        """
        if not self.session:
            raise RESTClientError("REST client not initialized, call start() first")
            
        # Create request object
        req = Request(
            method=method,
            url=url,
            params=params,
            data=data,
            headers=headers,
            auth=auth,
            timeout=timeout,
            require_auth=require_auth,
            rate_limit_key=rate_limit_key,
            circuit_breaker_key=circuit_breaker_key,
            compress=compress
        )
        
        # Check circuit breaker if provided
        if circuit_breaker_key:
            circuit_breaker = self._get_circuit_breaker(circuit_breaker_key)
            if not circuit_breaker.allow_request():
                self.logger.warning(f"Circuit breaker open for {circuit_breaker_key}, request blocked")
                raise CircuitBreakerError(f"Circuit breaker open for {circuit_breaker_key}")
        
        # Check rate limiter if provided
        if rate_limit_key:
            rate_limiter = self._get_rate_limiter(rate_limit_key)
            can_proceed = await rate_limiter.acquire()
            if not can_proceed:
                wait_time = rate_limiter.get_wait_time()
                self.logger.warning(f"Rate limit reached for {rate_limit_key}, need to wait {wait_time:.2f}s")
                # Instead of raising error, we'll wait if the wait time is reasonable
                if wait_time <= self.config.get('rest_client.max_rate_wait', 10):
                    self.logger.info(f"Waiting for {wait_time:.2f}s to respect rate limit")
                    await asyncio.sleep(wait_time)
                    # Try to acquire again after waiting
                    can_proceed = await rate_limiter.acquire()
                    if not can_proceed:
                        raise RateLimitError(f"Rate limit exceeded for {rate_limit_key}")
                else:
                    raise RateLimitError(f"Rate limit exceeded for {rate_limit_key}, wait time too long: {wait_time:.2f}s")
        
        # Retry loop
        retry_state = RetryState(max_retries=max_retries, retry_delay=retry_delay)
        
        while True:
            try:
                # Prepare the request (add auth, headers, etc.)
                prepared_req = await self._prepare_request(req)
                
                # Record metrics
                request_id = generate_nonce()
                self.metrics.increment(f"rest_client.requests.{method.lower()}")
                
                # Log request details at debug level
                self.logger.debug(
                    f"Sending {method} request to {url} "
                    f"(params: {params}, data: {data}, headers: {headers}, auth: {auth is not None})"
                )
                
                # Measure request time
                start_time = time.time()
                
                # Send the request
                async with self.session.request(
                    method=prepared_req.method,
                    url=prepared_req.url,
                    params=prepared_req.params,
                    data=prepared_req.data,
                    headers=prepared_req.headers,
                    auth=prepared_req.auth,
                    timeout=aiohttp.ClientTimeout(total=prepared_req.timeout)
                ) as aiohttp_response:
                    # Calculate request time
                    request_time = time.time() - start_time
                    
                    # Read the response
                    raw_response = await aiohttp_response.read()
                    
                    # Process response data
                    response_data = None
                    content_type = aiohttp_response.headers.get('Content-Type', '')
                    
                    if 'application/json' in content_type:
                        try:
                            response_data = await aiohttp_response.json()
                        except json.JSONDecodeError:
                            # Try to decode as JSON even if Content-Type is incorrect
                            try:
                                response_data = json.loads(raw_response.decode('utf-8'))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                # If it's not valid JSON, use raw text
                                response_data = raw_response.decode('utf-8', errors='replace')
                    else:
                        # For non-JSON responses, return the raw text
                        response_data = raw_response.decode('utf-8', errors='replace')
                    
                    # Create response object
                    response = Response(
                        status_code=aiohttp_response.status,
                        data=response_data,
                        headers=dict(aiohttp_response.headers),
                        url=str(aiohttp_response.url),
                        request_time=request_time,
                        raw_response=raw_response
                    )
                    
                    # Record metrics
                    self.metrics.timing(
                        f"rest_client.request_time.{method.lower()}",
                        request_time * 1000  # Convert to milliseconds
                    )
                    self.metrics.increment(
                        f"rest_client.status.{aiohttp_response.status // 100}xx"
                    )
                    
                    # Check for successful response
                    if response.is_success:
                        # Record success in circuit breaker if used
                        if circuit_breaker_key:
                            circuit_breaker = self._get_circuit_breaker(circuit_breaker_key)
                            circuit_breaker.record_success()
                            
                        return response
                    
                    # Handle error responses based on status code
                    if aiohttp_response.status == 401:
                        raise AuthenticationError(
                            f"Authentication failed: {response.data}", 
                            response=response
                        )
                    elif aiohttp_response.status == 429:
                        # Extract rate limit info from headers if available
                        retry_after = aiohttp_response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                retry_after = float(retry_after)
                                self.logger.warning(f"Rate limit hit, retry after {retry_after}s")
                                retry_state.retry_delay = max(retry_after, retry_state.retry_delay)
                            except (ValueError, TypeError):
                                pass
                                
                        raise RateLimitError(
                            f"Rate limit exceeded: {response.data}",
                            response=response,
                            retry_after=retry_after
                        )
                    elif 500 <= aiohttp_response.status < 600:
                        raise ServerError(
                            f"Server error {aiohttp_response.status}: {response.data}",
                            response=response
                        )
                    else:
                        raise RequestError(
                            f"Request error {aiohttp_response.status}: {response.data}",
                            response=response
                        )
                        
            except (RESTClientError, aiohttp.ClientError, asyncio.TimeoutError) as e:
                # Record failure in circuit breaker if used
                if circuit_breaker_key:
                    circuit_breaker = self._get_circuit_breaker(circuit_breaker_key)
                    circuit_breaker.record_failure()
                
                # Check if we should retry based on the error type
                should_retry = False
                retry_delay_override = None
                
                if isinstance(e, RateLimitError) and hasattr(e, 'retry_after') and e.retry_after:
                    # If we have a Retry-After header, use that as the delay
                    should_retry = True
                    retry_delay_override = float(e.retry_after)
                elif isinstance(e, ServerError):
                    # Retry server errors (5xx)
                    should_retry = True
                elif isinstance(e, aiohttp.ClientConnectorError):
                    # Retry connection errors
                    should_retry = True
                elif isinstance(e, asyncio.TimeoutError) or isinstance(e, aiohttp.ClientTimeoutError):
                    # Retry timeouts
                    should_retry = True
                    
                if should_retry and retry_state.can_retry():
                    # Calculate delay with exponential backoff
                    delay = retry_delay_override or retry_state.get_next_delay()
                    
                    self.logger.warning(
                        f"Request failed ({str(e)}), retrying in {delay:.2f}s "
                        f"(attempt {retry_state.attempts + 1}/{retry_state.max_retries})"
                    )
                    
                    # Sleep before retry
                    await asyncio.sleep(delay)
                    
                    # Increment retry counter
                    retry_state.attempts += 1
                    continue
                
                # If we can't retry, re-raise the exception
                # Wrap aiohttp exceptions in our own exception types
                if isinstance(e, asyncio.TimeoutError) or isinstance(e, aiohttp.ClientTimeoutError):
                    raise TimeoutError(f"Request timed out: {url}") from e
                elif isinstance(e, aiohttp.ClientError):
                    raise RequestError(f"Request failed: {str(e)}") from e
                else:
                    # Re-raise our own exceptions
                    raise
                    
    async def get(self, url: str, **kwargs) -> Response:
        """
        Send a GET request.
        
        Args:
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            Response: The API response
        """
        return await self.request('GET', url, **kwargs)
        
    async def post(self, url: str, **kwargs) -> Response:
        """
        Send a POST request.
        
        Args:
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            Response: The API response
        """
        return await self.request('POST', url, **kwargs)
        
    async def put(self, url: str, **kwargs) -> Response:
        """
        Send a PUT request.
        
        Args:
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            Response: The API response
        """
        return await self.request('PUT', url, **kwargs)
        
    async def delete(self, url: str, **kwargs) -> Response:
        """
        Send a DELETE request.
        
        Args:
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            Response: The API response
        """
        return await self.request('DELETE', url, **kwargs)
        
    async def batch_get(
        self, 
        urls: List[str], 
        concurrency: int = 5,
        **kwargs
    ) -> List[Response]:
        """
        Send multiple GET requests concurrently.
        
        Args:
            urls: List of request URLs
            concurrency: Maximum number of concurrent requests
            **kwargs: Additional request parameters shared across all requests
            
        Returns:
            List[Response]: List of API responses in the same order as the URLs
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def get_with_semaphore(url):
            async with semaphore:
                return await self.get(url, **kwargs)
                
        tasks = [get_with_semaphore(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
        
    async def batch_request(
        self, 
        requests: List[Request], 
        concurrency: int = 5
    ) -> List[Response]:
        """
        Send multiple requests concurrently.
        
        Args:
            requests: List of Request objects
            concurrency: Maximum number of concurrent requests
            
        Returns:
            List[Response]: List of API responses in the same order as the requests
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def request_with_semaphore(req):
            async with semaphore:
                return await self.request(
                    method=req.method,
                    url=req.url,
                    params=req.params,
                    data=req.data,
                    headers=req.headers,
                    auth=req.auth,
                    timeout=req.timeout,
                    require_auth=req.require_auth,
                    rate_limit_key=req.rate_limit_key,
                    circuit_breaker_key=req.circuit_breaker_key,
                    compress=req.compress
                )
                
        tasks = [request_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
