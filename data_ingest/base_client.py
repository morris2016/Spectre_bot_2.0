"""
Base Client for QuantumSpectre Elite Trading System.

This module provides an abstract base class for all data clients.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union

from common.config import settings
from common.logger import get_logger

logger = get_logger('base_client')

class BaseClient(ABC):
    """Abstract base class for all data clients."""
    
    def __init__(
        self, 
        name: str, 
        retry_limit: int = 5,
        retry_delay: float = 1.0,
        timeout: float = 10.0
    ):
        """
        Initialize base client.
        
        Args:
            name: Client name
            retry_limit: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            timeout: Operation timeout (seconds)
        """
        self.name = name
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.logger = get_logger(f'client.{name}')
        
        # Status tracking
        self.connected = False
        self.last_active = 0
        self.error_count = 0
        self.last_error = None
        
        # Performance metrics
        self.request_count = 0
        self.success_count = 0
        self.latency_sum = 0
        
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the data source.
        
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if client is connected.
        
        Returns:
            Connection status
        """
        pass
    
    async def execute_with_retry(
        self, 
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            Exception: If retry limit is reached
        """
        retry_count = 0
        last_error = None
        
        while retry_count <= self.retry_limit:
            try:
                start_time = time.time()
                self.request_count += 1
                
                # Execute function with timeout
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.timeout
                )
                
                # Update metrics
                elapsed = time.time() - start_time
                self.latency_sum += elapsed
                self.success_count += 1
                self.last_active = time.time()
                
                return result
                
            except asyncio.TimeoutError:
                retry_count += 1
                self.error_count += 1
                last_error = f"Operation timeout after {self.timeout}s"
                self.logger.warning(
                    f"Request timeout ({retry_count}/{self.retry_limit}): {last_error}"
                )
                
            except Exception as e:
                retry_count += 1
                self.error_count += 1
                last_error = str(e)
                self.logger.warning(
                    f"Request failed ({retry_count}/{self.retry_limit}): {last_error}"
                )
                
            # Wait before retry with exponential backoff
            if retry_count <= self.retry_limit:
                delay = min(
                    settings.MAX_RETRY_DELAY,
                    self.retry_delay * (2 ** (retry_count - 1))
                )
                await asyncio.sleep(delay)
        
        # If we get here, we've exhausted retries
        self.last_error = last_error
        error_message = f"Operation failed after {self.retry_limit} retries: {last_error}"
        self.logger.error(error_message)
        raise Exception(error_message)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get client performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        avg_latency = 0
        if self.success_count > 0:
            avg_latency = self.latency_sum / self.success_count
            
        success_rate = 0
        if self.request_count > 0:
            success_rate = (self.success_count / self.request_count) * 100
            
        return {
            'name': self.name,
            'connected': self.connected,
            'last_active': self.last_active,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'request_count': self.request_count,
            'success_count': self.success_count,
            'success_rate': success_rate,
            'avg_latency': avg_latency
        }
    
    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait for client to connect.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            Success status
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self.is_connected():
                return True
            await asyncio.sleep(0.5)
        
        self.logger.error(f"Timeout waiting for {self.name} connection")
        return False
