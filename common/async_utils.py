#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Async Utilities

This module provides async utility functions for the QuantumSpectre Elite Trading System.
"""

import asyncio
import functools
from contextlib import asynccontextmanager
import logging
import time
from typing import Any, List, Callable, Coroutine, Optional, TypeVar, Set

from common.exceptions import TimeoutError

# Type variables for better type hinting
T = TypeVar('T')
R = TypeVar('R')

# Logger for this module
logger = logging.getLogger(__name__)

async def run_with_timeout(
    coroutine: Coroutine[Any, Any, T], 
    timeout: float, 
    loop: Optional[asyncio.AbstractEventLoop] = None,
    error_message: Optional[str] = None
) -> T:
    """
    Run a coroutine with a timeout.

    Args:
        coroutine: Coroutine to run
        timeout: Timeout in seconds
        loop: Event loop to use (defaults to current loop)
        error_message: Custom error message on timeout

    Returns:
        Result of the coroutine

    Raises:
        TimeoutError: If the coroutine times out
    """
    if loop is None:
        loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(coroutine, timeout=timeout)
    except asyncio.TimeoutError:
        msg = error_message or f"Operation timed out after {timeout} seconds"
        logger.error(msg)
        raise TimeoutError(msg)


def run_in_executor(func, *args, **kwargs):
    """
    Run a synchronous function in an executor (thread pool).
    
    Args:
        func: The synchronous function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        A coroutine that will resolve to the function's result
    """
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(
        None, lambda: func(*args, **kwargs)
    )


async def create_task_with_error_handling(coro, name=None, exception_handler=None):
    """
    Create a task with error handling.
    
    Args:
        coro: The coroutine to run
        name: Optional name for the task
        exception_handler: Optional function to handle exceptions
        
    Returns:
        The created task
    """
    task = asyncio.create_task(coro, name=name)
    
    def _handle_task_result(task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if exception_handler:
                exception_handler(e)
            else:
                logging.exception(f"Unhandled exception in task {task.get_name()}: {e}")
    
    task.add_done_callback(_handle_task_result)
    return task


async def cancel_all_tasks(
    tasks: Set[asyncio.Task],
    timeout: float = 5.0,
    exception_handler: Optional[Callable[[Exception], None]] = None
) -> None:
    """
    Cancel all tasks and wait for them to complete.

    Args:
        tasks: Set of tasks to cancel
        timeout: Timeout for tasks to complete after cancellation
        exception_handler: Optional handler for exceptions during cancellation
    """
    if not tasks:
        return

    # Request cancellation for all tasks
    for task in tasks:
        if not task.done():
            task.cancel()

    # Wait for all tasks to complete or timeout
    pending = tasks
    try:
        done, pending = await asyncio.wait(tasks, timeout=timeout)
    except asyncio.CancelledError:
        # Handle the case where this function itself is cancelled
        logger.warning("Task cancellation was interrupted")
        # Re-raise to propagate the cancellation
        raise

    # Handle any remaining pending tasks
    if pending:
        logger.warning(f"{len(pending)} tasks did not complete cancellation within timeout")
        
    # Check for exceptions in completed tasks
    for task in done:
        if task.cancelled():
            continue
        if task.exception() and exception_handler:
            try:
                exception_handler(task.exception())
            except Exception as e:
                logger.error(f"Error in exception handler: {str(e)}")

async def gather_with_concurrency(
    n: int, 
    *coroutines: Coroutine, 
    return_exceptions: bool = False
) -> List[Any]:
    """
    Run coroutines with limited concurrency.

    Args:
        n: Maximum number of concurrent coroutines
        *coroutines: Coroutines to run
        return_exceptions: If True, exceptions are returned instead of raised

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(n)
    
    async def semaphore_coroutine(coroutine):
        async with semaphore:
            return await coroutine
    
    return await asyncio.gather(
        *(semaphore_coroutine(c) for c in coroutines),
        return_exceptions=return_exceptions
    )

async def safe_gather(*coroutines: Coroutine, return_exceptions: bool = True) -> List[Any]:
    """
    Safely gather coroutines, ensuring no exceptions propagate unless requested.

    Args:
        *coroutines: Coroutines to run
        return_exceptions: If True, exceptions are returned instead of raised

    Returns:
        List of results or exceptions
    """
    try:
        return await asyncio.gather(*coroutines, return_exceptions=return_exceptions)
    except Exception as e:
        logger.error(f"Error in safe_gather: {str(e)}")
        if return_exceptions:
            return [e] * len(coroutines)
        raise

async def run_in_threadpool(func: Callable[..., R], *args, **kwargs) -> R:
    """
    Run a blocking function in a thread pool executor.

    Args:
        func: Function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: func(*args, **kwargs)
    )


async def run_in_thread_pool(func: Callable[..., R], *args, **kwargs) -> R:
    """Alias for run_in_threadpool for backward compatibility."""

    return await run_in_threadpool(func, *args, **kwargs)

async def retry(
    coroutine_func: Callable[..., Coroutine],
    *args,
    retry_count: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    **kwargs
) -> Any:
    """
    Retry a coroutine with exponential backoff.

    Args:
        coroutine_func: Coroutine function to retry
        *args: Positional arguments to pass to the function
        retry_count: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff_factor: Backoff multiplier for subsequent retries
        exceptions: Tuple of exceptions to catch for retrying
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the coroutine function

    Raises:
        The last caught exception if all retries fail
    """
    last_exception = None
    current_delay = delay

    for i in range(retry_count + 1):
        try:
            return await coroutine_func(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if i < retry_count:
                logger.warning(f"Retry {i+1}/{retry_count} failed: {str(e)}. Retrying in {current_delay:.2f}s")
                await asyncio.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                logger.error(f"All {retry_count} retries failed for {coroutine_func.__name__}")
                break

    raise last_exception

class RateLimiter:
    """
    Asynchronous rate limiter to control API call rates.
    """
    
    def __init__(self, calls_per_second: float = 1.0, burst: int = 1):
        """
        Initialize the rate limiter.

        Args:
            calls_per_second: Maximum calls per second
            burst: Maximum number of tokens (calls) that can be accumulated
        """
        self.calls_per_second = calls_per_second
        self.burst = burst
        self.tokens = burst
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> float:
        """
        Acquire tokens for API call, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time spent waiting in seconds
        """
        start_time = time.monotonic()
        async with self.lock:
            # Update current token count based on elapsed time
            now = time.monotonic()
            elapsed = now - self.updated_at
            self.tokens = min(self.burst, self.tokens + elapsed * self.calls_per_second)
            self.updated_at = now
            
            # Calculate wait time if not enough tokens
            wait_time = 0.0
            if tokens > self.tokens:
                wait_time = (tokens - self.tokens) / self.calls_per_second
            
            # Wait if needed
            if wait_time > 0:
                self.updated_at += wait_time
                await asyncio.sleep(wait_time)
            
            # Consume tokens
            self.tokens -= tokens
            
            return time.monotonic() - start_time

class AsyncTimer:
    """
    Context manager for timing async code blocks.
    """
    
    def __init__(self, name: str = "timer", logger_name: Optional[str] = None):
        """
        Initialize the timer.

        Args:
            name: Timer name for identification
            logger_name: Logger name (if None, use this module's logger)
        """
        self.name = name
        self.start_time = 0
        self.logger = logging.getLogger(logger_name if logger_name else __name__)

    async def __aenter__(self) -> 'AsyncTimer':
        """Enter the context manager."""
        self.start_time = time.monotonic()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager."""
        elapsed = time.monotonic() - self.start_time
        self.logger.debug(f"{self.name} took {elapsed:.6f} seconds")

def async_retry(
    retry_count: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        retry_count: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff_factor: Backoff multiplier for subsequent retries
        exceptions: Tuple of exceptions to catch for retrying

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry(
                func, *args,
                retry_count=retry_count,
                delay=delay,
                backoff_factor=backoff_factor,
                exceptions=exceptions,
                **kwargs
            )
        return wrapper
    return decorator

async def retry_with_backoff(
    coro: Coroutine,
    retry_count: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Retry a coroutine with exponential backoff.
    
    Args:
        coro: Coroutine to retry
        retry_count: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Backoff multiplier for subsequent retries
        exceptions: Tuple of exceptions to catch for retrying
        
    Returns:
        Result of the coroutine
        
    Raises:
        The last caught exception if all retries fail
    """
    last_exception = None
    current_delay = initial_delay
    
    for i in range(retry_count + 1):
        try:
            return await coro
        except exceptions as e:
            last_exception = e
            if i < retry_count:
                logger.warning(f"Retry {i+1}/{retry_count} failed: {str(e)}. Retrying in {current_delay:.2f}s")
                await asyncio.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                logger.error(f"All {retry_count} retries failed")
                break
    
    raise last_exception


def retry_with_backoff_decorator(
    retry_count: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying a coroutine function with exponential backoff.
    
    Args:
        retry_count: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Backoff multiplier for subsequent retries
        exceptions: Tuple of exceptions to catch for retrying
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = initial_delay
            
            for i in range(retry_count + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if i < retry_count:
                        logger.warning(f"Retry {i+1}/{retry_count} failed: {str(e)}. Retrying in {current_delay:.2f}s")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"All {retry_count} retries failed")
                        break
            
            raise last_exception
        return wrapper
    return decorator

async def periodic_task(
    func: Callable[..., Coroutine],
    interval: float,
    *args,
    initial_delay: float = 0,
    stop_event: Optional[asyncio.Event] = None,
    **kwargs
) -> None:
    """
    Run a function periodically.

    Args:
        func: Coroutine function to run
        interval: Interval between runs in seconds
        *args: Positional arguments to pass to the function
        initial_delay: Delay before the first run
        stop_event: Event to signal stopping the periodic task
        **kwargs: Keyword arguments to pass to the function
    """
    if initial_delay > 0:
        await asyncio.sleep(initial_delay)
    
    while True:
        try:
            start_time = time.monotonic()
            await func(*args, **kwargs)
            
            # Calculate sleep time considering the execution time
            elapsed = time.monotonic() - start_time
            sleep_time = max(0, interval - elapsed)
            
            if stop_event and stop_event.is_set():
                break
                
            # Sleep until next interval
            await asyncio.sleep(sleep_time)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic task {func.__name__}: {str(e)}")
            # Still sleep before the next attempt
            await asyncio.sleep(interval)

@asynccontextmanager
async def create_task_group():
    tasks = []
    try:
        yield tasks
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Task {task} raised an exception: {str(e)}")
# Alias for compatibility with existing code
cancel_tasks = cancel_all_tasks

class AsyncBatcher:
    """Batches async operations to improve efficiency."""

    def __init__(self, batch_size: int = 100, batch_interval: float = 0.1):
        """
        Initialize AsyncBatcher.
        
        Args:
            batch_size: Maximum number of items in a batch
            batch_interval: Time interval between batch processing (seconds)
        """
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.batch = []
        self.batch_task = None
        self.logger = logging.getLogger(__name__)
        self.processing = False
        self._lock = asyncio.Lock()
        
    async def add(self, item: Any) -> asyncio.Future:
        """
        Add an item to the batch.
        
        Args:
            item: Item to add
            
        Returns:
            Future for the result
        """
        future = asyncio.Future()
        
        async with self._lock:
            self.batch.append((item, future))
            
            if len(self.batch) >= self.batch_size and not self.processing:
                self.processing = True
                asyncio.create_task(self._process_batch())
            elif self.batch_task is None:
                self.batch_task = asyncio.create_task(self._schedule_processing())
                
        return future
        
    async def _schedule_processing(self):
        """Schedule batch processing after an interval."""
        await asyncio.sleep(self.batch_interval)
        
        async with self._lock:
            self.batch_task = None
            if self.batch and not self.processing:
                self.processing = True
                asyncio.create_task(self._process_batch())
                
    async def _process_batch(self):
        """Process the current batch."""
        async with self._lock:
            current_batch = self.batch
            self.batch = []
            
        try:
            results = await self._process_items([item for item, _ in current_batch])
            
            for (_, future), result in zip(current_batch, results):
                if not future.cancelled():
                    future.set_result(result)
                    
        except Exception as e:
            self.logger.error(f"Batch processing error: {str(e)}")
            for _, future in current_batch:
                if not future.cancelled():
                    future.set_exception(e)
                    
        finally:
            self.processing = False
            
            # Check if new items were added during processing
            async with self._lock:
                if self.batch and self.batch_task is None:
                    self.processing = True
                    asyncio.create_task(self._process_batch())
                    
    async def _process_items(self, items: List[Any]) -> List[Any]:
        """
        Process batch items. To be overridden by subclasses.
        
        Args:
            items: Items to process
            
        Returns:
            Results for the items
        """
        raise NotImplementedError("Subclasses must implement this method")


class AsyncRateLimiter:
    """Rate limiter for async operations."""

    def __init__(self, rate_limit: float, time_period: float = 1.0):
        """
        Initialize AsyncRateLimiter.
        
        Args:
            rate_limit: Maximum number of operations per time period
            time_period: Time period in seconds (default 1 second)
        """
        self.rate_limit = rate_limit
        self.time_period = time_period
        self.tokens = rate_limit
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def acquire(self):
        """
        Acquire permission to proceed.
        
        This method blocks until permission is granted.
        """
        while True:
            async with self._lock:
                current_time = time.monotonic()
                time_passed = current_time - self.last_refill
                
                # Refill tokens based on time passed
                if time_passed > 0:
                    self.tokens = min(
                        self.rate_limit,
                        self.tokens + (time_passed * self.rate_limit / self.time_period)
                    )
                    self.last_refill = current_time
                    
                # Check if we can proceed
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                    
                # Calculate waiting time
                wait_time = (self.time_period / self.rate_limit) - time_passed
                
            # Wait and try again
            wait_time = max(0.001, wait_time)  # Ensure positive wait time
            self.logger.debug(f"Rate limited, waiting for {wait_time:.4f}s")
            await asyncio.sleep(wait_time)


class AsyncRetrier:
    """Retries async operations with exponential backoff."""

    def __init__(self, 
                max_retries: int = 3, 
                base_delay: float = 1.0,
                max_delay: float = 60.0,
                backoff_factor: float = 2.0,
                jitter: bool = True):
        """
        Initialize AsyncRetrier.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Exponential backoff factor
            jitter: Whether to add jitter to delay times
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.logger = logging.getLogger(__name__)
        
    async def execute(self, 
                    operation: Callable[..., Any],
                    *args,
                    retry_on: Any = Exception,
                    **kwargs) -> Any:
        """
        Execute an operation with retries.
        
        Args:
            operation: Async function to execute
            *args: Arguments for the operation
            retry_on: Exception(s) that trigger retry
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            The last exception if all retries fail
        """
        import random
        
        if not isinstance(retry_on, (list, tuple)):
            retry_on = [retry_on]
            
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = min(
                        self.max_delay,
                        self.base_delay * (self.backoff_factor ** (attempt - 1))
                    )
                    
                    if self.jitter:
                        delay = delay * (0.5 + random.random())
                        
                    self.logger.info(f"Retry attempt {attempt}/{self.max_retries}, waiting {delay:.2f}s")
                    await asyncio.sleep(delay)
                    
                return await operation(*args, **kwargs)
                
            except tuple(retry_on) as e:
                last_exception = e
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {str(e)}"
                )
                
                if attempt == self.max_retries:
                    self.logger.error(f"All retry attempts failed: {str(e)}")
                    raise
            except Exception as e:
                # Don't retry exceptions not in retry_on
                self.logger.error(f"Non-retryable error: {str(e)}")
                raise
# End of async_utils.py
def timed_cache(ttl_seconds=300):
    """
    Decorator for caching async function results with a time-to-live (TTL).
    
    Args:
        ttl_seconds: Time-to-live in seconds (default: 300)
        
    Returns:
        Decorated function
    """
    import time
    import functools
    
    def decorator(func):
        cache = {}
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a key from the function arguments
            key = str(args) + str(sorted(kwargs.items()))
            
            # Check if result is in cache and not expired
            current_time = time.time()
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return result
            
            # Call the function and cache the result
            result = await func(*args, **kwargs)
            cache[key] = (result, current_time)
            
            # Clean up expired entries
            for k in list(cache.keys()):
                if current_time - cache[k][1] > ttl_seconds:
                    del cache[k]
                    
            return result
            
    return wrapper


async def create_task_with_retry(
    coro: Callable[..., Coroutine[Any, Any, R]],
    *args: Any,
    retries: int = 3,
    delay: float = 1.0,
    **kwargs: Any,
) -> R:
    """Run a coroutine with retry logic."""
    attempt = 0
    while True:
        try:
            return await coro(*args, **kwargs)
        except Exception:
            attempt += 1
            if attempt > retries:
                raise
            await asyncio.sleep(delay * attempt)
        
    return decorator

class TaskGroup:
    """
    Manages a group of related asyncio tasks.
    
    This utility class helps with creating, tracking, and cancelling groups of tasks.
    """
    
    def __init__(self, name=None):
        """
        Initialize a task group.
        
        Args:
            name: Optional name for the task group for debugging
        """
        import asyncio
        self.name = name or "TaskGroup"
        self.tasks = set()
        self.logger = logging.getLogger(f"async.{self.name}")
        self._closed = False
        
    def create_task(self, coro, name=None):
        """
        Create and track an asyncio task.
        
        Args:
            coro: Coroutine to run as a task
            name: Optional name for the task
            
        Returns:
            The created asyncio.Task
        """
        import asyncio
        
        if self._closed:
            raise RuntimeError(f"TaskGroup {self.name} is closed")
            
        task = asyncio.create_task(coro)
        
        if name and hasattr(task, "set_name"):
            task.set_name(name)
            
        self.tasks.add(task)
        task.add_done_callback(self._task_done_callback)
        
        return task
        
    def _task_done_callback(self, task):
        """Remove task from set when it completes."""
        self.tasks.discard(task)
        
        # Check for exceptions
        if not task.cancelled() and task.exception():
            self.logger.error(f"Task error: {task.exception()}")
            
    async def cancel_all(self, timeout=10.0):
        """
        Cancel all tasks in this group and wait for them to complete.
        
        Args:
            timeout: Maximum time to wait for tasks to complete cancellation
            
        Returns:
            Set of tasks that did not complete within the timeout
        """
        import asyncio
        
        if not self.tasks:
            return set()
            
        # Request cancellation for all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                
        # Wait for all tasks to complete cancellation
        pending = self.tasks.copy()
        
        try:
            done, pending = await asyncio.wait(
                pending,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
        except asyncio.CancelledError:
            # If this function itself is cancelled, re-cancel all tasks
            for task in pending:
                if not task.done():
                    task.cancel()
            # Re-raise to propagate cancellation
            raise
            
        # Log warnings for tasks that didn't complete
        if pending:
            self.logger.warning(f"{len(pending)} tasks did not complete cancellation within {timeout}s")
            
        return pending
        
    async def wait_all(self, timeout=None):
        """
        Wait for all tasks in this group to complete.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            True if all tasks completed, False if timeout occurred
        """
        import asyncio
        
        if not self.tasks:
            return True
            
        try:
            done, pending = await asyncio.wait(
                self.tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            return len(pending) == 0
            
        except asyncio.CancelledError:
            # If this function itself is cancelled, re-cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            # Re-raise to propagate cancellation
            raise
            
    def is_empty(self):
        """
        Check if the task group is empty.
        
        Returns:
            True if there are no tasks in the group
        """
        return len(self.tasks) == 0
        
    def __len__(self):
        """Get the number of tasks in the group."""
        return len(self.tasks)
        
    async def close(self):
        """Cancel all tasks and mark the group as closed."""
        await self.cancel_all()
        self._closed = True
        
    async def __aenter__(self):
        """Enter async context."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and clean up tasks."""
        await self.close()


class PeriodicTask:
    """
    Run a coroutine function periodically at a specified interval.
    
    This utility class helps with scheduling periodic task execution.
    """
    
    def __init__(self, coro_func, interval, name=None, initial_delay=0, 
                error_interval=None, max_errors=None, task_group=None):
        """
        Initialize a periodic task.
        
        Args:
            coro_func: Coroutine function to call periodically
            interval: Interval between executions in seconds
            name: Optional name for the task
            initial_delay: Delay before first execution in seconds
            error_interval: Alternative interval after errors (defaults to interval)
            max_errors: Maximum consecutive errors before stopping (None for unlimited)
            task_group: Optional TaskGroup to add the task to
        """
        self.coro_func = coro_func
        self.interval = interval
        self.name = name or f"PeriodicTask({coro_func.__name__})"
        self.initial_delay = initial_delay
        self.error_interval = error_interval or interval
        self.max_errors = max_errors
        self.task_group = task_group
        
        self.running = False
        self.task = None
        self.stop_event = None
        self.consecutive_errors = 0
        self.logger = logging.getLogger(f"async.{self.name}")
        
    async def start(self):
        """Start the periodic task."""
        import asyncio
        
        if self.running:
            return
            
        self.running = True
        self.stop_event = asyncio.Event()
        
        if self.task_group:
            self.task = self.task_group.create_task(self._run(), name=self.name)
        else:
            self.task = asyncio.create_task(self._run())
            
        return self.task
        
    async def stop(self):
        """Stop the periodic task."""
        if not self.running:
            return
            
        self.running = False
        if self.stop_event:
            self.stop_event.set()
            
        if self.task and not self.task.done():
            self.task.cancel()
            
    async def _run(self):
        """Main task loop."""
        import asyncio
        
        try:
            # Apply initial delay
            if self.initial_delay > 0:
                await asyncio.sleep(self.initial_delay)
                
            while self.running and not self.stop_event.is_set():
                try:
                    # Execute the coroutine function
                    start_time = time.time()
                    await self.coro_func()
                    
                    # Reset consecutive errors counter
                    self.consecutive_errors = 0
                    
                    # Calculate sleep time
                    elapsed = time.time() - start_time
                    sleep_time = max(0, self.interval - elapsed)
                    
                    # Sleep until next execution or until stopped
                    try:
                        await asyncio.wait_for(self.stop_event.wait(), timeout=sleep_time)
                        if self.stop_event.is_set():
                            break
                    except asyncio.TimeoutError:
                        # Timeout is expected, continue with next iteration
                        pass
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.consecutive_errors += 1
                    self.logger.error(f"Error in periodic task {self.name}: {str(e)}")
                    
                    # Check if max consecutive errors reached
                    if self.max_errors is not None and self.consecutive_errors >= self.max_errors:
                        self.logger.error(f"Max consecutive errors ({self.max_errors}) reached, stopping task")
                        break
                        
                    # Use error interval for next retry
                    await asyncio.sleep(self.error_interval)
                    
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            pass
        finally:
            self.running = False
            
    async def __aenter__(self):
        """Enter async context."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        await self.stop()


class Throttler:
    """
    Rate limiter for coroutines to prevent exceeding API rate limits.
    
    This utility class helps with controlling execution rates.
    """
    
    def __init__(self, rate_limit, period=1.0, burst_limit=None):
        """
        Initialize a throttler.
        
        Args:
            rate_limit: Number of operations allowed per period
            period: Time period in seconds
            burst_limit: Maximum number of operations allowed in a burst (defaults to rate_limit)
        """
        self.rate_limit = rate_limit
        self.period = period
        self.burst_limit = burst_limit or rate_limit
        
        self.tokens = self.burst_limit
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self, tokens=1):
        """
        Acquire tokens from the throttler, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            
        Returns:
            True if tokens were acquired successfully
        """
        if tokens > self.burst_limit:
            raise ValueError(f"Cannot acquire {tokens} tokens (maximum burst is {self.burst_limit})")
            
        async with self.lock:
            await self._refill()
            
            if self.tokens >= tokens:
                # Enough tokens available immediately
                self.tokens -= tokens
                return True
                
            # Calculate wait time to get required tokens
            deficit = tokens - self.tokens
            wait_time = (deficit / self.rate_limit) * self.period
            
            # Release lock during wait
            self.lock.release()
            try:
                await asyncio.sleep(wait_time)
            finally:
                await self.lock.acquire()
                
            # Refill and acquire after waiting
            await self._refill()
            self.tokens -= tokens
            return True
            
    async def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed <= 0:
            return
            
        # Calculate tokens to add
        new_tokens = (elapsed / self.period) * self.rate_limit
        self.tokens = min(self.burst_limit, self.tokens + new_tokens)
        self.last_refill = now
        
    async def __aenter__(self):
        """Enter async context, acquiring one token."""
        await self.acquire(1)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        pass
class AsyncTaskManager:
    """
    Manages asynchronous tasks with lifecycle tracking, throttling, and error handling.
    
    This class provides a centralized way to create, monitor, and manage asynchronous
    tasks in the system. It supports task prioritization, throttling, and automatic
    cleanup of completed tasks.
    """
    
    def __init__(self, max_concurrent_tasks=100, task_timeout=300):
        """
        Initialize the AsyncTaskManager.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent tasks
            task_timeout: Default timeout for tasks in seconds
        """
        self.tasks = {}
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = task_timeout
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.logger = logging.getLogger(__name__)
        
    async def create_task(self, coro, name=None, priority=0, timeout=None):
        """
        Create and register a new task.
        
        Args:
            coro: Coroutine to run as a task
            name: Optional name for the task
            priority: Task priority (higher values = higher priority)
            timeout: Optional timeout override
            
        Returns:
            str: Task ID
        """
        task_id = str(uuid.uuid4())
        task_name = name or f"task-{task_id[:8]}"
        task_timeout = timeout or self.default_timeout
        
        async with self.lock:
            # Check if we're at capacity
            if len([t for t in self.tasks.values() if not t['task'].done()]) >= self.max_concurrent_tasks:
                self.logger.warning(f"Task limit reached ({self.max_concurrent_tasks}). Waiting for slot.")
                
            # Create the wrapped task
            wrapped_task = asyncio.create_task(
                self._run_task_with_timeout(coro, task_timeout, task_id, task_name),
                name=task_name
            )
            
            # Register the task
            self.tasks[task_id] = {
                'task': wrapped_task,
                'name': task_name,
                'created_at': time.time(),
                'priority': priority,
                'status': 'running'
            }
            
            # Set up done callback
            wrapped_task.add_done_callback(
                lambda t, tid=task_id: self._task_done_callback(tid, t)
            )
            
            self.logger.debug(f"Created task {task_name} (ID: {task_id})")
            return task_id
            
    async def _run_task_with_timeout(self, coro, timeout, task_id, task_name):
        """Run a task with timeout and semaphore control."""
        try:
            async with self.semaphore:
                return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.error(f"Task {task_name} (ID: {task_id}) timed out after {timeout}s")
            async with self.lock:
                if task_id in self.tasks:
                    self.tasks[task_id]['status'] = 'timeout'
            raise
        except asyncio.CancelledError:
            self.logger.info(f"Task {task_name} (ID: {task_id}) was cancelled")
            async with self.lock:
                if task_id in self.tasks:
                    self.tasks[task_id]['status'] = 'cancelled'
            raise
        except Exception as e:
            self.logger.error(f"Task {task_name} (ID: {task_id}) failed: {str(e)}")
            async with self.lock:
                if task_id in self.tasks:
                    self.tasks[task_id]['status'] = 'failed'
                    self.tasks[task_id]['error'] = str(e)
            raise
            
    def _task_done_callback(self, task_id, task):
        """Handle task completion."""
        try:
            # Update task status
            if task_id in self.tasks:
                if task.cancelled():
                    self.tasks[task_id]['status'] = 'cancelled'
                elif task.exception():
                    self.tasks[task_id]['status'] = 'failed'
                    self.tasks[task_id]['error'] = str(task.exception())
                else:
                    self.tasks[task_id]['status'] = 'completed'
                self.tasks[task_id]['completed_at'] = time.time()
                
                task_name = self.tasks[task_id]['name']
                status = self.tasks[task_id]['status']
                self.logger.debug(f"Task {task_name} (ID: {task_id}) {status}")
        except Exception as e:
            self.logger.error(f"Error in task completion callback: {str(e)}")
            
    async def cancel_task(self, task_id):
        """
        Cancel a running task by ID.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            bool: True if task was cancelled, False otherwise
        """
        async with self.lock:
            if task_id not in self.tasks:
                return False
                
            task_info = self.tasks[task_id]
            if task_info['task'].done():
                return False
                
            task_info['task'].cancel()
            return True
            
    async def get_task_status(self, task_id):
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            dict: Task status information or None if not found
        """
        async with self.lock:
            if task_id not in self.tasks:
                return None
                
            task_info = self.tasks[task_id]
            result = {
                'id': task_id,
                'name': task_info['name'],
                'status': task_info['status'],
                'created_at': task_info['created_at'],
                'priority': task_info['priority']
            }
            
            if 'completed_at' in task_info:
                result['completed_at'] = task_info['completed_at']
                result['duration'] = task_info['completed_at'] - task_info['created_at']
                
            if 'error' in task_info:
                result['error'] = task_info['error']
                
            return result
            
    async def get_all_tasks(self, include_completed=False):
        """
        Get information about all tasks.
        
        Args:
            include_completed: Whether to include completed tasks
            
        Returns:
            list: List of task information dictionaries
        """
        async with self.lock:
            result = []
            for task_id, task_info in self.tasks.items():
                if not include_completed and task_info['task'].done():
                    continue
                    
                status = {
                    'id': task_id,
                    'name': task_info['name'],
                    'status': task_info['status'],
                    'created_at': task_info['created_at'],
                    'priority': task_info['priority']
                }
                
                if 'completed_at' in task_info:
                    status['completed_at'] = task_info['completed_at']
                    status['duration'] = task_info['completed_at'] - task_info['created_at']
                    
                if 'error' in task_info:
                    status['error'] = task_info['error']
                    
                result.append(status)
                
            return result
            
    async def cleanup_completed_tasks(self, max_age=3600):
        """
        Remove completed tasks older than max_age seconds.
        
        Args:
            max_age: Maximum age in seconds for completed tasks
            
        Returns:
            int: Number of tasks cleaned up
        """
        now = time.time()
        to_remove = []
        
        async with self.lock:
            for task_id, task_info in self.tasks.items():
                if (task_info['task'].done() and 
                    'completed_at' in task_info and 
                    now - task_info['completed_at'] > max_age):
                    to_remove.append(task_id)
                    
            for task_id in to_remove:
                del self.tasks[task_id]
                
            return len(to_remove)
            
    async def wait_for_task(self, task_id, timeout=None):
        """
        Wait for a specific task to complete.
        
        Args:
            task_id: ID of the task to wait for
            timeout: Optional timeout in seconds
            
        Returns:
            Any: Task result
            
        Raises:
            KeyError: If task_id is not found
            asyncio.TimeoutError: If waiting times out
        """
        async with self.lock:
            if task_id not in self.tasks:
                raise KeyError(f"Task {task_id} not found")
                
            task = self.tasks[task_id]['task']
            
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for task {task_id}")
            raise
            
    async def cancel_all_tasks(self):
        """
        Cancel all running tasks.
        
        Returns:
            int: Number of tasks cancelled
        """
        count = 0
        async with self.lock:
            for task_id, task_info in self.tasks.items():
                if not task_info['task'].done():
                    task_info['task'].cancel()
                    count += 1
                    
        return count

def create_throttled_task(coro, delay=0):
    """
    Create a task that starts after a specified delay.
    
    Args:
        coro: Coroutine to run
        delay: Delay in seconds before starting
        
    Returns:
        asyncio.Task: The created task
    """
    async def delayed_coro():
        if delay > 0:
            await asyncio.sleep(delay)
        return await coro
        
    return asyncio.create_task(delayed_coro())

async def cancelable_periodic_task(coro, interval, cancel_event=None):
    """
    Run a coroutine periodically until canceled.
    
    Args:
        coro: Coroutine function to call
        interval: Interval between calls in seconds
        cancel_event: Optional event to signal cancellation
        
    Returns:
        None
    """
    while True:
        try:
            start_time = time.time()
            await coro()
            
            # Check if we should cancel
            if cancel_event and cancel_event.is_set():
                break
                
            # Sleep until next interval
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in periodic task: {str(e)}")
            await asyncio.sleep(interval)  # Still sleep on error
