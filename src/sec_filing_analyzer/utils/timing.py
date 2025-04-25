"""
Timing Utilities

This module provides utilities for measuring and logging execution times.
"""

import functools
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def timed_function(category: str = "function"):
    """
    Decorator to measure and log the execution time of a function.

    Args:
        category: Category of the timed operation (e.g., "llm", "tool", "processing")

    Returns:
        Decorated function that logs timing information
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get the instance if this is a method
            instance = args[0] if args else None

            # Get the logger - use instance logger if available, otherwise use module logger
            log = getattr(instance, "logger", logger) if instance else logger

            # Get function name
            func_name = func.__qualname__

            # Start timing
            start_time = time.time()

            # Log start
            log.debug(f"TIMING_START: {category}:{func_name}")

            try:
                # Call the function
                result = await func(*args, **kwargs)

                # Calculate duration
                duration = time.time() - start_time

                # Log completion
                log.info(f"TIMING: {category}:{func_name} completed in {duration:.3f}s")

                return result
            except Exception as e:
                # Calculate duration even if there's an error
                duration = time.time() - start_time

                # Log error
                log.error(f"TIMING: {category}:{func_name} failed after {duration:.3f}s: {str(e)}")

                # Re-raise the exception
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get the instance if this is a method
            instance = args[0] if args else None

            # Get the logger - use instance logger if available, otherwise use module logger
            log = getattr(instance, "logger", logger) if instance else logger

            # Get function name
            func_name = func.__qualname__

            # Start timing
            start_time = time.time()

            # Log start
            log.debug(f"TIMING_START: {category}:{func_name}")

            try:
                # Call the function
                result = func(*args, **kwargs)

                # Calculate duration
                duration = time.time() - start_time

                # Log completion
                log.info(f"TIMING: {category}:{func_name} completed in {duration:.3f}s")

                return result
            except Exception as e:
                # Calculate duration even if there's an error
                duration = time.time() - start_time

                # Log error
                log.error(f"TIMING: {category}:{func_name} failed after {duration:.3f}s: {str(e)}")

                # Re-raise the exception
                raise

        # Return the appropriate wrapper based on whether the function is async or not
        if asyncio_is_coroutine_function(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def asyncio_is_coroutine_function(func: Callable) -> bool:
    """
    Check if a function is a coroutine function (async).

    Args:
        func: Function to check

    Returns:
        True if the function is a coroutine function, False otherwise
    """
    import inspect

    return inspect.iscoroutinefunction(func)


class TimingContext:
    """
    Context manager for timing code blocks.

    Example:
        with TimingContext("database_query", logger=agent.logger) as timer:
            results = db.execute_query("SELECT * FROM data")
            timer.add_metadata({"rows": len(results)})
    """

    def __init__(self, operation: str, category: str = "block", logger: Optional[logging.Logger] = None):
        """
        Initialize the timing context.

        Args:
            operation: Name of the operation being timed
            category: Category of the operation
            logger: Logger to use (defaults to module logger)
        """
        self.operation = operation
        self.category = category
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.metadata = {}

    def __enter__(self):
        """Start timing when entering the context."""
        self.start_time = time.time()
        self.logger.debug(f"TIMING_START: {self.category}:{self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log timing information when exiting the context."""
        duration = time.time() - self.start_time

        # Format metadata if present
        metadata_str = ""
        if self.metadata:
            metadata_str = " " + " ".join(f"{k}={v}" for k, v in self.metadata.items())

        if exc_type is None:
            # No exception occurred
            self.logger.info(f"TIMING: {self.category}:{self.operation} completed in {duration:.3f}s{metadata_str}")
        else:
            # An exception occurred
            self.logger.error(
                f"TIMING: {self.category}:{self.operation} failed after {duration:.3f}s: {str(exc_val)}{metadata_str}"
            )

    def add_metadata(self, metadata: dict):
        """
        Add metadata to the timing information.

        Args:
            metadata: Dictionary of metadata to add
        """
        self.metadata.update(metadata)
