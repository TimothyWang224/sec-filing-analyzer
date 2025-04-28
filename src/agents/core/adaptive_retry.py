"""
Adaptive retry strategies for agent tool calls.

This module provides utilities for implementing adaptive retry strategies
based on error types and patterns.
"""

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, Dict

from .error_handling import ToolError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveRetryStrategy:
    """
    Implements adaptive retry strategies for tool calls.

    This class provides methods for retrying tool calls with different
    backoff strategies based on the type of error.
    """

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.5,
    ):
        """
        Initialize the adaptive retry strategy.

        Args:
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter_factor: Factor for random jitter (0-1)
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor

    async def retry_with_strategy(
        self,
        func: Callable[[], Awaitable[Any]],
        max_retries: int,
        error_classifier: Callable[[Exception], ToolError],
    ) -> Dict[str, Any]:
        """
        Retry a function with an adaptive strategy based on error type.

        Args:
            func: Async function to retry
            max_retries: Maximum number of retries
            error_classifier: Function to classify exceptions into tool errors

        Returns:
            Dictionary with result or error information
        """
        # Track previous errors to detect repeated identical errors
        previous_errors = []

        for retry in range(max_retries + 1):
            try:
                # Execute the function
                result = await func()
                return {"success": True, "result": result, "retries": retry}

            except Exception as e:
                # Classify the error
                error = error_classifier(e)

                # Log the error
                logger.error(f"Error on attempt {retry + 1}/{max_retries + 1}: {error}")

                # Check if this is a repeated identical error
                if previous_errors and str(error) == str(previous_errors[-1]):
                    logger.warning("Detected identical error on consecutive attempts - likely parameter issue")
                    # If we've seen this exact error before, it's likely that retrying won't help
                    # Return with a special flag indicating identical errors
                    return {
                        "success": False,
                        "error": error,
                        "retries": retry,
                        "identical_errors": True,
                        "error_message": str(error),
                    }

                # Add this error to the history
                previous_errors.append(error)

                # Check if we should retry
                if retry >= max_retries or not error.is_recoverable():
                    return {"success": False, "error": error, "retries": retry}

                # Get recovery strategy
                strategy = error.get_recovery_strategy()

                # Apply recovery strategy
                if strategy == "retry_with_backoff":
                    await self._apply_exponential_backoff(retry)
                elif strategy == "retry_with_longer_backoff":
                    await self._apply_longer_backoff(retry)
                else:
                    # Default backoff
                    await self._apply_default_backoff(retry)

    async def _apply_exponential_backoff(self, retry: int) -> None:
        """
        Apply exponential backoff with jitter.

        Args:
            retry: Current retry attempt (0-based)
        """
        # Calculate delay with exponential backoff
        delay = min(
            self.base_delay * (2**retry) + random.uniform(0, self.jitter_factor),
            self.max_delay,
        )

        logger.info(f"Retrying with exponential backoff in {delay:.2f}s...")
        await asyncio.sleep(delay)

    async def _apply_longer_backoff(self, retry: int) -> None:
        """
        Apply longer exponential backoff for rate limits.

        Args:
            retry: Current retry attempt (0-based)
        """
        # Calculate delay with more aggressive exponential backoff
        delay = min(
            self.base_delay * (4**retry) + random.uniform(0, self.jitter_factor * 2),
            self.max_delay,
        )

        logger.info(f"Rate limited. Retrying with longer backoff in {delay:.2f}s...")
        await asyncio.sleep(delay)

    async def _apply_default_backoff(self, retry: int) -> None:
        """
        Apply default backoff.

        Args:
            retry: Current retry attempt (0-based)
        """
        # Calculate delay with linear backoff
        delay = min(
            self.base_delay * (retry + 1) + random.uniform(0, self.jitter_factor),
            self.max_delay,
        )

        logger.info(f"Retrying with default backoff in {delay:.2f}s...")
        await asyncio.sleep(delay)
