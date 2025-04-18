"""
Adaptive rate limiter for API calls.

This module provides an adaptive rate limiter that can dynamically adjust
based on API response patterns, helping to maximize throughput while
avoiding rate limit errors.
"""

import time
import threading
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class AdaptiveRateLimiter:
    """
    Adaptive rate limiter for API calls.
    
    Features:
    - Dynamic rate adjustment based on success/failure patterns
    - Shared state across threads for coordinated rate limiting
    - Exponential backoff on failures
    - Gradual rate increase on sustained success
    """
    
    def __init__(
        self,
        initial_rate_limit: float = 0.5,  # Start conservative
        min_rate_limit: float = 0.1,      # Fastest we'll ever go
        max_rate_limit: float = 5.0,      # Slowest we'll ever go
        backoff_factor: float = 2.0,      # How much to slow down on failure
        success_factor: float = 0.95,     # How much to speed up on success
        success_threshold: int = 10,      # How many successes before speeding up
        shared_state: Optional[Dict[str, Any]] = None  # For sharing state across instances
    ):
        """Initialize the adaptive rate limiter.
        
        Args:
            initial_rate_limit: Starting rate limit in seconds
            min_rate_limit: Minimum rate limit in seconds
            max_rate_limit: Maximum rate limit in seconds
            backoff_factor: Factor to multiply rate limit by on failure
            success_factor: Factor to multiply rate limit by on success
            success_threshold: Number of consecutive successes before speeding up
            shared_state: Optional shared state dictionary for coordination
        """
        # Use shared state if provided, otherwise create new state
        if shared_state is not None:
            self.state = shared_state
        else:
            self.state = {
                "rate_limit": initial_rate_limit,
                "last_request_time": 0,
                "consecutive_successes": 0,
                "consecutive_failures": 0,
                "total_requests": 0,
                "total_successes": 0,
                "total_failures": 0,
                "lock": threading.Lock()
            }
        
        # Store configuration
        self.min_rate_limit = min_rate_limit
        self.max_rate_limit = max_rate_limit
        self.backoff_factor = backoff_factor
        self.success_factor = success_factor
        self.success_threshold = success_threshold
    
    def wait(self):
        """Wait for the appropriate amount of time based on the current rate limit."""
        with self.state["lock"]:
            current_time = time.time()
            time_since_last = current_time - self.state["last_request_time"]
            
            if time_since_last < self.state["rate_limit"]:
                wait_time = self.state["rate_limit"] - time_since_last
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s (current rate: {self.state['rate_limit']:.2f}s)")
                time.sleep(wait_time)
            
            self.state["last_request_time"] = time.time()
            self.state["total_requests"] += 1
    
    def report_success(self):
        """Report a successful API call, potentially adjusting the rate limit."""
        with self.state["lock"]:
            self.state["consecutive_successes"] += 1
            self.state["consecutive_failures"] = 0
            self.state["total_successes"] += 1
            
            # If we've had enough consecutive successes, speed up
            if self.state["consecutive_successes"] >= self.success_threshold:
                old_rate = self.state["rate_limit"]
                self.state["rate_limit"] = max(
                    self.min_rate_limit,
                    self.state["rate_limit"] * self.success_factor
                )
                
                if old_rate != self.state["rate_limit"]:
                    logger.info(f"Increasing API throughput: {old_rate:.2f}s → {self.state['rate_limit']:.2f}s")
                
                # Reset success counter
                self.state["consecutive_successes"] = 0
    
    def report_failure(self, status_code: Optional[int] = None):
        """Report a failed API call, adjusting the rate limit.
        
        Args:
            status_code: Optional HTTP status code for more specific handling
        """
        with self.state["lock"]:
            self.state["consecutive_failures"] += 1
            self.state["consecutive_successes"] = 0
            self.state["total_failures"] += 1
            
            # Apply backoff based on status code
            if status_code == 429:  # Too Many Requests
                # More aggressive backoff for rate limit errors
                backoff = self.backoff_factor * 2
            else:
                backoff = self.backoff_factor
            
            old_rate = self.state["rate_limit"]
            self.state["rate_limit"] = min(
                self.max_rate_limit,
                self.state["rate_limit"] * backoff
            )
            
            logger.warning(
                f"API call failed{f' with status {status_code}' if status_code else ''}, "
                f"reducing throughput: {old_rate:.2f}s → {self.state['rate_limit']:.2f}s"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics about the rate limiter.
        
        Returns:
            Dictionary with current rate limit and request statistics
        """
        with self.state["lock"]:
            return {
                "current_rate_limit": self.state["rate_limit"],
                "total_requests": self.state["total_requests"],
                "total_successes": self.state["total_successes"],
                "total_failures": self.state["total_failures"],
                "consecutive_successes": self.state["consecutive_successes"],
                "consecutive_failures": self.state["consecutive_failures"]
            }
    
    def __enter__(self):
        """Context manager entry point."""
        self.wait()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        if exc_type is None:
            self.report_success()
        else:
            # Try to extract status code if it's an API error
            status_code = None
            if hasattr(exc_val, "status_code"):
                status_code = exc_val.status_code
            
            self.report_failure(status_code)
