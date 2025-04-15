"""
Error handling for agent tool calls.

This module provides classes and utilities for handling errors in agent tool calls,
including error classification, recovery strategies, and circuit breaker patterns.
"""

import logging
import random
import time
import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union

from ...tools.registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolErrorType(Enum):
    """Types of errors that can occur during tool execution."""
    
    PARAMETER_ERROR = "parameter_error"      # Invalid or missing parameters
    NETWORK_ERROR = "network_error"          # Network connectivity issues
    AUTH_ERROR = "auth_error"                # Authentication/authorization issues
    RATE_LIMIT_ERROR = "rate_limit_error"    # Rate limiting or quota issues
    DATA_ERROR = "data_error"                # Data not found or invalid data
    SYSTEM_ERROR = "system_error"            # Internal system errors
    UNKNOWN_ERROR = "unknown_error"          # Unclassified errors


class ToolError:
    """
    Represents an error that occurred during tool execution.
    
    This class encapsulates information about the error, including its type,
    message, and the original exception that caused it.
    """
    
    def __init__(
        self, 
        error_type: ToolErrorType, 
        message: str, 
        tool_name: str,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize a tool error.
        
        Args:
            error_type: Type of the error
            message: Error message
            tool_name: Name of the tool that caused the error
            original_exception: Original exception that caused the error
        """
        self.error_type = error_type
        self.message = message
        self.tool_name = tool_name
        self.original_exception = original_exception
        self.timestamp = datetime.now()
        
    def is_recoverable(self) -> bool:
        """
        Determine if this error is potentially recoverable with retries.
        
        Returns:
            True if the error is potentially recoverable, False otherwise
        """
        return self.error_type in [
            ToolErrorType.NETWORK_ERROR, 
            ToolErrorType.RATE_LIMIT_ERROR,
            ToolErrorType.PARAMETER_ERROR
        ]
        
    def get_recovery_strategy(self) -> str:
        """
        Get the recommended recovery strategy for this error.
        
        Returns:
            Name of the recovery strategy to use
        """
        if self.error_type == ToolErrorType.PARAMETER_ERROR:
            return "fix_parameters"
        elif self.error_type == ToolErrorType.NETWORK_ERROR:
            return "retry_with_backoff"
        elif self.error_type == ToolErrorType.RATE_LIMIT_ERROR:
            return "retry_with_longer_backoff"
        elif self.error_type == ToolErrorType.AUTH_ERROR:
            return "check_credentials"
        else:
            return "report_error"
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary.
        
        Returns:
            Dictionary representation of the error
        """
        return {
            "type": self.error_type.value,
            "message": self.message,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp.isoformat()
        }
        
    def __str__(self) -> str:
        """String representation of the error."""
        return f"{self.error_type.value}: {self.message} (tool: {self.tool_name})"


class ErrorClassifier:
    """
    Classifies exceptions into tool error types.
    
    This class analyzes exceptions and determines the appropriate error type
    based on the exception message and type.
    """
    
    @staticmethod
    def classify_error(exception: Exception, tool_name: str) -> ToolError:
        """
        Classify an exception into a tool error.
        
        Args:
            exception: The exception to classify
            tool_name: Name of the tool that raised the exception
            
        Returns:
            Classified tool error
        """
        error_message = str(exception)
        error_type = type(exception).__name__
        
        # Check for parameter errors
        if any(keyword in error_message.lower() for keyword in 
               ["parameter", "argument", "missing", "required", "invalid", "type error"]):
            return ToolError(
                ToolErrorType.PARAMETER_ERROR,
                error_message,
                tool_name,
                exception
            )
            
        # Check for network errors
        elif any(keyword in error_message.lower() for keyword in 
                ["connection", "timeout", "network", "unreachable", "dns", "socket"]):
            return ToolError(
                ToolErrorType.NETWORK_ERROR,
                error_message,
                tool_name,
                exception
            )
            
        # Check for authentication errors
        elif any(keyword in error_message.lower() for keyword in 
                ["authentication", "authorization", "permission", "access denied", 
                 "forbidden", "unauthorized", "credentials"]):
            return ToolError(
                ToolErrorType.AUTH_ERROR,
                error_message,
                tool_name,
                exception
            )
            
        # Check for rate limit errors
        elif any(keyword in error_message.lower() for keyword in 
                ["rate limit", "quota", "too many requests", "throttle"]):
            return ToolError(
                ToolErrorType.RATE_LIMIT_ERROR,
                error_message,
                tool_name,
                exception
            )
            
        # Check for data errors
        elif any(keyword in error_message.lower() for keyword in 
                ["not found", "no data", "empty", "no results", "invalid data"]):
            return ToolError(
                ToolErrorType.DATA_ERROR,
                error_message,
                tool_name,
                exception
            )
            
        # Check for system errors
        elif any(keyword in error_message.lower() for keyword in 
                ["internal", "server error", "system", "unexpected"]):
            return ToolError(
                ToolErrorType.SYSTEM_ERROR,
                error_message,
                tool_name,
                exception
            )
            
        # Default to unknown error
        else:
            return ToolError(
                ToolErrorType.UNKNOWN_ERROR,
                error_message,
                tool_name,
                exception
            )


class ToolCircuitBreaker:
    """
    Circuit breaker for tool calls to prevent repeated failures.
    
    This class implements the circuit breaker pattern to prevent repeated calls
    to tools that are consistently failing.
    """
    
    def __init__(self, failure_threshold: int = 3, reset_timeout: int = 300):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening the circuit
            reset_timeout: Time in seconds before attempting to reset the circuit
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_counts = {}  # {tool_name: count}
        self.circuit_status = {}  # {tool_name: {"status": "open"|"closed", "last_failure": timestamp}}
        
    def record_failure(self, tool_name: str) -> None:
        """
        Record a tool failure.
        
        Args:
            tool_name: Name of the tool that failed
        """
        now = time.time()
        
        if tool_name not in self.failure_counts:
            self.failure_counts[tool_name] = 0
            self.circuit_status[tool_name] = {"status": "closed", "last_failure": now}
            
        self.failure_counts[tool_name] += 1
        self.circuit_status[tool_name]["last_failure"] = now
        
        # Check if we should open the circuit
        if self.failure_counts[tool_name] >= self.failure_threshold:
            self.circuit_status[tool_name]["status"] = "open"
            logger.warning(f"Circuit opened for tool {tool_name} after {self.failure_threshold} consecutive failures")
            
    def record_success(self, tool_name: str) -> None:
        """
        Record a tool success.
        
        Args:
            tool_name: Name of the tool that succeeded
        """
        if tool_name in self.failure_counts:
            self.failure_counts[tool_name] = 0
            self.circuit_status[tool_name]["status"] = "closed"
            
    def can_execute(self, tool_name: str) -> bool:
        """
        Check if a tool can be executed.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool can be executed, False otherwise
        """
        if tool_name not in self.circuit_status:
            return True
            
        status = self.circuit_status[tool_name]
        
        # If circuit is closed, allow execution
        if status["status"] == "closed":
            return True
            
        # If circuit is open, check if we should try to reset
        now = time.time()
        time_since_failure = now - status["last_failure"]
        
        if time_since_failure >= self.reset_timeout:
            # Allow a single test execution to see if the issue is resolved
            logger.info(f"Attempting to reset circuit for tool {tool_name}")
            return True
            
        return False
        
    def get_status(self, tool_name: str) -> Dict[str, Any]:
        """
        Get the status of a tool's circuit.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            Dictionary with circuit status information
        """
        if tool_name not in self.circuit_status:
            return {"status": "closed", "failure_count": 0}
            
        return {
            "status": self.circuit_status[tool_name]["status"],
            "failure_count": self.failure_counts.get(tool_name, 0),
            "last_failure": self.circuit_status[tool_name]["last_failure"]
        }


class ErrorAnalyzer:
    """
    Analyze patterns of errors to improve future tool calls.
    
    This class tracks error history and provides insights and suggestions
    based on common error patterns.
    """
    
    def __init__(self):
        """Initialize the error analyzer."""
        self.error_history = []  # List of ToolError objects
        
    def add_error(self, error: ToolError) -> None:
        """
        Add an error to the history.
        
        Args:
            error: The error to add
        """
        self.error_history.append(error)
        
    def get_common_errors(self, tool_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most common errors for a tool.
        
        Args:
            tool_name: Name of the tool to analyze
            limit: Maximum number of error types to return
            
        Returns:
            List of dictionaries with error type and count
        """
        tool_errors = [e for e in self.error_history if e.tool_name == tool_name]
        
        # Group errors by type
        error_types = {}
        for error in tool_errors:
            if error.error_type not in error_types:
                error_types[error.error_type] = 0
            error_types[error.error_type] += 1
            
        # Sort by frequency
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        
        return [{"type": e[0].value, "count": e[1]} for e in sorted_errors[:limit]]
        
    def get_error_suggestions(self, tool_name: str, error_type: ToolErrorType) -> List[str]:
        """
        Get suggestions for fixing common errors.
        
        Args:
            tool_name: Name of the tool
            error_type: Type of error
            
        Returns:
            List of suggestion strings
        """
        if error_type == ToolErrorType.PARAMETER_ERROR:
            return [
                "Check parameter types and formats",
                "Ensure required parameters are provided",
                "Verify date formats (use YYYY-MM-DD)"
            ]
        elif error_type == ToolErrorType.NETWORK_ERROR:
            return [
                "Check network connectivity",
                "Verify API endpoints are accessible",
                "Consider increasing timeout values"
            ]
        elif error_type == ToolErrorType.RATE_LIMIT_ERROR:
            return [
                "Reduce the frequency of requests",
                "Implement exponential backoff",
                "Consider upgrading API tier for higher limits"
            ]
        elif error_type == ToolErrorType.AUTH_ERROR:
            return [
                "Verify API credentials",
                "Check if API keys are expired",
                "Ensure proper permissions are set"
            ]
        elif error_type == ToolErrorType.DATA_ERROR:
            return [
                "Verify the requested data exists",
                "Check if data format has changed",
                "Try a different query or parameters"
            ]
        elif error_type == ToolErrorType.SYSTEM_ERROR:
            return [
                "Check system logs for details",
                "Verify system dependencies",
                "Contact system administrator"
            ]
        else:
            return ["No specific suggestions available for this error type"]
            
    def format_error_for_user(self, error: ToolError) -> str:
        """
        Format an error message for user display.
        
        Args:
            error: The error to format
            
        Returns:
            User-friendly error message
        """
        if error.error_type == ToolErrorType.PARAMETER_ERROR:
            return f"I couldn't complete this task because of an issue with the parameters: {error.message}. Please provide more specific information."
        elif error.error_type == ToolErrorType.DATA_ERROR:
            return f"I couldn't find the requested data: {error.message}. Please check if the information exists or try a different query."
        elif error.error_type == ToolErrorType.NETWORK_ERROR:
            return "I'm having trouble connecting to the data source. This might be a temporary issue. Please try again in a few moments."
        elif error.error_type == ToolErrorType.RATE_LIMIT_ERROR:
            return "I've reached the limit for data requests. Please try again in a few minutes."
        elif error.error_type == ToolErrorType.AUTH_ERROR:
            return "I don't have the necessary permissions to access this data. Please check your authentication settings."
        else:
            return f"I encountered an error while processing your request: {error.message}. Please try again or rephrase your question."
