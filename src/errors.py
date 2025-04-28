"""
Error classes for the SEC Filing Analyzer.

This module defines a hierarchy of error classes that can be used to
provide more specific error information and enable better error handling.
"""

from typing import Any, Dict, List, Optional


class ToolError(Exception):
    """Base class for all tool-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def user_message(self) -> str:
        """Return a user-friendly error message."""
        return self.message


class ParameterError(ToolError):
    """Error raised when tool parameters are invalid."""

    def user_message(self) -> str:
        """Return a user-friendly error message."""
        if "field" in self.details:
            return f"The parameter '{self.details['field']}' is invalid: {self.message}"
        return f"Invalid parameter: {self.message}"


class QueryTypeUnsupported(ToolError):
    """Error raised when a query type is not supported by a tool."""

    def __init__(
        self,
        query_type: str,
        tool_name: str,
        supported_types: Optional[List[str]] = None,
    ):
        self.query_type = query_type
        self.tool_name = tool_name
        self.supported_types = supported_types or []

        message = f"Query type '{query_type}' is not supported by the {tool_name} tool."
        if self.supported_types:
            message += f" Supported types are: {', '.join(self.supported_types)}"

        super().__init__(
            message,
            {
                "query_type": query_type,
                "tool_name": tool_name,
                "supported_types": self.supported_types,
            },
        )

    def user_message(self) -> str:
        """Return a user-friendly error message."""
        if self.supported_types:
            return f"I can't perform a '{self.query_type}' query with the {self.tool_name} tool. Try one of these instead: {', '.join(self.supported_types)}"
        return f"I can't perform a '{self.query_type}' query with the {self.tool_name} tool."


class StorageUnavailable(ToolError):
    """Error raised when a storage system is unavailable."""

    def __init__(self, storage_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.storage_type = storage_type

        full_message = f"{storage_type} storage is unavailable: {message}"
        super().__init__(full_message, details or {"storage_type": storage_type})

    def user_message(self) -> str:
        """Return a user-friendly error message."""
        return f"I'm having trouble accessing the {self.storage_type} data. Please try again later or contact support if the problem persists."


class DataNotFound(ToolError):
    """Error raised when requested data is not found."""

    def __init__(
        self,
        data_type: str,
        query_params: Dict[str, Any],
        message: Optional[str] = None,
    ):
        self.data_type = data_type
        self.query_params = query_params

        default_message = f"Could not find {data_type} data matching the query parameters."
        super().__init__(
            message or default_message,
            {"data_type": data_type, "query_params": query_params},
        )

    def user_message(self) -> str:
        """Return a user-friendly error message."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.query_params.items())
        return f"I couldn't find any {self.data_type} data for {params_str}. Please check your query or try different parameters."


# Error classification and mapping
ERROR_CLASSIFICATION = {
    ParameterError: "parameter_error",
    QueryTypeUnsupported: "query_type_error",
    StorageUnavailable: "storage_error",
    DataNotFound: "data_not_found",
    ToolError: "tool_error",
}


def classify_error(error: Exception) -> str:
    """
    Classify an error based on its type.

    Args:
        error: The error to classify

    Returns:
        A string classification of the error
    """
    for error_class, classification in ERROR_CLASSIFICATION.items():
        if isinstance(error, error_class):
            return classification

    return "unknown_error"


def get_user_message(error: Exception) -> str:
    """
    Get a user-friendly message for an error.

    Args:
        error: The error to get a message for

    Returns:
        A user-friendly error message
    """
    if isinstance(error, ToolError):
        return error.user_message()

    return str(error)
