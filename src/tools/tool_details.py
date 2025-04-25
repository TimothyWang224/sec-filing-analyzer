"""
Tool for getting detailed information about other tools.
"""

import logging
from typing import Any, Dict, Optional, Type

from ..contracts import BaseModel, field_validator
from .base import Tool
from .decorator import tool
from .registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define parameter models
class ToolDetailsParams(BaseModel):
    """Parameters for tool details queries."""

    tool_name: str

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Tool name must be a non-empty string")
        return v


# Map query types to parameter models
SUPPORTED_QUERIES: Dict[str, Type[BaseModel]] = {"tool_details": ToolDetailsParams}

# The tool registration is handled by the @tool decorator
# The ToolSpec will be created automatically by the ToolRegistry._register_tool method


@tool(name="tool_details", tags=["meta", "tools"], compact_description="Get detailed information about a specific tool")
class ToolDetailsTool(Tool):
    """Tool for getting detailed information about other tools."""

    def validate_args(self, query_type: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate the tool arguments.

        Args:
            query_type: Type of query to execute
            parameters: Parameters for the query

        Returns:
            True if arguments are valid, False otherwise
        """
        try:
            # Validate query type
            if query_type not in SUPPORTED_QUERIES:
                logger.error(f"Invalid query_type: must be one of {list(SUPPORTED_QUERIES.keys())}")
                return False

            # Validate parameters using the appropriate model
            param_model = SUPPORTED_QUERIES[query_type]
            if parameters is None:
                parameters = {}

            try:
                # Validate parameters
                param_model(**parameters)
                return True
            except Exception as e:
                logger.error(f"Parameter validation error: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False

    async def _execute(self, query_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.

        Args:
            query_type: Type of query to execute (e.g., "tool_details")
            parameters: Parameters for the query

        Returns:
            A standardized response dictionary with the following fields:
            - query_type: The type of query that was executed
            - parameters: The parameters that were used
            - results: Empty object (tool details are in other fields)
            - output_key: The tool's name
            - success: Boolean indicating whether the operation was successful
            - tool_name: The name of the requested tool
            - description: The tool's description
            - parameters: The tool's parameters
            - documentation: Detailed documentation for the tool

            Error responses will additionally have:
            - error or warning: The error message (depending on error_type)
            - available_tools: List of available tools (if tool not found)
        """
        # Ensure parameters is a dictionary
        if parameters is None:
            parameters = {}

        try:
            # Validate query type
            if query_type not in SUPPORTED_QUERIES:
                supported_types = list(SUPPORTED_QUERIES.keys())
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message=f"Unsupported query type: {query_type}. Supported types: {supported_types}",
                )

            # Validate parameters using the appropriate model
            param_model = SUPPORTED_QUERIES[query_type]

            try:
                # Validate parameters
                params = param_model(**parameters)
            except Exception as e:
                return self.format_error_response(
                    query_type=query_type, parameters=parameters, error_message=f"Parameter validation error: {str(e)}"
                )

            # Extract parameters
            tool_name = params.tool_name

            try:
                # Get tool info
                tool_info = ToolRegistry.get(tool_name)

                if not tool_info:
                    response = {
                        "query_type": query_type,
                        "parameters": parameters,
                        "warning": f"Tool '{tool_name}' not found",
                        "error": f"Tool '{tool_name}' not found",  # For backward compatibility
                        "results": [],
                        "output_key": self.name,
                        "success": False,
                        "available_tools": list(ToolRegistry.list_tools().keys()),
                    }
                    return response

                # Get formatted documentation
                detailed_docs = ToolRegistry.get_tool_documentation(name=tool_name, format="text")

                # Create a custom result with additional fields
                result = self.format_success_response(query_type=query_type, parameters=parameters, results={})

                # Add fields directly to the result
                result["tool_name"] = tool_name
                result["description"] = tool_info.get("description", "")
                result["parameters"] = tool_info.get("parameters", {})
                result["documentation"] = detailed_docs

                return result

            except Exception as e:
                logger.error(f"Error getting tool details: {str(e)}")
                return self.format_error_response(
                    query_type=query_type, parameters=parameters, error_message=f"Error getting tool details: {str(e)}"
                )

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return self.format_error_response(
                query_type=query_type, parameters=parameters, error_message=f"Unexpected error: {str(e)}"
            )
