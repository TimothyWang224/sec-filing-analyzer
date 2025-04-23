"""
Tool for getting detailed information about other tools.
"""

import logging
from typing import Dict, Any, Optional, Type

from .base import Tool
from .registry import ToolRegistry
from .decorator import tool
from ..contracts import BaseModel, ToolSpec, field_validator
from ..errors import ParameterError, QueryTypeUnsupported, StorageUnavailable, DataNotFound

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define parameter models
class ToolDetailsParams(BaseModel):
    """Parameters for tool details queries."""
    tool_name: str

    @field_validator('tool_name')
    @classmethod
    def validate_tool_name(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Tool name must be a non-empty string")
        return v

# Map query types to parameter models
SUPPORTED_QUERIES: Dict[str, Type[BaseModel]] = {
    "tool_details": ToolDetailsParams
}

# Register tool specification
ToolRegistry._tool_specs["tool_details"] = ToolSpec(
    name="tool_details",
    input_schema=SUPPORTED_QUERIES,
    output_key="tool_details",
    description="Tool for getting detailed information about other tools."
)

@tool(
    name="tool_details",
    tags=["meta", "tools"],
    compact_description="Get detailed information about a specific tool"
)
class ToolDetailsTool(Tool):
    """Tool for getting detailed information about other tools."""

    def validate_args(
        self,
        query_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
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


    async def _execute(
        self,
        query_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.

        Args:
            query_type: Type of query to execute (e.g., "tool_details")
            parameters: Parameters for the query

        Returns:
            Dictionary containing detailed information about the tool

        Raises:
            QueryTypeUnsupported: If the query type is not supported
            ParameterError: If the parameters are invalid
            DataNotFound: If the tool is not found
        """
        try:
            # Validate query type
            if query_type not in SUPPORTED_QUERIES:
                supported_types = list(SUPPORTED_QUERIES.keys())
                raise QueryTypeUnsupported(query_type, "tool_details", supported_types)

            # Validate parameters using the appropriate model
            param_model = SUPPORTED_QUERIES[query_type]
            if parameters is None:
                parameters = {}

            try:
                # Validate parameters
                params = param_model(**parameters)
            except Exception as e:
                raise ParameterError(str(e))

            # Extract parameters
            tool_name = params.tool_name

            # Get tool info
            tool_info = ToolRegistry.get(tool_name)

            if not tool_info:
                raise DataNotFound("tool", {"tool_name": tool_name})

            # Get formatted documentation
            detailed_docs = ToolRegistry.get_tool_documentation(name=tool_name, format="text")

            return {
                "success": True,
                "tool_name": tool_name,
                "description": tool_info.get("description", ""),
                "parameters": tool_info.get("parameters", {}),
                "documentation": detailed_docs,
                "output_key": "tool_details"
            }
        except DataNotFound as e:
            # Handle tool not found error
            return {
                "success": False,
                "error": f"Tool '{e.query_params.get('tool_name')}' not found",
                "available_tools": list(ToolRegistry.list_tools().keys()),
                "output_key": "tool_details"
            }
        except (QueryTypeUnsupported, ParameterError) as e:
            # Re-raise known errors
            raise
        except Exception as e:
            logger.error(f"Error getting tool details: {str(e)}")
            raise StorageUnavailable("tool_registry", f"Error getting tool details: {str(e)}")
