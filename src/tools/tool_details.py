"""
Tool for getting detailed information about other tools.
"""

from typing import Dict, Any

from .base import Tool
from .registry import ToolRegistry
from .decorator import tool

@tool(
    name="tool_details",
    tags=["meta", "tools"],
    compact_description="Get detailed information about a specific tool"
)
class ToolDetailsTool(Tool):
    """Tool for getting detailed information about other tools."""

    def validate_args(self, tool_name: str, **kwargs) -> bool:
        """Override the validate_args method to accept kwargs."""
        if not tool_name or not isinstance(tool_name, str):
            raise ValueError("Missing required parameter: tool_name")
        return True


    async def _execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.

        Args:
            tool_name: Name of the tool to get details for

        Returns:
            Dictionary containing detailed information about the tool
        """
        tool_info = ToolRegistry.get(tool_name)

        if not tool_info:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "available_tools": list(ToolRegistry.list_tools().keys())
            }

        # Get formatted documentation
        detailed_docs = ToolRegistry.get_tool_documentation(name=tool_name, format="text")

        return {
            "success": True,
            "tool_name": tool_name,
            "description": tool_info.get("description", ""),
            "parameters": tool_info.get("parameters", {}),
            "documentation": detailed_docs
        }
