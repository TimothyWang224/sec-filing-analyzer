"""
Tool for getting detailed information about other tools.
"""

from typing import Dict, Any

from .base import Tool
from .registry import ToolRegistry

class ToolDetailsTool(Tool):
    """Tool for getting detailed information about other tools."""
    
    _tool_name = "tool_details"
    _tool_tags = ["meta", "tools"]
    _compact_description = "Get detailed information about a specific tool"
    
    async def execute(self, tool_name: str) -> Dict[str, Any]:
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
