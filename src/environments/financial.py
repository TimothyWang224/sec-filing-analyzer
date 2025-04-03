from typing import Dict, Any, Optional
from .base import Environment
from ..tools.base import Tool
from ..tools.sec_data import SECDataTool

class FinancialEnvironment(Environment):
    """Environment specialized for financial analysis tasks."""
    
    def __init__(self):
        """Initialize the financial environment."""
        super().__init__()
        
        # Register default tools
        self.register_tool(SECDataTool())
        
    async def execute_action(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an action in the financial environment.
        
        Args:
            action: Action to execute
            context: Optional context for the action
            
        Returns:
            Dictionary containing action results
        """
        # Get the tool to use
        tool_name = action.get("tool")
        if not tool_name:
            raise ValueError("Action must specify a tool to use")
            
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
            
        # Validate tool arguments
        tool_args = action.get("args", {})
        if not tool.validate_args(**tool_args):
            raise ValueError(f"Invalid arguments for tool {tool_name}")
            
        # Execute the tool
        result = await tool.execute(**tool_args)
        
        # Update context if provided
        if context:
            context.update({
                "last_action": action,
                "last_result": result
            })
            
        return result
        
    def get_available_tools(self) -> Dict[str, Tool]:
        """
        Get all available tools in the environment.
        
        Returns:
            Dictionary mapping tool names to tool instances
        """
        return self.context.get("tools", {})
        
    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary containing tool metadata if found, None otherwise
        """
        tool = self.get_tool(tool_name)
        if tool:
            return tool.get_metadata()
        return None 