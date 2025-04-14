from typing import Dict, Any, Optional, Callable
from .base import Environment

class FinancialEnvironment(Environment):
    """Environment specialized for financial analysis tasks."""

    def __init__(self):
        """Initialize the financial environment."""
        # Define a filter to only include SEC tools
        def sec_tool_filter(name, info):
            return any(tag in info.get("tags", []) for tag in ["sec"])

        # Initialize with the SEC tool filter
        super().__init__(tool_filter=sec_tool_filter)

        # Print available tools
        print(f"Available tools: {list(self.tools.keys()) if self.tools else 'None'}")

    async def execute_action(
        self,
        action: Dict[str, Any] = None,
        agent: Any = None
    ) -> Dict[str, Any]:
        """
        Execute an action in the financial environment.

        Args:
            action: Action to execute
            agent: The agent executing the action (not used in this implementation)

        Returns:
            Dictionary containing action results
        """
        # Get the tool to use
        tool_name = action.get("tool")
        if not tool_name:
            raise ValueError("Action must specify a tool to use")

        # Debug: Print tool lookup
        print(f"Looking for tool: {tool_name}")
        print(f"Available tools in context: {[k for k in self.context.keys() if k.startswith('tool_')]}")

        tool = self.get_tool(tool_name)
        print(f"Tool found: {tool is not None}")

        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        # Validate tool arguments
        tool_args = action.get("args", {})
        if not tool.validate_args(**tool_args):
            raise ValueError(f"Invalid arguments for tool {tool_name}")

        # Execute the tool
        print(f"Executing tool: {tool_name} with args: {tool_args}")
        try:
            result = await tool.execute(**tool_args)
            print(f"Tool execution result: {result is not None}")
        except Exception as e:
            print(f"Error executing tool: {str(e)}")
            raise

        # Update environment context
        self.context.update({
            "last_action": action,
            "last_result": result
        })

        return result

    def get_available_tools(self):
        """
        Get all available tools in the environment.

        Returns:
            Dictionary mapping tool names to tool instances
        """
        return self.tools

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