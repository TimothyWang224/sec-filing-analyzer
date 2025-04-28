from typing import Any, Callable, Dict, Optional

from ..tools.base import Tool
from ..tools.registry import ToolRegistry


class Environment:
    """Unified environment for agent interactions with tools and context."""

    def __init__(self, tool_filter: Optional[Callable[[str, Dict[str, Any]], bool]] = None):
        """
        Initialize the environment.

        Args:
            tool_filter: Optional function to filter which tools to include
                         Function receives (tool_name, tool_info) and returns True to include
        """
        self.context: Dict[str, Any] = {}
        self.tools: Dict[str, Tool] = {}

        # Initialize tools from registry
        self._initialize_tools(tool_filter)

    def _initialize_tools(self, tool_filter: Optional[Callable] = None) -> None:
        """Initialize tools from the registry."""
        for name, info in ToolRegistry.list_tools().items():
            if tool_filter is None or tool_filter(name, info):
                try:
                    # Create an instance of the tool
                    tool_instance = info["class"]()

                    # Register the tool
                    self.register_tool(name, tool_instance)
                except Exception as e:
                    print(f"Error initializing tool {name}: {str(e)}")

    async def execute_action(self, action: Dict[str, Any]) -> Any:
        """
        Execute an action in the environment.

        Args:
            action: The action to execute with format:
                   {"tool": "tool_name", "args": {"param1": value1, ...}}

        Returns:
            The result of the action
        """
        tool_name = action.get("tool")
        args = action.get("args", {})

        if not tool_name:
            raise ValueError("Action must specify a tool name")

        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Validate arguments
        tool.validate_args(**args)

        # Execute the tool
        return await tool.execute(**args)

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            The result of the tool execution
        """
        return await self.execute_action({"tool": tool_name, "args": kwargs})

    def get_context(self) -> Dict[str, Any]:
        """Get the current environment context."""
        return self.context

    def update_context(self, updates: Dict[str, Any]) -> None:
        """
        Update the environment context.

        Args:
            updates: Dictionary of context updates
        """
        self.context.update(updates)

    def clear_context(self) -> None:
        """Clear the environment context."""
        self.context = {}

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            The tool if found, None otherwise
        """
        return self.tools.get(tool_name)

    def register_tool(self, tool_name: str, tool: Tool) -> None:
        """
        Register a tool in the environment.

        Args:
            tool_name: Name of the tool
            tool: The tool instance to register
        """
        self.tools[tool_name] = tool
        self.context[f"tool_{tool_name}"] = tool

    def get_available_tools(self) -> Dict[str, Tool]:
        """Get all available tools."""
        return self.tools.copy()
