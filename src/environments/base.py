from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

class Environment(ABC):
    """Base class for agent environments."""
    
    def __init__(self):
        """Initialize the environment."""
        self.context: Dict[str, Any] = {}
        
    @abstractmethod
    async def execute_action(self, agent: Any, action: Dict[str, Any]) -> Any:
        """
        Execute an action in the environment.
        
        Args:
            agent: The agent executing the action
            action: The action to execute
            
        Returns:
            The result of the action
        """
        pass
    
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
        
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            The tool if found, None otherwise
        """
        return self.context.get(f"tool_{tool_name}")
    
    def register_tool(self, tool_name: str, tool: Any) -> None:
        """
        Register a tool in the environment.
        
        Args:
            tool_name: Name of the tool
            tool: The tool instance to register
        """
        self.context[f"tool_{tool_name}"] = tool 