"""
Unified state management for agents.
"""

from typing import Dict, Any, List, Optional

class AgentState:
    """Unified state management for agents."""
    
    def __init__(self):
        """Initialize the agent state."""
        self.memory: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.current_iteration: int = 0
        
    def add_memory_item(self, item: Dict[str, Any]) -> None:
        """
        Add an item to the agent's memory.
        
        Args:
            item: Memory item to add
        """
        self.memory.append(item)
        
    def add_context(self, key: str, value: Any) -> None:
        """
        Add a key-value pair to the agent's context.
        
        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
        
    def get_memory(self) -> List[Dict[str, Any]]:
        """Get the agent's memory."""
        return self.memory
    
    def get_context(self) -> Dict[str, Any]:
        """Get the agent's context."""
        return self.context
    
    def get_full_state(self) -> Dict[str, Any]:
        """Get the agent's full state."""
        return {
            "memory": self.memory,
            "context": self.context,
            "current_iteration": self.current_iteration
        }
    
    def increment_iteration(self) -> None:
        """Increment the current iteration counter."""
        self.current_iteration += 1
        
    def clear(self) -> None:
        """Clear the agent's state."""
        self.memory = []
        self.context = {}
        self.current_iteration = 0
