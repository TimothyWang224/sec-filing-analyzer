"""
Unified state management for agents.
"""

from typing import Dict, Any, List, Optional, Literal
import time

class AgentState:
    """Unified state management for agents."""

    def __init__(self):
        """Initialize the agent state."""
        self.memory: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.current_iteration: int = 0
        self.start_time: float = time.time()
        self.current_phase: Literal['planning', 'execution', 'refinement'] = 'planning'
        self.phase_iterations: Dict[str, int] = {
            'planning': 0,
            'execution': 0,
            'refinement': 0
        }

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

    def update_context(self, updates: Dict[str, Any]) -> None:
        """
        Update the agent's context with multiple key-value pairs.

        Args:
            updates: Dictionary of context updates
        """
        self.context.update(updates)

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
        self.phase_iterations[self.current_phase] += 1

    def set_phase(self, phase: Literal['planning', 'execution', 'refinement']) -> None:
        """
        Set the current phase of the agent.

        Args:
            phase: The phase to set
        """
        self.current_phase = phase

    def clear(self) -> None:
        """Clear the agent's state."""
        self.memory = []
        self.context = {}
        self.current_iteration = 0
        self.start_time = time.time()
        self.current_phase = 'planning'
        self.phase_iterations = {
            'planning': 0,
            'execution': 0,
            'refinement': 0
        }
