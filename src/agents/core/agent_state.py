"""
Unified state management for agents.
"""

import time
from typing import Any, Dict, List, Literal, Optional


class AgentState:
    """Unified state management for agents."""

    # Default token budget for agents
    DEFAULT_TOKEN_BUDGET = {
        "planning": 25000,  # 10% for planning
        "execution": 100000,  # 40% for execution
        "refinement": 125000,  # 50% for refinement
    }

    def __init__(self):
        """Initialize the agent state."""
        self.memory: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.current_iteration: int = 0
        self.start_time: float = time.time()
        self.current_phase: Literal["planning", "execution", "refinement"] = "planning"
        self.phase_iterations: Dict[str, int] = {
            "planning": 0,
            "execution": 0,
            "refinement": 0,
        }
        self.tokens_used: Dict[str, int] = {
            "planning": 0,
            "execution": 0,
            "refinement": 0,
        }
        # Initialize with default token budget
        self.token_budget: Dict[str, int] = self.DEFAULT_TOKEN_BUDGET.copy()

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
            "current_iteration": self.current_iteration,
        }

    def increment_iteration(self) -> None:
        """Increment the current iteration counter."""
        self.current_iteration += 1
        self.phase_iterations[self.current_phase] += 1

    def set_phase(self, phase: Literal["planning", "execution", "refinement"]) -> None:
        """
        Set the current phase of the agent.

        Args:
            phase: The phase to set
        """
        # Only reset the phase iteration counter if we're changing phases
        if self.current_phase != phase:
            # Reset the phase iteration counter for the new phase
            self.phase_iterations[phase] = 0

        self.current_phase = phase

    def count_tokens(self, tokens: int, phase: Optional[str] = None) -> None:
        """
        Count tokens used in a specific phase.

        Args:
            tokens: Number of tokens to count
            phase: Phase to count tokens for (defaults to current phase)
        """
        if phase is None:
            phase = self.current_phase

        if phase in self.tokens_used:
            self.tokens_used[phase] += tokens

    def is_budget_exhausted(self, phase: Optional[str] = None) -> bool:
        """
        Check if the token budget for a phase is exhausted.

        Args:
            phase: Phase to check (defaults to current phase)

        Returns:
            True if the budget is exhausted, False otherwise
        """
        if phase is None:
            phase = self.current_phase

        # If no budget is set, use iteration-based control
        if not self.token_budget or phase not in self.token_budget:
            return False

        return self.tokens_used[phase] >= self.token_budget[phase]

    def clear(self) -> None:
        """Clear the agent's state."""
        self.memory = []
        self.context = {}
        self.current_iteration = 0
        self.start_time = time.time()
        self.current_phase = "planning"
        self.phase_iterations = {"planning": 0, "execution": 0, "refinement": 0}
        self.tokens_used = {"planning": 0, "execution": 0, "refinement": 0}
