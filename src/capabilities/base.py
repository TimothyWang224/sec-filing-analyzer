from abc import ABC, abstractmethod
from typing import Any, Dict


class Capability(ABC):
    """Base class for all agent capabilities."""

    def __init__(self, name: str, description: str):
        """
        Initialize a capability.

        Args:
            name: Name of the capability
            description: Description of what the capability does
        """
        self.name = name
        self.description = description

    @abstractmethod
    async def init(self, agent: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize the capability with the agent and context."""
        pass

    async def start_agent_loop(self, agent: Any, context: Dict[str, Any]) -> bool:
        """Called at the start of each agent loop iteration."""
        return True

    async def process_prompt(self, agent: Any, context: Dict[str, Any], prompt: str) -> str:
        """Process the prompt before it's sent to the LLM."""
        return prompt

    async def process_response(self, agent: Any, context: Dict[str, Any], response: str) -> str:
        """Process the response from the LLM."""
        return response

    async def process_action(self, agent: Any, context: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Process an action before it's executed."""
        return action

    async def process_result(
        self,
        agent: Any,
        context: Dict[str, Any],
        response: str,
        action: Dict[str, Any],
        result: Any,
    ) -> Any:
        """Process the result of an action."""
        return result

    async def process_new_memories(
        self,
        agent: Any,
        context: Dict[str, Any],
        response: str,
        result: Any,
        memories: list,
    ) -> list:
        """Process new memories before they're added."""
        return memories

    async def end_agent_loop(self, agent: Any, context: Dict[str, Any]):
        """Called at the end of each agent loop iteration."""
        pass

    async def should_terminate(self, agent: Any, context: Dict[str, Any], response: str) -> bool:
        """Determine if the agent should terminate."""
        return False

    async def terminate(self, agent: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up when the agent is terminating."""
        return {}
