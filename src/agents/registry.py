from typing import Dict, Optional, Type

from .base import Agent


class AgentRegistry:
    """Registry for managing agent types and their configurations."""

    _agents: Dict[str, Type[Agent]] = {}

    @classmethod
    def register(cls, name: str, agent_class: Type[Agent]) -> None:
        """
        Register a new agent type.

        Args:
            name: Unique identifier for the agent type
            agent_class: The agent class to register
        """
        cls._agents[name] = agent_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[Agent]]:
        """
        Get an agent class by name.

        Args:
            name: Name of the agent to retrieve

        Returns:
            The agent class if found, None otherwise
        """
        return cls._agents.get(name)

    @classmethod
    def list_agents(cls) -> Dict[str, Type[Agent]]:
        """
        Get all registered agents.

        Returns:
            Dictionary mapping agent names to their classes
        """
        return cls._agents.copy()

    @classmethod
    def create_agent(cls, name: str, **kwargs) -> Optional[Agent]:
        """
        Create an instance of a registered agent.

        Args:
            name: Name of the agent to create
            **kwargs: Arguments to pass to the agent constructor

        Returns:
            An instance of the agent if found, None otherwise
        """
        agent_class = cls.get(name)
        if agent_class:
            return agent_class(**kwargs)
        return None
