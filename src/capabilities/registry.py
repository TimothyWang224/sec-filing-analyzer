from typing import Dict, List, Optional, Type

from .base import Capability


class CapabilityRegistry:
    """Registry for managing agent capabilities."""

    _capabilities: Dict[str, Type[Capability]] = {}

    @classmethod
    def register(cls, name: str, capability_class: Type[Capability]) -> None:
        """
        Register a new capability.

        Args:
            name: Unique identifier for the capability
            capability_class: The capability class to register
        """
        cls._capabilities[name] = capability_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[Capability]]:
        """
        Get a capability class by name.

        Args:
            name: Name of the capability to retrieve

        Returns:
            The capability class if found, None otherwise
        """
        return cls._capabilities.get(name)

    @classmethod
    def list_capabilities(cls) -> Dict[str, Type[Capability]]:
        """
        Get all registered capabilities.

        Returns:
            Dictionary mapping capability names to their classes
        """
        return cls._capabilities.copy()

    @classmethod
    def create_capability(cls, name: str, **kwargs) -> Optional[Capability]:
        """
        Create an instance of a registered capability.

        Args:
            name: Name of the capability to create
            **kwargs: Arguments to pass to the capability constructor

        Returns:
            An instance of the capability if found, None otherwise
        """
        capability_class = cls.get(name)
        if capability_class:
            return capability_class(**kwargs)
        return None

    @classmethod
    def create_capabilities(cls, names: List[str], **kwargs) -> List[Capability]:
        """
        Create multiple capabilities by name.

        Args:
            names: List of capability names to create
            **kwargs: Arguments to pass to each capability constructor

        Returns:
            List of created capability instances
        """
        return [
            capability
            for capability in (cls.create_capability(name, **kwargs) for name in names)
            if capability is not None
        ]
