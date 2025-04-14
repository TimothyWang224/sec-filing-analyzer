from typing import Any, Dict, ClassVar
from abc import ABC, abstractmethod
from .registry import ToolRegistry

# Re-export the register_tool decorator from ToolRegistry
register_tool = ToolRegistry.register

class Tool(ABC):
    """Base class for all tools available to agents."""

    # Class variables for tool metadata
    _tool_name: ClassVar[str] = None
    _tool_tags: ClassVar[list] = None
    _compact_description: ClassVar[str] = None  # New class variable for compact description

    def __init__(self):
        """Initialize a tool."""
        # Get name from class if registered, otherwise use class name
        self.name = getattr(self.__class__, '_tool_name', self.__class__.__name__.lower().replace('tool', ''))

        # Get description from class docstring
        self.description = self.__class__.__doc__.strip().split('\n\n')[0].strip() if self.__class__.__doc__ else "No description available"

        # Get compact description or generate one from the full description
        self.compact_description = getattr(self.__class__, '_compact_description', self.description.split('.')[0])

        # Get tags from class if registered
        self.tags = getattr(self.__class__, '_tool_tags', [])

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Keyword arguments based on the tool's parameters

        Returns:
            The result of the tool's execution
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get the tool's metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags
        }

    @classmethod
    def get_documentation(cls, format: str = "text") -> str:
        """Get the tool's documentation."""
        tool_name = getattr(cls, '_tool_name', cls.__name__.lower().replace('tool', ''))
        return ToolRegistry.get_tool_documentation(tool_name, format)

    def validate_args(self, **kwargs) -> bool:
        """
        Validate the arguments before execution.

        Args:
            **kwargs: Keyword arguments to validate

        Returns:
            True if arguments are valid, False otherwise
        """
        # Get parameter info from registry
        tool_info = ToolRegistry.get(self.name)
        if not tool_info:
            return True  # Can't validate if not registered

        parameters = tool_info.get("parameters", {})

        # Check required parameters
        for param_name, param_info in parameters.items():
            if param_info.get("required", False) and param_name not in kwargs:
                raise ValueError(f"Missing required parameter: {param_name}")

        return True