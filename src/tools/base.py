from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
from functools import wraps

def register_tool(tags: Optional[list] = None):
    """
    Decorator to register a tool with optional tags.
    
    Args:
        tags: List of tags to categorize the tool
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
            
        # Add metadata to the function
        wrapper.tags = tags or []
        wrapper.is_tool = True
        return wrapper
    return decorator

class Tool(ABC):
    """Base class for all tools available to agents."""
    
    def __init__(self, name: str, description: str, tags: Optional[list] = None):
        """
        Initialize a tool.
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            tags: List of tags to categorize the tool
        """
        self.name = name
        self.description = description
        self.tags = tags or []
        
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool with the given arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
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
    
    def validate_args(self, *args, **kwargs) -> bool:
        """
        Validate the arguments before execution.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            True if arguments are valid, False otherwise
        """
        return True 