from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator

class BaseLLM(ABC):
    """Base class for LLM implementations."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The user prompt to generate a response for
            system_prompt: Optional system prompt to set context
            temperature: Controls randomness in the output (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            The generated response as a string
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM.
        
        Args:
            prompt: The user prompt to generate a response for
            system_prompt: Optional system prompt to set context
            temperature: Controls randomness in the output (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Chunks of the generated response as strings
        """
        pass 