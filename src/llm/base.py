"""
Base LLM interface for SEC Filing Analyzer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class LLM(ABC):
    """Base class for large language models."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate text from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt to set context
            temperature: Optional temperature parameter (0-1)
            max_tokens: Optional maximum number of tokens to generate
            stop: Optional stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    async def generate_with_json_output(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            output_schema: JSON schema defining the expected output structure
            system_prompt: Optional system prompt to set context
            temperature: Optional temperature parameter (0-1)
            max_tokens: Optional maximum number of tokens to generate
            **kwargs: Additional model-specific parameters

        Returns:
            Generated JSON object
        """
        pass
