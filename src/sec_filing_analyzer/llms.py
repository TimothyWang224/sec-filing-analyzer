"""
LLM wrappers for SEC Filing Analyzer.

This module provides wrappers for language models used in the SEC Filing Analyzer.
"""

import logging
import os
from typing import Any, Dict, Optional

from src.llm.openai import OpenAILLM

logger = logging.getLogger(__name__)


class ChatOpenAI:
    """
    Wrapper for OpenAI chat models.

    This class provides a simple interface for interacting with OpenAI chat models.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the ChatOpenAI wrapper.

        Args:
            model: The model to use
            temperature: The temperature for generation
            max_tokens: The maximum number of tokens to generate
            api_key: Optional API key (defaults to OPENAI_API_KEY environment variable)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Initialize the OpenAI LLM
        self.llm = OpenAILLM(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self.api_key,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response to the prompt.

        Args:
            prompt: The prompt to generate a response for

        Returns:
            The generated response
        """
        logger.info(f"Generating response for prompt: {prompt[:100]}...")

        try:
            response = self.llm.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"
