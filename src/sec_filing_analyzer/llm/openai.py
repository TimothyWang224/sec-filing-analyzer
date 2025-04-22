import os
import time
import logging
from typing import Any, AsyncGenerator, Optional

import openai
from dotenv import load_dotenv

from .base import BaseLLM
from ..utils.timing import timed_function, TimingContext

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation using the OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize the OpenAI LLM.

        Args:
            model: The OpenAI model to use
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var
            **kwargs: Additional configuration parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        # Create OpenAI client instance
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        )

    @timed_function(category="llm")
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs: Any
    ) -> str:
        """Generate a response using OpenAI's API.

        Args:
            prompt: The user prompt to generate a response for
            system_prompt: Optional system prompt to set context
            temperature: Controls randomness in the output (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            json_mode: If True, forces the model to return valid JSON
            **kwargs: Additional provider-specific parameters

        Returns:
            The generated response as a string
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Log the request details
        token_estimate = len(prompt.split()) // 4  # Rough estimate
        logger.debug(f"LLM Request: model={self.model}, tokens~{token_estimate}, temp={temperature}, json_mode={json_mode}")

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        # Add response_format for JSON mode if requested
        if json_mode:
            api_params["response_format"] = {"type": "json_object"}
            logger.debug("Using JSON response format")

        # Time the API call specifically
        with TimingContext("openai_api_call", category="api", logger=logger):
            response = self.client.chat.completions.create(**api_params)

        # Log completion info
        completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else "unknown"
        prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else "unknown"
        total_tokens = response.usage.total_tokens if hasattr(response, 'usage') else "unknown"

        logger.info(f"LLM Response: tokens={total_tokens} (prompt={prompt_tokens}, completion={completion_tokens})")

        return response.choices[0].message.content

    @timed_function(category="llm_stream")
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using OpenAI's API.

        Args:
            prompt: The user prompt to generate a response for
            system_prompt: Optional system prompt to set context
            temperature: Controls randomness in the output (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            json_mode: If True, forces the model to return valid JSON
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of the generated response as strings
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Log the request details
        token_estimate = len(prompt.split()) // 4  # Rough estimate
        logger.debug(f"LLM Stream Request: model={self.model}, tokens~{token_estimate}, temp={temperature}, json_mode={json_mode}")

        start_time = time.time()
        chunk_count = 0
        total_chars = 0

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs
        }

        # Add response_format for JSON mode if requested
        if json_mode:
            api_params["response_format"] = {"type": "json_object"}
            logger.debug("Using JSON response format for streaming")

        # Start the stream
        stream = self.client.chat.completions.create(**api_params)

        # Process the stream
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                chunk_count += 1
                total_chars += len(content)
                yield content

        # Log completion info
        duration = time.time() - start_time
        logger.info(f"LLM Stream completed: chunks={chunk_count}, chars={total_chars}, time={duration:.2f}s, rate={total_chars/duration:.1f} chars/sec")