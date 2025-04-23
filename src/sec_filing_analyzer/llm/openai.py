import os
import time
import logging
from typing import Any, AsyncGenerator, Optional, List, Dict, Union

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
        return_usage: bool = False,
        **kwargs: Any
    ) -> Union[str, Dict[str, Any]]:
        """Generate a response using OpenAI's API.

        Args:
            prompt: The user prompt to generate a response for
            system_prompt: Optional system prompt to set context
            temperature: Controls randomness in the output (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            json_mode: If True, forces the model to return valid JSON
            return_usage: If True, returns a dict with 'content' and 'usage' keys
            **kwargs: Additional provider-specific parameters

        Returns:
            If return_usage is False (default): The generated response as a string
            If return_usage is True: Dict with 'content' and 'usage' keys
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

        # Extract usage information
        usage = {
            "total_tokens": total_tokens if isinstance(total_tokens, int) else 0,
            "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, int) else 0,
            "completion_tokens": completion_tokens if isinstance(completion_tokens, int) else 0
        }

        # Return based on return_usage parameter
        if return_usage:
            return {
                "content": response.choices[0].message.content,
                "usage": usage
            }
        else:
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

    @timed_function(category="llm_function_call")
    async def generate_with_functions(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        return_usage: bool = False,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate a response using OpenAI's function calling API.

        Args:
            prompt: The user prompt to generate a response for
            functions: List of function definitions in OpenAI format
            system_prompt: Optional system prompt to set context
            temperature: Controls randomness in the output (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            function_call: Optional control over function calling behavior
                - "auto": Let the model decide whether to call a function
                - "none": Don't call a function
                - {"name": "function_name"}: Call the specified function
            return_usage: If True, includes token usage information in the result
            **kwargs: Additional provider-specific parameters

        Returns:
            Dictionary containing the response, function call information, and
            optionally token usage information if return_usage is True
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Log the request details
        token_estimate = len(prompt.split()) // 4  # Rough estimate
        logger.debug(f"LLM Function Call Request: model={self.model}, tokens~{token_estimate}, temp={temperature}, functions={len(functions)}")

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": [{
                "type": "function",
                "function": function
            } for function in functions],
            **kwargs
        }

        # Add function_call parameter if provided
        if function_call:
            if isinstance(function_call, str):
                api_params["tool_choice"] = function_call
            elif isinstance(function_call, dict) and "name" in function_call:
                api_params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": function_call["name"]}
                }

        # Time the API call specifically
        with TimingContext("openai_api_call", category="api", logger=logger):
            response = self.client.chat.completions.create(**api_params)

        # Log completion info
        completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else "unknown"
        prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else "unknown"
        total_tokens = response.usage.total_tokens if hasattr(response, 'usage') else "unknown"

        logger.info(f"LLM Function Call Response: tokens={total_tokens} (prompt={prompt_tokens}, completion={completion_tokens})")

        # Extract usage information
        usage = {
            "total_tokens": total_tokens if isinstance(total_tokens, int) else 0,
            "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, int) else 0,
            "completion_tokens": completion_tokens if isinstance(completion_tokens, int) else 0
        }

        # Extract response content and function call information
        message = response.choices[0].message
        result = {
            "content": message.content or "",
        }

        # Extract function call information if available
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            if tool_call.type == "function":
                result["function_call"] = {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }

        # Add usage information if requested
        if return_usage:
            result["usage"] = usage

        return result