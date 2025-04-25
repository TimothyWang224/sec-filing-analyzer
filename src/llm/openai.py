"""
OpenAI LLM implementation for SEC Filing Analyzer.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union

import openai
from openai import OpenAI

from .base import LLM


class OpenAILLM(LLM):
    """OpenAI LLM implementation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the OpenAI LLM.

        Args:
            model: OpenAI model to use
            temperature: Temperature parameter (0-1)
            max_tokens: Maximum number of tokens to generate
            api_key: Optional API key (defaults to OPENAI_API_KEY environment variable)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Use provided API key or get from environment
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate text from the OpenAI LLM.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt to set context
            temperature: Optional temperature parameter (0-1)
            max_tokens: Optional maximum number of tokens to generate
            stop: Optional stop sequences
            json_mode: If True, forces the model to return valid JSON
            **kwargs: Additional model-specific parameters

        Returns:
            Generated text
        """
        # Use provided parameters or fall back to instance defaults
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Prepare API call parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
            **kwargs,
        }

        # Add response_format for JSON mode if requested
        if json_mode:
            api_params["response_format"] = {"type": "json_object"}

        # Call the OpenAI API
        response = self.client.chat.completions.create(**api_params)

        # Extract and return the generated text
        return response.choices[0].message.content

    async def generate_with_json_output(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from the OpenAI LLM.

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
        # Add JSON instructions to the prompt
        json_prompt = (
            f"{prompt}\n\nRespond with a JSON object that matches this schema:\n{json.dumps(output_schema, indent=2)}"
        )

        # Add JSON instructions to the system prompt
        if system_prompt:
            system_prompt = (
                f"{system_prompt}\nYou must respond with a valid JSON object that matches the specified schema."
            )
        else:
            system_prompt = "You must respond with a valid JSON object that matches the specified schema."

        # Generate text with JSON mode enabled
        response_text = await self.generate(
            prompt=json_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,  # Force the model to return valid JSON
            **kwargs,
        )

        # Extract JSON from the response
        try:
            # Try to parse the entire response as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON using regex
            json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Try another common pattern
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass

            # If all extraction attempts fail, raise an error
            raise ValueError(f"Failed to extract valid JSON from LLM response: {response_text}")
