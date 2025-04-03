import os
from typing import Any, AsyncGenerator, Optional

import openai
from dotenv import load_dotenv

from .base import BaseLLM

# Load environment variables
load_dotenv()

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation using the OpenAI API."""
    
    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs: Any
    ):
        """Initialize the OpenAI LLM.
        
        Args:
            model: The OpenAI model to use
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var
            organization: OpenAI organization ID. If not provided, will look for OPENAI_ORG_ID env var
            **kwargs: Additional configuration parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
        # Configure OpenAI client
        openai.api_key = self.api_key
        if self.organization:
            openai.organization = self.organization
            
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """Generate a response using OpenAI's API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
        
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using OpenAI's API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        stream = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content 