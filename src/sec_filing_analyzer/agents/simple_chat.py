"""A bare-bones, synchronous chat agent for demo purposes.

This avoids the hierarchical planner: it just
 1. Receives user input,
 2. Lets the LLM decide whether to invoke a tool,
 3. Returns the final answer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agents.base import Agent, Goal

logger = logging.getLogger(__name__)


class SimpleChatAgent(Agent):
    """Single-shot agent. Assumes all tools expose a `.run(query)` method."""

    def __init__(
        self,
        tools: List[Any],
        llm: Any,
        max_iterations: int = 5,
    ):
        """
        Initialize the SimpleChatAgent.

        Args:
            tools: List of tools to use
            llm: LLM instance to use
            max_iterations: Maximum iterations for the agent
        """
        super().__init__(goal=Goal("Answer user finance questions"))
        self.tools = tools
        self.llm = llm
        self.max_iterations = max_iterations
        self.tool_map = {tool.__class__.__name__: tool for tool in tools}

        # Set up the system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt based on available tools."""
        tool_descriptions = []

        for tool in self.tools:
            name = tool.__class__.__name__
            desc = getattr(tool, "__doc__", "No description available").strip()
            tool_descriptions.append(f"- {name}: {desc}")

        return f"""
        You are an AI assistant specialized in analyzing SEC filings and financial data.
        You have access to the following tools:

        {chr(10).join(tool_descriptions)}

        When answering questions:
        1. Determine which tool(s) would be most appropriate for the question
        2. Call the tool(s) with the appropriate parameters
        3. Analyze the results and provide a clear, concise answer
        4. Always cite your sources (e.g., "According to the 2023 10-K filing...")
        5. If you're unsure or don't have enough information, be honest about it

        You are in DEMO MODE, which means you have access to a limited set of data and functionality.
        """

    def respond(self, user_input: str) -> str:
        """
        Generate a response to the user input.

        Args:
            user_input: The user's input

        Returns:
            The agent's response
        """
        logger.info(f"Received user input: {user_input}")

        # For now, just use the LLM directly
        # In a real implementation, we would use the tools
        try:
            # Simple prompt for demo purposes
            prompt = f"""
            User question: {user_input}

            Please respond to this question about SEC filings or financial data.
            If you need to use tools, describe which tools you would use and why.
            """

            response = self.llm.generate(prompt)
            logger.info(f"Generated response: {response[:100]}...")

            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I'm sorry, I encountered an error: {str(e)}"
