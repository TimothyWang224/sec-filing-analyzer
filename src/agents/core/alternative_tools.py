"""
Alternative tool selection for error recovery.

This module provides utilities for finding and using alternative tools
when a primary tool fails.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from sec_filing_analyzer.llm import BaseLLM

from ...tools.registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlternativeToolSelector:
    """
    Selects alternative tools when a primary tool fails.

    This class uses an LLM to suggest alternative tools that can achieve
    the same purpose as a failed tool.
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize the alternative tool selector.

        Args:
            llm: LLM instance to use for tool selection
        """
        self.llm = llm

    async def find_alternative_tool(self, failed_tool: str, original_purpose: str) -> Optional[str]:
        """
        Find an alternative tool that can achieve the same purpose.

        Args:
            failed_tool: Name of the failed tool
            original_purpose: Original purpose of the tool call

        Returns:
            Name of the alternative tool, or None if no suitable alternative
        """
        system_prompt = """You are an expert at selecting tools to accomplish tasks.
A tool has failed, and you need to suggest an alternative tool that can achieve the same purpose.
Return only the name of the alternative tool, or "none" if there is no suitable alternative."""

        # Get all available tools
        available_tools = self._get_available_tools()

        prompt = f"""
The following tool has failed: {failed_tool}

The original purpose was: {original_purpose}

Available tools:
{available_tools}

Please suggest an alternative tool that can achieve the same purpose.
If there is no suitable alternative, return "none".
"""

        response = await self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.2)

        # Extract the tool name from the response
        alternative = self._extract_tool_name(response)

        if alternative.lower() == "none":
            return None

        # Validate that the alternative tool exists
        if alternative in ToolRegistry.get_all_tool_names():
            return alternative

        return None

    async def map_parameters(
        self, source_tool: str, target_tool: str, original_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Map parameters from one tool to another.

        Args:
            source_tool: Name of the source tool
            target_tool: Name of the target tool
            original_params: Original parameters

        Returns:
            Mapped parameters for the target tool
        """
        # Get tool documentation
        source_doc = ToolRegistry.get_tool_documentation(source_tool, format="text")
        target_doc = ToolRegistry.get_tool_documentation(target_tool, format="text")

        system_prompt = """You are an expert at mapping parameters between different tools.
Your task is to map parameters from a source tool to a target tool.
Return only the mapped parameters as a JSON object."""

        prompt = f"""
Source Tool: {source_tool}
Source Tool Documentation:
{source_doc}

Target Tool: {target_tool}
Target Tool Documentation:
{target_doc}

Original Parameters:
{original_params}

Please map the parameters from the source tool to the target tool.
Return only the mapped parameters as a JSON object.
"""

        response = await self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.2)

        # Extract JSON from response
        mapped_params = self._extract_json(response)

        if not mapped_params:
            logger.warning(f"Failed to map parameters from {source_tool} to {target_tool}")
            return {}

        return mapped_params

    def _get_available_tools(self) -> str:
        """
        Get a formatted string of all available tools.

        Returns:
            Formatted string with tool names and descriptions
        """
        tools = ToolRegistry.list_tools()

        formatted_tools = []
        for name, info in tools.items():
            description = info.get("description", "No description available")
            formatted_tools.append(f"- {name}: {description}")

        return "\n".join(formatted_tools)

    def _extract_tool_name(self, response: str) -> str:
        """
        Extract a tool name from an LLM response.

        Args:
            response: LLM response

        Returns:
            Extracted tool name
        """
        # Clean up the response
        response = response.strip()

        # Check for common patterns
        patterns = [
            r"^([a-z_]+)$",  # Just the tool name
            r"^Tool: ([a-z_]+)$",  # Tool: name
            r"^The alternative tool is: ([a-z_]+)$",  # The alternative tool is: name
            r"^I suggest using ([a-z_]+)$",  # I suggest using name
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                return match.group(1)

        # If no pattern matches, return the first line
        return response.split("\n")[0].strip()

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """
        Extract a JSON object from an LLM response.

        Args:
            response: LLM response

        Returns:
            Extracted JSON object
        """
        import json

        # Try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find any JSON-like structure
            json_match = re.search(r'\{\s*".*"\s*:.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Use the whole response
                json_str = response

        try:
            # Clean up the JSON string
            json_str = re.sub(r"```.*?```", "", json_str, flags=re.DOTALL)

            # Parse the JSON
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            return {}
