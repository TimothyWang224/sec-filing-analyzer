"""
LLM-driven tool calling implementation.
"""

import json
import logging
from typing import Any, Dict, List

from sec_filing_analyzer.llm import BaseLLM

from ...environments.base import Environment
from ...tools.registry import ToolRegistry
from ...utils.json_utils import repair_json, safe_parse_json

logger = logging.getLogger(__name__)


class LLMToolCaller:
    """
    LLM-driven tool calling implementation.

    This class is responsible for selecting and executing tools using an LLM.
    """

    def __init__(self, llm: BaseLLM, environment: Environment):
        """
        Initialize the LLM tool caller.

        Args:
            llm: LLM instance to use for tool selection
            environment: Environment to use for tool execution
        """
        self.llm = llm
        self.environment = environment

    async def select_tools(self, input_text: str) -> List[Dict[str, Any]]:
        """
        Select tools to use based on input text.

        Args:
            input_text: Input text to select tools for

        Returns:
            List of tool calls
        """
        # Get available tools
        available_tools = ToolRegistry.get_available_tools()

        # Create tool selection prompt
        prompt = f"""
Based on the following input, select the appropriate tool(s) to use:

Input: {input_text}

Available tools:
{ToolRegistry.get_tool_documentation(format="text")}

Return a JSON array of tool calls, where each tool call is a JSON object with the following structure:
{{
    "tool": "tool_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2",
        ...
    }}
}}

Only include parameters that are relevant to the question. If a parameter is not mentioned in the question and doesn't have a default value, you can omit it.
If a company name is mentioned (e.g., "Apple"), include it in the appropriate parameter.
"""

        system_prompt = """You are an expert at selecting the right tools to answer questions.
Your task is to analyze the input and select the appropriate tool(s) to use.
Return a JSON array of tool calls, where each tool call includes the tool name and parameters.
"""

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,  # Low temperature for more deterministic tool selection
            json_mode=True,  # Force the model to return valid JSON
        )

        # Parse tool calls from response
        tool_calls = await self._parse_tool_calls(response)

        return tool_calls

    async def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a list of tool calls.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool results
        """
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("parameters", {})

            # Execute the tool
            try:
                result = await self.environment.execute_tool(tool_name, tool_args)
                results.append(
                    {
                        "tool": tool_name,
                        "parameters": tool_args,
                        "result": result,
                        "status": "success",
                    }
                )
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error executing tool {tool_name}: {error_message}")

                # Try to fix the tool call
                try:
                    fixed_args = await self._fix_tool_call(tool_name, tool_args, error_message)

                    # Execute the tool with fixed arguments
                    try:
                        result = await self.environment.execute_tool(tool_name, fixed_args)
                        results.append(
                            {
                                "tool": tool_name,
                                "parameters": fixed_args,
                                "result": result,
                                "status": "success",
                                "fixed": True,
                            }
                        )
                    except Exception as e2:
                        logger.error(f"Error executing tool {tool_name} with fixed arguments: {str(e2)}")
                        results.append(
                            {
                                "tool": tool_name,
                                "parameters": fixed_args,
                                "error": str(e2),
                                "status": "error",
                                "fixed": True,
                            }
                        )
                except Exception as e3:
                    logger.error(f"Error fixing tool call {tool_name}: {str(e3)}")
                    results.append(
                        {
                            "tool": tool_name,
                            "parameters": tool_args,
                            "error": error_message,
                            "status": "error",
                            "fixed": False,
                        }
                    )

        return results

    async def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response."""
        # First try to parse using the safe_parse_json utility
        try:
            # Use safe_parse_json with expected type "array"
            tool_calls = safe_parse_json(response, default_value=[], expected_type="array")

            # If parsing failed and we have an LLM instance, try to repair
            if not tool_calls and hasattr(self, "llm"):
                # Create a repair function that uses the LLM
                logger.info("Attempting to repair JSON tool calls")
                tool_calls = await repair_json(response, self.llm, default_value=[], expected_type="array")

            # Validate tool calls
            validated_calls = []
            for call in tool_calls:
                if isinstance(call, dict) and "tool" in call:
                    validated_calls.append(call)

            return validated_calls

        except Exception as e:
            logger.error(f"Error parsing tool calls: {str(e)}")
            logger.error(f"Response: {response}")
            return []

    async def _fix_tool_call(self, tool_name: str, tool_args: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """
        Use the LLM to fix a failed tool call.

        Args:
            tool_name: Name of the tool
            tool_args: Original tool arguments
            error_message: Error message from the failed call

        Returns:
            Fixed tool arguments
        """
        # Get tool documentation
        tool_doc = ToolRegistry.get_tool_documentation(tool_name, format="text")

        # Create prompt for fixing the tool call
        prompt = f"""
The following tool call failed:

Tool: {tool_name}
Arguments: {json.dumps(tool_args, indent=2)}
Error: {error_message}

Tool Documentation:
{tool_doc}

Please fix the tool arguments to make the call succeed.
Return only the fixed arguments as a JSON object.
"""

        system_prompt = """You are an expert at fixing failed tool calls.
Your task is to analyze the error message and tool documentation, and fix the tool arguments.
Return only the fixed arguments as a JSON object.
"""

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            json_mode=True,  # Force the model to return valid JSON
        )

        # Parse fixed arguments from response
        try:
            # Parse the JSON using our safe_parse_json utility
            fixed_args = safe_parse_json(response, default_value={}, expected_type="object")

            # If parsing failed, try to repair
            if not fixed_args:
                logger.info("Attempting to repair JSON fixed arguments")
                fixed_args = await repair_json(response, self.llm, default_value={}, expected_type="object")

            return fixed_args
        except Exception as e:
            logger.error(f"Error parsing fixed arguments: {str(e)}")
            logger.error(f"Response: {response}")
            return tool_args  # Return original arguments if parsing fails
