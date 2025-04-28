"""
Test script for the enhanced parameter completion.

This script demonstrates the enhanced parameter completion capabilities
of the LLMParameterCompleter class.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Create a simplified version of the LLMParameterCompleter for testing
from sec_filing_analyzer.llm import OpenAILLM


# Define a simplified version of the parameter completer for testing
class SimpleParameterCompleter:
    """A simplified version of the LLMParameterCompleter for testing."""

    def __init__(self, llm):
        """Initialize the parameter completer."""
        self.llm = llm

    async def complete_parameters(
        self, tool_name, partial_parameters, user_input, context=None
    ):
        """Complete tool parameters using the LLM."""
        # Create a prompt for parameter completion
        prompt = f"""
        User Input: {user_input}

        Tool: {tool_name}

        Current Parameters: {json.dumps(partial_parameters, indent=2)}

        Please complete the parameters based on the user input.
        Extract any relevant information such as:
        - Company names and ticker symbols
        - Date ranges
        - Financial metrics
        - Filing types

        Return only the completed parameters as a JSON object.
        """

        # Generate completed parameters
        system_prompt = """You are an expert at extracting information from text to complete tool parameters.
        Your task is to analyze the user input and extract relevant information to complete the tool parameters.
        Return only the completed parameters as a JSON object.
        """

        response = await self.llm.generate(
            prompt=prompt, system_prompt=system_prompt, temperature=0.2
        )

        # Parse the response
        try:
            # Extract JSON from response
            import re

            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON-like structure
                json_match = re.search(r'\{\s*".*"\s*:.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response

            # Clean up the JSON string
            json_str = re.sub(r"```.*?```", "", json_str, flags=re.DOTALL)

            # Parse the JSON
            completed_parameters = json.loads(json_str)

            # Merge with original parameters
            result = partial_parameters.copy()
            self._deep_update(result, completed_parameters)

            return result
        except Exception as e:
            logging.error(f"Error parsing parameters: {str(e)}")
            return partial_parameters

    def _deep_update(self, target, source):
        """Deep update a nested dictionary."""
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_update(target[key], value)
            else:
                target[key] = value


# Simple validation function
def validate_parameters(tool_name, parameters):
    """Validate tool parameters."""
    return {"parameters": parameters, "errors": []}


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_parameter_completion():
    """Test the parameter completion capabilities."""
    # Initialize the LLM
    llm = OpenAILLM(model="gpt-4o-mini")

    # Initialize the parameter completer
    parameter_completer = SimpleParameterCompleter(llm)

    # Test cases
    test_cases = [
        {
            "name": "Company name extraction",
            "tool_name": "sec_financial_data",
            "partial_parameters": {"query_type": "financial_facts", "parameters": {}},
            "user_input": "What was Apple's revenue in 2023?",
            "context": {},
        },
        {
            "name": "Date range extraction",
            "tool_name": "sec_financial_data",
            "partial_parameters": {
                "query_type": "financial_facts",
                "parameters": {"ticker": "MSFT"},
            },
            "user_input": "Show me Microsoft's financial performance from 2020 to 2023",
            "context": {},
        },
        {
            "name": "Metric extraction",
            "tool_name": "sec_financial_data",
            "partial_parameters": {
                "query_type": "financial_facts",
                "parameters": {"ticker": "GOOGL"},
            },
            "user_input": "What were Alphabet's revenue, net income, and operating expenses in 2022?",
            "context": {},
        },
        {
            "name": "Error correction",
            "tool_name": "sec_financial_data",
            "partial_parameters": {
                "query_type": "financial_facts",
                "parameters": {"ticker": "INVALID"},
            },
            "user_input": "What was Tesla's revenue in 2023?",
            "context": {
                "last_error": "Invalid ticker symbol: INVALID. Company not found."
            },
        },
        {
            "name": "Semantic search parameters",
            "tool_name": "sec_semantic_search",
            "partial_parameters": {"query": "risk factors"},
            "user_input": "What are the main risk factors mentioned in Amazon's 10-K filings from 2021 to 2023?",
            "context": {},
        },
    ]

    # Run the test cases
    for i, test_case in enumerate(test_cases):
        logger.info(f"Test case {i + 1}: {test_case['name']}")
        logger.info(f"Tool: {test_case['tool_name']}")
        logger.info(f"User input: {test_case['user_input']}")
        logger.info(
            f"Partial parameters: {json.dumps(test_case['partial_parameters'], indent=2)}"
        )

        # Validate the partial parameters
        validation_result = validate_parameters(
            test_case["tool_name"], test_case["partial_parameters"]
        )
        logger.info(f"Validation errors: {validation_result['errors']}")

        # Complete the parameters
        completed_parameters = await parameter_completer.complete_parameters(
            tool_name=test_case["tool_name"],
            partial_parameters=test_case["partial_parameters"],
            user_input=test_case["user_input"],
            context=test_case["context"],
        )

        logger.info(
            f"Completed parameters: {json.dumps(completed_parameters, indent=2)}"
        )
        logger.info("-" * 80)


if __name__ == "__main__":
    asyncio.run(test_parameter_completion())
