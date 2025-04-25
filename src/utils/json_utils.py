"""
JSON utilities for parsing and repairing JSON.

This module provides utilities for parsing and repairing JSON, especially
for handling JSON from LLM responses that may be malformed or wrapped in
Markdown code blocks.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Union

from sec_filing_analyzer.llm import BaseLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def repair_json(
    json_str: str, llm: BaseLLM, default_value: Any = None, expected_type: Optional[str] = None
) -> Any:
    """
    Repair malformed JSON using an LLM.

    Args:
        json_str: The potentially malformed JSON string
        llm: LLM instance to use for repair
        default_value: Default value to return if repair fails
        expected_type: Expected type of the JSON (e.g., "object", "array")

    Returns:
        Repaired JSON object or default value if repair fails
    """
    # Create a repair prompt
    type_hint = f"The result should be a valid JSON {expected_type}." if expected_type else ""

    prompt = f"""
    The text below is almost-valid JSON but does not parse.
    Fix it so that `json.loads` works. Return ONLY the fixed JSON.
    {type_hint}
    
    --- BEGIN ---
    {json_str}
    --- END ---
    """

    system_prompt = """You are an expert at fixing malformed JSON.
    Your task is to repair the JSON so it can be parsed by json.loads().
    Return ONLY the fixed JSON, with no additional text, explanations, or code blocks."""

    try:
        # Get repair from LLM
        response = await llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,  # Low temperature for more deterministic repair
        )

        # Clean up the response
        repaired_json = response.strip()

        # Remove any markdown code block markers
        repaired_json = re.sub(r"^```.*$", "", repaired_json, flags=re.MULTILINE)
        repaired_json = re.sub(r"^```$", "", repaired_json, flags=re.MULTILINE)

        # Try to parse the repaired JSON
        result = json.loads(repaired_json)
        logger.info("Successfully repaired JSON")
        return result

    except Exception as e:
        logger.error(f"Failed to repair JSON: {str(e)}")
        return default_value


def safe_parse_json(
    text: str, default_value: Any = None, expected_type: Optional[str] = None, repair_func: Optional[Callable] = None
) -> Any:
    """
    Safely parse JSON from text, handling common issues.

    Args:
        text: Text containing JSON
        default_value: Default value to return if parsing fails
        expected_type: Expected type of the JSON (e.g., "object", "array")
        repair_func: Optional function to call for repair if parsing fails

    Returns:
        Parsed JSON or default value if parsing fails
    """
    # Clean up the text
    cleaned_text = text.strip()

    # Extract JSON from markdown code blocks if present
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned_text, re.DOTALL)
    if json_match:
        cleaned_text = json_match.group(1)

    # Try to find JSON-like structures if no code block was found
    if not json_match:
        if expected_type == "object" or not expected_type:
            # Look for object pattern
            json_match = re.search(r'\{\s*".*"\s*:.*\}', cleaned_text, re.DOTALL)
            if json_match:
                cleaned_text = json_match.group(0)

        if expected_type == "array" or not expected_type:
            # Look for array pattern
            json_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned_text, re.DOTALL)
            if json_match:
                cleaned_text = json_match.group(0)

    # Remove any non-JSON characters
    if expected_type == "object" or not expected_type:
        cleaned_text = re.sub(r'[^\{\}"\':,\.\-\w\s\[\]]', "", cleaned_text)
    elif expected_type == "array":
        cleaned_text = re.sub(r'[^\[\]\{\}"\':,\.\-\w\s]', "", cleaned_text)

    # Ensure we have valid JSON structure
    cleaned_text = cleaned_text.strip()
    if expected_type == "object" and not cleaned_text.startswith("{"):
        cleaned_text = "{" + cleaned_text
    if expected_type == "object" and not cleaned_text.endswith("}"):
        cleaned_text = cleaned_text + "}"
    if expected_type == "array" and not cleaned_text.startswith("["):
        cleaned_text = "[" + cleaned_text
    if expected_type == "array" and not cleaned_text.endswith("]"):
        cleaned_text = cleaned_text + "]"

    try:
        # Try to parse the cleaned JSON
        return json.loads(cleaned_text)
    except Exception as e:
        logger.warning(f"Failed to parse JSON: {str(e)}")

        # If repair function is provided, try to repair
        if repair_func:
            logger.info("Attempting to repair JSON")
            return repair_func(cleaned_text, expected_type)

        return default_value
