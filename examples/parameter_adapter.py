"""
Parameter adapter for SEC Filing Analyzer tools.

This module provides functions to adapt parameters from the format used by agents
to the format expected by tools.

The SEC tools expect parameters in the format:
{
    "query_type": "metrics",
    "parameters": {
        "ticker": "NVDA",
        "filing_type": "10-K"
    }
}

But the agent might pass them as:
{
    "query_type": "metrics",
    "ticker": "NVDA",
    "filing_type": "10-K"
}

This adapter transforms the second format into the first.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def adapt_parameters(tool_name: str, query_type: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt parameters from the agent format to the tool format.

    Args:
        tool_name: Name of the tool
        query_type: Type of query to execute
        args: Arguments in the agent format

    Returns:
        Arguments in the tool format
    """
    logger.info(f"Adapting parameters for tool: {tool_name}, query_type: {query_type}")
    logger.info(f"Original args: {json.dumps(args, indent=2)}")

    # Check if parameters are already in the correct format
    if "parameters" in args and isinstance(args["parameters"], dict):
        logger.info("Parameters already in correct format, no adaptation needed")
        return args

    # Create a copy of the arguments to avoid modifying the original
    adapted_args = {"query_type": query_type}

    # Extract parameters from the arguments
    parameters = {}
    for key, value in args.items():
        if key != "query_type":
            parameters[key] = value

    # Add parameters to the adapted arguments
    adapted_args["parameters"] = parameters

    logger.info(f"Adapted args: {json.dumps(adapted_args, indent=2)}")
    return adapted_args
