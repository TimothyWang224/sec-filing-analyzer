"""
Test script for the parameter adapter.

This script tests the parameter adapter with different input formats.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.parameter_adapter import adapt_parameters
from src.tools.sec_data import SECDataTool
from src.tools.sec_financial_data import SECFinancialDataTool

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_sec_data_tool():
    """Test the SECDataTool with adapted parameters."""
    logger.info("\n\n=== Testing SECDataTool ===")

    # Create the tool
    tool = SECDataTool()

    # Original parameters (agent format)
    original_args = {
        "query_type": "sec_data",
        "ticker": "AAPL",
        "filing_type": "10-K",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
    }

    # Adapt parameters
    adapted_args = adapt_parameters("SECDataTool", "sec_data", original_args)

    # Test validation
    is_valid = tool.validate_args(**adapted_args)
    logger.info(f"Validation result: {is_valid}")

    if is_valid:
        # Execute the tool with adapted parameters
        result = await tool.execute(**adapted_args)
        logger.info(f"Execution result: {json.dumps(result, indent=2)}")
    else:
        logger.error("Validation failed, not executing tool")


async def test_sec_financial_data_tool():
    """Test the SECFinancialDataTool with adapted parameters."""
    logger.info("\n\n=== Testing SECFinancialDataTool ===")

    # Create the tool
    tool = SECFinancialDataTool()

    # Original parameters (agent format)
    original_args = {"query_type": "metrics", "ticker": "NVDA", "filing_type": "10-K"}

    # Adapt parameters
    adapted_args = adapt_parameters("SECFinancialDataTool", "metrics", original_args)

    # Test validation
    is_valid = tool.validate_args(**adapted_args)
    logger.info(f"Validation result: {is_valid}")

    if is_valid:
        # Execute the tool with adapted parameters
        result = await tool.execute(**adapted_args)
        logger.info(f"Execution result: {json.dumps(result, indent=2)}")
    else:
        logger.error("Validation failed, not executing tool")


async def test_already_adapted_parameters():
    """Test the adapter with parameters that are already in the correct format."""
    logger.info("\n\n=== Testing Already Adapted Parameters ===")

    # Parameters already in the correct format
    already_adapted_args = {"query_type": "metrics", "parameters": {"ticker": "NVDA", "filing_type": "10-K"}}

    # Adapt parameters (should return the same)
    result = adapt_parameters("SECFinancialDataTool", "metrics", already_adapted_args)

    # Check if the result is the same as the input
    is_same = result == already_adapted_args
    logger.info(f"Result is same as input: {is_same}")
    logger.info(f"Result: {json.dumps(result, indent=2)}")


async def main():
    """Run all tests."""
    await test_sec_data_tool()
    await test_sec_financial_data_tool()
    await test_already_adapted_parameters()


if __name__ == "__main__":
    asyncio.run(main())
