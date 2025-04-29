"""
Test script for the FixedChatAgent with parameter adaptation.

This script tests the FixedChatAgent with different queries to verify that
parameter adaptation is working correctly.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.fixed_chat_agent import FixedChatAgent
from src.agents.base import Goal

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_fixed_chat_agent():
    """Test the FixedChatAgent with different queries."""
    logger.info("Creating FixedChatAgent")

    # Create the agent
    agent = FixedChatAgent(
        goals=[
            Goal(
                name="Answer SEC filing questions",
                description="Answer user questions about SEC filings and financial data",
            )
        ],
        llm_model="gpt-4o-mini",
        llm_temperature=0.7,
        llm_max_tokens=4000,
        max_iterations=5,
        vector_store_path="data/vector_store",  # Path to the vector store
    )

    # Test queries
    test_queries = [
        "What was NVDA's revenue in 2023?",
        "Show me the financial metrics for Apple in their latest 10-K filing",
        "Find information about Microsoft's AI strategy in their recent filings",
    ]

    for query in test_queries:
        logger.info(f"\n\n=== Testing query: {query} ===")

        # Run the agent with the query
        result = await agent.run(query, chat_mode=True)

        # Log the result
        if isinstance(result, dict):
            if "response" in result:
                logger.info(f"Response: {result['response'][:200]}...")
            if "tool_calls" in result:
                logger.info(f"Tool calls: {json.dumps(result['tool_calls'], indent=2)}")
            if "adapted_tool_calls" in result:
                logger.info(f"Adapted tool calls: {json.dumps(result['adapted_tool_calls'], indent=2)}")
        else:
            logger.info(f"Result: {result}")


async def main():
    """Run the test."""
    await test_fixed_chat_agent()


if __name__ == "__main__":
    asyncio.run(main())
