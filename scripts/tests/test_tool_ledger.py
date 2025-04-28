"""
Test script for the Tool Ledger implementation.

This script demonstrates how the Tool Ledger tracks tool calls and their results,
making it easier for agents to reference previous tool calls and build on their results.
"""

import argparse
import asyncio
import json
import logging

from src.agents.qa_specialist import QASpecialistAgent
from src.capabilities.logging import LoggingCapability
from src.capabilities.planning import PlanningCapability
from src.environments.financial import FinancialEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_tool_ledger_test(question: str, max_iterations: int = 5):
    """Test the Tool Ledger implementation."""
    try:
        # Initialize environment
        environment = FinancialEnvironment()

        # Initialize capabilities
        planning = PlanningCapability()
        logging_capability = LoggingCapability(
            log_dir="data/logs/agents",
            log_level="INFO",
            log_to_console=True,
            log_to_file=True,
            include_memory=True,
            include_context=True,
            include_actions=True,
            include_results=True,
            include_prompts=True,
            include_responses=True,
        )

        # Initialize QA specialist agent with planning and logging
        agent = QASpecialistAgent(
            capabilities=[planning, logging_capability],
            environment=environment,
            max_iterations=max_iterations,  # Allow multiple iterations to see the effect
        )

        # Run the agent
        logger.info(f"Processing question: {question}")
        result = await agent.run(question)

        # Print results
        print("\n=== Tool Ledger Test Results ===")
        print(f"Question: {question}")
        print(f"\nAnswer: {result['answer']}")

        # Print tool ledger entries
        print("\n=== Tool Ledger Entries ===")
        for i, entry in enumerate(agent.tool_ledger.entries):
            print(f"\n--- Entry {i + 1} ---")
            print(f"Tool: {entry['tool']}")
            print(f"Args: {json.dumps(entry['args'], indent=2)}")
            print(f"Status: {entry['status']}")

            if entry["status"] == "success":
                print(f"Result: {str(entry['result'])[:200]}...")
            else:
                print(f"Error: {entry['error']}")

            print(f"Timestamp: {entry['timestamp']}")

        # Print formatted ledger
        print("\n=== Formatted Tool Ledger ===")
        print(agent.tool_ledger.format_for_prompt())

        return result
    except Exception as e:
        logger.error(f"Error running tool ledger test: {str(e)}")
        raise


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test the Tool Ledger implementation")
    parser.add_argument(
        "--question",
        type=str,
        default="What was Apple's revenue in 2023?",
        help="Question to process",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5, help="Maximum number of iterations"
    )

    args = parser.parse_args()

    asyncio.run(
        run_tool_ledger_test(question=args.question, max_iterations=args.max_iterations)
    )


if __name__ == "__main__":
    main()
