"""
Test script for the "one tool call per iteration" approach.

This script demonstrates how the agent executes a single tool call per iteration
and uses memory to track previous tool results.
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


async def run_single_tool_call_test(question: str, max_iterations: int = 5):
    """Test the single tool call per iteration approach."""
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
        print("\n=== Single Tool Call Test Results ===")
        print(f"Question: {question}")
        print(f"\nAnswer: {result['answer']}")

        # Print memory to show tool results
        print("\n=== Agent Memory ===")
        memory = result.get("memory", [])
        tool_results = [
            item for item in memory if item.get("type") in ["tool_result", "tool_error"]
        ]

        for i, result in enumerate(tool_results):
            print(f"\n--- Tool Call {i + 1} ---")
            print(f"Tool: {result.get('tool', 'unknown')}")
            print(f"Args: {json.dumps(result.get('args', {}), indent=2)}")

            if result.get("type") == "tool_result":
                print(f"Result: {str(result.get('result', ''))[:200]}...")
            else:  # tool_error
                print(f"Error: {result.get('error', 'Unknown error')}")

        return result
    except Exception as e:
        logger.error(f"Error running single tool call test: {str(e)}")
        raise


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(
        description="Test the single tool call per iteration approach"
    )
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
        run_single_tool_call_test(
            question=args.question, max_iterations=args.max_iterations
        )
    )


if __name__ == "__main__":
    main()
