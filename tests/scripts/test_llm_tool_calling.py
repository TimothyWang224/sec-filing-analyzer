#!/usr/bin/env python
"""
Test script for LLM-driven tool calling.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from sec_filing_analyzer.llm import get_agent_config
from src.agents.qa_specialist import QASpecialistAgent
from src.capabilities.logging import LoggingCapability
from src.capabilities.time_awareness import TimeAwarenessCapability
from src.environments.financial import FinancialEnvironment
from src.sec_filing_analyzer.utils.logging_utils import get_standard_log_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def process_question(
    question: str,
    log_level: str = "INFO",
    include_prompts: bool = False,
    use_llm_tool_calling: bool = True,
    max_iterations: int = 3,
) -> Dict[str, Any]:
    """
    Process a question using the QA Specialist Agent with LLM-driven tool calling.

    Args:
        question: The question to process
        log_level: Logging level
        include_prompts: Whether to include prompts and responses in logs
        use_llm_tool_calling: Whether to use LLM-driven tool calling
        max_iterations: Maximum number of iterations

    Returns:
        Dictionary containing the agent's response
    """
    try:
        # Set up logging level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logging.getLogger().setLevel(numeric_level)

        logger.info(f"Processing question with LLM-driven tool calling: {question}")

        # Initialize environment
        environment = FinancialEnvironment()

        # Get QA Specialist configuration from the master config
        llm_config = get_agent_config("qa_specialist")

        # Override with specific settings if needed
        llm_config.update(
            {
                "temperature": 0.7,  # Slightly higher for more creative responses
                "max_tokens": 1000,  # Limit response length
            }
        )

        # Initialize logging capability
        log_dir = str(get_standard_log_dir("tests"))
        os.makedirs(log_dir, exist_ok=True)

        logging_capability = LoggingCapability(
            log_dir=log_dir,
            log_level=log_level,
            log_to_console=True,
            log_to_file=True,
            include_memory=True,
            include_context=True,
            include_actions=True,
            include_results=True,
            include_prompts=include_prompts,
            include_responses=include_prompts,
            max_content_length=1000,
        )

        # Initialize time awareness capability
        time_awareness = TimeAwarenessCapability()

        # Initialize QA specialist agent with LLM-driven tool calling
        agent = QASpecialistAgent(
            capabilities=[logging_capability, time_awareness],
            environment=environment,
            llm_config=llm_config,
            max_iterations=max_iterations,
            use_llm_tool_calling=use_llm_tool_calling,
        )

        # Run the agent
        result = await agent.run(question)

        # Add iteration information
        result["iterations_completed"] = agent.current_iteration

        return result

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return {
            "error": str(e),
            "answer": "An error occurred while processing your question.",
        }


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test LLM-driven tool calling")
    parser.add_argument(
        "--question", type=str, required=True, help="Question to ask the agent"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--include_prompts",
        action="store_true",
        help="Include prompts and responses in logs",
    )
    parser.add_argument(
        "--disable_llm_tool_calling",
        action="store_true",
        help="Disable LLM-driven tool calling",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=3, help="Maximum number of iterations"
    )

    args = parser.parse_args()

    # Run the async function
    result = asyncio.run(
        process_question(
            args.question,
            args.log_level,
            args.include_prompts,
            not args.disable_llm_tool_calling,
            args.max_iterations,
        )
    )

    # Print the result
    print("\n=== Agent Results ===")
    print(f"Question: {args.question}\n")
    print(f"Answer: {result.get('answer', 'No answer generated')}")

    # Print iteration count
    print(f"\nCompleted in {result.get('iterations_completed', 'unknown')} iterations")


if __name__ == "__main__":
    main()
