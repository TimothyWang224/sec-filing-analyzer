#!/usr/bin/env python
"""
Test script for the unified architecture with LLM-driven tool calling.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.agents.qa_specialist import QASpecialistAgent
from src.capabilities.logging import LoggingCapability
from src.capabilities.time_awareness import TimeAwarenessCapability
from src.environments.base import Environment
from src.sec_filing_analyzer.utils.logging_utils import get_standard_log_dir
from src.tools.registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def process_question(
    question: str,
    log_level: str = "INFO",
    include_prompts: bool = False,
    max_iterations: int = 3,
    llm_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Process a question using the QA Specialist Agent with the unified architecture.

    Args:
        question: The question to process
        log_level: Logging level
        include_prompts: Whether to include prompts and responses in logs
        max_iterations: Maximum number of iterations
        llm_model: LLM model to use

    Returns:
        Dictionary containing the agent's response
    """
    try:
        # Set up logging level
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logging.getLogger().setLevel(numeric_level)

        logger.info(f"Processing question with unified architecture: {question}")

        # Initialize environment
        environment = Environment()

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

        # Initialize QA specialist agent
        agent = QASpecialistAgent(
            capabilities=[logging_capability, time_awareness],
            environment=environment,
            max_iterations=max_iterations,
            llm_model=llm_model,
            llm_temperature=0.7,
            llm_max_tokens=4000,
            max_tool_calls=3,
        )

        # Run the agent
        result = await agent.run(question)

        # Add iteration information
        result["iterations_completed"] = agent.state.current_iteration

        return result

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return {"error": str(e), "answer": "An error occurred while processing your question."}


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test the unified architecture")
    parser.add_argument("--question", type=str, required=True, help="Question to ask the agent")
    parser.add_argument(
        "--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )
    parser.add_argument("--include_prompts", action="store_true", help="Include prompts and responses in logs")
    parser.add_argument("--max_iterations", type=int, default=3, help="Maximum number of iterations")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="LLM model to use")

    args = parser.parse_args()

    # Print available tools
    print("Available tools:")
    for name, info in ToolRegistry.list_tools().items():
        print(f"  - {name}: {info['description']}")

    # Run the async function
    result = asyncio.run(
        process_question(args.question, args.log_level, args.include_prompts, args.max_iterations, args.llm_model)
    )

    # Print the result
    print("\n=== Agent Results ===")
    print(f"Question: {args.question}\n")

    # Print the answer
    answer = result.get("answer", "No answer generated")
    explanation = result.get("explanation", "")

    # Check if answer is a dictionary (the whole result) and extract just the answer text
    if isinstance(answer, dict) and "answer" in answer:
        explanation = answer.get("explanation", "")
        answer = answer["answer"]

    print(f"Answer: {answer}")
    print(f"Explanation: {explanation}")

    # Print supporting data
    supporting_data = result.get("supporting_data", {})
    if supporting_data.get("financial_data"):
        print("\nFinancial Data:")
        for item in supporting_data["financial_data"]:
            print(f"  - {item['metric']}: {item['value']} (Period: {item['period']}, Filing: {item['filing_type']})")

    # Print iteration count
    print(f"\nCompleted in {result.get('iterations_completed', 'unknown')} iterations")


if __name__ == "__main__":
    main()
