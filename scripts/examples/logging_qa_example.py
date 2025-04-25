"""
Example script demonstrating the Logging Capability with the QA Specialist Agent.

This script shows how to use the LoggingCapability to enhance
financial question answering with detailed logging.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.qa_specialist import QASpecialistAgent
from src.capabilities.logging import LoggingCapability
from src.capabilities.time_awareness import TimeAwarenessCapability
from src.environments.financial import FinancialEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def run_logging_qa(
    question: str, log_dir: str = "data/logs/agents", log_level: str = "INFO", include_prompts: bool = False
):
    """Run the QA specialist agent with logging capability."""
    try:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Initialize environment
        environment = FinancialEnvironment()

        # Initialize capabilities
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
            include_responses=include_prompts,  # Match prompts setting
        )

        time_awareness = TimeAwarenessCapability()

        # Initialize QA specialist agent with capabilities
        agent = QASpecialistAgent(capabilities=[logging_capability, time_awareness], environment=environment)

        # Run the agent
        logger.info(f"Processing question: {question}")
        result = await agent.run(question)

        # Print results
        print("\n=== QA Results with Logging ===")
        print(f"Question: {question}")
        print(f"\nAnswer: {result['answer']}")

        # Print logging information
        if "logging_stats" in result:
            print("\n=== Logging Information ===")
            for key, value in result["logging_stats"].items():
                print(f"{key}: {value}")

        # Print question analysis if available
        if "question_analysis" in result:
            print("\n=== Question Analysis ===")
            analysis = result["question_analysis"]

            if "companies" in analysis and analysis["companies"]:
                print(f"Companies: {', '.join(analysis['companies'])}")

            if "metrics" in analysis and analysis["metrics"]:
                print(f"Metrics: {', '.join(analysis['metrics'])}")

            if "filing_types" in analysis and analysis["filing_types"]:
                print(f"Filing Types: {', '.join(analysis['filing_types'])}")

            if "date_range" in analysis and analysis["date_range"]:
                print(f"Date Range: {analysis['date_range'][0]} to {analysis['date_range'][1]}")

            if "temporal_references" in analysis:
                print("\nTemporal References:")
                for key, value in analysis["temporal_references"].items():
                    if key != "date_range" and key != "fiscal_period":
                        print(f"- {key}: {value}")

        return result

    except Exception as e:
        logger.error(f"Error running QA with logging: {str(e)}")
        raise


def main():
    """Main function to run the example script."""
    parser = argparse.ArgumentParser(description="Logging QA Example")
    parser.add_argument(
        "--question",
        type=str,
        default="What was Apple's revenue growth in Q2 2023 compared to Q2 2022?",
        help="Financial question to ask",
    )
    parser.add_argument("--log_dir", type=str, default="data/logs/agents", help="Directory to store log files")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--include_prompts", action="store_true", help="Include LLM prompts in logs (may contain sensitive data)"
    )

    args = parser.parse_args()

    # Run the example
    asyncio.run(
        run_logging_qa(
            question=args.question, log_dir=args.log_dir, log_level=args.log_level, include_prompts=args.include_prompts
        )
    )


if __name__ == "__main__":
    main()
