"""
Test script for the Logging Capability.

This script tests the functionality of the LoggingCapability by
running a QA Specialist Agent with logging enabled.
"""

import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.agents.qa_specialist import QASpecialistAgent
from src.environments.financial import FinancialEnvironment
from src.capabilities.logging import LoggingCapability

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_logging_capability(
    question: str,
    log_dir: str = "data/logs/test",
    log_level: str = "DEBUG",
    include_prompts: bool = False
):
    """Test the logging capability with a QA specialist agent."""
    try:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Initialize environment
        environment = FinancialEnvironment()
        print(f"Environment created with tools: {[k for k in environment.context.keys() if k.startswith('tool_')]}")

        # Initialize logging capability
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
            max_content_length=500  # Shorter for testing
        )

        # Initialize QA specialist agent with logging capability
        agent = QASpecialistAgent(
            capabilities=[logging_capability],
            environment=environment
        )

        # Debug: Check if agent has the environment
        print(f"Agent environment has tools: {[k for k in agent.environment.context.keys() if k.startswith('tool_')]}")

        # Run the agent
        logger.info(f"Processing question with logging: {question}")
        result = await agent.run(question)

        # Print results
        print("\n=== Agent Results ===")
        print(f"Question: {question}")
        print(f"\nAnswer: {result['answer']}")

        # Print logging information
        if "logging_stats" in result:
            print("\n=== Logging Statistics ===")
            for key, value in result["logging_stats"].items():
                print(f"{key}: {value}")

            # Check if log file exists
            log_file = result["logging_stats"].get("log_file")
            if log_file and os.path.exists(log_file):
                print(f"\nLog file created at: {log_file}")

                # Print sample from log file
                with open(log_file, 'r') as f:
                    log_lines = f.readlines()
                    sample_size = min(5, len(log_lines))
                    print(f"\nSample from log file (first {sample_size} lines):")
                    for line in log_lines[:sample_size]:
                        print(f"  {line.strip()}")

            # Check if JSON log file exists
            json_log_file = str(log_file).replace(".log", ".json") if log_file else None
            if json_log_file and os.path.exists(json_log_file):
                print(f"\nJSON log file created at: {json_log_file}")

                # Print summary from JSON log file
                with open(json_log_file, 'r') as f:
                    log_data = json.load(f)
                    print(f"\nJSON log summary:")
                    print(f"  Session ID: {log_data.get('session_id')}")
                    print(f"  Agent Type: {log_data.get('agent_type')}")
                    print(f"  Start Time: {log_data.get('start_time')}")
                    print(f"  Log Entries: {len(log_data.get('logs', []))}")

        return result

    except Exception as e:
        logger.error(f"Error testing logging capability: {str(e)}")
        raise

def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test the Logging Capability")
    parser.add_argument("--question", type=str,
                        default="What was Apple's revenue in 2023?",
                        help="Financial question to ask")
    parser.add_argument("--log_dir", type=str,
                        default="data/logs/test",
                        help="Directory to store log files")
    parser.add_argument("--log_level", type=str,
                        default="DEBUG",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--include_prompts", action="store_true",
                        help="Include LLM prompts in logs (may contain sensitive data)")

    args = parser.parse_args()

    # Run the test
    asyncio.run(test_logging_capability(
        question=args.question,
        log_dir=args.log_dir,
        log_level=args.log_level,
        include_prompts=args.include_prompts
    ))

if __name__ == "__main__":
    main()
