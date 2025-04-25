"""
Test script for the Time Awareness Capability.

This script tests the functionality of the TimeAwarenessCapability by
processing financial questions with temporal aspects.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.qa_specialist import QASpecialistAgent
from src.capabilities.time_awareness import TimeAwarenessCapability
from src.environments.financial import FinancialEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_time_awareness(question: str):
    """Test the time awareness capability with a financial question."""
    try:
        # Initialize environment and capability
        environment = FinancialEnvironment()
        time_awareness = TimeAwarenessCapability()

        # Initialize agent with time awareness capability
        agent = QASpecialistAgent(capabilities=[time_awareness], environment=environment)

        # Extract temporal references from the question
        context = {}
        await time_awareness.init(agent, context)
        temporal_references = time_awareness._extract_temporal_references(question)

        # Generate time context
        time_context = time_awareness._generate_time_context(temporal_references)

        # Print results
        print("\n=== Time Awareness Test ===")
        print(f"Question: {question}")

        print("\n=== Temporal References ===")
        for key, value in temporal_references.items():
            print(f"{key}: {value}")

        print("\n=== Time Context ===")
        for key, value in time_context.items():
            print(f"{key}: {value}")

        # Process a sample prompt
        enhanced_prompt = await time_awareness.process_prompt(agent, context, question)

        print("\n=== Enhanced Prompt ===")
        print(enhanced_prompt)

        # Create a sample action
        sample_action = {
            "tool": "sec_financial_data",
            "args": {
                "query_type": "financial_facts",
                "parameters": {"ticker": "AAPL", "metrics": ["Revenue", "Net Income"]},
            },
        }

        # Process the action
        context["temporal_references"] = temporal_references
        enhanced_action = await time_awareness.process_action(agent, context, sample_action)

        print("\n=== Enhanced Action ===")
        print(json.dumps(enhanced_action, indent=2))

        return {
            "temporal_references": temporal_references,
            "time_context": time_context,
            "enhanced_prompt": enhanced_prompt,
            "enhanced_action": enhanced_action,
        }

    except Exception as e:
        logger.error(f"Error testing time awareness: {str(e)}")
        raise


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test the Time Awareness Capability")
    parser.add_argument(
        "--question",
        type=str,
        default="What was Apple's revenue growth in Q2 2023 compared to Q2 2022?",
        help="Financial question with temporal aspects",
    )

    args = parser.parse_args()

    # Run the test
    asyncio.run(test_time_awareness(args.question))


if __name__ == "__main__":
    main()
