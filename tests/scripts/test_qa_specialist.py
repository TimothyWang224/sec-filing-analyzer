"""
Test script for the QA Specialist Agent.

This script tests the functionality of the QA Specialist Agent by asking
financial questions and evaluating the responses.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.qa_specialist import QASpecialistAgent
from src.environments.financial import FinancialEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_qa_agent(question: str):
    """Test the QA specialist agent with a financial question."""
    try:
        # Initialize environment and agent
        environment = FinancialEnvironment()
        agent = QASpecialistAgent(
            environment=environment,
            max_execution_iterations=5,  # Increase from default 2 to 5
        )

        # Run the agent
        logger.info(f"Processing question: {question}")
        result = await agent.run(question)

        # Print results
        print("\n=== QA Specialist Agent Results ===")
        print(f"Question: {question}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nExplanation: {result.get('explanation', 'N/A')}")

        # Print supporting data if available
        if "supporting_data" in result:
            print("\n=== Supporting Data ===")

            # Print semantic context
            semantic_context = result["supporting_data"].get("semantic_context", [])
            if semantic_context:
                print("\n--- Semantic Context ---")
                for i, context in enumerate(semantic_context[:3]):  # Show top 3 for brevity
                    print(f"\nContext {i + 1}:")
                    print(f"Company: {context.get('company', 'N/A')}")
                    print(f"Filing: {context.get('filing_type', 'N/A')} ({context.get('filing_date', 'N/A')})")
                    print(f"Text: {context.get('text', 'N/A')[:200]}...")

            # Print financial data
            financial_data = result["supporting_data"].get("financial_data", [])
            if financial_data:
                print("\n--- Financial Data ---")
                for i, data in enumerate(financial_data):
                    print(f"\nMetric {i + 1}:")
                    print(f"Name: {data.get('metric', 'N/A')}")
                    print(f"Value: {data.get('value', 'N/A')}")
                    print(f"Period: {data.get('period', 'N/A')}")

            # Print filing info
            filing_info = result["supporting_data"].get("filing_info", [])
            if filing_info:
                print("\n--- Filing Information ---")
                for i, filing in enumerate(filing_info):
                    print(f"\nFiling {i + 1}:")
                    print(f"Type: {filing.get('filing_type', 'N/A')}")
                    print(f"Date: {filing.get('filing_date', 'N/A')}")

        # Print question analysis
        if "question_analysis" in result:
            print("\n=== Question Analysis ===")
            analysis = result["question_analysis"]
            print(f"Companies: {analysis.get('companies', [])}")
            print(f"Filing Types: {analysis.get('filing_types', [])}")
            print(f"Metrics: {analysis.get('metrics', [])}")
            print(f"Date Range: {analysis.get('date_range', 'N/A')}")

        return result

    except Exception as e:
        logger.error(f"Error testing QA agent: {str(e)}")
        raise


def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test the QA Specialist Agent")
    parser.add_argument(
        "--question", type=str, default="What was Apple's revenue growth in 2023?", help="Financial question to ask"
    )

    args = parser.parse_args()

    # Run the test
    asyncio.run(test_qa_agent(args.question))


if __name__ == "__main__":
    main()
