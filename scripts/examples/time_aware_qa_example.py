"""
Example script demonstrating the Time Awareness Capability with the QA Specialist Agent.

This script shows how to use the TimeAwarenessCapability to enhance
financial question answering with temporal awareness.
"""

import argparse
import asyncio
import logging

from src.agents.qa_specialist import QASpecialistAgent
from src.capabilities.time_awareness import TimeAwarenessCapability
from src.environments.financial import FinancialEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_time_aware_qa(question: str):
    """Run the QA specialist agent with time awareness capability."""
    try:
        # Initialize environment
        environment = FinancialEnvironment()

        # Initialize time awareness capability
        time_awareness = TimeAwarenessCapability()

        # Initialize QA specialist agent with time awareness
        agent = QASpecialistAgent(
            capabilities=[time_awareness], environment=environment
        )

        # Run the agent
        logger.info(f"Processing question: {question}")
        result = await agent.run(question)

        # Print results
        print("\n=== Time-Aware QA Results ===")
        print(f"Question: {question}")
        print(f"\nAnswer: {result['answer']}")

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
                print(
                    f"Date Range: {analysis['date_range'][0]} to {analysis['date_range'][1]}"
                )

            if "fiscal_year" in analysis and "fiscal_quarter" in analysis:
                print(
                    f"Fiscal Period: FY{analysis['fiscal_year']} {analysis['fiscal_quarter']}"
                )

            if "temporal_references" in analysis:
                print("\nTemporal References:")
                for key, value in analysis["temporal_references"].items():
                    if key != "date_range" and key != "fiscal_period":
                        print(f"- {key}: {value}")

        # Print supporting data if available
        if "supporting_data" in result:
            print("\n=== Supporting Data ===")

            # Print semantic context
            semantic_context = result["supporting_data"].get("semantic_context", [])
            if semantic_context:
                print("\n--- Semantic Context ---")
                for i, context in enumerate(
                    semantic_context[:3]
                ):  # Show top 3 for brevity
                    print(f"\nContext {i + 1}:")
                    print(f"Company: {context.get('company', 'N/A')}")
                    print(
                        f"Filing: {context.get('filing_type', 'N/A')} ({context.get('filing_date', 'N/A')})"
                    )
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

        return result

    except Exception as e:
        logger.error(f"Error running time-aware QA: {str(e)}")
        raise


def main():
    """Main function to run the example script."""
    parser = argparse.ArgumentParser(description="Time-Aware QA Example")
    parser.add_argument(
        "--question",
        type=str,
        default="How did Apple's revenue in Q2 2023 compare to the same quarter last year?",
        help="Financial question with temporal aspects",
    )

    args = parser.parse_args()

    # Run the example
    asyncio.run(run_time_aware_qa(args.question))


if __name__ == "__main__":
    main()
