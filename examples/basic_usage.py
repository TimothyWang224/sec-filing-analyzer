import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use absolute import instead of relative import
from src.api import SECFilingAnalyzer


async def main():
    # Initialize the analyzer
    analyzer = SECFilingAnalyzer()

    # Example company ticker
    ticker = "AAPL"

    # Perform comprehensive diligence
    print(f"\nPerforming comprehensive diligence for {ticker}...")
    diligence_results = await analyzer.perform_diligence(ticker)
    print("\nDiligence Results:")
    print(f"Executive Summary: {diligence_results['diligence_report']['executive_summary']}")
    print("\nKey Findings:")
    for finding in diligence_results["diligence_report"]["key_findings"]:
        print(f"- {finding}")
    print("\nRecommendations:")
    for rec in diligence_results["diligence_report"]["recommendations"]:
        print(f"- {rec}")

    # Ask a specific question
    question = "What is the company's revenue growth trend?"
    print(f"\nAsking question: {question}")
    qa_results = await analyzer.answer_question(question)
    print("\nAnswer:")
    print(qa_results["answer"])

    # Get memory summary
    print("\nMemory Summary:")
    memory_summary = analyzer.get_memory_summary()
    print(f"Total Items: {memory_summary['total_items']}")
    print(f"Financial Metrics: {memory_summary['financial_metrics']}")
    print(f"Risk Assessments: {memory_summary['risk_assessments']}")
    print(f"Insights: {memory_summary['insights']}")


if __name__ == "__main__":
    asyncio.run(main())
