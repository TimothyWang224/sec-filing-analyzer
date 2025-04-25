"""
Test script for the agent implementations.
"""

import asyncio
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agents import FinancialAnalystAgent, FinancialDiligenceCoordinator, QASpecialistAgent, RiskAnalystAgent
from src.capabilities.logging import LoggingCapability
from src.capabilities.time_awareness import TimeAwarenessCapability
from src.environments.financial import FinancialEnvironment


async def test_qa_specialist():
    """Test the QA Specialist agent."""
    print("\n=== Testing QA Specialist Agent ===")

    # Create environment
    environment = FinancialEnvironment()

    # Create agent with capabilities
    agent = QASpecialistAgent(
        capabilities=[LoggingCapability(), TimeAwarenessCapability()], environment=environment, max_iterations=1
    )

    # Run the agent
    result = await agent.run("What was Apple's revenue in 2023?")

    # Print the result
    print("\nQA Specialist Result:")
    print(f"Answer: {result.get('answer', {}).get('answer', 'No answer')}")
    print(f"Confidence: {result.get('answer', {}).get('confidence', 0)}")

    return result


async def test_financial_analyst():
    """Test the Financial Analyst agent."""
    print("\n=== Testing Financial Analyst Agent ===")

    # Create environment
    environment = FinancialEnvironment()

    # Create agent with capabilities
    agent = FinancialAnalystAgent(
        capabilities=[LoggingCapability(), TimeAwarenessCapability()], environment=environment, max_iterations=1
    )

    # Run the agent
    result = await agent.run("Analyze Apple's financial performance")

    # Print the result
    print("\nFinancial Analyst Result:")
    print(f"Analysis: {result.get('analysis', {}).get('analysis', 'No analysis')[:200]}...")
    print(f"Metrics: {result.get('analysis', {}).get('metrics', {})}")

    return result


async def test_risk_analyst():
    """Test the Risk Analyst agent."""
    print("\n=== Testing Risk Analyst Agent ===")

    # Create environment
    environment = FinancialEnvironment()

    # Create agent with capabilities
    agent = RiskAnalystAgent(
        capabilities=[LoggingCapability(), TimeAwarenessCapability()], environment=environment, max_iterations=1
    )

    # Run the agent
    result = await agent.run("Identify risks for Apple")

    # Print the result
    print("\nRisk Analyst Result:")
    print(f"Analysis: {result.get('risk_analysis', {}).get('analysis', 'No analysis')[:200]}...")
    print(f"Risk Factors: {result.get('risk_analysis', {}).get('risk_factors', {})}")

    return result


async def test_coordinator():
    """Test the Financial Diligence Coordinator agent."""
    print("\n=== Testing Financial Diligence Coordinator Agent ===")

    # Create environment
    environment = FinancialEnvironment()

    # Create agent with capabilities
    agent = FinancialDiligenceCoordinator(
        capabilities=[LoggingCapability(), TimeAwarenessCapability()], environment=environment, max_iterations=1
    )

    # Run the agent
    result = await agent.run("Provide a comprehensive analysis of Apple's financial health and risks")

    # Print the result
    print("\nCoordinator Result:")
    print(f"Executive Summary: {result.get('diligence_report', {}).get('executive_summary', 'No summary')[:200]}...")
    print(f"Key Findings: {result.get('diligence_report', {}).get('key_findings', [])}")

    return result


async def main():
    """Run all agent tests."""
    # Test QA Specialist
    await test_qa_specialist()

    # Test Financial Analyst
    await test_financial_analyst()

    # Test Risk Analyst
    await test_risk_analyst()

    # Test Coordinator
    await test_coordinator()


if __name__ == "__main__":
    asyncio.run(main())
