"""
Test script for the Planning capability.

This script tests the Planning capability with the FinancialDiligenceCoordinator agent.
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents import FinancialDiligenceCoordinator
from src.capabilities import PlanningCapability, LoggingCapability
from src.environments import FinancialEnvironment

async def test_planning_with_coordinator():
    """Test the Planning capability with the FinancialDiligenceCoordinator agent."""
    print("\n=== Testing Planning Capability with Coordinator Agent ===")

    # Create environment
    environment = FinancialEnvironment()

    # Create agent with capabilities
    agent = FinancialDiligenceCoordinator(
        capabilities=[
            PlanningCapability(
                enable_dynamic_replanning=True,
                enable_step_reflection=True,
                min_steps_before_reflection=2,
                max_plan_steps=5,
                plan_detail_level="high"
            ),
            LoggingCapability(log_to_console=True)
        ],
        environment=environment,
        max_iterations=2  # Allow for plan execution
    )

    # Run the agent with a complex query that requires planning
    result = await agent.run("Analyze Apple's financial performance over the last 3 years, focusing on revenue growth, profit margins, and compare it to Microsoft")

    # Print the result
    print("\nCoordinator Result:")
    print(f"Report: {result.get('diligence_report', {}).get('executive_summary', 'No summary')[:200]}...")
    
    # Print the plan that was generated
    memory = result.get('memory', [])
    plan_items = [item for item in memory if item.get('type') == 'plan']
    
    if plan_items:
        print("\nGenerated Plan:")
        plan = plan_items[0].get('content', {})
        print(f"Goal: {plan.get('goal', 'No goal')}")
        print("Steps:")
        for step in plan.get('steps', []):
            print(f"  {step.get('step_id', '?')}. {step.get('description', 'No description')} - Status: {step.get('status', 'unknown')}")
    
    return result

async def test_planning_with_financial_analyst():
    """Test the Planning capability with the FinancialAnalystAgent."""
    print("\n=== Testing Planning Capability with Financial Analyst Agent ===")

    # Create environment
    environment = FinancialEnvironment()

    # Create agent with capabilities
    from src.agents import FinancialAnalystAgent
    
    agent = FinancialAnalystAgent(
        capabilities=[
            PlanningCapability(
                enable_dynamic_replanning=True,
                enable_step_reflection=True,
                min_steps_before_reflection=1,
                max_plan_steps=3,
                plan_detail_level="medium"
            ),
            LoggingCapability(log_to_console=True)
        ],
        environment=environment,
        max_iterations=2  # Allow for plan execution
    )

    # Run the agent with a query that requires planning
    result = await agent.run("Analyze Apple's revenue growth trends and profit margins for the last 2 years")

    # Print the result
    print("\nFinancial Analyst Result:")
    print(f"Analysis: {result.get('analysis', {}).get('analysis', 'No analysis')[:200]}...")
    
    # Print the plan that was generated
    memory = result.get('memory', [])
    plan_items = [item for item in memory if item.get('type') == 'plan']
    
    if plan_items:
        print("\nGenerated Plan:")
        plan = plan_items[0].get('content', {})
        print(f"Goal: {plan.get('goal', 'No goal')}")
        print("Steps:")
        for step in plan.get('steps', []):
            print(f"  {step.get('step_id', '?')}. {step.get('description', 'No description')} - Status: {step.get('status', 'unknown')}")
    
    return result

async def main():
    """Run all tests."""
    # Test planning with coordinator
    await test_planning_with_coordinator()
    
    # Test planning with financial analyst
    await test_planning_with_financial_analyst()

if __name__ == "__main__":
    asyncio.run(main())
