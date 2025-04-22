"""
Test script for the chat mode of the Financial Diligence Coordinator.

This script tests the chat mode functionality of the coordinator agent,
which is used in the dedicated chat app.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agents import FinancialDiligenceCoordinator
from src.environments import FinancialEnvironment
from src.capabilities import TimeAwarenessCapability, LoggingCapability, PlanningCapability

async def test_chat_mode():
    """Test the chat mode of the coordinator agent."""
    print("\n=== Testing Chat Mode ===")
    
    # Create environment
    environment = FinancialEnvironment()
    
    # Create capabilities
    capabilities = [
        TimeAwarenessCapability(),
        LoggingCapability(),
        PlanningCapability(
            enable_dynamic_replanning=True,
            enable_step_reflection=True,
            max_plan_steps=5,
            plan_detail_level="medium"
        )
    ]
    
    # Create coordinator agent
    agent = FinancialDiligenceCoordinator(
        environment=environment,
        capabilities=capabilities,
        llm_model="gpt-4o-mini",  # Use a smaller model for testing
        llm_temperature=0.7,
        llm_max_tokens=2000,
        max_planning_iterations=1,
        max_execution_iterations=1,
        max_refinement_iterations=1
    )
    
    # Test questions
    questions = [
        "What was Apple's revenue in 2023?",
        "Analyze NVIDIA's financial performance.",
        "What are the main risks for Microsoft?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        
        # Run the agent with chat mode enabled
        result = await agent.run(question, chat_mode=True)
        
        # Print the formatted response
        if "response" in result:
            print("\nFormatted Response:")
            print(result["response"][:300] + "..." if len(result["response"]) > 300 else result["response"])
        else:
            print("\nNo formatted response found.")
            print("Raw result keys:", result.keys())
        
        print("\n" + "-" * 80)

if __name__ == "__main__":
    asyncio.run(test_chat_mode())
