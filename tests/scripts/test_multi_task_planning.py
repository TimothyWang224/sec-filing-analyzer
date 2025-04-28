"""
Test script for the multi-task planning capability.

This script demonstrates how the financial analyst agent can handle multiple tasks
using the multi-task planning capability.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agents.financial_analyst import FinancialAnalystAgent
from src.capabilities.multi_task_planning import MultiTaskPlanningCapability
from src.llm.openai import OpenAILLM


async def main():
    """Run the test script."""
    # Create an LLM instance
    llm = OpenAILLM(model="gpt-4o-mini", temperature=0.3, max_tokens=1000)

    # Create a multi-task planning capability
    multi_task_planning = MultiTaskPlanningCapability(
        enable_dynamic_replanning=True,
        enable_step_reflection=True,
        min_steps_before_reflection=1,
        max_plan_steps=5,
        plan_detail_level="medium",
    )

    # Create a financial analyst agent with multi-task planning
    agent = FinancialAnalystAgent(
        capabilities=[multi_task_planning],
        llm_model="gpt-4o-mini",
        llm_temperature=0.3,
        llm_max_tokens=1000,
        max_iterations=10,  # Increase max iterations to handle multiple tasks
        max_duration_seconds=300,  # Increase max duration to handle multiple tasks
    )

    # Multi-task input with 5 distinct tasks
    user_input = """
    I need you to perform the following financial analysis tasks:

    1. Analyze Apple's revenue growth trends over the last 2 years
    2. Calculate and interpret key profitability ratios for Microsoft
    3. Compare the debt-to-equity ratios of Google and Amazon
    4. Evaluate Tesla's cash flow situation and liquidity
    5. Identify potential red flags in Netflix's financial statements
    """

    print(f"Starting financial analysis with multi-task planning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {user_input}")
    print("-" * 80)

    # Run the agent
    result = await agent.run(user_input)

    # Print the results
    print("\nResults:")
    print("-" * 80)
    print(f"Status: {result['status']}")
    print(f"Completed tasks: {result['completed_tasks']} / {result['total_tasks']}")

    # Print each task result
    for i, task in enumerate(result["tasks"]):
        print(f"\nTask {i + 1}: {task['input']}")
        print(f"Status: {task['status']}")
        print("Analysis:")
        print("-" * 40)
        if isinstance(task["result"], dict) and "analysis" in task["result"]:
            print(
                task["result"]["analysis"][:500] + "..."
                if len(task["result"]["analysis"]) > 500
                else task["result"]["analysis"]
            )
        else:
            print(task["result"])

    # Save the full results to a file
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"multi_task_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
