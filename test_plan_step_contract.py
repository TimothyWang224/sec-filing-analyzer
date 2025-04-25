import asyncio
import json
import logging

from src.agents.core.agent_state import AgentState
from src.agents.qa_specialist import QASpecialistAgent
from src.capabilities.planning import PlanningCapability

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


async def test_plan_step_contract():
    """Test the plan-step ↔ tool contract implementation."""
    print("\n\n" + "=" * 80)
    print("TESTING PLAN-STEP ↔ TOOL CONTRACT")
    print("=" * 80 + "\n")

    # Create a QA Specialist Agent
    agent = QASpecialistAgent(max_planning_iterations=2, max_execution_iterations=5, max_refinement_iterations=3)

    # Add planning capability
    planning_capability = PlanningCapability(agent)
    agent.capabilities.append(planning_capability)

    # Create a test query
    query = "What was Apple's revenue in 2022?"

    # Run the agent
    result = await agent.run(query)

    # Print the plan
    planning_context = agent.state.get_context().get("planning", {})
    plan = planning_context.get("plan", {})

    print("\nPLAN:")
    print("=" * 80)
    print(json.dumps(plan, indent=2))

    # Print the memory items
    memory_items = agent.get_memory()
    plan_items = [item for item in memory_items if item.get("type") == "plan"]

    if plan_items:
        print("\nPLAN FROM MEMORY:")
        print("=" * 80)
        for i, plan_item in enumerate(plan_items):
            print(f"Plan {i + 1}:")
            print(json.dumps(plan_item.get("content", {}), indent=2))

    # Check if each step has a done_check field
    steps = plan.get("steps", [])
    for step in steps:
        print(f"\nStep {step['step_id']}: {step['description']}")
        print(f"  Tool: {step.get('tool', 'N/A')}")
        print(f"  Expected Key: {step.get('expected_key', 'N/A')}")
        print(f"  Output Path: {step.get('output_path', 'N/A')}")
        print(f"  Done Check: {step.get('done_check', 'N/A')}")

    # Print the final result
    print("\nRESULT:")
    print("=" * 80)
    print(result.get("response", "No response"))

    return result


if __name__ == "__main__":
    asyncio.run(test_plan_step_contract())
