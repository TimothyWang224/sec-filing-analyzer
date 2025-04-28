import asyncio
import json
import logging

from src.agents.qa_specialist import QASpecialistAgent
from src.capabilities.planning import PlanningCapability
from src.contracts import extract_value

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_pydantic_plan_contract():
    """Test the plan-step ↔ tool contract implementation with Pydantic models."""
    print("\n\n" + "=" * 80)
    print("TESTING PLAN-STEP ↔ TOOL CONTRACT WITH PYDANTIC MODELS")
    print("=" * 80 + "\n")

    # Create a QA Specialist Agent
    agent = QASpecialistAgent(
        max_planning_iterations=2,
        max_execution_iterations=5,
        max_refinement_iterations=3,
    )

    # Add planning capability
    planning_capability = PlanningCapability(agent)
    agent.capabilities.append(planning_capability)

    # Create a test query
    query = "What was Apple's revenue in 2022?"

    # Run the agent
    result = await agent.run(query)

    # Print the plan
    planning_context = agent.state.get_context().get("planning", {})
    plan_dict = planning_context.get("plan", {})

    print("\nPLAN (Dictionary):")
    print("=" * 80)
    print(json.dumps(plan_dict, indent=2))

    # Convert the plan to a Pydantic model
    plan = planning_capability._dict_to_plan(plan_dict)

    print("\nPLAN (Pydantic Model):")
    print("=" * 80)
    print(f"Goal: {plan.goal}")
    print(f"Status: {plan.status}")
    print(f"Owner: {plan.owner}")
    print(f"Can Modify: {plan.can_modify}")
    print(f"Created At: {plan.created_at}")
    print(f"Completed At: {plan.completed_at}")
    print(f"Number of Steps: {len(plan.steps)}")

    # Print each step
    for step in plan.steps:
        print(f"\nStep {step.step_id}: {step.description}")
        print(f"  Tool: {step.tool}")
        print(f"  Agent: {step.agent}")
        print(f"  Parameters: {step.parameters}")
        print(f"  Dependencies: {step.dependencies}")
        print(f"  Expected Key: {step.expected_key}")
        print(f"  Output Path: {step.output_path}")
        print(f"  Done Check: {step.done_check}")
        print(f"  Status: {step.status}")
        print(f"  Completed At: {step.completed_at}")
        print(f"  Skipped: {step.skipped}")

    # Test the extract_value function
    test_data = {"results": [{"value": 123, "name": "test"}]}

    value = extract_value(test_data, ["results", 0, "value"])
    print(f"\nExtract Value Test: {value}")

    # Print the final result
    print("\nRESULT:")
    print("=" * 80)
    print(result.get("response", "No response"))

    return result


if __name__ == "__main__":
    asyncio.run(test_pydantic_plan_contract())
