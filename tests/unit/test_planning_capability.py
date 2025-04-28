import asyncio
import json
import logging

from src.agents.qa_specialist import QASpecialistAgent
from src.capabilities.planning import PlanningCapability

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_planning_capability():
    """Test the PlanningCapability with Pydantic models."""
    print("\n\n" + "=" * 80)
    print("TESTING PLANNING CAPABILITY WITH PYDANTIC MODELS")
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

    # Initialize the planning capability
    context = {"input": query}
    context = await planning_capability.init(agent, context)

    # Create a plan
    plan = await planning_capability._create_plan(query)

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

    # Convert plan to dictionary
    plan_dict = planning_capability._plan_to_dict(plan)

    print("\nPLAN (Dictionary):")
    print("=" * 80)
    print(json.dumps(plan_dict, indent=2))

    # Convert back to Plan object
    plan2 = planning_capability._dict_to_plan(plan_dict)

    print("\nPLAN (Converted back to Pydantic Model):")
    print("=" * 80)
    print(f"Goal: {plan2.goal}")
    print(f"Status: {plan2.status}")
    print(f"Owner: {plan2.owner}")
    print(f"Can Modify: {plan2.can_modify}")
    print(f"Created At: {plan2.created_at}")
    print(f"Completed At: {plan2.completed_at}")
    print(f"Number of Steps: {len(plan2.steps)}")

    return plan


if __name__ == "__main__":
    asyncio.run(test_planning_capability())
