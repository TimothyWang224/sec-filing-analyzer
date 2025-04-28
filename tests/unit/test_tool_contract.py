import asyncio
import json
import logging

from src.agents.qa_specialist import QASpecialistAgent
from src.contracts import Plan, PlanStep, ToolSpec, extract_value
from src.tools.registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


async def test_tool_contract():
    """Test the Plan-Step ↔ Tool Contract implementation."""
    print("\n\n" + "=" * 80)
    print("TESTING PLAN-STEP ↔ TOOL CONTRACT")
    print("=" * 80 + "\n")

    # List all registered tools
    tools = ToolRegistry.list_tools()
    print(f"Found {len(tools)} registered tools:")
    for name in tools:
        print(f"- {name}")

    # Get a tool spec for a specific tool
    tool_name = "sec_financial_data"
    tool_spec = ToolRegistry.get_tool_spec(tool_name)

    if tool_spec:
        print(f"\nTool Spec for {tool_name}:")
        print(f"Name: {tool_spec.name}")
        print(f"Description: {tool_spec.description}")
        print(f"Output Key: {tool_spec.output_key}")
        print(f"Input Schema: {json.dumps(tool_spec.input_schema, indent=2)}")
    else:
        print(f"\nNo tool spec found for {tool_name}")

    # Test the extract_value function
    test_data = {"results": [{"value": 123, "name": "test"}]}

    value = extract_value(test_data, ["results", 0, "value"])
    print(f"\nExtract Value Test: {value}")

    # Create a PlanStep
    plan_step = PlanStep(
        step_id=1,
        description="Test step",
        tool="sec_financial_data",
        parameters={"query_type": "metrics", "parameters": {"ticker": "AAPL", "year": 2022}},
        expected_key="financial_data",
        output_path=["results"],
        done_check="financial_data is not None",
        dependencies=[],
        status="pending",
    )

    print("\nPlan Step:")
    print(f"Step ID: {plan_step.step_id}")
    print(f"Description: {plan_step.description}")
    print(f"Tool: {plan_step.tool}")
    print(f"Parameters: {plan_step.parameters}")
    print(f"Expected Key: {plan_step.expected_key}")
    print(f"Output Path: {plan_step.output_path}")
    print(f"Done Check: {plan_step.done_check}")
    print(f"Status: {plan_step.status}")

    # Create a Plan
    plan = Plan(goal="Test plan", steps=[plan_step], status="pending", owner="agent", can_modify=True)

    print("\nPlan:")
    print(f"Goal: {plan.goal}")
    print(f"Status: {plan.status}")
    print(f"Owner: {plan.owner}")
    print(f"Can Modify: {plan.can_modify}")
    print(f"Number of Steps: {len(plan.steps)}")

    # Create a QA Specialist Agent
    agent = QASpecialistAgent(max_planning_iterations=2, max_execution_iterations=5, max_refinement_iterations=3)

    # Run the agent with a simple query
    query = "What was Apple's revenue in 2022?"
    print(f"\nRunning agent with query: {query}")

    result = await agent.run(query)

    print("\nAgent Result:")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    asyncio.run(test_tool_contract())
