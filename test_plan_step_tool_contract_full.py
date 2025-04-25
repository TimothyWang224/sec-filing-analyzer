import asyncio
import json
import logging
from datetime import datetime

from src.agents.qa_specialist import QASpecialistAgent
from src.capabilities.planning import PlanningCapability
from src.contracts import FinancialFactsParams, Plan, PlanStep, ToolInput
from src.environments.base import Environment
from src.errors import ParameterError, QueryTypeUnsupported
from src.tools.registry import ToolRegistry
from src.tools.validator import validate_call

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


async def test_plan_step_tool_contract():
    """Test the Plan-Step ↔ Tool Contract with all tools."""
    print("\n\n" + "=" * 80)
    print("TESTING PLAN-STEP ↔ TOOL CONTRACT")
    print("=" * 80 + "\n")

    # Create a QA Specialist Agent
    agent = QASpecialistAgent(max_planning_iterations=2, max_execution_iterations=5, max_refinement_iterations=3)

    # Add planning capability
    planning_capability = PlanningCapability(
        enable_dynamic_replanning=True,
        enable_step_reflection=True,
        min_steps_before_reflection=2,
        max_plan_steps=10,
        plan_detail_level="high",
    )
    agent.capabilities.append(planning_capability)

    # Initialize the planning capability
    context = {"input": "What was Apple's revenue in 2022?"}
    context = await planning_capability.init(agent, context)

    # Create a plan
    plan = await planning_capability._create_plan("What was Apple's revenue in 2022?")

    print("\nPLAN:")
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

    # Test the contract with a specific tool
    print("\nTESTING CONTRACT WITH SEC_FINANCIAL_DATA TOOL:")
    print("=" * 80)

    # Create a step for the sec_financial_data tool
    step = PlanStep(
        step_id=1,
        description="Get Apple's revenue for 2022",
        tool="sec_financial_data",
        parameters={
            "query_type": "financial_facts",
            "parameters": {
                "ticker": "AAPL",
                "metrics": ["Revenue"],
                "start_date": "2022-01-01",
                "end_date": "2022-12-31",
            },
        },
        expected_key="revenue_data",
        output_path=["data", "Revenue"],
        done_check="True",
        dependencies=[],
        status="pending",
    )

    # Get the tool spec
    tool_spec = ToolRegistry.get_tool_spec("sec_financial_data")
    print(f"Tool Spec: {tool_spec}")
    print(f"  Output Key: {tool_spec.output_key}")

    # Create an environment
    environment = Environment()

    # Execute the tool
    try:
        print(f"Executing tool: {step.tool}")
        print(f"Parameters: {json.dumps(step.parameters, indent=2)}")

        result = await environment.execute_action({"tool": step.tool, "args": step.parameters})

        print(f"Result: {json.dumps(result, indent=2)}")

        # Check if the result has an output_key
        if isinstance(result, dict) and "output_key" in result:
            print(f"Output Key in Result: {result['output_key']}")
            print(f"Expected Output Key: {tool_spec.output_key}")

            # Verify that the output_key matches the tool_spec.output_key
            assert result["output_key"] == tool_spec.output_key, "Output key mismatch"
            print("✅ Output key matches tool spec")
        else:
            print("❌ Output key not found in result")
    except Exception as e:
        print(f"Error executing tool: {str(e)}")

    # Test the contract with a skipped step
    print("\nTESTING CONTRACT WITH SKIPPED STEP:")
    print("=" * 80)

    # Create a plan with a step that should be skipped
    plan = Plan(
        goal="Test skipped step",
        steps=[step],
        status="in_progress",
        created_at=datetime.now().isoformat(),
        owner="agent",
        can_modify=True,
    )

    # Add the step result to memory
    agent.add_to_memory(
        {
            "type": "tool_result",
            "tool": "sec_financial_data",
            "result": {"data": {"Revenue": 123456789}, "output_key": "sec_financial_data"},
            "expected_key": "revenue_data",
        }
    )

    # Add a step_skipped memory item
    agent.add_to_memory(
        {
            "type": "step_skipped",
            "step_id": 1,
            "tool": "sec_financial_data",
            "expected_key": "revenue_data",
            "reason": "Success criterion already satisfied",
            "timestamp": datetime.now().isoformat(),
        }
    )

    # Check if the step should be skipped
    print(f"Checking if step should be skipped...")
    # Initialize the agent's context
    agent.state.context = {"planning": {"plan": plan.model_dump(), "current_step": step.model_dump()}}

    # Use the _should_skip method from the base Agent class
    should_skip = agent._should_skip(step.model_dump())
    print(f"Should Skip: {should_skip}")

    # Verify that the step was skipped
    if should_skip:
        print("✅ Step was correctly skipped")

        # Check if the step was added to memory
        memory = agent.get_memory()
        step_skipped_found = False

        for item in memory:
            if item.get("type") == "step_skipped":
                step_skipped_found = True
                print(f"Memory Item: {json.dumps(item, indent=2)}")
                print("✅ Step skipped item found in memory")
                break

        if not step_skipped_found:
            print("❌ Step skipped item not found in memory")
    else:
        print("❌ Step was not skipped")

    # Test the validator
    print("\nTESTING VALIDATOR:")
    print("=" * 80)

    # Test with valid parameters
    try:
        print("Testing with valid parameters:")
        validate_call(
            "sec_financial_data",
            "financial_facts",
            {"ticker": "AAPL", "metrics": ["Revenue"], "start_date": "2022-01-01", "end_date": "2022-12-31"},
        )
        print("✅ Validation passed")
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")

    # Test with invalid parameters
    try:
        print("\nTesting with invalid parameters (missing ticker):")
        validate_call(
            "sec_financial_data",
            "financial_facts",
            {"metrics": ["Revenue"], "start_date": "2022-01-01", "end_date": "2022-12-31"},
        )
        print("❌ Validation passed (should have failed)")
    except ParameterError as e:
        print(f"✅ Validation failed as expected: {str(e)}")
        print(f"User message: {e.user_message()}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

    # Test with invalid query type
    try:
        print("\nTesting with invalid query type:")
        validate_call("sec_financial_data", "invalid_query_type", {"ticker": "AAPL"})
        print("❌ Validation passed (should have failed)")
    except QueryTypeUnsupported as e:
        print(f"✅ Validation failed as expected: {str(e)}")
        print(f"User message: {e.user_message()}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

    # Test direct model validation
    print("\nTESTING DIRECT MODEL VALIDATION:")
    print("=" * 80)

    # Test with valid parameters
    try:
        print("Testing FinancialFactsParams with valid parameters:")
        params = FinancialFactsParams(
            ticker="AAPL", metrics=["Revenue"], start_date="2022-01-01", end_date="2022-12-31"
        )
        print(f"✅ Validation passed: {params.model_dump()}")
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")

    # Test with invalid parameters
    try:
        print("\nTesting FinancialFactsParams with invalid parameters (missing ticker):")
        params = FinancialFactsParams(metrics=["Revenue"], start_date="2022-01-01", end_date="2022-12-31")
        print(f"❌ Validation passed (should have failed): {params.model_dump()}")
    except Exception as e:
        print(f"✅ Validation failed as expected: {str(e)}")

    # Test ToolInput
    print("\nTESTING TOOL INPUT:")
    print("=" * 80)

    # Test with valid parameters
    try:
        print("Testing ToolInput with valid parameters:")
        tool_input = ToolInput(
            query_type="financial_facts",
            parameters={"ticker": "AAPL", "metrics": ["Revenue"], "start_date": "2022-01-01", "end_date": "2022-12-31"},
        )
        print(f"✅ Validation passed: {tool_input.model_dump()}")
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")

    return plan


if __name__ == "__main__":
    asyncio.run(test_plan_step_tool_contract())
