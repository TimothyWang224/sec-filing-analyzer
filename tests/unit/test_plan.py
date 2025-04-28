import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from src.contracts import Plan, PlanStep

    # Create a test plan
    plan_step = PlanStep(
        step_id=1,
        description="Test step",
        tool="test_tool",
        parameters={"param1": "value1"},
        expected_key="test_result",
        output_path=["data", "result"],
        done_check="True",
        dependencies=[],
        status="pending",
    )

    plan = Plan(
        goal="Test plan",
        steps=[plan_step],
        status="pending",
        created_at="2023-01-01T00:00:00",
        owner="agent",
        can_modify=True,
    )

    # Test accessing plan attributes
    print(f"Plan goal: {plan.goal}")
    print(f"Plan steps: {len(plan.steps)}")
    print(f"First step description: {plan.steps[0].description}")

    # Test accessing plan as a dictionary (this should fail)
    try:
        print(f"Plan goal via dict access: {plan['goal']}")
    except TypeError as e:
        print(f"Expected error: {e}")

    # Test converting plan to dictionary
    plan_dict = plan.model_dump()
    print(f"Plan as dict: {plan_dict['goal']}")

    print("All tests passed!")

except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
