"""
Test script to verify Plan-Object integration.

This script tests the Plan-Object integration by creating a Plan object,
converting it to a dictionary and back, and verifying that the object
attributes are preserved.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from src.agents.base import _to_dict, _to_object
    from src.contracts import Plan, PlanStep

    # Create a test plan
    plan_step = PlanStep(
        step_id=1,
        description="Test step",
        tool="sec_financial_data",
        parameters={"query_type": "financial_facts", "parameters": {"ticker": "MSFT"}},
        expected_key="financial_data",
        output_path=["results"],
        done_check="results is not None and len(results) > 0",
        dependencies=[],
        status="pending",
    )

    plan = Plan(
        goal="Get Microsoft financial data",
        steps=[plan_step],
        status="pending",
        created_at=datetime.now().isoformat(),
        owner="agent",
        can_modify=True,
    )

    # Test converting to dictionary and back
    logger.info("Testing Plan-Object conversion...")

    # Convert to dictionary
    plan_dict = _to_dict(plan)
    logger.info(f"Plan as dictionary: {plan_dict['goal']}")

    # Convert back to Plan object
    plan_obj = _to_object(plan_dict, Plan)
    logger.info(f"Plan as object: {plan_obj.goal}")

    # Verify attributes are preserved
    assert plan_obj.goal == plan.goal, "Goal attribute not preserved"
    assert len(plan_obj.steps) == len(plan.steps), "Steps count not preserved"
    assert plan_obj.steps[0].tool == plan.steps[0].tool, "Tool attribute not preserved"

    # Test accessing attributes
    logger.info(f"Plan goal: {plan.goal}")
    logger.info(f"Plan steps: {len(plan.steps)}")
    logger.info(f"First step description: {plan.steps[0].description}")
    logger.info(f"First step tool: {plan.steps[0].tool}")

    # Test accessing as dictionary (this should fail)
    try:
        logger.info(f"Plan goal via dict access: {plan['goal']}")
        assert False, "Dictionary access should have failed"
    except TypeError as e:
        logger.info(f"Expected error: {e}")

    # Test the _to_dict helper
    plan_dict = _to_dict(plan)
    logger.info(f"Plan as dict via _to_dict: {plan_dict['goal']}")

    # Test the _to_object helper
    plan_obj = _to_object(plan_dict, Plan)
    logger.info(f"Plan as object via _to_object: {plan_obj.goal}")

    logger.info("All tests passed!")

except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
