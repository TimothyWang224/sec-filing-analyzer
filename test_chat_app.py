import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    # Import agent components
    from src.agents import FinancialDiligenceCoordinator
    from src.capabilities import LoggingCapability, PlanningCapability, TimeAwarenessCapability
    from src.environments import FinancialEnvironment

    # Create environment
    environment = FinancialEnvironment()

    # Create capabilities
    capabilities = [
        TimeAwarenessCapability(),
        LoggingCapability(include_prompts=True, include_responses=True, max_content_length=10000),
        PlanningCapability(
            enable_dynamic_replanning=True, enable_step_reflection=True, max_plan_steps=10, plan_detail_level="high"
        ),
    ]

    # Create coordinator agent
    agent = FinancialDiligenceCoordinator(
        environment=environment,
        capabilities=capabilities,
        llm_model="gpt-4o-mini",
        llm_temperature=0.7,
        llm_max_tokens=4000,
        max_iterations=30,
        max_planning_iterations=5,
        max_execution_iterations=10,
        max_refinement_iterations=3,
        max_tool_retries=2,
        tools_per_iteration=1,
    )

    print("Agent initialized successfully!")

except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
