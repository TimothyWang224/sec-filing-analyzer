import sys
from pathlib import Path
import logging

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from src.contracts import Plan, PlanStep
    from datetime import datetime
    
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
        status="pending"
    )
    
    plan = Plan(
        goal="Test plan",
        steps=[plan_step],
        status="pending",
        created_at=datetime.now().isoformat(),
        owner="agent",
        can_modify=True
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
    
    # Test the PlanningCapability class
    from src.capabilities.planning import PlanningCapability
    
    # Create a mock agent
    class MockAgent:
        def __init__(self):
            self.state = type('obj', (object,), {
                'current_phase': 'planning',
                'count_tokens': lambda x, y: None
            })
            self.llm = type('obj', (object,), {
                'generate': lambda **kwargs: {'content': '{}', 'usage': {'total_tokens': 100}}
            })
            self.add_to_memory = lambda x: None
            self.get_memory = lambda: []
            self.environment = type('obj', (object,), {
                'get_available_tools': lambda: {'test_tool': {}}
            })
    
    # Create a planning capability
    planning = PlanningCapability()
    
    # Initialize the capability
    agent = MockAgent()
    context = {}
    planning.agent = agent
    planning.context = context
    
    # Set the current plan
    planning.current_plan = plan
    
    # Test accessing the plan
    print(f"Planning capability plan goal: {planning.current_plan.goal}")
    print(f"Planning capability plan steps: {len(planning.current_plan.steps)}")
    
    # Test updating the context
    planning.current_step_index = 0
    context = {}
    planning._update_context(context)
    
    print(f"Context has plan: {context['planning']['has_plan']}")
    print(f"Context plan goal: {context['planning']['plan'].goal}")
    print(f"Context current step: {context['planning']['current_step'].description}")
    
    print("All tests passed!")
    
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
