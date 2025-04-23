import asyncio
import logging
import json
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath("."))

from src.agents.qa_specialist import QASpecialistAgent
from src.environments import Environment

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_token_budgeting():
    """Test the token budgeting system with the QA Specialist Agent."""

    # Create environment
    environment = Environment()

    # Create QA Specialist Agent
    agent = QASpecialistAgent(
        environment=environment,
        llm_model="gpt-4o-mini",  # Use a smaller model for testing
        llm_temperature=0.5,
        # Keep iteration limits as a safety mechanism
        max_planning_iterations=2,
        max_execution_iterations=5,
        max_refinement_iterations=3
    )

    # Manually set token budgets in the agent's state
    agent.max_total_tokens = 5000
    agent.state.token_budget = {
        "planning": 1500,    # 30% for planning
        "execution": 1500,   # 30% for execution
        "refinement": 2000   # 40% for refinement
    }

    # Run the agent with a simple question
    user_query = "What was Microsoft's revenue in 2023?"

    print(f"\n\n{'='*80}")
    print(f"Running QA Specialist Agent with query: {user_query}")
    print(f"{'='*80}\n")

    result = await agent.run(user_query)

    # Print the result
    print(f"\n\n{'='*80}")
    print("RESULT:")
    print(f"{'='*80}")
    print(json.dumps(result, indent=2))

    # Print token usage
    print(f"\n\n{'='*80}")
    print("TOKEN USAGE:")
    print(f"{'='*80}")
    print(f"Planning:   {agent.state.tokens_used['planning']} / {agent.state.token_budget['planning']} tokens")
    print(f"Execution:  {agent.state.tokens_used['execution']} / {agent.state.token_budget['execution']} tokens")
    print(f"Refinement: {agent.state.tokens_used['refinement']} / {agent.state.token_budget['refinement']} tokens")

    # Calculate total tokens used
    total_used = sum(agent.state.tokens_used.values())
    total_budget = sum(agent.state.token_budget.values())
    print(f"Total:      {total_used} / {total_budget} tokens")

    # Print phase iterations
    print(f"\n\n{'='*80}")
    print("PHASE ITERATIONS:")
    print(f"{'='*80}")
    print(f"Planning:   {agent.state.phase_iterations['planning']} / {agent.max_planning_iterations}")
    print(f"Execution:  {agent.state.phase_iterations['execution']} / {agent.max_execution_iterations}")
    print(f"Refinement: {agent.state.phase_iterations['refinement']} / {agent.max_refinement_iterations}")

    return result

if __name__ == "__main__":
    asyncio.run(test_token_budgeting())
