"""
Test script for the new agent parameters.

This script demonstrates how the new parameters affect agent behavior,
including tool retries, phase-specific iterations, and dynamic termination.
"""

import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse

from src.agents.qa_specialist import QASpecialistAgent
from src.environments.financial import FinancialEnvironment
from src.capabilities.planning import PlanningCapability
from src.capabilities.logging import LoggingCapability

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_agent_parameters_test(
    question: str,
    max_planning_iterations: int = 2,
    max_execution_iterations: int = 3,
    max_refinement_iterations: int = 1,
    max_tool_retries: int = 2,
    tools_per_iteration: int = 1,
    enable_dynamic_termination: bool = False
):
    """Test the new agent parameters."""
    try:
        # Initialize environment
        environment = FinancialEnvironment()
        
        # Initialize capabilities
        planning = PlanningCapability()
        logging_capability = LoggingCapability(
            log_dir="data/logs/agents",
            log_level="INFO",
            log_to_console=True,
            log_to_file=True,
            include_memory=True,
            include_context=True,
            include_actions=True,
            include_results=True,
            include_prompts=True,
            include_responses=True
        )
        
        # Initialize QA specialist agent with planning and logging
        agent = QASpecialistAgent(
            capabilities=[planning, logging_capability],
            environment=environment,
            # Agent iteration parameters
            max_planning_iterations=max_planning_iterations,
            max_execution_iterations=max_execution_iterations,
            max_refinement_iterations=max_refinement_iterations,
            # Tool execution parameters
            max_tool_retries=max_tool_retries,
            tools_per_iteration=tools_per_iteration,
            # Termination parameters
            enable_dynamic_termination=enable_dynamic_termination,
            min_confidence_threshold=0.8
        )
        
        # Set the initial phase
        agent.state.set_phase('planning')
        
        # Run the agent
        logger.info(f"Processing question: {question}")
        result = await agent.run(question)
        
        # Print results
        print("\n=== Agent Parameters Test Results ===")
        print(f"Question: {question}")
        print(f"\nAnswer: {result['answer']}")
        
        # Print phase iterations
        print("\n=== Phase Iterations ===")
        print(f"Planning: {agent.state.phase_iterations['planning']}")
        print(f"Execution: {agent.state.phase_iterations['execution']}")
        print(f"Refinement: {agent.state.phase_iterations['refinement']}")
        print(f"Total: {agent.state.current_iteration}")
        
        # Print tool ledger entries
        print("\n=== Tool Ledger Entries ===")
        for i, entry in enumerate(agent.tool_ledger.entries):
            print(f"\n--- Entry {i+1} ---")
            print(f"Tool: {entry['tool']}")
            print(f"Args: {json.dumps(entry['args'], indent=2)}")
            print(f"Status: {entry['status']}")
            
            if "retries" in entry.get("metadata", {}):
                print(f"Retries: {entry['metadata']['retries']}")
            
            if entry["status"] == "success":
                print(f"Result: {str(entry['result'])[:200]}...")
            else:
                print(f"Error: {entry['error']}")
        
        return result
    except Exception as e:
        logger.error(f"Error running agent parameters test: {str(e)}")
        raise

async def run_phase_transition_demo():
    """Demonstrate phase transitions."""
    # Initialize environment
    environment = FinancialEnvironment()
    
    # Initialize agent
    agent = QASpecialistAgent(
        environment=environment,
        max_planning_iterations=2,
        max_execution_iterations=3,
        max_refinement_iterations=1
    )
    
    # Demonstrate phase transitions
    print("\n=== Phase Transition Demo ===")
    
    # Planning phase
    agent.state.set_phase('planning')
    print(f"Current phase: {agent.state.current_phase}")
    
    for i in range(3):  # Exceeds max_planning_iterations
        agent.state.increment_iteration()
        print(f"Iteration {agent.state.current_iteration}, Planning iterations: {agent.state.phase_iterations['planning']}")
        should_terminate = agent.should_terminate()
        print(f"Should terminate: {should_terminate}")
        
        if should_terminate:
            break
    
    # Execution phase
    agent.state.set_phase('execution')
    print(f"\nCurrent phase: {agent.state.current_phase}")
    
    for i in range(4):  # Exceeds max_execution_iterations
        agent.state.increment_iteration()
        print(f"Iteration {agent.state.current_iteration}, Execution iterations: {agent.state.phase_iterations['execution']}")
        should_terminate = agent.should_terminate()
        print(f"Should terminate: {should_terminate}")
        
        if should_terminate:
            break
    
    # Refinement phase
    agent.state.set_phase('refinement')
    print(f"\nCurrent phase: {agent.state.current_phase}")
    
    for i in range(2):  # Exceeds max_refinement_iterations
        agent.state.increment_iteration()
        print(f"Iteration {agent.state.current_iteration}, Refinement iterations: {agent.state.phase_iterations['refinement']}")
        should_terminate = agent.should_terminate()
        print(f"Should terminate: {should_terminate}")
        
        if should_terminate:
            break

def main():
    """Main function to run the test script."""
    parser = argparse.ArgumentParser(description="Test the new agent parameters")
    parser.add_argument("--question", type=str, default="What was Apple's revenue in 2023?",
                        help="Question to process")
    parser.add_argument("--max-planning-iterations", type=int, default=2,
                        help="Maximum planning iterations")
    parser.add_argument("--max-execution-iterations", type=int, default=3,
                        help="Maximum execution iterations")
    parser.add_argument("--max-refinement-iterations", type=int, default=1,
                        help="Maximum refinement iterations")
    parser.add_argument("--max-tool-retries", type=int, default=2,
                        help="Maximum tool retries")
    parser.add_argument("--tools-per-iteration", type=int, default=1,
                        help="Tools per iteration")
    parser.add_argument("--enable-dynamic-termination", action="store_true",
                        help="Enable dynamic termination")
    parser.add_argument("--demo-phases", action="store_true",
                        help="Run phase transition demo")
    
    args = parser.parse_args()
    
    if args.demo_phases:
        asyncio.run(run_phase_transition_demo())
    else:
        asyncio.run(run_agent_parameters_test(
            question=args.question,
            max_planning_iterations=args.max_planning_iterations,
            max_execution_iterations=args.max_execution_iterations,
            max_refinement_iterations=args.max_refinement_iterations,
            max_tool_retries=args.max_tool_retries,
            tools_per_iteration=args.tools_per_iteration,
            enable_dynamic_termination=args.enable_dynamic_termination
        ))

if __name__ == "__main__":
    main()
