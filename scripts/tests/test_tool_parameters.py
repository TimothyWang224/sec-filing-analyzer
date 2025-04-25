"""
Test script for tool parameter handling.

This script isolates the tool parameter handling process to understand
how parameters flow from the agent to the tools.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.agents.financial_analyst import FinancialAnalystAgent
from src.capabilities.planning import PlanningCapability
from src.environments.base import Environment
from src.llm.openai import OpenAILLM
from src.tools.base import Tool
from src.tools.registry import ToolRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("scripts/tests/outputs/tool_parameters_test.log")],
)
logger = logging.getLogger(__name__)


# Create a simple test tool
class TestFinancialDataTool(Tool):
    """Tool for testing financial data retrieval."""

    name = "test_financial_data"
    description = "Retrieve financial data for testing purposes"
    tags = ["test", "financial"]

    async def execute(self, query_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the test financial data tool.

        Args:
            query_type: Type of query to execute (revenue, profit, metrics)
            parameters: Parameters for the query, must include ticker

        Returns:
            Dictionary containing the query results
        """
        logger.info(f"TestFinancialDataTool.execute called with query_type={query_type}, parameters={parameters}")

        # Validate parameters
        if not self.validate_args(query_type, parameters):
            return {"error": "Invalid parameters", "query_type": query_type, "parameters": parameters}

        # Process based on query_type
        if query_type == "revenue":
            return {
                "query_type": query_type,
                "parameters": parameters,
                "result": {"ticker": parameters.get("ticker"), "revenue": {"2022": 100000000, "2023": 120000000}},
            }
        elif query_type == "profit":
            return {
                "query_type": query_type,
                "parameters": parameters,
                "result": {"ticker": parameters.get("ticker"), "profit": {"2022": 20000000, "2023": 25000000}},
            }
        elif query_type == "metrics":
            return {
                "query_type": query_type,
                "parameters": parameters,
                "result": {"ticker": parameters.get("ticker"), "metrics": {"pe_ratio": 15.2, "debt_to_equity": 0.8}},
            }
        else:
            return {
                "error": f"Unsupported query_type: {query_type}",
                "query_type": query_type,
                "parameters": parameters,
            }

    def validate_args(self, query_type: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate the tool arguments.

        Args:
            query_type: Type of query to execute
            parameters: Parameters for the query

        Returns:
            True if arguments are valid, False otherwise
        """
        logger.info(f"TestFinancialDataTool.validate_args called with query_type={query_type}, parameters={parameters}")

        # Validate query_type
        valid_query_types = ["revenue", "profit", "metrics"]
        if query_type not in valid_query_types:
            logger.error(f"Invalid query_type: {query_type}. Must be one of {valid_query_types}")
            return False

        # Validate parameters
        if parameters is None:
            logger.error("Parameters cannot be None")
            return False

        # Check for required ticker parameter
        if "ticker" not in parameters:
            logger.error("Missing required parameter: ticker")
            return False

        return True


# Create a custom environment that logs tool execution
class LoggingEnvironment(Environment):
    """Environment that logs tool execution for debugging."""

    async def execute_action(self, action: Dict[str, Any]) -> Any:
        """
        Execute an action in the environment with detailed logging.

        Args:
            action: The action to execute

        Returns:
            The result of the action
        """
        tool_name = action.get("tool")
        args = action.get("args", {})

        logger.info(
            f"LoggingEnvironment.execute_action called with tool={tool_name}, args={json.dumps(args, indent=2)}"
        )

        try:
            result = await super().execute_action(action)
            logger.info(f"Tool execution result: {json.dumps(result, indent=2) if result is not None else 'None'}")
            return result
        except Exception as e:
            logger.error(f"Error executing tool: {str(e)}")
            raise


# Create a custom planning capability that logs parameter handling
class LoggingPlanningCapability(PlanningCapability):
    """Planning capability that logs parameter handling."""

    async def process_action(self, agent: Any, context: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an action with detailed logging.

        Args:
            agent: The agent processing the action
            context: Current context
            action: Action to process

        Returns:
            Processed action
        """
        logger.info(f"LoggingPlanningCapability.process_action called with action={json.dumps(action, indent=2)}")

        # Call the parent method
        processed_action = await super().process_action(agent, context, action)

        logger.info(f"Processed action: {json.dumps(processed_action, indent=2)}")
        return processed_action


# Register the test tool
ToolRegistry._register_tool(TestFinancialDataTool, name="test_financial_data", tags=["test", "financial"])


async def main():
    """Run the test script."""
    # Create an LLM instance
    llm = OpenAILLM(model="gpt-4o-mini", temperature=0.3, max_tokens=1000)

    # Create a logging planning capability
    planning_capability = LoggingPlanningCapability(
        enable_dynamic_replanning=False, enable_step_reflection=False, max_plan_steps=3, plan_detail_level="high"
    )

    # Create a logging environment
    environment = LoggingEnvironment()

    # Create a financial analyst agent with the logging planning capability
    agent = FinancialAnalystAgent(
        capabilities=[planning_capability],
        llm_model="gpt-4o-mini",
        llm_temperature=0.3,
        llm_max_tokens=1000,
        max_iterations=3,
        environment=environment,
    )

    # Simple input that should trigger the test_financial_data tool
    user_input = "Analyze Apple's revenue growth for the last 2 years"

    print(f"Starting test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {user_input}")
    print("-" * 80)

    # Run the agent
    result = await agent.run(user_input)

    # Save the result to a file
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"tool_parameters_result_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Test completed. Results saved to {output_file}")
    print("-" * 80)

    # Print a summary of the tool calls
    tool_calls = result.get("tool_calls", [])
    print(f"Number of tool calls: {len(tool_calls)}")

    for i, call in enumerate(tool_calls):
        print(f"\nTool Call {i + 1}:")
        print(f"Tool: {call.get('tool')}")
        print(f"Args: {json.dumps(call.get('args', {}), indent=2)}")
        print(f"Success: {call.get('success', False)}")
        if not call.get("success", False):
            print(f"Error: {call.get('error', 'Unknown error')}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
