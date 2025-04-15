"""
Test script for the enhanced error handling and recovery.

This script demonstrates the enhanced error handling and recovery capabilities
of the agent system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.core.error_handling import ToolErrorType, ToolError, ErrorClassifier
from src.agents.core.adaptive_retry import AdaptiveRetryStrategy
from src.agents.core.alternative_tools import AlternativeToolSelector
from src.agents.core.error_recovery import ErrorRecoveryManager
from src.agents.base import Agent, Goal
from src.tools.registry import ToolRegistry
from sec_filing_analyzer.llm import OpenAILLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestTool:
    """Test tool that can be configured to fail in different ways."""
    
    def __init__(self, name: str, behavior: str = "success"):
        """
        Initialize the test tool.
        
        Args:
            name: Name of the tool
            behavior: Behavior of the tool (success, parameter_error, network_error, etc.)
        """
        self.name = name
        self.behavior = behavior
        self.call_count = 0
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with the given arguments.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool result
            
        Raises:
            Exception: If the tool is configured to fail
        """
        self.call_count += 1
        
        # Simulate different behaviors
        if self.behavior == "success":
            return {"result": f"Success! Called {self.call_count} times with {kwargs}"}
        elif self.behavior == "parameter_error":
            raise ValueError(f"Invalid parameter: {kwargs}")
        elif self.behavior == "network_error":
            raise ConnectionError("Failed to connect to the server")
        elif self.behavior == "rate_limit_error":
            raise Exception("Rate limit exceeded. Try again later.")
        elif self.behavior == "auth_error":
            raise PermissionError("Authentication failed")
        elif self.behavior == "data_error":
            raise KeyError("Data not found")
        elif self.behavior == "system_error":
            raise RuntimeError("Internal system error")
        elif self.behavior == "success_after_retry" and self.call_count > 2:
            return {"result": f"Success after {self.call_count} retries!"}
        elif self.behavior == "success_after_retry":
            raise ConnectionError(f"Temporary failure (attempt {self.call_count})")
        else:
            raise Exception(f"Unknown behavior: {self.behavior}")


async def test_error_classifier():
    """Test the error classifier."""
    logger.info("=== Testing Error Classifier ===")
    
    # Create test exceptions
    exceptions = [
        ValueError("Missing required parameter: ticker"),
        ConnectionError("Failed to connect to the server"),
        PermissionError("Authentication failed"),
        Exception("Rate limit exceeded. Try again later."),
        KeyError("Data not found"),
        RuntimeError("Internal system error"),
        Exception("Unknown error")
    ]
    
    # Classify each exception
    for exception in exceptions:
        error = ErrorClassifier.classify_error(exception, "test_tool")
        logger.info(f"Exception: {type(exception).__name__}: {str(exception)}")
        logger.info(f"Classified as: {error.error_type.value}")
        logger.info(f"Recoverable: {error.is_recoverable()}")
        logger.info(f"Recovery strategy: {error.get_recovery_strategy()}")
        logger.info("")


async def test_adaptive_retry():
    """Test the adaptive retry strategy."""
    logger.info("=== Testing Adaptive Retry Strategy ===")
    
    # Create a retry strategy
    retry_strategy = AdaptiveRetryStrategy(base_delay=0.1, max_delay=1.0)
    
    # Test with a tool that succeeds immediately
    success_tool = TestTool("success_tool", "success")
    
    async def execute_success():
        return await success_tool.execute(param="value")
        
    def classify_error(exception):
        return ErrorClassifier.classify_error(exception, "success_tool")
        
    result = await retry_strategy.retry_with_strategy(
        execute_success,
        max_retries=3,
        error_classifier=classify_error
    )
    
    logger.info(f"Success tool result: {result}")
    logger.info(f"Success tool call count: {success_tool.call_count}")
    logger.info("")
    
    # Test with a tool that succeeds after retries
    retry_tool = TestTool("retry_tool", "success_after_retry")
    
    async def execute_retry():
        return await retry_tool.execute(param="value")
        
    def classify_retry_error(exception):
        return ErrorClassifier.classify_error(exception, "retry_tool")
        
    result = await retry_strategy.retry_with_strategy(
        execute_retry,
        max_retries=3,
        error_classifier=classify_retry_error
    )
    
    logger.info(f"Retry tool result: {result}")
    logger.info(f"Retry tool call count: {retry_tool.call_count}")
    logger.info("")
    
    # Test with a tool that always fails
    fail_tool = TestTool("fail_tool", "system_error")
    
    async def execute_fail():
        return await fail_tool.execute(param="value")
        
    def classify_fail_error(exception):
        return ErrorClassifier.classify_error(exception, "fail_tool")
        
    result = await retry_strategy.retry_with_strategy(
        execute_fail,
        max_retries=3,
        error_classifier=classify_fail_error
    )
    
    logger.info(f"Fail tool result: {result}")
    logger.info(f"Fail tool call count: {fail_tool.call_count}")
    logger.info("")


async def test_error_recovery_manager():
    """Test the error recovery manager."""
    logger.info("=== Testing Error Recovery Manager ===")
    
    # Create an LLM
    llm = OpenAILLM(model="gpt-4o-mini", temperature=0.2)
    
    # Create an error recovery manager
    recovery_manager = ErrorRecoveryManager(
        llm=llm,
        max_retries=2,
        base_delay=0.1,
        circuit_breaker_threshold=3,
        circuit_breaker_reset_timeout=5
    )
    
    # Create test tools
    success_tool = TestTool("success_tool", "success")
    retry_tool = TestTool("retry_tool", "success_after_retry")
    param_error_tool = TestTool("param_error_tool", "parameter_error")
    network_error_tool = TestTool("network_error_tool", "network_error")
    
    # Test with a tool that succeeds immediately
    async def execute_tool(tool_name, args):
        if tool_name == "success_tool":
            return await success_tool.execute(**args)
        elif tool_name == "retry_tool":
            return await retry_tool.execute(**args)
        elif tool_name == "param_error_tool":
            return await param_error_tool.execute(**args)
        elif tool_name == "network_error_tool":
            return await network_error_tool.execute(**args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    # Test with success tool
    result = await recovery_manager.execute_tool_with_recovery(
        tool_name="success_tool",
        tool_args={"param": "value"},
        execute_func=execute_tool,
        user_input="Test with success tool",
        context={"test": "context"}
    )
    
    logger.info(f"Success tool result: {result}")
    logger.info(f"Success tool call count: {success_tool.call_count}")
    logger.info("")
    
    # Test with retry tool
    result = await recovery_manager.execute_tool_with_recovery(
        tool_name="retry_tool",
        tool_args={"param": "value"},
        execute_func=execute_tool,
        user_input="Test with retry tool",
        context={"test": "context"}
    )
    
    logger.info(f"Retry tool result: {result}")
    logger.info(f"Retry tool call count: {retry_tool.call_count}")
    logger.info("")
    
    # Test with parameter error tool
    result = await recovery_manager.execute_tool_with_recovery(
        tool_name="param_error_tool",
        tool_args={"param": "value"},
        execute_func=execute_tool,
        user_input="Test with parameter error tool",
        context={"test": "context"}
    )
    
    logger.info(f"Parameter error tool result: {json.dumps(result, indent=2, default=str)}")
    logger.info(f"Parameter error tool call count: {param_error_tool.call_count}")
    logger.info("")
    
    # Test with network error tool
    result = await recovery_manager.execute_tool_with_recovery(
        tool_name="network_error_tool",
        tool_args={"param": "value"},
        execute_func=execute_tool,
        user_input="Test with network error tool",
        context={"test": "context"}
    )
    
    logger.info(f"Network error tool result: {json.dumps(result, indent=2, default=str)}")
    logger.info(f"Network error tool call count: {network_error_tool.call_count}")
    logger.info("")


async def main():
    """Main function."""
    logger.info("Testing Enhanced Error Handling and Recovery")
    
    await test_error_classifier()
    await test_adaptive_retry()
    await test_error_recovery_manager()


if __name__ == "__main__":
    asyncio.run(main())
