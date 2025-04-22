"""
Comprehensive error recovery for agent tool calls.

This module provides a comprehensive error recovery manager that combines
various recovery strategies to handle tool call failures.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, Awaitable, List, Tuple

from sec_filing_analyzer.llm import BaseLLM
from ...tools.registry import ToolRegistry
from ...tools.llm_parameter_completer import LLMParameterCompleter
from .error_handling import ToolError, ToolErrorType, ErrorClassifier, ErrorAnalyzer, ToolCircuitBreaker
from .adaptive_retry import AdaptiveRetryStrategy
from .alternative_tools import AlternativeToolSelector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorRecoveryManager:
    """
    Comprehensive error recovery manager for agent tool calls.

    This class combines various recovery strategies to handle tool call failures,
    including parameter fixing, adaptive retries, circuit breaking, and alternative
    tool selection.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_retries: int = 2,
        base_delay: float = 1.0,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_reset_timeout: int = 300
    ):
        """
        Initialize the error recovery manager.

        Args:
            llm: LLM instance to use for recovery strategies
            max_retries: Maximum number of retries for tool calls
            base_delay: Base delay in seconds for retry strategies
            circuit_breaker_threshold: Number of consecutive failures before opening the circuit
            circuit_breaker_reset_timeout: Time in seconds before attempting to reset the circuit
        """
        self.llm = llm
        self.max_retries = max_retries

        # Initialize components
        self.parameter_completer = LLMParameterCompleter(llm)
        self.retry_strategy = AdaptiveRetryStrategy(base_delay=base_delay)
        self.circuit_breaker = ToolCircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            reset_timeout=circuit_breaker_reset_timeout
        )
        self.alternative_selector = AlternativeToolSelector(llm)
        self.error_analyzer = ErrorAnalyzer()

    async def execute_tool_with_recovery(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        execute_func: Callable[[str, Dict[str, Any]], Awaitable[Any]],
        user_input: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool with comprehensive error recovery.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Tool arguments
            execute_func: Function to execute the tool
            user_input: User input for parameter completion
            context: Additional context for recovery

        Returns:
            Dictionary with execution result or error information
        """
        # Check if the circuit is closed for this tool
        if not self.circuit_breaker.can_execute(tool_name):
            logger.warning(f"Circuit open for tool {tool_name}, skipping execution")
            circuit_status = self.circuit_breaker.get_status(tool_name)

            # Create a circuit open error
            error = ToolError(
                ToolErrorType.SYSTEM_ERROR,
                f"Circuit open for tool {tool_name} due to repeated failures",
                tool_name
            )

            return {
                "success": False,
                "error": error,
                "circuit_status": circuit_status,
                "user_message": self.error_analyzer.format_error_for_user(error)
            }

        # Create a function to execute the tool with the current arguments
        async def execute_with_current_args():
            return await execute_func(tool_name, tool_args)

        # Create an error classifier function
        def classify_error(exception):
            return ErrorClassifier.classify_error(exception, tool_name)

        # Execute the tool with adaptive retry
        result = await self.retry_strategy.retry_with_strategy(
            execute_with_current_args,
            self.max_retries,
            classify_error
        )

        # If successful, record success and return result
        if result["success"]:
            self.circuit_breaker.record_success(tool_name)
            return result

        # If failed, start recovery process
        error = result["error"]
        self.error_analyzer.add_error(error)
        self.circuit_breaker.record_failure(tool_name)

        # Check if we had identical errors during retry attempts
        if result.get("identical_errors", False):
            logger.warning("Skipping parameter fixing due to identical errors in consecutive attempts")
            # Create enhanced context with schema information
            enhanced_context = context.copy() if context else {}
            enhanced_context["last_error"] = str(error)
            enhanced_context["tool_schema"] = ToolRegistry.get_schema(tool_name)
            enhanced_context["identical_errors"] = True

            # Try recovery strategies with enhanced context
            recovery_result = await self._apply_recovery_strategies(
                tool_name,
                tool_args,
                error,
                execute_func,
                user_input,
                enhanced_context
            )
        else:
            # Try normal recovery strategies
            recovery_result = await self._apply_recovery_strategies(
                tool_name,
                tool_args,
                error,
                execute_func,
                user_input,
                context
            )

        # If recovery succeeded, return the result
        if recovery_result["success"]:
            self.circuit_breaker.record_success(tool_name)
            return recovery_result

        # If all recovery strategies failed, return error with suggestions
        suggestions = self.error_analyzer.get_error_suggestions(tool_name, error.error_type)
        user_message = self.error_analyzer.format_error_for_user(error)

        return {
            "success": False,
            "error": error,
            "user_message": user_message,
            "suggestions": suggestions,
            "recovery_attempted": True
        }

    async def _apply_recovery_strategies(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        error: ToolError,
        execute_func: Callable[[str, Dict[str, Any]], Awaitable[Any]],
        user_input: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply various recovery strategies to handle a tool error.

        Args:
            tool_name: Name of the failed tool
            tool_args: Original tool arguments
            error: Tool error
            execute_func: Function to execute the tool
            user_input: User input for parameter completion
            context: Additional context for recovery

        Returns:
            Dictionary with recovery result
        """
        # Strategy 1: Fix parameters if it's a parameter error
        if error.error_type == ToolErrorType.PARAMETER_ERROR:
            logger.info(f"Attempting to fix parameters for tool: {tool_name}")

            # Create context with error information
            error_context = context.copy() if context else {}
            error_context["last_error"] = str(error)

            try:
                # Complete parameters using the LLM with error context
                fixed_args = await self.parameter_completer.complete_parameters(
                    tool_name=tool_name,
                    partial_parameters=tool_args,
                    user_input=user_input,
                    context=error_context
                )

                # If parameters were fixed, try again
                if fixed_args != tool_args:
                    logger.info(f"Parameters fixed: {tool_args} -> {fixed_args}")

                    # Execute with fixed parameters
                    async def execute_with_fixed_args():
                        return await execute_func(tool_name, fixed_args)

                    # Create an error classifier function
                    def classify_error(exception):
                        return ErrorClassifier.classify_error(exception, tool_name)

                    # Execute with fixed parameters
                    result = await self.retry_strategy.retry_with_strategy(
                        execute_with_fixed_args,
                        1,  # Only try once with fixed parameters
                        classify_error
                    )

                    if result["success"]:
                        result["recovery_strategy"] = "parameter_fixing"
                        return result
            except Exception as e:
                logger.error(f"Error fixing parameters: {str(e)}")

        # Strategy 2: Try an alternative tool
        try:
            # Get the purpose of the original tool
            tool_purpose = self._get_tool_purpose(tool_name, tool_args)

            # Find an alternative tool
            alternative_tool = await self.alternative_selector.find_alternative_tool(
                tool_name,
                tool_purpose
            )

            if alternative_tool:
                logger.info(f"Trying alternative tool: {alternative_tool}")

                # Map parameters to the alternative tool
                mapped_args = await self.alternative_selector.map_parameters(
                    tool_name,
                    alternative_tool,
                    tool_args
                )

                # Execute the alternative tool
                async def execute_alternative():
                    return await execute_func(alternative_tool, mapped_args)

                # Create an error classifier function
                def classify_error(exception):
                    return ErrorClassifier.classify_error(exception, alternative_tool)

                # Execute the alternative tool
                result = await self.retry_strategy.retry_with_strategy(
                    execute_alternative,
                    1,  # Only try once with the alternative tool
                    classify_error
                )

                if result["success"]:
                    result["recovery_strategy"] = "alternative_tool"
                    result["alternative_tool"] = alternative_tool
                    return result
        except Exception as e:
            logger.error(f"Error trying alternative tool: {str(e)}")

        # If all recovery strategies fail, return failure
        return {
            "success": False,
            "error": error,
            "recovery_attempted": True
        }

    def _get_tool_purpose(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Get a description of the purpose of a tool call.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments

        Returns:
            Description of the tool's purpose
        """
        # Get tool documentation
        tool_info = ToolRegistry.get(tool_name)
        if not tool_info:
            return f"Using {tool_name} with arguments {tool_args}"

        description = tool_info.get("description", "")

        # Create a purpose description
        purpose = f"Using {tool_name} to {description.lower()}"

        # Add key arguments if available
        if tool_args:
            arg_str = ", ".join([f"{k}={v}" for k, v in tool_args.items() if k != "parameters"])
            purpose += f" with {arg_str}"

        return purpose
