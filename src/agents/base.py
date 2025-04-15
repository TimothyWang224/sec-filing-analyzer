from typing import List, Dict, Any, Optional, Tuple, Union, Type
import json
import re
import logging
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .core.agent_state import AgentState
from .core.tool_ledger import ToolLedger
from .core.error_handling import ToolError, ToolErrorType, ErrorClassifier, ErrorAnalyzer, ToolCircuitBreaker
from .core.adaptive_retry import AdaptiveRetryStrategy
from .core.alternative_tools import AlternativeToolSelector
from .core.error_recovery import ErrorRecoveryManager
from ..environments.base import Environment
from ..tools.registry import ToolRegistry
from ..tools.llm_parameter_completer import LLMParameterCompleter
from sec_filing_analyzer.llm import BaseLLM, OpenAILLM
from sec_filing_analyzer.llm.llm_config import LLMConfigFactory
from ..sec_filing_analyzer.utils.logging_utils import get_standard_log_dir
from ..sec_filing_analyzer.utils.timing import timed_function, TimingContext

# Try to import the ConfigProvider
try:
    from sec_filing_analyzer.config import ConfigProvider, AgentConfig
    HAS_CONFIG_PROVIDER = True
except ImportError:
    HAS_CONFIG_PROVIDER = False

logger = logging.getLogger(__name__)

@dataclass
class Goal:
    """Represents a goal for an agent to achieve."""
    name: str
    description: str

class Agent(ABC):
    """Base class for all agents in the system with integrated LLM tool calling."""

    def __init__(
        self,
        goals: List[Goal],
        capabilities: List[Any] = None,
        # Agent iteration parameters
        max_iterations: Optional[int] = None,  # Legacy parameter, still used for backward compatibility
        max_planning_iterations: Optional[int] = None,
        max_execution_iterations: Optional[int] = None,
        max_refinement_iterations: Optional[int] = None,
        # Tool execution parameters
        max_tool_retries: Optional[int] = None,
        tools_per_iteration: Optional[int] = None,  # Default to 1 for single tool call approach
        # Runtime parameters
        max_duration_seconds: Optional[int] = None,
        # LLM parameters
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
        # Environment
        environment: Optional[Environment] = None,
        # Termination parameters
        enable_dynamic_termination: Optional[bool] = None,
        min_confidence_threshold: Optional[float] = None,
        # Agent type for configuration
        agent_type: Optional[str] = None
    ):
        """
        Initialize an agent with its goals and capabilities.

        Args:
            goals: List of goals the agent aims to achieve
            capabilities: List of capabilities that extend agent behavior

            # Agent iteration parameters
            max_iterations: Legacy parameter for backward compatibility
            max_planning_iterations: Maximum iterations for the planning phase
            max_execution_iterations: Maximum iterations for the execution phase
            max_refinement_iterations: Maximum iterations for result refinement

            # Tool execution parameters
            max_tool_retries: Number of times to retry a failed tool call
            tools_per_iteration: Number of tools to execute per iteration

            # Runtime parameters
            max_duration_seconds: Maximum runtime in seconds

            # LLM parameters
            llm_model: LLM model to use
            llm_temperature: Temperature for LLM generation
            llm_max_tokens: Maximum tokens for LLM generation

            # Environment
            environment: Optional environment to use

            # Termination parameters
            enable_dynamic_termination: Whether to allow early termination
            min_confidence_threshold: Minimum confidence score for satisfactory results
        """
        self.goals = goals
        self.capabilities = capabilities or []

        # Get configuration based on agent type if available
        config = self._get_config(agent_type)

        # Set iteration parameters with fallbacks
        self.max_iterations = max_iterations if max_iterations is not None else config.get("max_iterations", 3)  # Legacy parameter
        self.max_planning_iterations = max_planning_iterations if max_planning_iterations is not None else config.get("max_planning_iterations", 2)
        self.max_execution_iterations = max_execution_iterations if max_execution_iterations is not None else config.get("max_execution_iterations", 3)
        self.max_refinement_iterations = max_refinement_iterations if max_refinement_iterations is not None else config.get("max_refinement_iterations", 1)

        # Set tool execution parameters with fallbacks
        self.max_tool_retries = max_tool_retries if max_tool_retries is not None else config.get("max_tool_retries", 2)
        self.tools_per_iteration = tools_per_iteration if tools_per_iteration is not None else config.get("tools_per_iteration", 1)

        # Set runtime parameters with fallbacks
        self.max_duration_seconds = max_duration_seconds if max_duration_seconds is not None else config.get("max_duration_seconds", 180)

        # Set termination parameters with fallbacks
        self.enable_dynamic_termination = enable_dynamic_termination if enable_dynamic_termination is not None else config.get("enable_dynamic_termination", False)
        self.min_confidence_threshold = min_confidence_threshold if min_confidence_threshold is not None else config.get("min_confidence_threshold", 0.8)

        # Get LLM parameters with fallbacks
        llm_model = llm_model if llm_model is not None else config.get("model", config.get("llm_model", "gpt-4o-mini"))
        llm_temperature = llm_temperature if llm_temperature is not None else config.get("temperature", config.get("llm_temperature", 0.7))
        llm_max_tokens = llm_max_tokens if llm_max_tokens is not None else config.get("max_tokens", config.get("llm_max_tokens", 4000))

        # Initialize state and tool ledger
        self.state = AgentState()
        self.tool_ledger = ToolLedger()

        # Initialize LLM
        self.llm = OpenAILLM(
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens
        )

        # Initialize parameter completer
        self.parameter_completer = LLMParameterCompleter(self.llm)

        # Initialize error recovery manager
        self.error_recovery_manager = ErrorRecoveryManager(
            llm=self.llm,
            max_retries=self.max_tool_retries,
            base_delay=1.0,
            circuit_breaker_threshold=3,
            circuit_breaker_reset_timeout=300
        )

        # Initialize environment
        self.environment = environment or Environment()

        # Set up session ID for logging
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Set up basic logging
        self.logger = self._setup_basic_logger()

    def _get_config(self, agent_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for this agent.

        Args:
            agent_type: Type of agent to get configuration for

        Returns:
            Dictionary containing agent configuration
        """
        # Try to use ConfigProvider if available
        if HAS_CONFIG_PROVIDER:
            try:
                if agent_type:
                    return ConfigProvider.get_agent_config(agent_type)
                else:
                    # Use the class name as the agent type if not specified
                    class_name = self.__class__.__name__
                    # Convert CamelCase to snake_case and remove 'Agent' suffix
                    agent_type = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
                    agent_type = agent_type.replace('_agent', '')
                    return ConfigProvider.get_agent_config(agent_type)
            except Exception as e:
                logger.warning(f"Error getting config from ConfigProvider: {str(e)}")

        # Try to use LLMConfigFactory if available
        try:
            if agent_type:
                return LLMConfigFactory.create_config(agent_type)
            else:
                # Use default config
                return {}
        except Exception as e:
            logger.warning(f"Error getting config from LLMConfigFactory: {str(e)}")

        # Return empty config as fallback
        return {}

    async def select_tools(self, input_text: str) -> List[Dict[str, Any]]:
        """
        Use LLM to select which tools to call and what parameters to pass.

        Args:
            input_text: User's input text

        Returns:
            List of tool call specifications
        """
        # Store the input text in the agent state for parameter completion
        self.state.update_context({"input": input_text})

        # Get compact tool documentation
        compact_tool_docs = ToolRegistry.get_compact_tool_documentation(format="text")

        # Create prompt for initial tool selection
        initial_prompt = f"""
        Question: {input_text}

        Available Tools:
        {compact_tool_docs}

        Based on the question, which tool(s) might be useful to answer it?
        If you need more details about a specific tool, you can use the tool_details tool.

        Return your answer as a JSON array of tool names or tool detail requests.
        For example:
        ```json
        [
          {{"tool": "tool_details", "args": {{"tool_name": "sec_semantic_search"}}}},
          {{"tool": "sec_financial_data", "args": {{"query_type": "financial_facts"}}}}
        ]
        ```
        """

        # Generate initial tool selection
        system_prompt = """You are an expert at selecting the right tools to answer questions about SEC filings and financial data.
        Your task is to analyze the question and identify which tools might be useful.
        You can request more details about tools you think might be relevant.
        """

        try:
            # Get initial tool selection
            self.logger.info(f"Selecting tools for input: {input_text}")
            initial_response = await self.llm.generate(
                prompt=initial_prompt,
                system_prompt=system_prompt,
                temperature=0.2
            )

            # Parse tool calls from response
            initial_tool_calls = self._parse_tool_calls(initial_response)
            self.logger.info(f"Initial tool selection: {initial_tool_calls}")

            # Process tool detail requests
            final_tool_calls = []
            tool_details = {}

            for call in initial_tool_calls:
                tool_name = call.get("tool")

                if tool_name == "tool_details":
                    # Get details for the requested tool
                    requested_tool = call.get("args", {}).get("tool_name")
                    if requested_tool:
                        self.logger.info(f"Requesting details for tool: {requested_tool}")
                        tool_details[requested_tool] = await self.environment.execute_action({
                            "tool": "tool_details",
                            "args": {"tool_name": requested_tool}
                        })
                else:
                    # Add to final tool calls
                    final_tool_calls.append(call)

            # If we have tool details, make a second call to get specific parameters
            if tool_details:
                # Create prompt with detailed tool information
                details_prompt = f"""
                Question: {input_text}

                You previously requested details about these tools:

                {json.dumps(tool_details, indent=2)}

                Now, specify the exact tool calls with all necessary parameters.
                Return your answer as a JSON array of tool calls.
                For example:
                ```json
                [
                  {{"tool": "sec_semantic_search", "args": {{"query": "Apple's financial performance", "companies": ["AAPL"]}}}}
                ]
                ```
                """

                self.logger.info(f"Making second call for detailed tool selection")
                # Generate final tool selection with parameters
                details_response = await self.llm.generate(
                    prompt=details_prompt,
                    system_prompt=system_prompt,
                    temperature=0.2
                )

                # Parse final tool calls
                detailed_tool_calls = self._parse_tool_calls(details_response)
                self.logger.info(f"Detailed tool selection: {detailed_tool_calls}")

                # Add detailed tool calls to final list
                final_tool_calls.extend(detailed_tool_calls)

            return final_tool_calls
        except Exception as e:
            self.logger.error(f"Error selecting tools: {str(e)}")
            return []

    @timed_function(category="tool_execution")
    async def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute tool calls based on the tools_per_iteration parameter.

        Args:
            tool_calls: List of tool call specifications

        Returns:
            List containing the results of the executed tool calls
        """
        results = []

        # If no tool calls, return empty results
        if not tool_calls:
            return results

        # Limit the number of tool calls to execute based on tools_per_iteration
        tool_calls_to_execute = tool_calls[:self.tools_per_iteration]

        for tool_call in tool_calls_to_execute:
            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("args", {})

            # Complete tool parameters using the parameter completer
            try:
                # Get the current user input from state context
                user_input = self.state.get_context().get("input", "")

                # Complete parameters using the LLM
                self.logger.info(f"Completing parameters for tool: {tool_name}")
                completed_args = await self.parameter_completer.complete_parameters(
                    tool_name=tool_name,
                    partial_parameters=tool_args,
                    user_input=user_input,
                    context=self.state.get_context()
                )

                # Update tool arguments with completed parameters
                if completed_args != tool_args:
                    self.logger.info(f"Parameters completed: {tool_args} -> {completed_args}")
                    tool_args = completed_args
            except Exception as e:
                self.logger.error(f"Error completing parameters: {str(e)}")
                # Continue with original parameters if completion fails

            self.logger.info(f"Executing tool call: {tool_name}")
            self.logger.info(f"Tool arguments: {tool_args}")

            # Define a function to execute the tool
            async def execute_tool(tool_name, args):
                return await self.environment.execute_action({
                    "tool": tool_name,
                    "args": args
                })

            # Get the current user input from state context
            user_input = self.state.get_context().get("input", "")

            # Execute the tool with comprehensive error recovery
            tool_start_time = time.time()
            recovery_result = await self.error_recovery_manager.execute_tool_with_recovery(
                tool_name=tool_name,
                tool_args=tool_args,
                execute_func=execute_tool,
                user_input=user_input,
                context=self.state.get_context()
            )
            tool_duration = time.time() - tool_start_time

            # Process the result based on success/failure
            if recovery_result["success"]:
                # Create result object
                result_obj = {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": recovery_result["result"],
                    "success": True,
                    "duration": tool_duration
                }

                # Add recovery strategy information if available
                if "recovery_strategy" in recovery_result:
                    result_obj["recovery_strategy"] = recovery_result["recovery_strategy"]

                # Add alternative tool information if available
                if "alternative_tool" in recovery_result:
                    result_obj["alternative_tool"] = recovery_result["alternative_tool"]

                # Add to results list
                results.append(result_obj)

                # Record in tool ledger
                self.tool_ledger.record_tool_call(
                    tool_name=tool_name,
                    args=tool_args,
                    result=recovery_result["result"],
                    metadata={
                        "duration": tool_duration,
                        "retries": recovery_result.get("retries", 0),
                        "recovery_strategy": recovery_result.get("recovery_strategy", None)
                    }
                )

                # Add to agent memory for future reference
                self.add_to_memory({
                    "type": "tool_result",
                    "tool": tool_name,
                    "args": tool_args,
                    "result": recovery_result["result"],
                    "timestamp": datetime.now().isoformat(),
                    "recovery_info": {
                        "strategy": recovery_result.get("recovery_strategy", None),
                        "alternative_tool": recovery_result.get("alternative_tool", None)
                    }
                })
            else:
                # Get error information
                error = recovery_result["error"]
                error_message = error.message if hasattr(error, "message") else str(error)
                error_type = error.error_type.value if hasattr(error, "error_type") else "unknown_error"

                # Get user-friendly error message
                user_message = recovery_result.get("user_message", error_message)

                # Create error result object
                error_obj = {
                    "tool": tool_name,
                    "args": tool_args,
                    "error": error_message,
                    "error_type": error_type,
                    "user_message": user_message,
                    "success": False,
                    "duration": tool_duration
                }

                # Add suggestions if available
                if "suggestions" in recovery_result:
                    error_obj["suggestions"] = recovery_result["suggestions"]

                # Add circuit status if available
                if "circuit_status" in recovery_result:
                    error_obj["circuit_status"] = recovery_result["circuit_status"]

                # Add to results list
                results.append(error_obj)

                # Record in tool ledger
                self.tool_ledger.record_tool_call(
                    tool_name=tool_name,
                    args=tool_args,
                    error=error_message,
                    metadata={
                        "duration": tool_duration,
                        "error_type": error_type,
                        "recovery_attempted": recovery_result.get("recovery_attempted", False)
                    }
                )

                # Add to agent memory for future reference
                self.add_to_memory({
                    "type": "tool_error",
                    "tool": tool_name,
                    "args": tool_args,
                    "error": error_message,
                    "error_type": error_type,
                    "user_message": user_message,
                    "timestamp": datetime.now().isoformat(),
                    "recovery_attempted": recovery_result.get("recovery_attempted", False),
                    "suggestions": recovery_result.get("suggestions", [])
                })

        return results



    def _setup_basic_logger(self):
        """Set up basic logging that's always available."""
        logger_name = f"{self.__class__.__name__}.{self.session_id}"
        agent_logger = logging.getLogger(logger_name)

        # Only set up handlers if they don't exist
        if not agent_logger.handlers:
            agent_logger.setLevel(logging.INFO)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            agent_logger.addHandler(console_handler)

            # File handler (basic)
            log_dir = Path(get_standard_log_dir("agents"))
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / f"{self.__class__.__name__}_{self.session_id}.log")
            file_handler.setFormatter(formatter)
            agent_logger.addHandler(file_handler)

        return agent_logger

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response."""
        # Extract JSON array from response
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find any JSON-like structure
            json_str = response

        try:
            # Clean up the JSON string
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            json_str = re.sub(r'[^\[\{\}\]"\':,\.\-\w\s]', '', json_str)

            # Parse the JSON
            tool_calls = json.loads(json_str)

            # Validate tool calls
            validated_calls = []
            for call in tool_calls:
                if isinstance(call, dict) and "tool" in call:
                    validated_calls.append(call)

            return validated_calls
        except Exception as e:
            self.logger.error(f"Error parsing tool calls: {str(e)}")
            self.logger.error(f"Response: {response}")
            return []

    @abstractmethod
    async def run(self, user_input: str, memory: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run the agent with the given input.

        Args:
            user_input: The input to process
            memory: Optional memory to initialize with

        Returns:
            Dictionary containing the agent's response and any additional data
        """
        pass

    def add_to_memory(self, content: Dict[str, Any]):
        """Add an item to the agent's memory."""
        self.state.add_memory_item(content)

    def get_memory(self) -> List[Dict[str, Any]]:
        """Get the agent's current memory."""
        return self.state.get_memory()

    def should_terminate(self) -> bool:
        """
        Check if the agent should terminate based on iteration count, duration, or result quality.

        This method considers:
        1. Current iteration count vs. max_iterations (legacy parameter)
        2. Current phase and its specific iteration limit
        3. Dynamic termination based on result quality (if enabled)
        4. Maximum runtime duration

        Returns:
            True if the agent should terminate, False otherwise
        """
        # Check iteration count (legacy parameter)
        if self.state.current_iteration >= self.max_iterations:
            self.logger.info(f"Terminating: Reached max iterations ({self.max_iterations})")
            return True

        # Check phase-specific iteration limits
        if hasattr(self.state, 'current_phase'):
            phase = self.state.current_phase
            if phase == 'planning' and self.state.current_iteration >= self.max_planning_iterations:
                self.logger.info(f"Terminating: Reached max planning iterations ({self.max_planning_iterations})")
                return True
            elif phase == 'execution' and self.state.current_iteration >= self.max_execution_iterations:
                self.logger.info(f"Terminating: Reached max execution iterations ({self.max_execution_iterations})")
                return True
            elif phase == 'refinement' and self.state.current_iteration >= self.max_refinement_iterations:
                self.logger.info(f"Terminating: Reached max refinement iterations ({self.max_refinement_iterations})")
                return True

        # Check dynamic termination based on result quality
        if self.enable_dynamic_termination and self.state.current_iteration > 0:
            # Get the latest memory item
            memory = self.get_memory()
            if memory:
                latest_item = memory[-1]
                if 'confidence' in latest_item and latest_item['confidence'] >= self.min_confidence_threshold:
                    self.logger.info(f"Terminating: Reached confidence threshold ({latest_item['confidence']})")
                    return True

        # Check maximum runtime duration
        if hasattr(self.state, 'start_time'):
            elapsed_time = time.time() - self.state.start_time
            if elapsed_time >= self.max_duration_seconds:
                self.logger.info(f"Terminating: Reached max duration ({elapsed_time:.1f}s)")
                return True

        return False

    def increment_iteration(self):
        """Increment the current iteration counter."""
        self.state.increment_iteration()

    @timed_function(category="process")
    async def process_with_llm_tools(self, input_text: str) -> Dict[str, Any]:
        """
        Process input using LLM-driven tool calling.

        Args:
            input_text: User's input text

        Returns:
            Dictionary containing tool call results and other information
        """
        process_start_time = time.time()
        self.logger.info(f"Processing input: {input_text[:100]}{'...' if len(input_text) > 100 else ''}")

        # 1. Select tools to call
        tool_selection_start = time.time()
        tool_calls = await self.select_tools(input_text)
        tool_selection_duration = time.time() - tool_selection_start
        self.logger.info(f"Tool selection completed in {tool_selection_duration:.3f}s, selected {len(tool_calls)} tools")

        # 2. Execute tool calls
        execution_start = time.time()
        results = await self.execute_tool_calls(tool_calls)
        execution_duration = time.time() - execution_start
        self.logger.info(f"Tool execution completed in {execution_duration:.3f}s")

        # 3. Return results with timing information
        total_duration = time.time() - process_start_time
        self.logger.info(f"Total processing completed in {total_duration:.3f}s")

        return {
            "input": input_text,
            "tool_calls": tool_calls,
            "results": results,
            "timing": {
                "total": total_duration,
                "tool_selection": tool_selection_duration,
                "tool_execution": execution_duration
            }
        }