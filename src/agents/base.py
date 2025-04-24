from typing import List, Dict, Any, Optional, Tuple, Union, Type
import json
import re
import logging
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..contracts import extract_value, PlanStep, Plan
from ..tools.registry import ToolRegistry

from ..utils.json_utils import safe_parse_json, repair_json


def _to_dict(obj: Any) -> Dict[str, Any]:
    """
    Convert an object to a dictionary.

    This helper function ensures consistent conversion of objects to dictionaries,
    handling both Pydantic models and plain dictionaries.

    Args:
        obj: The object to convert (PlanStep, Plan, or dictionary)

    Returns:
        Dictionary representation of the object
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, PlanStep):
        return obj.model_dump()
    # fallback for full Plan model
    return obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)


def _to_object(obj: Any, model_class: Type = None) -> Any:
    """
    Convert a dictionary to a Pydantic object.

    This helper function ensures consistent conversion of dictionaries to Pydantic objects,
    handling both dictionaries and existing Pydantic models.

    Args:
        obj: The object to convert (dictionary or Pydantic model)
        model_class: The Pydantic model class to use for conversion (PlanStep or Plan)

    Returns:
        Pydantic model instance
    """
    if model_class is None:
        return obj

    if isinstance(obj, model_class):
        return obj

    if isinstance(obj, dict):
        return model_class.model_validate(obj)

    return obj

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
from ..sec_filing_analyzer.utils.logging_utils import get_standard_log_dir, SessionLogger, get_current_session_id, set_current_session_id
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
        # Token budget parameters
        max_total_tokens: Optional[int] = None,
        token_budgets: Optional[Dict[str, int]] = None,
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

            # Token budget parameters
            max_total_tokens: Maximum total tokens to use across all phases
            token_budgets: Dictionary mapping phases to token budgets

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
        self.max_iterations = max_iterations if max_iterations is not None else config.get("max_iterations", None)  # Legacy parameter
        self.max_planning_iterations = max_planning_iterations if max_planning_iterations is not None else config.get("max_planning_iterations", 2)
        self.max_execution_iterations = max_execution_iterations if max_execution_iterations is not None else config.get("max_execution_iterations", 3)
        self.max_refinement_iterations = max_refinement_iterations if max_refinement_iterations is not None else config.get("max_refinement_iterations", 1)

        # Compute effective max iterations
        self.max_iterations_effective = self._compute_effective_max_iterations()

        # Set tool execution parameters with fallbacks
        self.max_tool_retries = max_tool_retries if max_tool_retries is not None else config.get("max_tool_retries", 2)
        self.tools_per_iteration = tools_per_iteration if tools_per_iteration is not None else config.get("tools_per_iteration", 1)

        # Set runtime parameters with fallbacks
        self.max_duration_seconds = max_duration_seconds if max_duration_seconds is not None else config.get("max_duration_seconds", 180)

        # Set token budget parameters with fallbacks
        self.max_total_tokens = max_total_tokens if max_total_tokens is not None else config.get("max_total_tokens", 3000)
        self.token_budgets = token_budgets if token_budgets is not None else config.get("token_budgets", None)

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
        current_session_id = get_current_session_id()
        if current_session_id:
            # Use the existing session ID if available
            self.session_id = current_session_id
        else:
            # Generate a new session ID and set it as current
            self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            set_current_session_id(self.session_id)

        # Set up basic logging
        self.logger = self._setup_basic_logger()

        # Log effective max iterations
        self.logger.info(f"Effective max iterations: {self.max_iterations_effective} (derived from planning={self.max_planning_iterations}, execution={self.max_execution_iterations}, refinement={self.max_refinement_iterations})")

        # Configure token budgets
        self._configure_token_budgets()

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

        # Check if we have a plan and should respect it
        if hasattr(self, 'state') and hasattr(self.state, 'get_context'):
            planning_context = self.state.get_context().get("planning", {})
            current_plan = planning_context.get("plan", {})
            current_step = planning_context.get("current_step", {})

            # If we have a plan that we can't modify and a current step with a tool
            if current_plan and not current_plan.get("can_modify", True) and current_step:
                self.logger.info("Using tools from coordinator's plan instead of selecting new ones")

                # Extract tool from the current step
                tool_name = current_step.get("tool")
                if tool_name:
                    # Create a tool call from the plan step
                    tool_calls = [{
                        "tool": tool_name,
                        "args": current_step.get("parameters", {})
                    }]
                    self.logger.info(f"Using tool from plan: {tool_name}")
                    return tool_calls

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
                temperature=0.2,
                json_mode=True,  # Force the model to return valid JSON
                return_usage=True  # Get token usage information
            )

            # Count tokens
            if isinstance(initial_response, dict) and "usage" in initial_response:
                self.state.count_tokens(initial_response["usage"]["total_tokens"])
                initial_response = initial_response["content"]


            # Parse tool calls from response
            initial_tool_calls = await self._parse_tool_calls(initial_response)
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
                    temperature=0.2,
                    json_mode=True,  # Force the model to return valid JSON
                    return_usage=True  # Get token usage information
                )

                # Count tokens
                if isinstance(details_response, dict) and "usage" in details_response:
                    self.state.count_tokens(details_response["usage"]["total_tokens"])
                    details_response = details_response["content"]

                # Parse final tool calls
                detailed_tool_calls = await self._parse_tool_calls(details_response)
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

                # Special handling for specific tools to ensure required parameters are present
                if tool_name == "sec_semantic_search" and "query" not in tool_args:
                    self.logger.info("Adding missing query parameter to sec_semantic_search tool")
                    # Use the user input as the default query if not provided
                    tool_args["query"] = user_input
            except Exception as e:
                self.logger.error(f"Error completing parameters: {str(e)}")
                # Continue with original parameters if completion fails

            self.logger.info(f"Executing tool call: {tool_name}")
            self.logger.info(f"Tool arguments: {json.dumps(tool_args, indent=2)}")

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

                # Log the tool result
                try:
                    # Format the result for logging
                    result_str = json.dumps(recovery_result["result"], indent=2)
                    if len(result_str) > 1000:
                        # Truncate long results
                        result_str = result_str[:997] + "..."
                    self.logger.info(f"Tool result: {result_str}")
                except Exception as e:
                    # Handle non-serializable results
                    self.logger.info(f"Tool result: {str(recovery_result['result'])}")
                    self.logger.warning(f"Could not serialize tool result: {str(e)}")

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
                result_data = recovery_result["result"]
                memory_item = {
                    "type": "tool_result",
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result_data,
                    "timestamp": datetime.now().isoformat(),
                    "recovery_info": {
                        "strategy": recovery_result.get("recovery_strategy", None),
                        "alternative_tool": recovery_result.get("alternative_tool", None)
                    }
                }

                # Check if the result has an output_key
                if isinstance(result_data, dict) and "output_key" in result_data:
                    memory_item["output_key"] = result_data["output_key"]

                    # Get the expected_key from the current plan step if available
                    current_plan = self.state.get_context().get("planning", {}).get("plan", {})
                    current_step = self.state.get_context().get("planning", {}).get("current_step", {})

                    if current_step and "expected_key" in current_step:
                        expected_key = current_step["expected_key"]

                        # Store the result in memory using the expected_key
                        self.state.memory[expected_key] = result_data
                        self.logger.info(f"Stored result for {expected_key} in memory with output_key {result_data['output_key']}")

                self.add_to_memory(memory_item)
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

                # Log the error
                self.logger.error(f"Tool error: {error_message}")
                if "suggestions" in recovery_result:
                    self.logger.info(f"Error suggestions: {recovery_result['suggestions']}")

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



    def _should_skip(self, step: Union[Dict[str, Any], PlanStep]) -> bool:
        """
        Check if a step should be skipped based on success criteria.

        This method checks if the expected output of a step is already in memory,
        allowing us to skip redundant steps and move directly to the next step.
        It uses the done_check condition if available, otherwise falls back to checking
        if the expected_key exists in the result.

        Args:
            step: The plan step to check (dictionary or PlanStep object)

        Returns:
            True if the step should be skipped, False otherwise
        """
        # Convert to dictionary if it's a PlanStep object
        step_dict = _to_dict(step)

        # If no expected_key is specified, we can't skip
        if "expected_key" not in step_dict:
            return False

        expected_key = step_dict["expected_key"]
        output_path = step_dict.get("output_path", [])
        done_check = step_dict.get("done_check", None)

        # Get all memory items
        memory = self.get_memory()

        # Check tool results in memory
        for item in memory:
            if item.get("type") == "tool_result":
                result = item.get("result", {})

                # If output_path is specified, navigate to the specific location in the result
                if output_path:
                    try:
                        # Use the extract_value function from contracts module
                        value = extract_value(result, output_path)

                        # If we found a value at the specified path, check the done_check condition
                        if value is not None:
                            # If done_check is specified, evaluate it
                            if done_check:
                                # Create a local context with the expected_key and value
                                local_context = {expected_key: value}
                                try:
                                    # Evaluate the done_check condition in the local context
                                    is_done = eval(done_check, {}, local_context)
                                    if is_done:
                                        self.logger.info(f"Success criterion met: {done_check} evaluated to True")
                                        return True
                                except Exception as e:
                                    self.logger.warning(f"Error evaluating done_check condition: {str(e)}")
                            else:
                                # No done_check, just check if value exists
                                self.logger.info(f"Success criterion met: Found {expected_key} at path {output_path}")
                                return True
                    except Exception as e:
                        self.logger.warning(f"Error checking output path: {str(e)}")
                        continue
                # Otherwise, check if the expected key is in the result
                elif expected_key in str(result):
                    # If done_check is specified, evaluate it
                    if done_check:
                        # Create a local context with the expected_key and result
                        local_context = {expected_key: result}
                        try:
                            # Evaluate the done_check condition in the local context
                            is_done = eval(done_check, {}, local_context)
                            if is_done:
                                self.logger.info(f"Success criterion met: {done_check} evaluated to True")
                                return True
                        except Exception as e:
                            self.logger.warning(f"Error evaluating done_check condition: {str(e)}")
                    else:
                        # No done_check, just check if expected_key exists in result
                        self.logger.info(f"Success criterion met: Found {expected_key} in result")
                        return True

        # If we reach here, success criteria are not met
        return False

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

            # Register with session logger for consolidated logging
            session_logger = SessionLogger.get_logger(self.session_id)
            session_logger.register_agent(self.__class__.__name__, agent_logger)

        return agent_logger

    def _configure_token_budgets(self) -> None:
        """
        Configure token budgets for each phase based on the agent's configuration.

        This method sets up the token budgets for planning, execution, and refinement phases
        based on the agent's configuration. If no specific budgets are provided, it uses
        default percentages of the total token budget.
        """
        # Default token budget percentages
        DEFAULT_PERCENTAGES = {
            "planning": 0.10,   # 10% for planning
            "execution": 0.40,  # 40% for execution
            "refinement": 0.50  # 50% for refinement
        }

        # Default total token budget
        DEFAULT_TOTAL_TOKENS = 250000

        # Get total token budget from config or use default
        total_tokens = getattr(self, "max_total_tokens", DEFAULT_TOTAL_TOKENS)

        # If token_budgets is explicitly provided, use it
        if hasattr(self, "token_budgets") and self.token_budgets is not None:
            # Set token budgets in state
            self.state.token_budget = self.token_budgets
        else:
            # If max_total_tokens is explicitly provided, calculate budgets based on percentages
            if hasattr(self, "max_total_tokens") and self.max_total_tokens is not None:
                # Calculate budgets based on percentages
                calculated_budgets = {
                    phase: int(total_tokens * percentage)
                    for phase, percentage in DEFAULT_PERCENTAGES.items()
                }
                self.state.token_budget = calculated_budgets
            # Otherwise, use the default token budget from AgentState
            # (which is already set in the AgentState constructor)

        # Log the token budgets
        self.logger.info(f"Token budgets configured: {self.state.token_budget}")

    def _compute_effective_max_iterations(self) -> int:
        """Compute the effective max iterations based on phase iterations."""
        # If max_iterations is explicitly set, use it
        if self.max_iterations is not None:
            return self.max_iterations

        # Otherwise, compute from phase iterations with a small buffer
        phase_sum = (
            self.max_planning_iterations +
            self.max_execution_iterations +
            self.max_refinement_iterations
        )

        # Add a small buffer (10%) to account for potential phase transitions
        # or other edge cases, with a minimum of 1 extra iteration
        buffer = max(1, int(phase_sum * 0.1))

        return phase_sum + buffer

    async def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response."""
        # First try to parse using the safe_parse_json utility
        try:
            # Use safe_parse_json with expected type "array"
            tool_calls = safe_parse_json(response, default_value=[], expected_type="array")

            # If parsing failed and we have an LLM instance, try to repair
            if not tool_calls and hasattr(self, 'llm'):
                # Create a repair function that uses the agent's LLM
                async def repair_with_agent_llm(text, expected_type):
                    return await repair_json(text, self.llm, default_value=[], expected_type=expected_type)

                # Try to repair and parse again
                tool_calls = await repair_with_agent_llm(response, "array")

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

    def _write_to_memory(self, step: Union[Dict[str, Any], PlanStep], raw_result: Dict[str, Any]):
        """
        Write the result of a step to memory using the expected_key and output_path.

        Args:
            step: The plan step (dictionary or PlanStep object)
            raw_result: The raw result from the tool
        """
        # Convert to dictionary if it's a PlanStep object
        step_dict = _to_dict(step)

        # Check if we have expected_key and output_path
        if "expected_key" in step_dict and step_dict.get("output_path") is not None:
            expected_key = step_dict["expected_key"]
            output_path = step_dict["output_path"]

            # Extract the value from the result using the output_path
            value = extract_value(raw_result, output_path)

            # Store the value in memory
            self.state.memory[expected_key] = value

            # Log that we stored the result
            self.logger.info(f"Stored result for {expected_key} in memory with value {value}")

    async def _execute_current_step(self, step: Union[Dict[str, Any], PlanStep]) -> bool:
        """
        Execute the current step in the plan.

        This method checks if the step should be skipped based on success criteria,
        and if not, executes the step using the appropriate tool or agent.

        Args:
            step: The plan step to execute (dictionary or PlanStep object)

        Returns:
            True if the step was executed or skipped successfully, False otherwise
        """
        # Convert to PlanStep if it's a dictionary
        step_obj = _to_object(step, PlanStep) if isinstance(step, dict) else step

        # Also keep a dictionary version for backward compatibility
        step_dict = _to_dict(step_obj)

        # 1. Success short-circuit: Check if we should skip this step based on success criteria
        if self._should_skip(step_obj):
            self.logger.info(f"Skipping step {step_dict['step_id']} - success criterion already satisfied")

            # Mark the step as completed
            if isinstance(step, dict):
                step["status"] = "completed"
                step["completed_at"] = datetime.now().isoformat()
                step["skipped"] = True
            else:
                step_obj.status = "completed"
                step_obj.completed_at = datetime.now().isoformat()
                step_obj.skipped = True

            # Add to memory that we skipped this step
            self.add_to_memory({
                "type": "step_skipped",
                "step_id": step_dict["step_id"],
                "description": step_dict["description"],
                "expected_key": step_dict.get("expected_key"),
                "output_path": step_dict.get("output_path"),
                "done_check": step_dict.get("done_check"),
                "timestamp": datetime.now().isoformat()
            })

            # If we have a tool and expected_key, store the result in memory
            if ("tool" in step_dict and "expected_key" in step_dict) or \
               (hasattr(step_obj, 'tool') and step_obj.tool and hasattr(step_obj, 'expected_key') and step_obj.expected_key):
                tool_name = step_dict["tool"]
                expected_key = step_dict["expected_key"]

                # Get the tool spec
                tool_spec = ToolRegistry.get_tool_spec(tool_name)
                if tool_spec:
                    # Store the result in memory using the expected_key
                    self.state.memory[expected_key] = {
                        "skipped": True,
                        "reason": "Success criterion already satisfied",
                        "output_key": tool_spec.output_key
                    }

                    # Add a memory item for the skipped step
                    self.add_to_memory({
                        "type": "step_skipped",
                        "step_id": step_dict.get("step_id"),
                        "tool": tool_name,
                        "expected_key": expected_key,
                        "reason": "Success criterion already satisfied",
                        "timestamp": datetime.now().isoformat()
                    })

                    # Log that we stored the result
                    self.logger.info(f"Stored result for {expected_key} in memory with output_key {tool_spec.output_key}")

            return True

        # 2. Validation: If the step has a tool, validate the parameters
        if ("tool" in step_dict and step_dict["tool"]) or (hasattr(step_obj, 'tool') and step_obj.tool):
            tool_name = step_dict["tool"]
            parameters = step_dict.get("parameters", {})

            try:
                # Import the validator
                from ..tools.validator import validate_call

                # Get the query_type and parameters
                query_type = parameters.get("query_type")
                tool_params = parameters.get("parameters", {})

                if query_type:
                    # Validate the call
                    try:
                        validate_call(tool_name, query_type, tool_params)
                    except Exception as e:
                        from ..errors import ToolError, get_user_message

                        # Get a user-friendly error message
                        error_message = get_user_message(e)

                        # Mark the step as failed
                        if isinstance(step, dict):
                            step["status"] = "failed"
                            step["error"] = error_message
                        else:
                            step_obj.status = "failed"
                            # We can't set error directly on PlanStep as it's not a field
                            # But we can add it to the dictionary for memory

                        # Add to memory that the step failed
                        self.add_to_memory({
                            "type": "step_failed",
                            "step_id": step_dict["step_id"],
                            "tool": tool_name,
                            "error": error_message,
                            "timestamp": datetime.now().isoformat()
                        })

                        # Log the error
                        self.logger.error(f"Step {step_dict['step_id']} failed validation: {error_message}")

                        return False
            except ImportError:
                # If validation fails, continue with the step
                self.logger.warning("Could not import validator, skipping validation")

        # If we shouldn't skip, execute the step normally
        # This is handled by the planning capability and the agent's execution phase
        return False

    def get_memory(self) -> List[Dict[str, Any]]:
        """Get the agent's current memory."""
        return self.state.get_memory()

    def should_terminate(self) -> bool:
        """
        Check if the agent should terminate based on iteration count, token budget, duration, or result quality.

        This method considers:
        1. Current iteration count vs. max_iterations_effective (derived from phase iterations or legacy max_iterations)
        2. Current phase and its specific iteration limit
        3. Token budget exhaustion for the current phase
        4. Dynamic termination based on result quality (if enabled)
        5. Maximum runtime duration

        Returns:
            True if the agent should terminate, False otherwise
        """
        # Check iteration count against effective max iterations
        if self.state.current_iteration >= self.max_iterations_effective:
            self.logger.info(f"Terminating: Reached max iterations ({self.state.current_iteration}/{self.max_iterations_effective})")
            return True

        # Check phase-specific iteration limits
        if hasattr(self.state, 'current_phase'):
            phase = self.state.current_phase
            if phase == 'planning' and self.state.phase_iterations[phase] >= self.max_planning_iterations:
                self.logger.info(f"Terminating: Reached max planning iterations ({self.state.phase_iterations[phase]}/{self.max_planning_iterations})")
                return True
            elif phase == 'execution' and self.state.phase_iterations[phase] >= self.max_execution_iterations:
                self.logger.info(f"Terminating: Reached max execution iterations ({self.state.phase_iterations[phase]}/{self.max_execution_iterations})")
                return True
            elif phase == 'refinement' and self.state.phase_iterations[phase] >= self.max_refinement_iterations:
                self.logger.info(f"Terminating: Reached max refinement iterations ({self.state.phase_iterations[phase]}/{self.max_refinement_iterations})")
                return True

        # Check token budget exhaustion
        if self.state.is_budget_exhausted():
            phase = self.state.current_phase
            tokens_used = self.state.tokens_used[phase]
            budget = self.state.token_budget[phase]
            self.logger.info(f"Terminating: Token budget exhausted for {phase} phase ({tokens_used}/{budget} tokens)")
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

    def _rollover_token_surplus(self, from_phase: str, to_phase: str) -> None:
        """
        Roll over unused tokens from one phase to another.

        Args:
            from_phase: Phase to roll over tokens from
            to_phase: Phase to roll over tokens to
        """
        if from_phase not in self.state.token_budget or to_phase not in self.state.token_budget:
            return

        # Calculate surplus
        tokens_used = self.state.tokens_used[from_phase]
        budget = self.state.token_budget[from_phase]
        surplus = max(0, budget - tokens_used)

        if surplus > 0:
            # Add surplus to the target phase
            self.state.token_budget[to_phase] += surplus
            self.logger.info(f"Rolling over {surplus} unused tokens from {from_phase} to {to_phase} phase")
            self.logger.info(f"New {to_phase} budget: {self.state.token_budget[to_phase]} tokens")

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

        # Update the agent's context with the input text
        try:
            # Try direct assignment first
            self.state.context["input"] = input_text

            # If update_context method exists, use it as well
            if hasattr(self.state, 'update_context'):
                self.state.update_context({"input": input_text})
        except Exception as e:
            self.logger.error(f"Error updating context: {str(e)}")
            # Fallback: create context if it doesn't exist
            if not hasattr(self.state, 'context'):
                self.state.context = {}
                self.state.context["input"] = input_text

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