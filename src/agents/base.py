from typing import List, Dict, Any, Optional, Tuple, Union
import json
import re
import logging
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .core.agent_state import AgentState
from ..environments.base import Environment
from ..tools.registry import ToolRegistry
from sec_filing_analyzer.llm import BaseLLM, OpenAILLM
from ..sec_filing_analyzer.utils.logging_utils import get_standard_log_dir
from ..sec_filing_analyzer.utils.timing import timed_function, TimingContext

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
        max_iterations: int = 3,
        max_duration_seconds: int = 180,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 4000,
        environment: Optional[Environment] = None,
        max_tool_calls: int = 3
    ):
        """
        Initialize an agent with its goals and capabilities.

        Args:
            goals: List of goals the agent aims to achieve
            capabilities: List of capabilities that extend agent behavior
            max_iterations: Maximum number of action loops
            max_duration_seconds: Maximum runtime in seconds
            llm_model: LLM model to use
            llm_temperature: Temperature for LLM generation
            llm_max_tokens: Maximum tokens for LLM generation
            environment: Optional environment to use
            max_tool_calls: Maximum number of tool calls per iteration
        """
        self.goals = goals
        self.capabilities = capabilities or []
        self.max_iterations = max_iterations
        self.max_duration_seconds = max_duration_seconds
        self.max_tool_calls = max_tool_calls

        # Initialize state
        self.state = AgentState()

        # Initialize LLM
        self.llm = OpenAILLM(
            model=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens
        )

        # Initialize environment
        self.environment = environment or Environment()

        # Set up session ID for logging
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Set up basic logging
        self.logger = self._setup_basic_logger()

    async def select_tools(self, input_text: str) -> List[Dict[str, Any]]:
        """
        Use LLM to select which tools to call and what parameters to pass.

        Args:
            input_text: User's input text

        Returns:
            List of tool call specifications
        """
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
        Execute a list of tool calls.

        Args:
            tool_calls: List of tool call specifications

        Returns:
            List of tool call results
        """
        results = []
        start_time = time.time()
        self.logger.info(f"Starting execution of {len(tool_calls)} tool calls")

        for i, tool_call in enumerate(tool_calls):
            if i >= self.max_tool_calls:
                self.logger.warning(f"Reached maximum number of tool calls ({self.max_tool_calls}). Skipping remaining calls.")
                break

            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("args", {})

            # Execute the tool call with timing
            tool_start_time = time.time()
            try:
                self.logger.info(f"Executing tool call {i+1}/{len(tool_calls)}: {tool_name}")
                self.logger.info(f"Tool arguments: {tool_args}")

                with TimingContext(f"tool_{tool_name}", category="tool", logger=self.logger):
                    result = await self.environment.execute_action({
                        "tool": tool_name,
                        "args": tool_args
                    })

                tool_duration = time.time() - tool_start_time
                self.logger.info(f"Tool {tool_name} completed in {tool_duration:.3f}s")

                results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                    "success": True,
                    "duration": tool_duration
                })

            except Exception as e:
                tool_duration = time.time() - tool_start_time
                self.logger.error(f"Error executing tool call: {str(e)}")
                self.logger.error(f"Tool {tool_name} failed after {tool_duration:.3f}s")

                results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "error": str(e),
                    "success": False,
                    "duration": tool_duration
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
        """Check if the agent should terminate based on iteration count or duration."""
        return self.state.current_iteration >= self.max_iterations

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