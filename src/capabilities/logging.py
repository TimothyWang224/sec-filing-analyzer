"""
Logging Capability

This module provides a capability for agents to log their activities, decisions,
and reasoning in a structured and configurable way.
"""

import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..agents.base import Agent
from ..sec_filing_analyzer.utils.logging_utils import get_standard_log_dir
from .base import Capability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoggingCapability(Capability):
    """Capability for logging agent activities, decisions, and reasoning."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        include_memory: bool = True,
        include_context: bool = True,
        include_actions: bool = True,
        include_results: bool = True,
        include_prompts: bool = False,
        include_responses: bool = False,
        max_content_length: int = 1000,
        custom_formatters: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize the logging capability.

        Args:
            log_dir: Directory to store log files (default: data/logs/agents)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            log_format: Format string for log messages
            include_memory: Whether to log agent memory
            include_context: Whether to log agent context
            include_actions: Whether to log agent actions
            include_results: Whether to log action results
            include_prompts: Whether to log LLM prompts (may contain sensitive data)
            include_responses: Whether to log LLM responses (may be verbose)
            max_content_length: Maximum length for content in logs (to avoid huge logs)
            custom_formatters: Custom formatter functions for specific content types
        """
        super().__init__(
            name="logging",
            description="Logs agent activities, decisions, and reasoning",
        )

        # Set logging configuration
        self.log_dir = log_dir or str(get_standard_log_dir("agents"))
        self.log_level = self._get_log_level(log_level)
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.log_format = log_format

        # Set content configuration
        self.include_memory = include_memory
        self.include_context = include_context
        self.include_actions = include_actions
        self.include_results = include_results
        self.include_prompts = include_prompts
        self.include_responses = include_responses
        self.max_content_length = max_content_length

        # Set custom formatters
        self.custom_formatters = custom_formatters or {}

        # Initialize logger
        self.agent_logger = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.log_count = 0

    async def init(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the capability with agent and context.

        Args:
            agent: The agent this capability belongs to
            context: Initial context for the capability

        Returns:
            Updated context
        """
        self.agent = agent
        self.context = context

        # Set up agent-specific logger
        agent_type = agent.__class__.__name__
        self.agent_logger = self._setup_logger(agent_type)

        # Log initialization
        self.agent_logger.info(
            f"Initializing {agent_type} with session ID: {self.session_id}"
        )

        # Log goals
        if hasattr(agent, "goals") and agent.goals:
            goals_str = ", ".join(
                [f"{goal.name}: {goal.description}" for goal in agent.goals]
            )
            self.agent_logger.info(f"Agent goals: {goals_str}")

        # Log capabilities
        if hasattr(agent, "capabilities") and agent.capabilities:
            capabilities_str = ", ".join(
                [cap.__class__.__name__ for cap in agent.capabilities]
            )
            self.agent_logger.info(f"Agent capabilities: {capabilities_str}")

        # Add logging info to context
        context["logging"] = {
            "session_id": self.session_id,
            "agent_type": agent_type,
            "start_time": datetime.now().isoformat(),
            "log_file": self.log_file if hasattr(self, "log_file") else None,
        }

        return context

    async def start_agent_loop(self, agent: Agent, context: Dict[str, Any]) -> bool:
        """
        Called at the start of each agent loop iteration.

        Args:
            agent: The agent
            context: Current context

        Returns:
            Whether to continue the loop
        """
        # Log iteration start
        self.agent_logger.info(
            f"Starting iteration {agent.state.current_iteration + 1}/{agent.max_iterations}"
        )

        # Log memory if enabled
        if self.include_memory and hasattr(agent, "state"):
            self._log_memory(agent.state.get_memory())

        # Log context if enabled
        if self.include_context:
            self._log_context(context)

        return True

    async def process_prompt(
        self, agent: Agent, context: Dict[str, Any], prompt: str
    ) -> str:
        """
        Process the prompt before it's sent to the LLM.

        Args:
            agent: The agent processing the prompt
            context: Current context
            prompt: Original prompt

        Returns:
            Processed prompt
        """
        # Log prompt if enabled
        if self.include_prompts:
            self.agent_logger.info(f"Prompt: {self._truncate(prompt)}")

        return prompt

    async def process_response(
        self, agent: Agent, context: Dict[str, Any], response: str
    ) -> str:
        """
        Process the response from the LLM.

        Args:
            agent: The agent processing the response
            context: Current context
            response: Response from LLM

        Returns:
            Processed response
        """
        # Log response if enabled
        if self.include_responses:
            self.agent_logger.info(f"LLM Response: {self._truncate(response)}")

        return response

    async def process_action(
        self, agent: Agent, context: Dict[str, Any], action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an action before it's executed.

        Args:
            agent: The agent processing the action
            context: Current context
            action: Action to process

        Returns:
            Processed action
        """
        # Log action if enabled
        if self.include_actions:
            # Format action for logging
            formatted_action = self._format_action(action)
            self.agent_logger.info(f"Action: {formatted_action}")

        return action

    async def process_result(
        self,
        agent: Agent,
        context: Dict[str, Any],
        response: str,
        action: Dict[str, Any],
        result: Any,
    ) -> Any:
        """
        Process the result of an action.

        Args:
            agent: The agent processing the result
            context: Current context
            response: Original response
            action: Action that produced the result
            result: Result to process

        Returns:
            Processed result
        """
        # Log result if enabled
        if self.include_results:
            # Format result for logging
            formatted_result = self._format_result(result)
            self.agent_logger.info(f"Result: {formatted_result}")

        return result

    async def end_agent_loop(self, agent: Agent, context: Dict[str, Any]):
        """
        Called at the end of each agent loop iteration.

        Args:
            agent: The agent
            context: Current context
        """
        # Log iteration end
        self.agent_logger.info(
            f"Completed iteration {agent.state.current_iteration + 1}"
        )

        # Log memory changes if enabled
        if self.include_memory and hasattr(agent, "state"):
            self._log_memory_changes(agent.state.get_memory())

    async def terminate(self, agent: Agent, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean up when the agent is terminating.

        Args:
            agent: The agent
            context: Current context

        Returns:
            Final context updates
        """
        # Calculate execution time
        execution_time = time.time() - self.start_time

        # Log termination
        self.agent_logger.info(f"Agent terminated after {execution_time:.2f} seconds")
        self.agent_logger.info(f"Total log entries: {self.log_count}")

        # Log final memory if enabled
        if self.include_memory and hasattr(agent, "state"):
            self._log_memory(agent.state.get_memory(), is_final=True)

        # Return execution stats
        return {
            "logging_stats": {
                "session_id": self.session_id,
                "execution_time": execution_time,
                "log_count": self.log_count,
                "log_file": self.log_file if hasattr(self, "log_file") else None,
            }
        }

    def _setup_logger(self, agent_type: str) -> logging.Logger:
        """
        Set up a logger for the agent.

        Args:
            agent_type: Type of agent

        Returns:
            Configured logger
        """
        # Create logger
        agent_logger = logging.getLogger(f"agent.{agent_type}.{self.session_id}")
        agent_logger.setLevel(self.log_level)

        # Remove existing handlers to avoid duplicates
        for handler in agent_logger.handlers[:]:
            agent_logger.removeHandler(handler)

        # Create formatters
        formatter = logging.Formatter(self.log_format)

        # Add console handler if enabled
        if self.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            agent_logger.addHandler(console_handler)

        # Add file handler if enabled
        if self.log_to_file:
            # Create log directory if it doesn't exist
            log_dir = Path(self.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create log file
            self.log_file = log_dir / f"{agent_type}_{self.session_id}.log"
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            agent_logger.addHandler(file_handler)

            # Create JSON log file for structured logging
            self.json_log_file = log_dir / f"{agent_type}_{self.session_id}.json"
            with open(self.json_log_file, "w") as f:
                json.dump(
                    {
                        "session_id": self.session_id,
                        "agent_type": agent_type,
                        "start_time": datetime.now().isoformat(),
                        "logs": [],
                    },
                    f,
                    indent=2,
                    default=str,
                )

        return agent_logger

    def _log_memory(self, memory: List[Dict[str, Any]], is_final: bool = False):
        """
        Log agent memory.

        Args:
            memory: Agent memory
            is_final: Whether this is the final memory log
        """
        if not memory:
            return

        prefix = "Final" if is_final else "Current"
        self.agent_logger.debug(f"{prefix} memory size: {len(memory)} items")

        # Log memory items
        for i, item in enumerate(
            memory[-3:]
        ):  # Log only the last 3 items to avoid verbosity
            # Format memory item
            formatted_item = self._format_memory_item(item)
            self.agent_logger.debug(
                f"Memory item {len(memory) - 3 + i + 1}: {formatted_item}"
            )

        # Log to JSON file
        if hasattr(self, "json_log_file"):
            self._append_to_json_log(
                {
                    "type": "memory",
                    "timestamp": datetime.now().isoformat(),
                    "is_final": is_final,
                    "memory_size": len(memory),
                    "memory_items": memory[-3:] if len(memory) > 3 else memory,
                }
            )

    def _log_memory_changes(self, memory: List[Dict[str, Any]]):
        """
        Log changes to agent memory.

        Args:
            memory: Agent memory
        """
        if not memory:
            return

        # Log only the last memory item (most recent change)
        if len(memory) > 0:
            last_item = memory[-1]
            formatted_item = self._format_memory_item(last_item)
            self.agent_logger.debug(f"Memory updated: {formatted_item}")

            # Log to JSON file
            if hasattr(self, "json_log_file"):
                self._append_to_json_log(
                    {
                        "type": "memory_update",
                        "timestamp": datetime.now().isoformat(),
                        "memory_item": last_item,
                    }
                )

    def _log_context(self, context: Dict[str, Any]):
        """
        Log agent context.

        Args:
            context: Agent context
        """
        if not context:
            return

        # Format context for logging
        formatted_context = self._format_context(context)
        self.agent_logger.debug(f"Context: {formatted_context}")

        # Log to JSON file
        if hasattr(self, "json_log_file"):
            self._append_to_json_log(
                {
                    "type": "context",
                    "timestamp": datetime.now().isoformat(),
                    "context": context,
                }
            )

    def _format_memory_item(self, item: Dict[str, Any]) -> str:
        """
        Format a memory item for logging.

        Args:
            item: Memory item

        Returns:
            Formatted memory item
        """
        # Use custom formatter if available
        if "memory" in self.custom_formatters:
            return self.custom_formatters["memory"](item)

        # Default formatting
        if isinstance(item, dict):
            item_type = item.get("type", "unknown")
            content = item.get("content", {})

            if isinstance(content, dict):
                # Truncate content if needed
                truncated_content = {k: self._truncate(v) for k, v in content.items()}
                return f"{item_type}: {truncated_content}"
            else:
                return f"{item_type}: {self._truncate(content)}"
        else:
            return self._truncate(str(item))

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context for logging.

        Args:
            context: Context

        Returns:
            Formatted context
        """
        # Use custom formatter if available
        if "context" in self.custom_formatters:
            return self.custom_formatters["context"](context)

        # Default formatting
        if isinstance(context, dict):
            # Extract key information
            keys = list(context.keys())
            return f"Keys: {keys}"
        else:
            return self._truncate(str(context))

    def _format_action(self, action: Dict[str, Any]) -> str:
        """
        Format an action for logging.

        Args:
            action: Action

        Returns:
            Formatted action
        """
        # Use custom formatter if available
        if "action" in self.custom_formatters:
            return self.custom_formatters["action"](action)

        # Default formatting
        if isinstance(action, dict):
            tool = action.get("tool", "unknown")
            args = action.get("args", {})

            # Truncate args if needed
            if isinstance(args, dict):
                truncated_args = {k: self._truncate(v) for k, v in args.items()}
                return f"{tool}: {truncated_args}"
            else:
                return f"{tool}: {self._truncate(args)}"
        else:
            return self._truncate(str(action))

    def _format_result(self, result: Any) -> str:
        """
        Format a result for logging.

        Args:
            result: Result

        Returns:
            Formatted result
        """
        # Use custom formatter if available
        if "result" in self.custom_formatters:
            return self.custom_formatters["result"](result)

        # Default formatting
        if isinstance(result, dict):
            # Extract key information
            keys = list(result.keys())
            return f"Keys: {keys}"
        else:
            return self._truncate(str(result))

    def _truncate(self, content: Any) -> str:
        """
        Truncate content to avoid huge logs.

        Args:
            content: Content to truncate

        Returns:
            Truncated content
        """
        if content is None:
            return "None"

        content_str = str(content)
        if len(content_str) > self.max_content_length:
            return content_str[: self.max_content_length] + "..."
        return content_str

    def _get_log_level(self, level_str: str) -> int:
        """
        Get logging level from string.

        Args:
            level_str: Logging level string

        Returns:
            Logging level
        """
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return levels.get(level_str.upper(), logging.INFO)

    def _append_to_json_log(self, log_entry: Dict[str, Any]):
        """
        Append an entry to the JSON log file.

        Args:
            log_entry: Log entry
        """
        if not hasattr(self, "json_log_file"):
            return

        try:
            # Convert Pydantic objects to dictionaries
            log_entry = self._prepare_for_serialization(log_entry)

            # Read existing log
            with open(self.json_log_file, "r") as f:
                log_data = json.load(f)

            # Append new entry
            log_data["logs"].append(log_entry)

            # Write updated log with default=str to handle non-serializable objects
            with open(self.json_log_file, "w") as f:
                json.dump(log_data, f, indent=2, default=str)

            # Increment log count
            self.log_count += 1

        except Exception as e:
            # Log error but don't crash
            logger.error(f"Error appending to JSON log: {str(e)}")
            logger.debug(traceback.format_exc())

    def _prepare_for_serialization(self, obj: Any) -> Any:
        """
        Prepare an object for JSON serialization by converting Pydantic models to dictionaries.

        Args:
            obj: Object to prepare

        Returns:
            JSON-serializable object
        """
        # If it's a dictionary, process each value
        if isinstance(obj, dict):
            return {k: self._prepare_for_serialization(v) for k, v in obj.items()}

        # If it's a list, process each item
        elif isinstance(obj, list):
            return [self._prepare_for_serialization(item) for item in obj]

        # If it's a Pydantic model, convert to dictionary
        elif hasattr(obj, "model_dump"):
            return obj.model_dump()

        # Otherwise, return as is
        return obj
