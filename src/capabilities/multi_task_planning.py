"""
Multi-Task Planning Capability

This module provides an enhanced planning capability that can handle multiple tasks.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict

from ..agents.base import Agent
from ..agents.task_queue import TaskQueue
from ..tools.tool_parameter_helper import validate_tool_parameters
from .planning import PlanningCapability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTaskPlanningCapability(PlanningCapability):
    """Enhanced planning capability for handling multiple tasks."""

    def __init__(
        self,
        enable_dynamic_replanning: bool = True,
        enable_step_reflection: bool = True,
        min_steps_before_reflection: int = 2,
        max_plan_steps: int = 10,
        plan_detail_level: str = "high",  # "low", "medium", "high"
    ):
        """
        Initialize the multi-task planning capability.

        Args:
            enable_dynamic_replanning: Whether to allow plan updates during execution
            enable_step_reflection: Whether to reflect on step results and adjust the plan
            min_steps_before_reflection: Minimum steps to execute before reflecting
            max_plan_steps: Maximum number of steps in a plan
            plan_detail_level: Level of detail in the generated plan
        """
        super().__init__(
            enable_dynamic_replanning=enable_dynamic_replanning,
            enable_step_reflection=enable_step_reflection,
            min_steps_before_reflection=min_steps_before_reflection,
            max_plan_steps=max_plan_steps,
            plan_detail_level=plan_detail_level,
        )

        # Override the name and description
        self.name = "multi_task_planning"
        self.description = "Creates and manages execution plans for multiple tasks"

        # Task queue
        self.task_queue = None

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

        # Reset plan state
        self.current_plan = None
        self.current_step_index = 0
        self.completed_steps = []
        self.step_results = {}
        self.plan_created_at = None
        self.last_reflection_at = None

        # Initialize task queue if it doesn't exist
        if self.task_queue is None:
            self.task_queue = context.get("task_queue")

        # Add planning to context
        if "planning" not in context:
            context["planning"] = {
                "has_plan": False,
                "current_step": None,
                "completed_steps": [],
                "plan_status": "not_started",
                "task_queue": self.task_queue,
            }

        return context

    async def start_agent_loop(self, agent: Agent, context: Dict[str, Any]) -> bool:
        """
        Called at the start of each agent loop iteration.
        Creates a plan for the current task if one doesn't exist yet.

        Args:
            agent: The agent
            context: Current context

        Returns:
            Whether to continue the loop
        """
        # Ensure planning context exists
        if "planning" not in context:
            context["planning"] = {
                "has_plan": False,
                "current_step": None,
                "completed_steps": [],
                "plan_status": "not_started",
                "task_queue": self.task_queue,
            }

        # Get the task queue from context if it exists
        if self.task_queue is None and "task_queue" in context:
            self.task_queue = context["task_queue"]
            context["planning"]["task_queue"] = self.task_queue

        # If we have a task queue but no current task, select the next task
        if (
            self.task_queue
            and not self.task_queue.has_current_task()
            and self.task_queue.has_pending_tasks()
        ):
            self.task_queue._select_next_task()

            # Reset plan state for the new task
            self.current_plan = None
            self.current_step_index = 0
            self.completed_steps = []
            self.step_results = {}
            self.plan_created_at = None
            self.last_reflection_at = None

        # If we have a current task but no plan, create one
        if (
            self.task_queue
            and self.task_queue.has_current_task()
            and not self.current_plan
        ):
            current_task = self.task_queue.get_current_task()

            # Mark the task as started
            self.task_queue.mark_current_task_started()

            # Create a plan for the current task
            self.current_plan = await self._create_plan(current_task.input_text)
            self.plan_created_at = datetime.now()

            # Store the plan in the task
            current_task.plan = self.current_plan

            # Update context with plan information
            context["planning"] = {
                "has_plan": True,
                "plan": self.current_plan,
                "current_step": self.current_plan["steps"][0]
                if self.current_plan["steps"]
                else None,
                "completed_steps": [],
                "plan_status": "in_progress",
                "task_queue": self.task_queue,
                "current_task": current_task.to_dict(),
            }

            # Log the plan
            logger.info(
                f"Created plan for task {current_task.task_id} with {len(self.current_plan['steps'])} steps"
            )
            logger.info(f"Plan: {json.dumps(self.current_plan, indent=2)}")

            # Add plan to agent memory
            agent.add_to_memory(
                {
                    "type": "plan",
                    "content": self.current_plan,
                    "task_id": current_task.task_id,
                }
            )

        # If we have a plan, update the current step
        if self.current_plan and self.current_step_index < len(
            self.current_plan["steps"]
        ):
            current_step = self.current_plan["steps"][self.current_step_index]
            context["planning"]["current_step"] = current_step

            # Log the current step
            logger.info(
                f"Current step ({self.current_step_index + 1}/{len(self.current_plan['steps'])}): {current_step['description']}"
            )

        return True

    async def process_action(
        self, agent: Agent, context: Dict[str, Any], action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an action to align with the current plan step.

        Args:
            agent: The agent processing the action
            context: Current context
            action: Action to process

        Returns:
            Processed action aligned with the plan
        """
        # Ensure planning context exists
        if "planning" not in context and self.current_plan:
            context["planning"] = {
                "has_plan": True,
                "plan": self.current_plan,
                "current_step": self.current_plan["steps"][self.current_step_index]
                if self.current_step_index < len(self.current_plan["steps"])
                else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan["status"],
                "task_queue": self.task_queue,
            }

        # If we have a plan and a current step, enhance the action
        if self.current_plan and self.current_step_index < len(
            self.current_plan["steps"]
        ):
            current_step = self.current_plan["steps"][self.current_step_index]

            # If the current step recommends a specific tool, suggest it
            if "tool" in current_step and "tool" not in action:
                tool_name = current_step["tool"]
                action["tool"] = tool_name

            # If the current step has specific parameters, suggest them
            if "parameters" in current_step and "args" not in action:
                tool_name = current_step.get("tool")
                parameters = current_step["parameters"]

                # Validate and fix parameters if a tool is specified
                if tool_name:
                    validation_result = validate_tool_parameters(tool_name, parameters)
                    if validation_result["errors"]:
                        logger.warning(
                            f"Parameter validation errors for {tool_name}: {validation_result['errors']}"
                        )
                    action["args"] = validation_result["parameters"]
                else:
                    action["args"] = parameters

            # Add plan context to the action
            action["plan_context"] = {
                "step_id": current_step["step_id"],
                "description": current_step["description"],
                "task_id": self.task_queue.get_current_task().task_id
                if self.task_queue and self.task_queue.has_current_task()
                else None,
            }

        return action

    async def process_prompt(
        self, agent: Agent, context: Dict[str, Any], prompt: str
    ) -> str:
        """
        Process the prompt to include planning information.

        Args:
            agent: The agent processing the prompt
            context: Current context
            prompt: Original prompt

        Returns:
            Enhanced prompt with planning information
        """
        # Ensure planning context exists
        if "planning" not in context:
            context["planning"] = {
                "has_plan": False,
                "current_step": None,
                "completed_steps": [],
                "plan_status": "not_started",
                "task_queue": self.task_queue,
            }

        # If we have a task queue and a current task, add task information to the prompt
        if self.task_queue and self.task_queue.has_current_task():
            current_task = self.task_queue.get_current_task()

            # Add task information to the prompt
            enhanced_prompt = f"{prompt}\n\nCurrent Task: {current_task.input_text}\n"

            # If we have a plan and a current step, add step information to the prompt
            if self.current_plan and self.current_step_index < len(
                self.current_plan["steps"]
            ):
                current_step = self.current_plan["steps"][self.current_step_index]

                # Add planning context to the prompt
                enhanced_prompt += f"\nCurrent Plan Step ({self.current_step_index + 1}/{len(self.current_plan['steps'])}):\n"
                enhanced_prompt += f"- Description: {current_step['description']}\n"

                if "tool" in current_step:
                    enhanced_prompt += f"- Recommended Tool: {current_step['tool']}\n"

                if "agent" in current_step:
                    enhanced_prompt += f"- Recommended Agent: {current_step['agent']}\n"

                if "dependencies" in current_step and current_step["dependencies"]:
                    enhanced_prompt += (
                        "- Dependencies: "
                        + ", ".join([str(dep) for dep in current_step["dependencies"]])
                        + "\n"
                    )

                return enhanced_prompt

            return enhanced_prompt

        # If we have a plan but no task queue, fall back to the original behavior
        if self.current_plan and self.current_step_index < len(
            self.current_plan["steps"]
        ):
            current_step = self.current_plan["steps"][self.current_step_index]

            # Add planning context to the prompt
            enhanced_prompt = f"{prompt}\n\nCurrent Plan Step ({self.current_step_index + 1}/{len(self.current_plan['steps'])}):\n"
            enhanced_prompt += f"- Description: {current_step['description']}\n"

            if "tool" in current_step:
                enhanced_prompt += f"- Recommended Tool: {current_step['tool']}\n"

            if "agent" in current_step:
                enhanced_prompt += f"- Recommended Agent: {current_step['agent']}\n"

            if "dependencies" in current_step and current_step["dependencies"]:
                enhanced_prompt += (
                    "- Dependencies: "
                    + ", ".join([str(dep) for dep in current_step["dependencies"]])
                    + "\n"
                )

            return enhanced_prompt

        return prompt

    async def process_result(
        self,
        agent: Agent,
        context: Dict[str, Any],
        prompt: str,
        action: Dict[str, Any],
        result: Any,
    ) -> Any:
        """
        Process the result to update the plan status.

        Args:
            agent: The agent processing the result
            context: Current context
            prompt: Original prompt
            action: Action that produced the result
            result: Result to process

        Returns:
            Processed result with plan updates
        """
        # Ensure planning context exists
        if "planning" not in context:
            context["planning"] = {
                "has_plan": bool(self.current_plan),
                "plan": self.current_plan,
                "current_step": self.current_plan["steps"][self.current_step_index]
                if self.current_plan
                and self.current_step_index < len(self.current_plan["steps"])
                else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan["status"]
                if self.current_plan
                else "not_started",
                "task_queue": self.task_queue,
            }

        # If we have a plan and a current step, update the plan status
        if self.current_plan and self.current_step_index < len(
            self.current_plan["steps"]
        ):
            current_step = self.current_plan["steps"][self.current_step_index]

            # Mark the current step as completed
            current_step["status"] = "completed"
            current_step["completed_at"] = datetime.now().isoformat()

            # Store the result
            self.step_results[current_step["step_id"]] = result

            # Add to completed steps
            self.completed_steps.append(current_step)

            # Update context
            context["planning"]["completed_steps"] = self.completed_steps

            # Move to the next step
            self.current_step_index += 1

            # Check if we need to reflect on the plan
            if (
                self.enable_step_reflection
                and len(self.completed_steps) >= self.min_steps_before_reflection
                and (
                    not self.last_reflection_at
                    or len(self.completed_steps) - self.last_reflection_at
                    >= self.min_steps_before_reflection
                )
            ):
                # Reflect on the plan and potentially update it
                if self.enable_dynamic_replanning:
                    updated_plan = await self._reflect_and_update_plan(
                        self.current_plan,
                        self.completed_steps,
                        self.step_results,
                        prompt,
                    )

                    # If the plan was updated, use the new plan
                    if updated_plan != self.current_plan:
                        logger.info("Plan updated based on reflection")
                        self.current_plan = updated_plan

                        # Update context with the new plan
                        context["planning"]["plan"] = self.current_plan

                        # If we have a task queue and a current task, update the task's plan
                        if self.task_queue and self.task_queue.has_current_task():
                            current_task = self.task_queue.get_current_task()
                            current_task.plan = self.current_plan

                        # Add updated plan to agent memory
                        agent.add_to_memory(
                            {"type": "updated_plan", "content": self.current_plan}
                        )

                # Update last reflection time
                self.last_reflection_at = len(self.completed_steps)

            # Check if the plan is completed
            if self.current_step_index >= len(self.current_plan["steps"]):
                logger.info("Plan completed")
                self.current_plan["status"] = "completed"
                self.current_plan["completed_at"] = datetime.now().isoformat()

                # Update context
                context["planning"]["plan_status"] = "completed"

                # If we have a task queue and a current task, mark the task as completed
                if self.task_queue and self.task_queue.has_current_task():
                    current_task = self.task_queue.get_current_task()

                    # Only mark as completed if not already completed
                    if current_task.status != "completed":
                        self.task_queue.mark_current_task_completed(result)

                        # Add task completion to agent memory, but check for duplicates first
                        memory_items = agent.get_memory()
                        task_already_in_memory = False

                        for item in memory_items:
                            if (
                                item.get("type") == "task_completed"
                                and item.get("content", {}).get("task_id")
                                == current_task.task_id
                            ):
                                task_already_in_memory = True
                                break

                        if not task_already_in_memory:
                            agent.add_to_memory(
                                {
                                    "type": "task_completed",
                                    "content": {
                                        "task_id": current_task.task_id,
                                        "input_text": current_task.input_text,
                                        "result": result,
                                    },
                                }
                            )

                # Add plan completion to agent memory, but check for duplicates first
                memory_items = agent.get_memory()
                plan_already_in_memory = False

                for item in memory_items:
                    if item.get("type") == "plan_completed" and item.get(
                        "content", {}
                    ).get("plan", {}).get("goal") == self.current_plan.get("goal"):
                        plan_already_in_memory = True
                        break

                if not plan_already_in_memory:
                    agent.add_to_memory(
                        {
                            "type": "plan_completed",
                            "content": {
                                "plan": self.current_plan,
                                "results": self.step_results,
                                "task_id": current_task.task_id
                                if current_task
                                else None,
                            },
                        }
                    )

                # Reset plan state for the next task
                self.current_plan = None
                self.current_step_index = 0
                self.completed_steps = []
                self.step_results = {}
                self.plan_created_at = None
                self.last_reflection_at = None
            else:
                # Update current step in context
                context["planning"]["current_step"] = self.current_plan["steps"][
                    self.current_step_index
                ]

        # Add plan information to the result
        if isinstance(result, dict):
            result["plan_status"] = {
                "current_step": self.current_step_index + 1 if self.current_plan else 0,
                "total_steps": len(self.current_plan["steps"])
                if self.current_plan
                else 0,
                "completed_steps": len(self.completed_steps),
                "plan_status": self.current_plan["status"]
                if self.current_plan
                else "not_started",
            }

            # Add task queue information if available
            if self.task_queue:
                result["task_status"] = {
                    "current_task": self.task_queue.get_current_task().to_dict()
                    if self.task_queue.has_current_task()
                    else None,
                    "completed_tasks": len(self.task_queue.get_completed_tasks()),
                    "pending_tasks": len(self.task_queue.get_pending_tasks()),
                    "failed_tasks": len(self.task_queue.get_failed_tasks()),
                    "total_tasks": len(self.task_queue.get_all_tasks()),
                }

        return result

    async def should_terminate(
        self, agent: Agent, context: Dict[str, Any], response: str
    ) -> bool:
        """
        Determine if the agent should terminate based on task and plan completion.

        Args:
            agent: The agent
            context: Current context
            response: Agent's response

        Returns:
            Whether the agent should terminate
        """
        # Ensure planning context exists
        if "planning" not in context and self.current_plan:
            context["planning"] = {
                "has_plan": True,
                "plan": self.current_plan,
                "current_step": self.current_plan["steps"][self.current_step_index]
                if self.current_step_index < len(self.current_plan["steps"])
                else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan["status"],
                "task_queue": self.task_queue,
            }

        # If we have a task queue, check if all tasks are completed
        if self.task_queue:
            if (
                not self.task_queue.has_pending_tasks()
                and not self.task_queue.has_current_task()
            ):
                logger.info("Suggesting termination: All tasks completed")
                return True

        # If we don't have a task queue but the plan is completed, suggest termination
        elif self.current_plan and self.current_plan.get("status") == "completed":
            logger.info("Suggesting termination: Plan completed")
            return True

        return False

    def set_task_queue(self, task_queue: TaskQueue) -> None:
        """
        Set the task queue.

        Args:
            task_queue: The task queue to use
        """
        self.task_queue = task_queue
