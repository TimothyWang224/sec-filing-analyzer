"""
Planning Capability

This module provides a capability for agents to create, manage, and execute plans
for complex tasks, particularly useful for coordinator agents that need to break down
tasks into manageable steps.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .base import Capability
from ..agents.base import Agent
from ..tools.tool_parameter_helper import validate_tool_parameters, generate_tool_parameter_prompt
from ..contracts import PlanStep, Plan, extract_value

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlanningCapability(Capability):
    """Capability for creating and managing execution plans for complex tasks."""

    def __init__(
        self,
        enable_dynamic_replanning: bool = True,
        enable_step_reflection: bool = True,
        min_steps_before_reflection: int = 2,
        max_plan_steps: int = 10,
        plan_detail_level: str = "high",  # "low", "medium", "high"
        planning_instructions: Optional[str] = None,
        is_coordinator: bool = False,
        respect_existing_plan: bool = True,
        max_planning_iterations: int = 1,  # Cap planning iterations to reduce token usage
        enable_plan_caching: bool = True  # Enable plan caching to avoid regenerating plans
    ):
        """
        Initialize the planning capability.

        Args:
            enable_dynamic_replanning: Whether to allow plan updates during execution
            enable_step_reflection: Whether to reflect on step results and adjust the plan
            min_steps_before_reflection: Minimum steps to execute before reflecting
            max_plan_steps: Maximum number of steps in a plan
            plan_detail_level: Level of detail in the generated plan
            planning_instructions: Optional custom instructions for plan generation
            is_coordinator: Whether this capability belongs to a coordinator agent
            respect_existing_plan: Whether to respect an existing plan from a coordinator
        """
        super().__init__(
            name="planning",
            description="Creates and manages execution plans for complex tasks"
        )
        self.enable_dynamic_replanning = enable_dynamic_replanning
        self.enable_step_reflection = enable_step_reflection
        self.min_steps_before_reflection = min_steps_before_reflection
        self.max_plan_steps = max_plan_steps
        self.plan_detail_level = plan_detail_level
        self.planning_instructions = planning_instructions
        self.is_coordinator = is_coordinator
        self.respect_existing_plan = respect_existing_plan

        # Plan state
        self.current_plan = None
        self.current_step_index = 0
        self.completed_steps = []
        self.step_results = {}
        self.plan_created_at = None
        self.last_reflection_at = None
        self.plan_owner = "coordinator" if is_coordinator else "agent"
        self.max_planning_iterations = max_planning_iterations
        self.enable_plan_caching = enable_plan_caching
        self.plan_cache = {}  # Cache plans by user input

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

        # Add planning to context
        context["planning"] = {
            "has_plan": False,
            "current_step": None,
            "completed_steps": [],
            "plan_status": "not_started"
        }

        return context

    def _update_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the context with the current plan state.

        Args:
            context: Current context

        Returns:
            Updated context
        """
        if self.current_plan:
            current_step = None
            if self.current_step_index < len(self.current_plan.steps):
                current_step = self.current_plan.steps[self.current_step_index]

            context["planning"] = {
                "has_plan": True,
                "plan": self.current_plan,
                "current_step": current_step,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan.status,
                "phase": self.agent.state.current_phase if hasattr(self.agent.state, 'current_phase') else "planning"
            }
        else:
            context["planning"] = {
                "has_plan": False,
                "current_step": None,
                "completed_steps": [],
                "plan_status": "not_started"
            }

        return context

    async def start_agent_loop(self, agent: Agent, context: Dict[str, Any]) -> bool:
        """
        Called at the start of each agent loop iteration.
        Creates a plan if one doesn't exist yet and manages phase transitions.

        Args:
            agent: The agent
            context: Current context

        Returns:
            Whether to continue the loop
        """
        # Check if we have a plan from a coordinator in the context
        coordinator_plan = None
        if self.respect_existing_plan and not self.is_coordinator:
            # Look for a plan in memory from the coordinator
            memory = agent.get_memory() if hasattr(agent, 'get_memory') else []
            for item in memory:
                if item.get("type") == "execution_plan" and "content" in item:
                    coordinator_plan = item["content"]
                    logger.info("Found coordinator plan in memory")
                    break

        # If we don't have a plan yet, create one or use the coordinator's plan
        if not self.current_plan:
            user_input = context.get("input", "")

            # Check if we have a cached plan for this input
            if self.enable_plan_caching and user_input in self.plan_cache:
                logger.info("Using cached plan for this input")
                self.current_plan = self.plan_cache[user_input]
            elif coordinator_plan and self.respect_existing_plan and not self.is_coordinator:
                # Use the coordinator's plan
                logger.info("Using coordinator's plan instead of creating a new one")

                # Create a Plan object from the coordinator's plan
                plan_dict = {
                    "goal": coordinator_plan.get("task_objective", f"Process: {user_input}"),
                    "steps": [],
                    "status": "in_progress",
                    "created_at": datetime.now().isoformat(),
                    "owner": "coordinator",
                    "can_modify": False
                }

                # Convert coordinator plan format to our plan format
                if "steps" in coordinator_plan:
                    for i, step in enumerate(coordinator_plan.get("steps", [])):
                        plan_dict["steps"].append({
                            "step_id": i + 1,
                            "description": step.get("description", f"Step {i+1}"),
                            "tool": step.get("tool"),
                            "parameters": step.get("parameters", {}),
                            "dependencies": step.get("dependencies", []),
                            "status": "pending"
                        })
                else:
                    # Create a single step from the task objective
                    plan_dict["steps"] = [{
                        "step_id": 1,
                        "description": coordinator_plan.get("task_objective", "Execute task"),
                        "status": "pending"
                    }]

                # Convert to Plan object
                self.current_plan = self._dict_to_plan(plan_dict)
            else:
                # Create a new plan
                self.current_plan = await self._create_plan(user_input)

                # Set owner and can_modify attributes
                self.current_plan.owner = self.plan_owner
                self.current_plan.can_modify = self.is_coordinator or not self.respect_existing_plan

                # Cache the plan if caching is enabled
                if self.enable_plan_caching:
                    self.plan_cache[user_input] = self.current_plan.model_copy()

            self.plan_created_at = datetime.now()

            # Initialize planning context
            if "planning" not in context:
                context["planning"] = {}

            # Update context with plan information
            context["planning"] = {
                "has_plan": True,
                "plan": self.current_plan,
                "current_step": self.current_plan.steps[0] if self.current_plan.steps else None,
                "completed_steps": [],
                "plan_status": "in_progress",
                "phase": agent.state.current_phase if hasattr(agent.state, 'current_phase') else "planning",
                "plan_owner": self.current_plan.owner if hasattr(self.current_plan, 'owner') else self.plan_owner,
                "can_modify": self.current_plan.can_modify if hasattr(self.current_plan, 'can_modify') else True
            }

        # Update the planning context with the current phase
        if "planning" in context and hasattr(agent.state, 'current_phase'):
            context["planning"]["phase"] = agent.state.current_phase

            # Log the plan
            logger.info(f"Created plan with {len(self.current_plan.steps)} steps")
            plan_dict = self._plan_to_dict(self.current_plan)
            logger.info(f"Plan: {json.dumps(plan_dict, indent=2)}")

            # Add plan to agent memory - store the Plan object, not a dictionary
            agent.add_to_memory({
                "type": "plan",
                "content": self.current_plan  # Store the Plan object directly
            })

            # Guard against infinite planning loops
            # If we already have a plan and we're still in planning phase, check if we should move to execution
            if hasattr(agent.state, 'current_phase') and agent.state.current_phase == 'planning':
                # If the plan is in progress and we've done at least one planning iteration, move to execution
                if self.current_plan.status == "in_progress" and agent.state.phase_iterations.get('planning', 0) > 0:
                    logger.info("Plan already exists and is in progress. Moving to execution phase.")
                    agent.state.set_phase('execution')

                    # Roll over unused tokens from planning to execution if the agent supports it
                    if hasattr(agent, '_rollover_token_surplus'):
                        agent._rollover_token_surplus('planning', 'execution')

        # Ensure planning context exists
        if "planning" not in context:
            context["planning"] = {
                "has_plan": bool(self.current_plan),
                "plan": self.current_plan,
                "current_step": self.current_plan.steps[self.current_step_index] if self.current_plan and self.current_step_index < len(self.current_plan.steps) else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan.status if self.current_plan else "not_started"
            }

        # If we have a plan, update the current step
        if self.current_plan and self.current_step_index < len(self.current_plan.steps):
            current_step = self.current_plan.steps[self.current_step_index]
            context["planning"]["current_step"] = current_step

            # Log the current step
            logger.info(f"Current step ({self.current_step_index + 1}/{len(self.current_plan.steps)}): {current_step.description}")

        return True

    async def process_prompt(self, agent: Agent, context: Dict[str, Any], prompt: str) -> str:
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
        if "planning" not in context and self.current_plan:
            context["planning"] = {
                "has_plan": True,
                "plan": self.current_plan,
                "current_step": self.current_plan.steps[self.current_step_index] if self.current_step_index < len(self.current_plan.steps) else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan.status,
                "phase": agent.state.current_phase if hasattr(agent.state, 'current_phase') else "planning"
            }

        # If we have a plan, add the current step to the prompt
        if self.current_plan and self.current_step_index < len(self.current_plan.steps):
            current_step = self.current_plan.steps[self.current_step_index]
            current_phase = agent.state.current_phase if hasattr(agent.state, 'current_phase') else "planning"

            # Add planning context to the prompt
            enhanced_prompt = f"{prompt}\n\nCurrent Phase: {current_phase.upper()}\n"
            enhanced_prompt += f"Current Plan Step ({self.current_step_index + 1}/{len(self.current_plan.steps)}):\n"
            enhanced_prompt += f"- Description: {current_step.description}\n"

            if current_step.tool:
                enhanced_prompt += f"- Recommended Tool: {current_step.tool}\n"

            if current_step.agent:
                enhanced_prompt += f"- Recommended Agent: {current_step.agent}\n"

            if current_step.dependencies:
                enhanced_prompt += "- Dependencies: " + ", ".join([str(dep) for dep in current_step.dependencies]) + "\n"

            # Add phase-specific guidance
            if current_phase == "planning":
                enhanced_prompt += "\nIn the PLANNING phase, focus on understanding the task and creating a detailed plan.\n"
                enhanced_prompt += "Analyze the question carefully and identify the key information needed.\n"
            elif current_phase == "execution":
                enhanced_prompt += "\nIn the EXECUTION phase, focus on gathering data and generating an initial answer.\n"
                enhanced_prompt += "Use the appropriate tools to collect the necessary information.\n"
            elif current_phase == "refinement":
                enhanced_prompt += "\nIn the REFINEMENT phase, focus on improving the answer quality.\n"
                enhanced_prompt += "Make the answer more concise, accurate, and directly responsive to the question.\n"

            # Add previous tool results from the tool ledger
            if hasattr(agent, 'tool_ledger'):
                enhanced_prompt += "\n" + agent.tool_ledger.format_for_prompt(limit=3)

            return enhanced_prompt

        return prompt

    async def process_action(
        self,
        agent: Agent,
        context: Dict[str, Any],
        action: Dict[str, Any]
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
                "current_step": self.current_plan.steps[self.current_step_index] if self.current_step_index < len(self.current_plan.steps) else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan.status
            }

        # If we have a plan and a current step, enhance the action
        if self.current_plan and self.current_step_index < len(self.current_plan.steps):
            current_step = self.current_plan.steps[self.current_step_index]

            # Check if all dependencies for this step are satisfied
            dependencies_satisfied = self._check_dependencies(current_step)
            if not dependencies_satisfied:
                # If dependencies are not satisfied, log a warning and modify the action
                missing_deps = self._get_missing_dependencies(current_step)
                logger.warning(f"Dependencies not satisfied for step {current_step.step_id}: {missing_deps}")

                # If the action is trying to use a tool that depends on previous steps,
                # modify it to get the missing dependency data first
                if "tool" in action:
                    # Find a step that would satisfy the dependency
                    dependency_step = self._find_dependency_step(missing_deps[0] if missing_deps else None)
                    if dependency_step and dependency_step.tool:
                        # Replace the current tool with the dependency tool
                        logger.info(f"Replacing tool {action.get('tool')} with dependency tool {dependency_step.tool}")
                        action["tool"] = dependency_step.tool
                        if dependency_step.parameters:
                            tool_name = dependency_step.tool
                            parameters = dependency_step.parameters
                            validation_result = validate_tool_parameters(tool_name, parameters)
                            action["args"] = validation_result["parameters"]

                        # Add dependency context to the action
                        action["dependency_context"] = {
                            "original_step_id": current_step.step_id,
                            "dependency_step_id": dependency_step.step_id,
                            "missing_dependencies": missing_deps
                        }

                        return action

            # If the current step recommends a specific tool, suggest it
            if current_step.tool and "tool" not in action:
                tool_name = current_step.tool
                action["tool"] = tool_name

            # If the current step has specific parameters, suggest them
            if current_step.parameters and "args" not in action:
                tool_name = current_step.tool
                parameters = current_step.parameters

                # Validate and fix parameters if a tool is specified
                if tool_name:
                    validation_result = validate_tool_parameters(tool_name, parameters)
                    if validation_result["errors"]:
                        logger.warning(f"Parameter validation errors for {tool_name}: {validation_result['errors']}")
                    action["args"] = validation_result["parameters"]
                else:
                    action["args"] = parameters

            # Add plan context to the action
            action["plan_context"] = {
                "step_id": current_step.step_id,
                "description": current_step.description
            }

        return action

    def _check_dependencies(self, step: PlanStep) -> bool:
        """Check if all dependencies for a step are satisfied."""
        if not step.dependencies:
            return True  # No dependencies to satisfy

        # Check each dependency
        for dep_id in step.dependencies:
            # Find the dependency step in completed steps
            dep_satisfied = False
            for completed_step in self.completed_steps:
                if completed_step.step_id == dep_id:
                    dep_satisfied = True
                    break

            if not dep_satisfied:
                return False  # At least one dependency is not satisfied

        return True  # All dependencies are satisfied

    def _get_missing_dependencies(self, step: PlanStep) -> List[int]:
        """Get a list of missing dependencies for a step."""
        missing_deps = []

        if not step.dependencies:
            return missing_deps  # No dependencies to check

        # Check each dependency
        for dep_id in step.dependencies:
            # Find the dependency step in completed steps
            dep_satisfied = False
            for completed_step in self.completed_steps:
                if completed_step.step_id == dep_id:
                    dep_satisfied = True
                    break

            if not dep_satisfied:
                missing_deps.append(dep_id)

        return missing_deps

    def _find_dependency_step(self, dep_id: Optional[int]) -> Optional[PlanStep]:
        """Find a step that would satisfy a dependency."""
        if dep_id is None:
            return None

        # Look for the dependency step in the plan
        if self.current_plan and self.current_plan.steps:
            for step in self.current_plan.steps:
                if step.step_id == dep_id:
                    return step

        return None

    async def process_result(
        self,
        agent: Agent,
        context: Dict[str, Any],
        prompt: str,
        action: Dict[str, Any],
        result: Any
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
                "current_step": self.current_plan.steps[self.current_step_index] if self.current_plan and self.current_step_index < len(self.current_plan.steps) else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan.status if self.current_plan else "not_started"
            }

        # If we have a plan and a current step, update the plan status
        if self.current_plan and self.current_step_index < len(self.current_plan.steps):
            current_step = self.current_plan.steps[self.current_step_index]

            # Mark the current step as completed
            current_step.status = "completed"
            current_step.completed_at = datetime.now().isoformat()

            # Store the result
            self.step_results[current_step.step_id] = result

            # Add to completed steps
            self.completed_steps.append(current_step)

            # Update context
            context["planning"]["completed_steps"] = self.completed_steps

            # Move to the next step
            self.current_step_index += 1

            # Check if we should skip the next step based on success criteria
            if self.current_step_index < len(self.current_plan.steps):
                next_step = self.current_plan.steps[self.current_step_index]
                # Use the agent's _execute_current_step method to check if we should skip
                if hasattr(agent, '_execute_current_step'):
                    should_skip = await agent._execute_current_step(next_step)
                    if should_skip:
                                # If we skipped, update the step_results and completed_steps
                        self.step_results[next_step.step_id] = {"skipped": True, "reason": "Success criterion already satisfied"}
                        self.completed_steps.append(next_step)
                        # Move to the next step
                        self.current_step_index += 1

            # Check if we need to reflect on the plan
            if (self.enable_step_reflection and
                len(self.completed_steps) >= self.min_steps_before_reflection and
                (not self.last_reflection_at or
                 len(self.completed_steps) - self.last_reflection_at >= self.min_steps_before_reflection)):

                # Reflect on the plan and potentially update it if allowed
                can_modify = self.current_plan.can_modify if hasattr(self.current_plan, 'can_modify') else True
                if self.enable_dynamic_replanning and can_modify:
                    updated_plan = await self._reflect_and_update_plan(
                        self.current_plan,
                        self.completed_steps,
                        self.step_results,
                        prompt
                    )

                    # If the plan was updated, use the new plan
                    if updated_plan != self.current_plan:
                        logger.info("Plan updated based on reflection")
                        self.current_plan = updated_plan

                        # Preserve ownership and modification settings
                        self.current_plan.owner = self.plan_owner
                        self.current_plan.can_modify = can_modify

                        # Update context with the new plan
                        context["planning"]["plan"] = self.current_plan

                        # Add updated plan to agent memory - store the Plan object, not a dictionary
                        agent.add_to_memory({
                            "type": "updated_plan",
                            "content": self.current_plan  # Store the Plan object directly
                        })
                elif not can_modify and self.enable_dynamic_replanning:
                    logger.info("Plan reflection skipped - plan is not modifiable (owned by coordinator)")

                # Update last reflection time
                self.last_reflection_at = len(self.completed_steps)

            # Check if the plan is completed
            if self.current_step_index >= len(self.current_plan.steps):
                logger.info("Plan completed")
                self.current_plan.status = "completed"
                self.current_plan.completed_at = datetime.now().isoformat()

                # Update context
                context["planning"]["plan_status"] = "completed"

                # Add plan completion to agent memory - store the Plan object, not a dictionary
                agent.add_to_memory({
                    "type": "plan_completed",
                    "content": {
                        "plan": self.current_plan,  # Store the Plan object directly
                        "results": self.step_results
                    }
                })
            else:
                # Update current step in context
                context["planning"]["current_step"] = self.current_plan.steps[self.current_step_index]

        # Add plan information to the result
        if isinstance(result, dict):
            result["plan_status"] = {
                "current_step": self.current_step_index + 1 if self.current_plan else 0,
                "total_steps": len(self.current_plan.steps) if self.current_plan else 0,
                "completed_steps": len(self.completed_steps),
                "plan_status": self.current_plan.status if self.current_plan else "not_started"
            }

        return result

    async def should_terminate(self, agent: Agent, context: Dict[str, Any], response: str) -> bool:
        """
        Determine if the agent should terminate based on plan completion.

        Args:
            agent: The agent
            context: Current context
            response: Agent's response

        Returns:
            Whether the agent should terminate
        """
        # Ensure planning context exists
        if "planning" not in context and self.current_plan:
            # Convert Plan object to dictionary for context
            plan_dict = self._plan_to_dict(self.current_plan)
            current_step = None
            if self.current_step_index < len(self.current_plan.steps):
                current_step = self._plan_step_to_dict(self.current_plan.steps[self.current_step_index])

            context["planning"] = {
                "has_plan": True,
                "plan": plan_dict,
                "current_step": current_step,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan.status
            }

        # If the plan is completed, suggest termination
        if self.current_plan and self.current_plan.status == "completed":
            logger.info("Suggesting termination: Plan completed")
            return True

        return False

    async def _create_plan(self, user_input: str) -> Plan:
        """
        Create a plan based on the user input.

        Args:
            user_input: User's input

        Returns:
            Plan object
        """
        # Create a prompt for plan generation
        prompt = self._create_plan_generation_prompt(user_input)

        # Generate the plan using the agent's LLM
        system_prompt = """You are an expert financial planning assistant.
Your task is to create a detailed, step-by-step plan to address the user's request.
Each step should be specific, actionable, and include the necessary tools or agents.
Focus on efficiency - do NOT create redundant validation steps that re-check data already retrieved.
The system automatically validates tool outputs using the done_check condition.
Return your plan in the exact JSON format requested."""

        plan_response = await self.agent.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            return_usage=True
        )

        # Count tokens if return_usage is True
        if isinstance(plan_response, dict) and "usage" in plan_response:
            self.agent.state.count_tokens(plan_response["usage"]["total_tokens"], "planning")
            plan_response = plan_response["content"]

        # Parse the plan from the response
        try:
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', plan_response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(0))
            else:
                # Create a default plan if parsing fails
                plan = self._create_default_plan(user_input)
                logger.warning("Failed to parse plan from LLM response, using default plan")
        except Exception as e:
            logger.error(f"Error parsing plan: {str(e)}")
            plan = self._create_default_plan(user_input)

        # Validate and clean up the plan
        plan_dict = self._validate_and_clean_plan(plan, user_input)

        # Convert to Plan object
        return self._dict_to_plan(plan_dict)

    def _create_plan_generation_prompt(self, user_input: str) -> str:
        """Create a prompt for plan generation."""
        # Get available tools and agents
        available_tools = []
        if hasattr(self.agent, 'environment') and hasattr(self.agent.environment, 'get_available_tools'):
            available_tools = list(self.agent.environment.get_available_tools().keys())

        available_agents = ["financial_analyst", "risk_analyst", "qa_specialist"]

        # Create tool parameter documentation
        tool_parameter_docs = ""
        for tool_name in available_tools:
            tool_parameter_docs += generate_tool_parameter_prompt(tool_name) + "\n\n"

        # Create the prompt
        if self.planning_instructions:
            # Use custom planning instructions if provided
            prompt = f"""
Based on the following user request, create a detailed step-by-step plan:

User Request: {user_input}

Available Tools: {', '.join(available_tools)}
Available Agents: {', '.join(available_agents)}

Tool Parameter Documentation:
{tool_parameter_docs}

{self.planning_instructions}
"""
        else:
            # Use default planning instructions
            prompt = f"""
Based on the following user request, create a detailed step-by-step plan:

User Request: {user_input}

Available Tools: {', '.join(available_tools)}
Available Agents: {', '.join(available_agents)}

Tool Parameter Documentation:
{tool_parameter_docs}

For SEC filing analysis, typical steps include:
1. Retrieving relevant documents or data
2. Analyzing financial metrics and statements
3. Identifying risks and opportunities
4. Comparing with industry benchmarks
5. Generating insights and recommendations

Your plan should include:
1. A clear goal based on the user's request
2. A sequence of steps to achieve that goal
3. For each step, specify:
   - A description of what to do
   - Which tool or agent to use (if applicable)
   - Any dependencies on previous steps
   - Parameters or inputs needed
   - Expected key to check in results (for automatic success detection)
   - Output path to the expected data (for automatic success detection)
   - A "done" check condition that specifies when this step is considered complete

Each step must have a clear contract with its tool:
- If using a tool, specify exactly what output key you expect from that tool
- Define a simple condition to check if the step is complete (e.g., "value not None", "list length > 0")
- Make sure the expected_key and output_path match the actual output structure of the tool

IMPORTANT: Do NOT create separate validation steps to verify data you've already retrieved. The system automatically validates tool outputs using the done_check condition. Creating redundant validation steps wastes resources and time.

Return your plan as a JSON object with this structure:
```json
{{
  "goal": "Clear statement of the overall goal",
  "steps": [
    {{
      "step_id": 1,
      "description": "Detailed description of what to do",
      "tool": "optional_tool_name",
      "agent": "optional_agent_name",
      "parameters": {{"param1": "value1", "param2": "value2"}},
      "dependencies": [],
      "expected_key": "optional_key_to_check_in_results",
      "output_path": ["optional", "path", "to", "result"],
      "done_check": "value not None",
      "status": "pending"
    }},
    ...
  ],
  "status": "in_progress",
  "created_at": "{datetime.now().isoformat()}"
}}
```

The plan should be detailed but concise, with no more than {self.max_plan_steps} steps.
"""
        return prompt

    def _create_default_plan(self, user_input: str) -> Dict[str, Any]:
        """Create a default plan dictionary if plan generation fails."""
        # Parse the user input to extract potential ticker and metrics
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', user_input.upper())
        ticker = ticker_match.group(1) if ticker_match else "AAPL"

        # Look for common financial metrics
        metrics = []
        for metric in ["revenue", "income", "profit", "earnings", "eps", "assets", "liabilities", "debt"]:
            if metric.lower() in user_input.lower():
                metrics.append(metric.capitalize())

        # If no metrics found, default to Revenue
        if not metrics:
            metrics = ["Revenue"]

        # Extract year if present
        year_match = re.search(r'(20\d{2})', user_input)
        year = year_match.group(1) if year_match else str(datetime.now().year - 1)

        # Create a more specific default plan based on the extracted information
        return {
            "goal": f"Analyze {', '.join(metrics)} for {ticker} in {year}",
            "steps": [
                {
                    "step_id": 1,
                    "description": f"Retrieve {', '.join(metrics)} data for {ticker} in {year}",
                    "tool": "sec_financial_data",
                    "parameters": {
                        "query_type": "financial_facts",
                        "parameters": {
                            "ticker": ticker,
                            "metrics": metrics,
                            "start_date": f"{year}-01-01",
                            "end_date": f"{year}-12-31"
                        }
                    },
                    "dependencies": [],
                    "expected_key": f"{ticker}_{'_'.join(metrics)}_{year}",
                    "output_path": ["results"],
                    "done_check": "results is not None and len(results) > 0",
                    "status": "pending"
                },
                {
                    "step_id": 2,
                    "description": f"Search for context about {ticker}'s {', '.join(metrics)} in {year}",
                    "tool": "sec_semantic_search",
                    "parameters": {
                        "query": f"{', '.join(metrics)} performance in {year}",
                        "companies": [ticker],
                        "top_k": 3
                    },
                    "dependencies": [],
                    "expected_key": f"{ticker}_search_{year}",
                    "output_path": ["results"],
                    "done_check": "results is not None and len(results) > 0",
                    "status": "pending"
                },
                {
                    "step_id": 3,
                    "description": f"Analyze {ticker}'s {', '.join(metrics)} performance",
                    "agent": "financial_analyst",
                    "parameters": {
                        "analysis_type": "metric_performance",
                        "ticker": ticker,
                        "metrics": metrics,
                        "year": year
                    },
                    "dependencies": [1, 2],
                    "expected_key": f"{ticker}_analysis_{year}",
                    "done_check": f"{ticker}_analysis_{year} is not None",
                    "status": "pending"
                }
            ],
            "status": "in_progress",
            "created_at": datetime.now().isoformat()
        }

    def _validate_and_clean_plan(self, plan: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """Validate and clean up the plan."""
        # Ensure required fields exist
        if "goal" not in plan:
            plan["goal"] = f"Analyze financial information based on: {user_input}"

        if "steps" not in plan or not isinstance(plan["steps"], list) or not plan["steps"]:
            plan["steps"] = self._create_default_plan(user_input)["steps"]

        if "status" not in plan:
            plan["status"] = "in_progress"

        if "created_at" not in plan:
            plan["created_at"] = datetime.now().isoformat()

        # Validate and clean each step
        for i, step in enumerate(plan["steps"]):
            # Ensure step_id exists and is sequential
            if "step_id" not in step:
                step["step_id"] = i + 1

            # Ensure description exists
            if "description" not in step:
                step["description"] = f"Step {i + 1}"

            # Ensure status exists
            if "status" not in step:
                step["status"] = "pending"

            # Ensure dependencies is a list of integers
            if "dependencies" not in step:
                step["dependencies"] = []
            elif not isinstance(step["dependencies"], list):
                step["dependencies"] = []
            else:
                # Convert dependencies to integers
                dependencies = []
                for dep in step["dependencies"]:
                    if isinstance(dep, int):
                        dependencies.append(dep)
                    elif isinstance(dep, dict) and "step_id" in dep:
                        # If it's a dictionary with step_id, extract the step_id
                        dependencies.append(dep["step_id"])
                    elif isinstance(dep, str) and dep.isdigit():
                        # If it's a string that can be converted to an integer
                        dependencies.append(int(dep))
                step["dependencies"] = dependencies

            # Ensure expected_key exists if tool is specified
            if "tool" in step and "expected_key" not in step:
                # Extract parameters for more specific keys
                parameters = step.get("parameters", {})
                query_type = parameters.get("query_type", "")
                params = parameters.get("parameters", {})

                # Set a default expected key based on the tool and parameters
                if step["tool"] == "sec_financial_data":
                    ticker = params.get("ticker", "")
                    metrics = params.get("metrics", [])
                    start_date = params.get("start_date", "")
                    end_date = params.get("end_date", "")

                    # Extract year from date if available
                    year = ""
                    if start_date:
                        year_match = re.search(r"((?:19|20)\d{2})", start_date)
                        if year_match:
                            year = year_match.group(1)

                    # Create a specific key based on parameters
                    if ticker and metrics and len(metrics) == 1 and year:
                        # Single metric with year
                        step["expected_key"] = f"{ticker}_{metrics[0]}_{year}"
                    elif ticker and metrics and len(metrics) == 1:
                        # Single metric without year
                        step["expected_key"] = f"{ticker}_{metrics[0]}"
                    elif ticker and year:
                        # Multiple metrics with year
                        step["expected_key"] = f"{ticker}_financial_facts_{year}"
                    elif ticker:
                        # Multiple metrics without year
                        step["expected_key"] = f"{ticker}_financial_facts"
                    else:
                        # Fallback
                        step["expected_key"] = "financial_facts"
                elif step["tool"] == "sec_semantic_search":
                    query = params.get("query", "")
                    ticker = params.get("ticker", "")

                    # Create a specific key based on parameters
                    if ticker and query:
                        # Shorten and clean query for key
                        short_query = re.sub(r'[^a-zA-Z0-9]', '_', query[:20]).strip('_')
                        step["expected_key"] = f"{ticker}_search_{short_query}"
                    elif ticker:
                        step["expected_key"] = f"{ticker}_semantic_search"
                    else:
                        step["expected_key"] = "semantic_search"
                else:
                    # Default key for other tools
                    step["expected_key"] = f"{step['tool']}_result"

            # Ensure output_path exists if expected_key exists
            if "expected_key" in step and "output_path" not in step:
                # Set a default output path based on the tool and query type
                if "tool" in step:
                    parameters = step.get("parameters", {})
                    query_type = parameters.get("query_type", "")

                    if step["tool"] == "sec_financial_data":
                        if query_type == "financial_facts":
                            # Financial facts have a nested structure with metrics and years
                            params = parameters.get("parameters", {})
                            metrics = params.get("metrics", [])

                            if len(metrics) == 1:
                                # For a single metric, point directly to it
                                step["output_path"] = ["results", metrics[0]]
                            else:
                                # For multiple metrics, point to results
                                step["output_path"] = ["results"]
                        else:
                            # Default path for other query types
                            step["output_path"] = ["results"]
                    elif step["tool"] == "sec_semantic_search":
                        # Semantic search returns results in a list
                        step["output_path"] = ["results"]
                    else:
                        # Default path for other tools
                        step["output_path"] = ["results"]

            # Ensure done_check exists if expected_key exists
            if "expected_key" in step and "done_check" not in step:
                expected_key = step["expected_key"]
                # Set a default done_check based on the expected_key
                step["done_check"] = f"{expected_key} is not None"

        # Limit the number of steps
        if len(plan["steps"]) > self.max_plan_steps:
            plan["steps"] = plan["steps"][:self.max_plan_steps]

        return plan

    def _dict_to_plan_step(self, step_dict: Dict[str, Any]) -> PlanStep:
        """Convert a dictionary-based plan step to a PlanStep model."""
        # Ensure dependencies is a list of integers
        if "dependencies" in step_dict:
            # Handle case where dependencies might be a list of dictionaries or other non-integer values
            dependencies = []
            for dep in step_dict["dependencies"]:
                if isinstance(dep, int):
                    dependencies.append(dep)
                elif isinstance(dep, dict) and "step_id" in dep:
                    # If it's a dictionary with step_id, extract the step_id
                    dependencies.append(dep["step_id"])
                elif isinstance(dep, str) and dep.isdigit():
                    # If it's a string that can be converted to an integer
                    dependencies.append(int(dep))

            # Replace the dependencies in the step_dict
            step_dict["dependencies"] = dependencies

        try:
            return PlanStep(**step_dict)
        except Exception as e:
            logger.error(f"Error creating PlanStep from dictionary: {e}")
            logger.error(f"Step dictionary: {step_dict}")
            # Create a minimal valid PlanStep
            return PlanStep(
                step_id=step_dict.get("step_id", 0),
                description=step_dict.get("description", "Unknown step"),
                tool=step_dict.get("tool"),
                parameters=step_dict.get("parameters", {}),
                dependencies=[]
            )

    def _plan_step_to_dict(self, plan_step: PlanStep) -> Dict[str, Any]:
        """Convert a PlanStep model to a dictionary."""
        return plan_step.model_dump()

    def _dict_to_plan(self, plan_dict: Dict[str, Any]) -> Plan:
        """Convert a dictionary-based plan to a Plan model."""
        # Convert steps to PlanStep models
        steps = [self._dict_to_plan_step(step) for step in plan_dict.get("steps", [])]

        # Create a Plan model
        return Plan(
            goal=plan_dict.get("goal", "Process the input"),
            steps=steps,
            status=plan_dict.get("status", "pending"),
            created_at=plan_dict.get("created_at"),
            completed_at=plan_dict.get("completed_at"),
            owner=plan_dict.get("owner", "agent"),
            can_modify=plan_dict.get("can_modify", True)
        )

    def _plan_to_dict(self, plan: Plan) -> Dict[str, Any]:
        """Convert a Plan model to a dictionary."""
        return plan.model_dump()

    async def _reflect_and_update_plan(
        self,
        current_plan: Plan,
        completed_steps: List[Dict[str, Any]],
        step_results: Dict[int, Any],
        original_input: str
    ) -> Plan:
        """
        Reflect on the plan execution so far and potentially update the plan.

        Args:
            current_plan: Current plan
            completed_steps: Steps completed so far
            step_results: Results of completed steps
            original_input: Original user input

        Returns:
            Updated plan
        """
        # Convert Plan object to dictionary for the prompt
        plan_dict = self._plan_to_dict(current_plan)

        # Create a prompt for plan reflection
        prompt = f"""
I'm executing a plan to address this request: "{original_input}"

Current plan:
{json.dumps(plan_dict, indent=2)}

Completed steps ({len(completed_steps)}/{len(plan_dict['steps'])}):
{json.dumps(completed_steps, indent=2)}

Based on the results so far, should I:
1. Continue with the current plan
2. Modify the remaining steps
3. Add new steps
4. Remove unnecessary steps

IMPORTANT GUIDELINES:
- Do NOT add redundant validation steps that re-check data already retrieved
- The system automatically validates tool outputs using the done_check condition
- Focus on adding steps that provide new information or analysis
- Remove any steps that simply verify data we already have

If modifications are needed, provide the updated plan in the same JSON format.
If no changes are needed, return the current plan unchanged.
"""

        system_prompt = """You are an expert at adaptive planning.
Analyze the plan execution so far and determine if changes are needed.
Consider what has been learned from completed steps and whether the plan still addresses the original goal.
Return the updated plan in the exact JSON format of the original plan."""

        reflection_response = await self.agent.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            return_usage=True
        )

        # Count tokens if return_usage is True
        if isinstance(reflection_response, dict) and "usage" in reflection_response:
            self.agent.state.count_tokens(reflection_response["usage"]["total_tokens"], "planning")
            reflection_response = reflection_response["content"]

        # Parse the updated plan from the response
        try:
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', reflection_response, re.DOTALL)
            if json_match:
                updated_plan_dict = json.loads(json_match.group(0))

                # Validate and clean the updated plan
                updated_plan_dict = self._validate_and_clean_plan(updated_plan_dict, original_input)

                # Preserve completed steps
                for step in updated_plan_dict["steps"]:
                    step_id = step["step_id"]
                    # If this step was already completed, mark it as such
                    for completed_step in completed_steps:
                        if isinstance(completed_step, dict) and completed_step.get("step_id") == step_id:
                            step["status"] = "completed"
                            step["completed_at"] = completed_step.get("completed_at")
                        elif hasattr(completed_step, 'step_id') and completed_step.step_id == step_id:
                            step["status"] = "completed"
                            step["completed_at"] = completed_step.completed_at if hasattr(completed_step, 'completed_at') else None

                # Convert to Plan object
                return self._dict_to_plan(updated_plan_dict)
            else:
                # No valid JSON found, return the current plan
                return current_plan
        except Exception as e:
            logger.error(f"Error parsing updated plan: {str(e)}")
            return current_plan
