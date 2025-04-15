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
        plan_detail_level: str = "high"  # "low", "medium", "high"
    ):
        """
        Initialize the planning capability.

        Args:
            enable_dynamic_replanning: Whether to allow plan updates during execution
            enable_step_reflection: Whether to reflect on step results and adjust the plan
            min_steps_before_reflection: Minimum steps to execute before reflecting
            max_plan_steps: Maximum number of steps in a plan
            plan_detail_level: Level of detail in the generated plan
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

        # Plan state
        self.current_plan = None
        self.current_step_index = 0
        self.completed_steps = []
        self.step_results = {}
        self.plan_created_at = None
        self.last_reflection_at = None

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
        # If we don't have a plan yet, create one
        if not self.current_plan:
            user_input = context.get("input", "")
            self.current_plan = await self._create_plan(user_input)
            self.plan_created_at = datetime.now()

            # Initialize planning context
            if "planning" not in context:
                context["planning"] = {}

            # Update context with plan information
            context["planning"] = {
                "has_plan": True,
                "plan": self.current_plan,
                "current_step": self.current_plan["steps"][0] if self.current_plan["steps"] else None,
                "completed_steps": [],
                "plan_status": "in_progress",
                "phase": agent.state.current_phase if hasattr(agent.state, 'current_phase') else "planning"
            }

        # Update the planning context with the current phase
        if "planning" in context and hasattr(agent.state, 'current_phase'):
            context["planning"]["phase"] = agent.state.current_phase

            # Log the plan
            logger.info(f"Created plan with {len(self.current_plan['steps'])} steps")
            logger.info(f"Plan: {json.dumps(self.current_plan, indent=2)}")

            # Add plan to agent memory
            agent.add_to_memory({
                "type": "plan",
                "content": self.current_plan
            })

        # Ensure planning context exists
        if "planning" not in context:
            context["planning"] = {
                "has_plan": bool(self.current_plan),
                "plan": self.current_plan,
                "current_step": self.current_plan["steps"][self.current_step_index] if self.current_plan and self.current_step_index < len(self.current_plan["steps"]) else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan["status"] if self.current_plan else "not_started"
            }

        # If we have a plan, update the current step
        if self.current_plan and self.current_step_index < len(self.current_plan["steps"]):
            current_step = self.current_plan["steps"][self.current_step_index]
            context["planning"]["current_step"] = current_step

            # Log the current step
            logger.info(f"Current step ({self.current_step_index + 1}/{len(self.current_plan['steps'])}): {current_step['description']}")

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
                "current_step": self.current_plan["steps"][self.current_step_index] if self.current_step_index < len(self.current_plan["steps"]) else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan["status"],
                "phase": agent.state.current_phase if hasattr(agent.state, 'current_phase') else "planning"
            }

        # If we have a plan, add the current step to the prompt
        if self.current_plan and self.current_step_index < len(self.current_plan["steps"]):
            current_step = self.current_plan["steps"][self.current_step_index]
            current_phase = agent.state.current_phase if hasattr(agent.state, 'current_phase') else "planning"

            # Add planning context to the prompt
            enhanced_prompt = f"{prompt}\n\nCurrent Phase: {current_phase.upper()}\n"
            enhanced_prompt += f"Current Plan Step ({self.current_step_index + 1}/{len(self.current_plan['steps'])}):\n"
            enhanced_prompt += f"- Description: {current_step['description']}\n"

            if "tool" in current_step:
                enhanced_prompt += f"- Recommended Tool: {current_step['tool']}\n"

            if "agent" in current_step:
                enhanced_prompt += f"- Recommended Agent: {current_step['agent']}\n"

            if "dependencies" in current_step and current_step["dependencies"]:
                enhanced_prompt += "- Dependencies: " + ", ".join([str(dep) for dep in current_step["dependencies"]]) + "\n"

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
                "current_step": self.current_plan["steps"][self.current_step_index] if self.current_step_index < len(self.current_plan["steps"]) else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan["status"]
            }

        # If we have a plan and a current step, enhance the action
        if self.current_plan and self.current_step_index < len(self.current_plan["steps"]):
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
                        logger.warning(f"Parameter validation errors for {tool_name}: {validation_result['errors']}")
                    action["args"] = validation_result["parameters"]
                else:
                    action["args"] = parameters

            # Add plan context to the action
            action["plan_context"] = {
                "step_id": current_step["step_id"],
                "description": current_step["description"]
            }

        return action

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
                "current_step": self.current_plan["steps"][self.current_step_index] if self.current_plan and self.current_step_index < len(self.current_plan["steps"]) else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan["status"] if self.current_plan else "not_started"
            }

        # If we have a plan and a current step, update the plan status
        if self.current_plan and self.current_step_index < len(self.current_plan["steps"]):
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
            if (self.enable_step_reflection and
                len(self.completed_steps) >= self.min_steps_before_reflection and
                (not self.last_reflection_at or
                 len(self.completed_steps) - self.last_reflection_at >= self.min_steps_before_reflection)):

                # Reflect on the plan and potentially update it
                if self.enable_dynamic_replanning:
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

                        # Update context with the new plan
                        context["planning"]["plan"] = self.current_plan

                        # Add updated plan to agent memory
                        agent.add_to_memory({
                            "type": "updated_plan",
                            "content": self.current_plan
                        })

                # Update last reflection time
                self.last_reflection_at = len(self.completed_steps)

            # Check if the plan is completed
            if self.current_step_index >= len(self.current_plan["steps"]):
                logger.info("Plan completed")
                self.current_plan["status"] = "completed"
                self.current_plan["completed_at"] = datetime.now().isoformat()

                # Update context
                context["planning"]["plan_status"] = "completed"

                # Add plan completion to agent memory
                agent.add_to_memory({
                    "type": "plan_completed",
                    "content": {
                        "plan": self.current_plan,
                        "results": self.step_results
                    }
                })
            else:
                # Update current step in context
                context["planning"]["current_step"] = self.current_plan["steps"][self.current_step_index]

        # Add plan information to the result
        if isinstance(result, dict):
            result["plan_status"] = {
                "current_step": self.current_step_index + 1 if self.current_plan else 0,
                "total_steps": len(self.current_plan["steps"]) if self.current_plan else 0,
                "completed_steps": len(self.completed_steps),
                "plan_status": self.current_plan["status"] if self.current_plan else "not_started"
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
            context["planning"] = {
                "has_plan": True,
                "plan": self.current_plan,
                "current_step": self.current_plan["steps"][self.current_step_index] if self.current_step_index < len(self.current_plan["steps"]) else None,
                "completed_steps": self.completed_steps.copy(),
                "plan_status": self.current_plan["status"]
            }

        # If the plan is completed, suggest termination
        if self.current_plan and self.current_plan.get("status") == "completed":
            logger.info("Suggesting termination: Plan completed")
            return True

        return False

    async def _create_plan(self, user_input: str) -> Dict[str, Any]:
        """
        Create a plan based on the user input.

        Args:
            user_input: User's input

        Returns:
            Plan dictionary
        """
        # Create a prompt for plan generation
        prompt = self._create_plan_generation_prompt(user_input)

        # Generate the plan using the agent's LLM
        system_prompt = """You are an expert financial planning assistant.
Your task is to create a detailed, step-by-step plan to address the user's request.
Each step should be specific, actionable, and include the necessary tools or agents.
Return your plan in the exact JSON format requested."""

        plan_response = await self.agent.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )

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
        plan = self._validate_and_clean_plan(plan, user_input)

        return plan

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
        """Create a default plan if plan generation fails."""
        return {
            "goal": f"Analyze financial information based on: {user_input}",
            "steps": [
                {
                    "step_id": 1,
                    "description": "Retrieve relevant financial data",
                    "tool": "sec_financial_data",
                    "parameters": {},
                    "dependencies": [],
                    "status": "pending"
                },
                {
                    "step_id": 2,
                    "description": "Search for relevant context in SEC filings",
                    "tool": "sec_semantic_search",
                    "parameters": {},
                    "dependencies": [],
                    "status": "pending"
                },
                {
                    "step_id": 3,
                    "description": "Analyze financial metrics and trends",
                    "agent": "financial_analyst",
                    "parameters": {},
                    "dependencies": [1, 2],
                    "status": "pending"
                },
                {
                    "step_id": 4,
                    "description": "Generate comprehensive report",
                    "dependencies": [3],
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

            # Ensure dependencies is a list
            if "dependencies" not in step or not isinstance(step["dependencies"], list):
                step["dependencies"] = []

        # Limit the number of steps
        if len(plan["steps"]) > self.max_plan_steps:
            plan["steps"] = plan["steps"][:self.max_plan_steps]

        return plan

    async def _reflect_and_update_plan(
        self,
        current_plan: Dict[str, Any],
        completed_steps: List[Dict[str, Any]],
        step_results: Dict[int, Any],
        original_input: str
    ) -> Dict[str, Any]:
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
        # Create a prompt for plan reflection
        prompt = f"""
I'm executing a plan to address this request: "{original_input}"

Current plan:
{json.dumps(current_plan, indent=2)}

Completed steps ({len(completed_steps)}/{len(current_plan["steps"])}):
{json.dumps(completed_steps, indent=2)}

Based on the results so far, should I:
1. Continue with the current plan
2. Modify the remaining steps
3. Add new steps
4. Remove unnecessary steps

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
            temperature=0.3
        )

        # Parse the updated plan from the response
        try:
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', reflection_response, re.DOTALL)
            if json_match:
                updated_plan = json.loads(json_match.group(0))

                # Validate and clean the updated plan
                updated_plan = self._validate_and_clean_plan(updated_plan, original_input)

                # Preserve completed steps
                for step in updated_plan["steps"]:
                    step_id = step["step_id"]
                    # If this step was already completed, mark it as such
                    for completed_step in completed_steps:
                        if completed_step["step_id"] == step_id:
                            step["status"] = "completed"
                            step["completed_at"] = completed_step.get("completed_at")

                return updated_plan
            else:
                # No valid JSON found, return the current plan
                return current_plan
        except Exception as e:
            logger.error(f"Error parsing updated plan: {str(e)}")
            return current_plan
