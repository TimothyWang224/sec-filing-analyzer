import json
import os
import re  # Used for regex pattern matching in JSON extraction
from typing import Any, Dict, List, Optional

from ..capabilities.base import Capability
from ..capabilities.planning import PlanningCapability
from ..capabilities.time_awareness import TimeAwarenessCapability
from ..environments.financial import FinancialEnvironment
from .base import Agent, Goal
from .financial_analyst import FinancialAnalystAgent
from .qa_specialist import QASpecialistAgent
from .risk_analyst import RiskAnalystAgent

# Try to import from the config files, but use defaults if not available
# These imports are used in the configuration fallback mechanism
try:
    from sec_filing_analyzer.config import AGENT_CONFIG  # Used for global config fallback
    from sec_filing_analyzer.llm import get_agent_config  # Used for agent-specific config

    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


class FinancialDiligenceCoordinator(Agent):
    """Coordinates multiple agents for comprehensive financial diligence."""

    def __init__(
        self,
        capabilities: Optional[List[Capability]] = None,
        # Agent iteration parameters
        max_iterations: Optional[int] = None,  # Legacy parameter, still used for backward compatibility
        max_planning_iterations: Optional[int] = None,
        max_execution_iterations: Optional[int] = None,
        max_refinement_iterations: Optional[int] = None,
        # Tool execution parameters
        max_tool_retries: Optional[int] = None,
        tools_per_iteration: Optional[int] = None,
        # Runtime parameters
        max_duration_seconds: Optional[int] = None,
        # LLM parameters
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
        # Environment
        environment: Optional[FinancialEnvironment] = None,
        # Termination parameters
        enable_dynamic_termination: Optional[bool] = None,
        min_confidence_threshold: Optional[float] = None,
    ):
        """
        Initialize the financial diligence coordinator.

        Args:
            capabilities: List of capabilities to extend coordinator behavior
            max_iterations: Maximum number of action loops
            max_duration_seconds: Maximum runtime in seconds
            environment: Optional environment to use
            llm_model: LLM model to use
            llm_temperature: Temperature for LLM generation
            llm_max_tokens: Maximum tokens for LLM generation
            max_tool_calls: Maximum number of tool calls per iteration
        """
        goals = [
            Goal(name="coordination", description="Coordinate multiple agents for comprehensive analysis"),
            Goal(name="synthesis", description="Synthesize insights from multiple agents"),
            Goal(name="reporting", description="Generate comprehensive diligence reports"),
        ]

        # Initialize the base agent with agent type for configuration
        super().__init__(
            goals=goals,
            capabilities=capabilities,
            # Agent iteration parameters
            max_iterations=max_iterations,
            max_planning_iterations=max_planning_iterations,
            max_execution_iterations=max_execution_iterations,
            max_refinement_iterations=max_refinement_iterations,
            # Tool execution parameters
            max_tool_retries=max_tool_retries,
            tools_per_iteration=tools_per_iteration,
            # Runtime parameters
            max_duration_seconds=max_duration_seconds,
            # LLM parameters
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            # Environment
            environment=environment,
            # Termination parameters
            enable_dynamic_termination=enable_dynamic_termination,
            min_confidence_threshold=min_confidence_threshold,
            # Agent type for configuration
            agent_type="coordinator",
        )

        # Initialize environment
        self.environment = environment or FinancialEnvironment()

        # Add TimeAwarenessCapability if not already present
        has_time_awareness = any(isinstance(cap, TimeAwarenessCapability) for cap in self.capabilities)
        if not has_time_awareness:
            self.capabilities.append(TimeAwarenessCapability())

        # Add PlanningCapability if not already present
        has_planning = any(isinstance(cap, PlanningCapability) for cap in self.capabilities)
        if not has_planning:
            self.capabilities.append(
                PlanningCapability(
                    enable_dynamic_replanning=True,
                    enable_step_reflection=True,
                    min_steps_before_reflection=2,
                    max_plan_steps=20,  # Increased to allow more detailed plans for sub-agents
                    plan_detail_level="high",
                    is_coordinator=True,  # This is a coordinator agent
                    respect_existing_plan=False,  # Coordinator creates its own plans
                    # Enhanced planning instructions for hybrid planning approach
                    planning_instructions="""You are a financial diligence coordinator responsible for creating high-level plans and delegating tasks.
                Your goal is to create strategic plans that outline WHAT needs to be done, while allowing specialized agents to determine HOW to do it.

                For each user request:
                1. Analyze the request to understand the overall objective
                2. Break down the objective into high-level tasks
                3. Determine which specialized agent should handle each task
                4. Provide clear objectives and success criteria for each task
                5. Avoid specifying exact tools or parameters - let the specialized agents decide those details

                Available specialized agents and their capabilities:
                - Financial Analyst Agent: Analyzes financial statements, calculates metrics, identifies trends, and assesses financial health
                  * Strengths: Financial ratio analysis, trend identification, performance evaluation
                  * Example tasks: "Analyze AAPL's revenue growth", "Evaluate MSFT's profitability"

                - Risk Analyst Agent: Identifies and analyzes financial and operational risks
                  * Strengths: Risk factor identification, risk assessment, mitigation recommendations
                  * Example tasks: "Identify key risk factors for NVDA", "Assess GOOG's regulatory risks"

                - QA Specialist Agent: Answers specific financial questions and retrieves factual information
                  * Strengths: Information retrieval, fact-finding, data gathering
                  * Example tasks: "Find available companies in the database", "Retrieve AAPL's recent quarterly earnings"

                For each task in your plan, specify:
                1. A clear task number and description
                2. Which agent should perform the task
                3. The specific objective of the task
                4. Success criteria for the task
                5. Any dependencies on other tasks

                Your plan should provide clear direction while allowing agents to use their expertise to determine the best approach.

                Common query types and recommended task allocation:
                1. Listing available companies: Task for QA Specialist - "Retrieve a list of all companies available in the database"
                2. Financial metrics for a company: Task for Financial Analyst - "Analyze key financial metrics for [COMPANY]"
                3. Risk assessment for a company: Task for Risk Analyst - "Identify and assess major risk factors for [COMPANY]"
                4. General questions about a company: Task for QA Specialist - "Find information about [SPECIFIC QUESTION] for [COMPANY]"
                5. Comparing multiple companies: Multiple tasks with dependencies - QA Specialist gathers data, Financial Analyst compares metrics

                Remember: Focus on WHAT needs to be done, not HOW to do it. The specialized agents will determine the best tools and approaches for their tasks.
                """,
                )
            )

        # Initialize default values for agent configurations
        # Financial analyst defaults
        financial_analyst_model = "gpt-4o-mini"
        financial_analyst_temp = 0.3
        financial_analyst_tokens = 1000
        financial_analyst_planning = 1
        financial_analyst_execution = 2
        financial_analyst_refinement = 1

        # Risk analyst defaults
        risk_analyst_model = "gpt-4o-mini"
        risk_analyst_temp = 0.3
        risk_analyst_tokens = 1000
        risk_analyst_planning = 1
        risk_analyst_execution = 2
        risk_analyst_refinement = 1

        # QA specialist defaults
        qa_specialist_model = "gpt-4o-mini"
        qa_specialist_temp = 0.5
        qa_specialist_tokens = 4000
        qa_specialist_max_iterations = 5  # Increased from 3 to 5 for more complex queries
        qa_specialist_planning = 1
        qa_specialist_execution = 5  # Increased from 2 to 5 to allow more tool calls
        qa_specialist_refinement = 2  # Increased from 1 to 2 for better answer refinement

        # Try to use configuration values if available
        use_config = False
        if "HAS_CONFIG" in globals() and HAS_CONFIG:
            try:
                # Get configuration for financial analyst
                financial_analyst_config = get_agent_config("financial_analyst")
                financial_analyst_model = financial_analyst_config.get(
                    "model", AGENT_CONFIG.get("llm_model", financial_analyst_model)
                )
                financial_analyst_temp = financial_analyst_config.get(
                    "temperature", AGENT_CONFIG.get("llm_temperature", financial_analyst_temp)
                )
                financial_analyst_tokens = financial_analyst_config.get(
                    "max_tokens", AGENT_CONFIG.get("llm_max_tokens", financial_analyst_tokens)
                )
                financial_analyst_planning = financial_analyst_config.get(
                    "max_planning_iterations", AGENT_CONFIG.get("max_planning_iterations", financial_analyst_planning)
                )
                financial_analyst_execution = financial_analyst_config.get(
                    "max_execution_iterations",
                    AGENT_CONFIG.get("max_execution_iterations", financial_analyst_execution),
                )
                financial_analyst_refinement = financial_analyst_config.get(
                    "max_refinement_iterations",
                    AGENT_CONFIG.get("max_refinement_iterations", financial_analyst_refinement),
                )

                # Get configuration for risk analyst
                risk_analyst_config = get_agent_config("risk_analyst")
                risk_analyst_model = risk_analyst_config.get("model", AGENT_CONFIG.get("llm_model", risk_analyst_model))
                risk_analyst_temp = risk_analyst_config.get(
                    "temperature", AGENT_CONFIG.get("llm_temperature", risk_analyst_temp)
                )
                risk_analyst_tokens = risk_analyst_config.get(
                    "max_tokens", AGENT_CONFIG.get("llm_max_tokens", risk_analyst_tokens)
                )
                risk_analyst_planning = risk_analyst_config.get(
                    "max_planning_iterations", AGENT_CONFIG.get("max_planning_iterations", risk_analyst_planning)
                )
                risk_analyst_execution = risk_analyst_config.get(
                    "max_execution_iterations", AGENT_CONFIG.get("max_execution_iterations", risk_analyst_execution)
                )
                risk_analyst_refinement = risk_analyst_config.get(
                    "max_refinement_iterations", AGENT_CONFIG.get("max_refinement_iterations", risk_analyst_refinement)
                )

                # Get configuration for QA specialist
                qa_specialist_config = get_agent_config("qa_specialist")
                qa_specialist_model = qa_specialist_config.get(
                    "model", AGENT_CONFIG.get("llm_model", qa_specialist_model)
                )
                qa_specialist_temp = qa_specialist_config.get(
                    "temperature", AGENT_CONFIG.get("llm_temperature", qa_specialist_temp)
                )
                qa_specialist_tokens = qa_specialist_config.get(
                    "max_tokens", AGENT_CONFIG.get("llm_max_tokens", qa_specialist_tokens)
                )
                qa_specialist_planning = qa_specialist_config.get(
                    "max_planning_iterations", AGENT_CONFIG.get("max_planning_iterations", qa_specialist_planning)
                )
                qa_specialist_execution = qa_specialist_config.get(
                    "max_execution_iterations", AGENT_CONFIG.get("max_execution_iterations", qa_specialist_execution)
                )
                qa_specialist_refinement = qa_specialist_config.get(
                    "max_refinement_iterations", AGENT_CONFIG.get("max_refinement_iterations", qa_specialist_refinement)
                )

                use_config = True
            except Exception as e:
                # If there's an error getting the config, use default values
                print(f"Error getting agent config: {str(e)}. Using default values.")

        # Use environment variables if config was not available or failed
        if not use_config:
            # Financial analyst defaults
            financial_analyst_model = os.getenv("FINANCIAL_ANALYST_MODEL", "gpt-4o-mini")
            financial_analyst_temp = float(os.getenv("FINANCIAL_ANALYST_TEMPERATURE", "0.3"))
            financial_analyst_tokens = int(os.getenv("FINANCIAL_ANALYST_MAX_TOKENS", "1000"))
            financial_analyst_planning = int(os.getenv("FINANCIAL_ANALYST_PLANNING_ITERATIONS", "1"))
            financial_analyst_execution = int(os.getenv("FINANCIAL_ANALYST_EXECUTION_ITERATIONS", "2"))
            financial_analyst_refinement = int(os.getenv("FINANCIAL_ANALYST_REFINEMENT_ITERATIONS", "1"))

            # Risk analyst defaults
            risk_analyst_model = os.getenv("RISK_ANALYST_MODEL", "gpt-4o-mini")
            risk_analyst_temp = float(os.getenv("RISK_ANALYST_TEMPERATURE", "0.3"))
            risk_analyst_tokens = int(os.getenv("RISK_ANALYST_MAX_TOKENS", "1000"))
            risk_analyst_planning = int(os.getenv("RISK_ANALYST_PLANNING_ITERATIONS", "1"))
            risk_analyst_execution = int(os.getenv("RISK_ANALYST_EXECUTION_ITERATIONS", "2"))
            risk_analyst_refinement = int(os.getenv("RISK_ANALYST_REFINEMENT_ITERATIONS", "1"))

            # QA specialist defaults
            qa_specialist_model = os.getenv("QA_SPECIALIST_MODEL", "gpt-4o-mini")
            qa_specialist_temp = float(os.getenv("QA_SPECIALIST_TEMPERATURE", "0.5"))
            qa_specialist_tokens = int(os.getenv("QA_SPECIALIST_MAX_TOKENS", "4000"))
            qa_specialist_max_iterations = int(os.getenv("QA_SPECIALIST_MAX_ITERATIONS", "5"))  # Increased from 3 to 5
            qa_specialist_planning = int(os.getenv("QA_SPECIALIST_PLANNING_ITERATIONS", "1"))
            qa_specialist_execution = int(os.getenv("QA_SPECIALIST_EXECUTION_ITERATIONS", "5"))  # Increased from 2 to 5
            qa_specialist_refinement = int(
                os.getenv("QA_SPECIALIST_REFINEMENT_ITERATIONS", "2")
            )  # Increased from 1 to 2

        # Initialize specialized agents with the configuration values
        self.financial_analyst = FinancialAnalystAgent(
            environment=self.environment,
            llm_model=financial_analyst_model,
            llm_temperature=financial_analyst_temp,
            llm_max_tokens=financial_analyst_tokens,
            max_planning_iterations=financial_analyst_planning,
            max_execution_iterations=financial_analyst_execution,
            max_refinement_iterations=financial_analyst_refinement,
        )

        self.risk_analyst = RiskAnalystAgent(
            environment=self.environment,
            llm_model=risk_analyst_model,
            llm_temperature=risk_analyst_temp,
            llm_max_tokens=risk_analyst_tokens,
            max_planning_iterations=risk_analyst_planning,
            max_execution_iterations=risk_analyst_execution,
            max_refinement_iterations=risk_analyst_refinement,
        )

        self.qa_specialist = QASpecialistAgent(
            environment=self.environment,
            llm_model=qa_specialist_model,
            llm_temperature=qa_specialist_temp,
            llm_max_tokens=qa_specialist_tokens,
            max_iterations=qa_specialist_max_iterations,  # Added max_iterations parameter
            max_planning_iterations=qa_specialist_planning,
            max_execution_iterations=qa_specialist_execution,
            max_refinement_iterations=qa_specialist_refinement,
        )

    async def run(
        self, user_input: str, memory: Optional[List[Dict]] = None, chat_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Run the financial diligence coordinator.

        Args:
            user_input: The input to process (e.g., company name or diligence request)
            memory: Optional memory to initialize with
            chat_mode: Whether to format the response for chat interface

        Returns:
            Dictionary containing comprehensive diligence results
        """
        # Initialize memory if provided
        if memory:
            for item in memory:
                self.state.add_memory_item(item)

        # Initialize capabilities
        for capability in self.capabilities:
            await capability.init(self, {"input": user_input})

        # Log the start of processing
        self.logger.info(f"Processing diligence request: {user_input}")

        # Set initial phase to planning
        self.state.set_phase("planning")
        self.logger.info(f"Starting planning phase")

        # Initialize variables for results
        financial_analysis = None
        risk_analysis = None
        qa_response = None
        diligence_report = None

        # Phase 1: Planning
        while not self.should_terminate() and self.state.current_phase == "planning":
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break

            # In planning phase, we focus on understanding the request and creating a plan
            self.logger.info(f"Planning diligence analysis for: {user_input}")

            # Check if we have a plan from the planning capability
            planning_context = self.state.get_context().get("planning", {})
            current_step = planning_context.get("current_step", {})

            # Determine which agents to run
            agent_selection = await self._select_agents(user_input)

            # Add agent selection to memory
            self.add_to_memory({"type": "agent_selection", "content": agent_selection})

            self.increment_iteration()

            # If we've done enough planning, move to execution phase
            if self.state.phase_iterations["planning"] >= self.max_planning_iterations:
                # Roll over unused tokens from planning to execution
                self._rollover_token_surplus("planning", "execution")

                self.state.set_phase("execution")
                self.logger.info(f"Moving to execution phase")

        # Phase 2: Execution
        while not self.should_terminate() and self.state.current_phase == "execution":
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break

            # In execution phase, we run the specialized agents
            self.logger.info(f"Executing diligence analysis for: {user_input}")

            # Check if we have a plan from the planning capability
            planning_context = self.state.get_context().get("planning", {})
            current_step = planning_context.get("current_step", {})

            # Get the agent selection from memory
            memory_items = self.get_memory()
            agent_selection_items = [item for item in memory_items if item.get("type") == "agent_selection"]
            agent_selection = agent_selection_items[-1].get("content", {}) if agent_selection_items else {}

            # Log if a tool is specified in the current step (but don't execute it directly)
            if current_step and "tool" in current_step:
                tool_name = current_step.get("tool")

                self.logger.info(f"Tool specified in plan step: {tool_name}")
                self.logger.info(f"This will be delegated to a specialized agent")

            # If we have a plan with a current step, follow it
            if current_step:
                self.logger.info(f"Executing plan step: {current_step.get('description')}")

                # If the step specifies an agent, run that agent
                if "agent" in current_step:
                    agent_name = current_step.get("agent")
                    self.logger.info(f"Running {agent_name} as specified in the plan...")

                    # Extract high-level task information from the step
                    task_objective = current_step.get("objective", "")
                    task_description = current_step.get("description", "")
                    success_criteria = current_step.get("success_criteria", [])

                    # Create a high-level task plan for the agent
                    task_plan = {
                        "task_objective": task_objective or task_description,
                        "success_criteria": success_criteria,
                        "context": {
                            "user_query": user_input,
                            "step_number": current_step.get("step_number"),
                            "total_steps": len(planning_context.get("plan", {}).get("steps", [])),
                        },
                    }

                    self.logger.info(f"Running {agent_name} with high-level task: {task_objective or task_description}")

                    if agent_name == "financial_analyst":
                        financial_analysis = await self.financial_analyst.run(user_input, plan=task_plan)
                    elif agent_name == "risk_analyst":
                        risk_analysis = await self.risk_analyst.run(user_input, plan=task_plan)
                    elif agent_name == "qa_specialist":
                        qa_response = await self.qa_specialist.run(user_input, plan=task_plan)
                else:
                    # Run selected agents based on the agent selection
                    # Get the plan context for potential agent instructions
                    planning_context = self.state.get_context().get("planning", {})
                    plan = planning_context.get("plan", {})
                    steps = plan.get("steps", [])

                    # Find high-level tasks for each agent type
                    financial_analyst_task = None
                    risk_analyst_task = None
                    qa_specialist_task = None

                    for step in steps:
                        if step.get("agent") == "financial_analyst":
                            financial_analyst_task = {
                                "task_objective": step.get("objective", step.get("description", "")),
                                "success_criteria": step.get("success_criteria", []),
                                "context": {
                                    "user_query": user_input,
                                    "step_number": step.get("step_number"),
                                    "total_steps": len(steps),
                                },
                            }
                        elif step.get("agent") == "risk_analyst":
                            risk_analyst_task = {
                                "task_objective": step.get("objective", step.get("description", "")),
                                "success_criteria": step.get("success_criteria", []),
                                "context": {
                                    "user_query": user_input,
                                    "step_number": step.get("step_number"),
                                    "total_steps": len(steps),
                                },
                            }
                        elif step.get("agent") == "qa_specialist":
                            qa_specialist_task = {
                                "task_objective": step.get("objective", step.get("description", "")),
                                "success_criteria": step.get("success_criteria", []),
                                "context": {
                                    "user_query": user_input,
                                    "step_number": step.get("step_number"),
                                    "total_steps": len(steps),
                                },
                            }

                    if agent_selection.get("financial_analyst", False) and not financial_analysis:
                        task_desc = (
                            financial_analyst_task.get("task_objective")
                            if financial_analyst_task
                            else "Analyze financial information"
                        )
                        self.logger.info(f"Running Financial Analyst Agent with task: {task_desc}")
                        financial_analysis = await self.financial_analyst.run(user_input, plan=financial_analyst_task)

                    if agent_selection.get("risk_analyst", False) and not risk_analysis:
                        task_desc = (
                            risk_analyst_task.get("task_objective") if risk_analyst_task else "Analyze risk factors"
                        )
                        self.logger.info(f"Running Risk Analyst Agent with task: {task_desc}")
                        risk_analysis = await self.risk_analyst.run(user_input, plan=risk_analyst_task)

                    if agent_selection.get("qa_specialist", False) and not qa_response:
                        task_desc = (
                            qa_specialist_task.get("task_objective") if qa_specialist_task else "Retrieve information"
                        )
                        self.logger.info(f"Running QA Specialist Agent with task: {task_desc}")
                        qa_response = await self.qa_specialist.run(user_input, plan=qa_specialist_task)
            else:
                # Run selected agents based on the agent selection
                # Get the plan context for potential agent instructions
                planning_context = self.state.get_context().get("planning", {})
                plan = planning_context.get("plan", {})
                steps = plan.get("steps", [])

                # Create default high-level tasks for each agent type
                default_tasks = {
                    "financial_analyst": {
                        "task_objective": "Analyze financial information related to the user query",
                        "success_criteria": [
                            "Identify key financial metrics",
                            "Assess financial performance",
                            "Provide financial insights",
                        ],
                        "context": {"user_query": user_input},
                    },
                    "risk_analyst": {
                        "task_objective": "Identify and assess risk factors related to the user query",
                        "success_criteria": [
                            "Identify key risk factors",
                            "Assess risk severity",
                            "Provide risk mitigation recommendations",
                        ],
                        "context": {"user_query": user_input},
                    },
                    "qa_specialist": {
                        "task_objective": "Retrieve and provide information related to the user query",
                        "success_criteria": [
                            "Find relevant information",
                            "Provide accurate answers",
                            "Present information clearly",
                        ],
                        "context": {"user_query": user_input},
                    },
                }

                # Find high-level tasks from the plan if available
                financial_analyst_task = None
                risk_analyst_task = None
                qa_specialist_task = None

                for step in steps:
                    if step.get("agent") == "financial_analyst":
                        financial_analyst_task = {
                            "task_objective": step.get("objective", step.get("description", "")),
                            "success_criteria": step.get("success_criteria", []),
                            "context": {
                                "user_query": user_input,
                                "step_number": step.get("step_number"),
                                "total_steps": len(steps),
                            },
                        }
                    elif step.get("agent") == "risk_analyst":
                        risk_analyst_task = {
                            "task_objective": step.get("objective", step.get("description", "")),
                            "success_criteria": step.get("success_criteria", []),
                            "context": {
                                "user_query": user_input,
                                "step_number": step.get("step_number"),
                                "total_steps": len(steps),
                            },
                        }
                    elif step.get("agent") == "qa_specialist":
                        qa_specialist_task = {
                            "task_objective": step.get("objective", step.get("description", "")),
                            "success_criteria": step.get("success_criteria", []),
                            "context": {
                                "user_query": user_input,
                                "step_number": step.get("step_number"),
                                "total_steps": len(steps),
                            },
                        }

                # Use tasks from plan if available, otherwise use defaults
                if agent_selection.get("financial_analyst", False) and not financial_analysis:
                    task = financial_analyst_task or default_tasks["financial_analyst"]
                    task_desc = task.get("task_objective")
                    self.logger.info(f"Running Financial Analyst Agent with task: {task_desc}")
                    financial_analysis = await self.financial_analyst.run(user_input, plan=task)

                if agent_selection.get("risk_analyst", False) and not risk_analysis:
                    task = risk_analyst_task or default_tasks["risk_analyst"]
                    task_desc = task.get("task_objective")
                    self.logger.info(f"Running Risk Analyst Agent with task: {task_desc}")
                    risk_analysis = await self.risk_analyst.run(user_input, plan=task)

                if agent_selection.get("qa_specialist", False) and not qa_response:
                    task = qa_specialist_task or default_tasks["qa_specialist"]
                    task_desc = task.get("task_objective")
                    self.logger.info(f"Running QA Specialist Agent with task: {task_desc}")
                    qa_response = await self.qa_specialist.run(user_input, plan=task)

            self.increment_iteration()

            # If we've run all the selected agents, move to refinement phase
            all_agents_run = True
            if agent_selection.get("financial_analyst", False) and not financial_analysis:
                all_agents_run = False
            if agent_selection.get("risk_analyst", False) and not risk_analysis:
                all_agents_run = False
            if agent_selection.get("qa_specialist", False) and not qa_response:
                all_agents_run = False

            if all_agents_run or self.state.phase_iterations["execution"] >= self.max_execution_iterations:
                # Roll over unused tokens from execution to refinement
                self._rollover_token_surplus("execution", "refinement")

                self.state.set_phase("refinement")
                self.logger.info(f"Moving to refinement phase")

        # Phase 3: Refinement
        while not self.should_terminate() and self.state.current_phase == "refinement":
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break

            # In refinement phase, we generate and refine the diligence report
            self.logger.info(f"Refining diligence report for: {user_input}")

            # Generate comprehensive report if we haven't already
            if not diligence_report:
                # Generate a report using the specialized agent results
                diligence_report = await self._generate_diligence_report(
                    user_input, financial_analysis, risk_analysis, qa_response
                )

                # Add result to memory
                self.add_to_memory({"type": "diligence_report", "content": diligence_report})
            else:
                # Refine the existing report
                diligence_report = await self._refine_diligence_report(
                    user_input, diligence_report, financial_analysis, risk_analysis, qa_response
                )

                # Add refined result to memory
                self.add_to_memory({"type": "refined_diligence_report", "content": diligence_report})

            # Process result with capabilities
            for capability in self.capabilities:
                diligence_report = await capability.process_result(
                    self, {"input": user_input}, user_input, {"type": "diligence_report"}, diligence_report
                )

            self.increment_iteration()

            # If we've done enough refinement, we're done
            if self.state.phase_iterations["refinement"] >= self.max_refinement_iterations:
                break

        result = {
            "status": "completed",
            "diligence_report": diligence_report,
            "memory": self.get_memory(),
            "phase_iterations": self.state.phase_iterations,
        }

        # Format for chat if requested
        if chat_mode:
            result["response"] = self._format_for_chat(user_input, diligence_report)

        return result

    async def _select_agents(self, user_input: str) -> Dict[str, bool]:
        """
        Determine which specialized agents to run based on the input.

        Args:
            user_input: The user's input

        Returns:
            Dictionary mapping agent names to boolean indicating whether to run them
        """
        # Create a prompt for agent selection that focuses on high-level task allocation
        prompt = f"""
        Based on the following user input, determine which specialized agents should be involved in addressing the request:

        User Input: "{user_input}"

        Available Agents and their specialties:
        1. Financial Analyst Agent
           - Financial statement analysis
           - Financial ratio calculation and interpretation
           - Performance trend identification
           - Financial health assessment
           - Comparative financial analysis

        2. Risk Analyst Agent
           - Risk factor identification
           - Risk severity and likelihood assessment
           - Regulatory compliance analysis
           - Risk mitigation recommendations
           - Forward-looking risk projections

        3. QA Specialist Agent
           - Factual information retrieval
           - Database querying
           - Specific question answering
           - Data gathering and organization
           - Information synthesis

        Consider the following when making your selection:
        - What is the primary objective of the user's request?
        - What types of analysis or information are needed?
        - Which agent's specialties best align with these needs?
        - Would multiple agents working together provide a more comprehensive response?

        For each agent, determine if they should be involved (true/false) based on the user input.
        Return your decision as a JSON object with the following structure:
        {{"financial_analyst": true/false, "risk_analyst": true/false, "qa_specialist": true/false}}
        """

        # Generate agent selection
        response = await self.llm.generate(prompt=prompt, temperature=0.3)

        # Parse agent selection from response
        try:
            # Extract JSON from response
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                agent_selection = json.loads(json_match.group(0))
            else:
                # Default to running all agents if parsing fails
                agent_selection = {"financial_analyst": True, "risk_analyst": True, "qa_specialist": True}
        except Exception as e:
            print(f"Error parsing agent selection: {str(e)}")
            # Default to running all agents if parsing fails
            agent_selection = {"financial_analyst": True, "risk_analyst": True, "qa_specialist": True}

        return agent_selection

    async def _refine_diligence_report(
        self,
        input: str,
        current_report: Dict[str, Any],
        financial_analysis: Optional[Dict[str, Any]],  # Passed for potential future use
        risk_analysis: Optional[Dict[str, Any]],  # Passed for potential future use
        qa_response: Optional[Dict[str, Any]],  # Passed for potential future use
    ) -> Dict[str, Any]:
        """
        Refine the diligence report based on all available information.

        Args:
            input: The input to process
            current_report: The current diligence report
            financial_analysis: Results from the financial analyst
            risk_analysis: Results from the risk analyst
            qa_response: Results from the QA specialist

        Returns:
            Dictionary containing refined diligence report
        """
        try:
            # Extract the current report components
            executive_summary = current_report.get("executive_summary", "")
            # These components are extracted for potential future use
            _ = current_report.get("financial_health", {})
            _ = current_report.get("risk_profile", {})
            key_findings = current_report.get("key_findings", [])
            recommendations = current_report.get("recommendations", [])

            # Log the analyses that were passed in (to use the parameters and avoid IDE warnings)
            if financial_analysis:
                print(f"Financial analysis data available: {len(str(financial_analysis))} characters")
            if risk_analysis:
                print(f"Risk analysis data available: {len(str(risk_analysis))} characters")
            if qa_response:
                print(f"QA response data available: {len(str(qa_response))} characters")

            # Generate a refinement prompt
            refinement_prompt = f"""
            I need to refine the following diligence report for: "{input}"

            Current executive summary: "{executive_summary}"

            Please improve this diligence report by:
            1. Making the executive summary more concise and focused on key findings
            2. Ensuring all major financial and risk aspects are covered
            3. Providing more specific and actionable recommendations
            4. Improving the clarity and organization of information
            5. Ensuring numerical data is presented clearly

            Please provide only the refined executive summary.
            """

            # Generate the refined executive summary using the LLM
            refined_summary = await self.llm.generate(prompt=refinement_prompt)

            # Refine key findings
            findings_prompt = f"""
            Based on the refined executive summary, please improve the key findings for {input}.

            Current key findings: {json.dumps(key_findings, indent=2)}

            Please provide a more focused set of key findings that highlight the most important aspects.
            Format your response as a JSON array of finding strings.
            """

            findings_response = await self.llm.generate(prompt=findings_prompt)

            # Try to parse key findings as JSON
            try:
                # Extract JSON from the response
                import re

                json_match = re.search(r"\[.*\]", findings_response, re.DOTALL)
                refined_findings = json.loads(json_match.group(0)) if json_match else key_findings
            except:
                refined_findings = key_findings

            # Refine recommendations
            recommendations_prompt = f"""
            Based on the refined executive summary and key findings, please improve the recommendations for {input}.

            Current recommendations: {json.dumps(recommendations, indent=2)}

            Please provide more specific, actionable recommendations that address the key findings.
            Format your response as a JSON array of recommendation strings.
            """

            recommendations_response = await self.llm.generate(prompt=recommendations_prompt)

            # Try to parse recommendations as JSON
            try:
                # Extract JSON from the response
                json_match = re.search(r"\[.*\]", recommendations_response, re.DOTALL)
                refined_recommendations = json.loads(json_match.group(0)) if json_match else recommendations
            except:
                refined_recommendations = recommendations

            # Create the refined report object
            refined_report = current_report.copy()
            refined_report["executive_summary"] = refined_summary.strip()
            refined_report["key_findings"] = refined_findings
            refined_report["recommendations"] = refined_recommendations
            refined_report["refinement_iteration"] = refined_report.get("refinement_iteration", 0) + 1

            # Add confidence score if dynamic termination is enabled
            if self.enable_dynamic_termination:
                confidence_prompt = f"""
                On a scale of 0.0 to 1.0, how confident are you that the following diligence report fully and accurately addresses the request for: "{input}"

                Executive Summary: "{refined_summary.strip()}"

                Please respond with only a number between 0.0 and 1.0.
                """

                confidence_response = await self.llm.generate(prompt=confidence_prompt)
                try:
                    confidence = float(confidence_response.strip())
                    refined_report["confidence"] = min(max(confidence, 0.0), 1.0)  # Ensure it's between 0 and 1
                except ValueError:
                    refined_report["confidence"] = 0.5  # Default if parsing fails

            return refined_report

        except Exception as e:
            self.logger.error(f"Error refining diligence report: {str(e)}")
            # Return the original report with an error note
            current_report["refinement_error"] = str(e)
            return current_report

    async def _generate_diligence_report(
        self,
        input: str,
        financial_analysis: Optional[Dict[str, Any]],
        risk_analysis: Optional[Dict[str, Any]],
        qa_response: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive diligence report.

        Args:
            input: Input to analyze (e.g., company name)
            financial_analysis: Results from financial analyst (optional)
            risk_analysis: Results from risk analyst (optional)
            qa_response: Results from QA specialist (optional)

        Returns:
            Dictionary containing comprehensive diligence report
        """
        try:
            # Extract financial health information
            financial_health = {}
            if financial_analysis:
                financial_health = {
                    "metrics": financial_analysis.get("analysis", {}).get("metrics", {}),
                    "trends": financial_analysis.get("analysis", {}).get("trends", []),
                    "insights": financial_analysis.get("analysis", {}).get("insights", []),
                }

            # Extract risk profile information
            risk_profile = {}
            if risk_analysis:
                risk_profile = {
                    "risk_factors": risk_analysis.get("risk_analysis", {}).get("risk_factors", {}),
                    "risk_trends": risk_analysis.get("risk_analysis", {}).get("risk_trends", []),
                    "recommendations": risk_analysis.get("risk_analysis", {}).get("recommendations", []),
                }

            # Extract QA information
            qa_info = {}
            if qa_response:
                qa_info = {"question": input, "answer": qa_response.get("answer", {}).get("answer", "")}

            # Generate executive summary using the LLM
            summary_prompt = f"""
            Generate a comprehensive executive summary for {input} based on the following information:

            Financial Health:
            {json.dumps(financial_health, indent=2) if financial_health else "No financial analysis available."}

            Risk Profile:
            {json.dumps(risk_profile, indent=2) if risk_profile else "No risk analysis available."}

            Q&A Information:
            {json.dumps(qa_info, indent=2) if qa_info else "No Q&A information available."}

            The executive summary should be concise but comprehensive, highlighting the most important findings.
            """

            executive_summary = await self.llm.generate(prompt=summary_prompt)

            # Generate key findings
            findings_prompt = f"""
            Based on the financial analysis and risk assessment for {input}, identify the key findings.
            Format your response as a JSON array of finding descriptions.
            """

            findings_response = await self.llm.generate(prompt=findings_prompt)

            # Try to parse key findings as JSON
            try:
                # Extract JSON from the response
                json_match = re.search(r"\[.*\]", findings_response, re.DOTALL)
                key_findings = json.loads(json_match.group(0)) if json_match else []
            except:
                key_findings = ["Insufficient data to determine key findings"]

            # Generate recommendations
            recommendations_prompt = f"""
            Based on the financial analysis and risk assessment for {input}, provide strategic recommendations.
            Format your response as a JSON array of recommendation descriptions.
            """

            recommendations_response = await self.llm.generate(prompt=recommendations_prompt)

            # Try to parse recommendations as JSON
            try:
                # Extract JSON from the response
                json_match = re.search(r"\[.*\]", recommendations_response, re.DOTALL)
                recommendations = json.loads(json_match.group(0)) if json_match else []
            except:
                recommendations = ["Insufficient data to provide meaningful recommendations"]

            # Return the comprehensive diligence report
            return {
                "input": input,
                "executive_summary": executive_summary.strip(),
                "financial_health": financial_health,
                "risk_profile": risk_profile,
                "qa_response": qa_info.get("answer", "") if qa_info else "",
                "key_findings": key_findings,
                "recommendations": recommendations,
            }

        except Exception as e:
            # Return error information
            return {
                "input": input,
                "error": str(e),
                "executive_summary": "I encountered an error while generating the diligence report. Please try again or provide more specific information.",
                "financial_health": {},
                "risk_profile": {},
                "qa_response": "",
                "key_findings": ["Error in report generation"],
                "recommendations": [f"Error details: {str(e)}"],
            }

    async def _get_available_companies(self) -> List[str]:
        """
        Get a list of available companies in the database.

        Returns:
            List of company tickers available for analysis
        """
        try:
            # Use the financial data tool to query available companies
            companies_result = await self.environment.execute_action(
                {"tool": "sec_financial_data", "args": {"query_type": "companies"}}
            )

            # Extract company tickers from the result
            self.logger.info(f"Company info result: {companies_result}")

            if isinstance(companies_result, dict):
                if "companies" in companies_result:
                    return companies_result["companies"]
                elif "results" in companies_result and isinstance(companies_result["results"], list):
                    # Extract tickers from the results
                    tickers = []
                    for company in companies_result["results"]:
                        if isinstance(company, dict) and "ticker" in company:
                            tickers.append(company["ticker"])
                    return tickers
            elif isinstance(companies_result, list):
                # If it's a list of dictionaries with ticker field
                tickers = []
                for company in companies_result:
                    if isinstance(company, dict) and "ticker" in company:
                        tickers.append(company["ticker"])
                return tickers

            self.logger.warning(f"Unexpected companies result format: {type(companies_result)}")
            # Fallback to hardcoded list of companies we know are in the database
            return ["AAPL", "GOOG", "MSFT", "NVDA"]
        except Exception as e:
            self.logger.error(f"Error getting available companies: {str(e)}")
            # Fallback to hardcoded list of companies we know are in the database
            self.logger.info("Using fallback hardcoded company list")
            return ["AAPL", "GOOG", "MSFT", "NVDA"]

    def _format_for_chat(self, user_input: str, diligence_report: Dict[str, Any]) -> str:
        """
        Format the diligence report for chat interface.

        Args:
            user_input: The original user input (used for context in some formatting decisions)
            diligence_report: The diligence report to format

        Returns:
            Formatted response string for chat interface
        """
        # Log that we're formatting the response
        self.logger.info(f"Formatting response for: {user_input[:50]}{'...' if len(user_input) > 50 else ''}")
        if not diligence_report:
            return "I'm sorry, I couldn't generate a report based on your request. Please try again with more specific information."

        if "error" in diligence_report:
            return (
                f"I encountered an error while analyzing your request: {diligence_report.get('error', 'Unknown error')}"
            )

        # Check if this is a QA-type request
        if diligence_report.get("qa_response") and len(diligence_report.get("qa_response", "")) > 10:
            return diligence_report.get("qa_response")

        # Format the diligence report for chat
        response = ""

        # Add executive summary
        if "executive_summary" in diligence_report and diligence_report["executive_summary"]:
            response += diligence_report["executive_summary"]

        # Add key findings if available
        if "key_findings" in diligence_report and diligence_report["key_findings"]:
            response += "\n\n**Key Findings:**\n"
            for i, finding in enumerate(diligence_report["key_findings"]):
                response += f"\n{i + 1}. {finding}"

        # Add recommendations if available
        if "recommendations" in diligence_report and diligence_report["recommendations"]:
            response += "\n\n**Recommendations:**\n"
            for i, rec in enumerate(diligence_report["recommendations"]):
                response += f"\n{i + 1}. {rec}"

        return response
