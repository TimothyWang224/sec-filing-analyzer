import json
import os
from typing import Dict, Any, List, Optional
from .base import Agent, Goal
from .task_parser import TaskParser
from ..capabilities.base import Capability
from ..capabilities.time_awareness import TimeAwarenessCapability
from ..capabilities.planning import PlanningCapability
from ..capabilities.multi_task_planning import MultiTaskPlanningCapability
from ..environments.financial import FinancialEnvironment

class FinancialAnalystAgent(Agent):
    """Agent specialized in analyzing financial statements and metrics."""

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
        min_confidence_threshold: Optional[float] = None
    ):
        """
        Initialize the financial analyst agent.

        Args:
            capabilities: List of capabilities to extend agent behavior
            max_iterations: Maximum number of action loops
            max_duration_seconds: Maximum runtime in seconds
            environment: Optional environment to use
            llm_model: LLM model to use
            llm_temperature: Temperature for LLM generation
            llm_max_tokens: Maximum tokens for LLM generation
            max_tool_calls: Maximum number of tool calls per iteration
        """
        goals = [
            Goal(
                name="financial_analysis",
                description="Analyze financial statements and metrics to provide insights"
            ),
            Goal(
                name="ratio_calculation",
                description="Calculate and interpret key financial ratios"
            ),
            Goal(
                name="trend_analysis",
                description="Identify trends and changes in financial metrics"
            )
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
            agent_type="financial_analyst"
        )

        # Initialize environment
        self.environment = environment or FinancialEnvironment()

        # Add TimeAwarenessCapability if not already present
        has_time_awareness = any(isinstance(cap, TimeAwarenessCapability) for cap in self.capabilities)
        if not has_time_awareness:
            self.capabilities.append(TimeAwarenessCapability())

        # Add MultiTaskPlanningCapability if not already present
        has_planning = any(isinstance(cap, (PlanningCapability, MultiTaskPlanningCapability)) for cap in self.capabilities)
        if not has_planning:
            self.capabilities.append(MultiTaskPlanningCapability(
                enable_dynamic_replanning=True,
                enable_step_reflection=True,
                min_steps_before_reflection=1,
                max_plan_steps=5,
                plan_detail_level="medium"
            ))

    async def run(self, user_input: str, memory: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run the financial analyst agent.

        Args:
            user_input: The input to process (e.g., ticker symbol or analysis request)
            memory: Optional memory to initialize with

        Returns:
            Dictionary containing analysis results and insights
        """
        # Initialize memory if provided
        if memory:
            for item in memory:
                self.state.add_memory_item(item)

        # Parse tasks from user input
        task_parser = TaskParser(self.llm)
        task_queue = await task_parser.parse_tasks(user_input)

        # Find the MultiTaskPlanningCapability
        multi_task_planning = None
        for capability in self.capabilities:
            if isinstance(capability, MultiTaskPlanningCapability):
                multi_task_planning = capability
                multi_task_planning.set_task_queue(task_queue)
                break

        # Initialize context with task queue
        context = {
            "input": user_input,
            "task_queue": task_queue
        }

        # Initialize capabilities
        for capability in self.capabilities:
            await capability.init(self, context)

        # Initialize results container with a task map to avoid duplicates
        task_results = {}
        all_results = {
            "input": user_input,
            "tasks": [],
            "completed_tasks": 0,
            "total_tasks": len(task_queue.get_all_tasks())
        }

        # Log the start of processing
        self.logger.info(f"Processing financial analysis request: {user_input}")

        # Set initial phase to planning
        self.state.set_phase('planning')
        self.logger.info(f"Starting planning phase")

        # Phase 1: Planning
        while not self.should_terminate() and self.state.current_phase == 'planning':
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, context):
                    break

            # Get the current task
            current_task = task_queue.get_current_task()

            # If there's no current task, we're done with planning
            if not current_task:
                self.state.set_phase('execution')
                self.logger.info(f"No tasks to plan for, moving to execution phase")
                break

            # In planning phase, we focus on understanding the financial analysis tasks
            self.logger.info(f"Planning financial analysis for task: {current_task.input_text}")

            # Add task analysis to memory
            self.add_to_memory({
                "type": "task_analysis",
                "task_id": current_task.task_id,
                "content": {
                    "task": current_task.input_text,
                    "analysis": "Planning financial analysis"
                }
            })

            self.increment_iteration()

            # If we've done enough planning, move to execution phase
            if self.state.phase_iterations['planning'] >= self.max_planning_iterations:
                self.state.set_phase('execution')
                self.logger.info(f"Moving to execution phase")
                break

        # Phase 2: Execution
        while not self.should_terminate() and self.state.current_phase == 'execution':
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, context):
                    break

            # Get the current task
            current_task = task_queue.get_current_task()

            # If there's no current task, we're done with execution
            if not current_task:
                self.state.set_phase('refinement')
                self.logger.info(f"No tasks to execute, moving to refinement phase")
                break

            # In execution phase, we gather data and generate initial financial analysis
            self.logger.info(f"Executing financial analysis for task: {current_task.input_text}")

            # Check if we have a plan from the planning capability
            planning_context = self.state.get_context().get("planning", {})
            current_step = planning_context.get("current_step", {})

            # If we have a plan with a current step, follow it
            if current_step:
                self.logger.info(f"Executing plan step: {current_step.get('description')}")

                # Process the input and generate analysis based on the current step
                analysis_result = await self._analyze_financials(current_task.input_text, current_step)
            else:
                # If we don't have a plan or current step, fall back to the original behavior
                analysis_result = await self._analyze_financials(current_task.input_text)

            # Add result to memory
            self.add_to_memory({
                "type": "financial_analysis",
                "task_id": current_task.task_id,
                "content": analysis_result
            })

            # Process result with capabilities
            for capability in self.capabilities:
                analysis_result = await capability.process_result(
                    self,
                    context,
                    current_task.input_text,
                    {"type": "financial_analysis", "task_id": current_task.task_id},
                    analysis_result
                )

            # Store the result for this task in the task_results dictionary to avoid duplicates
            task_results[current_task.task_id] = {
                "task_id": current_task.task_id,
                "input": current_task.input_text,
                "result": analysis_result,
                "status": current_task.status
            }

            # Update the completed tasks count
            all_results["completed_tasks"] = len(task_queue.get_completed_tasks())

            self.increment_iteration()

            # Mark the task as completed
            task_queue.mark_task_completed(current_task.task_id)

            # If we've done enough execution, move to refinement phase
            if self.state.phase_iterations['execution'] >= self.max_execution_iterations:
                self.state.set_phase('refinement')
                self.logger.info(f"Moving to refinement phase")
                break

        # Phase 3: Refinement
        while not self.should_terminate() and self.state.current_phase == 'refinement':
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, context):
                    break

            # In refinement phase, we improve the financial analysis results
            self.logger.info(f"Refining financial analysis results")

            # Get all completed task results
            completed_tasks = list(task_results.values())

            if completed_tasks:
                # Refine each task result
                for task_result in completed_tasks:
                    task_id = task_result["task_id"]
                    current_analysis = task_result["result"]

                    # Refine the analysis
                    refined_analysis = await self._refine_financial_analysis(task_result["input"], current_analysis)

                    # Update the task result
                    task_results[task_id]["result"] = refined_analysis

                    # Add refined result to memory
                    self.add_to_memory({
                        "type": "refined_financial_analysis",
                        "task_id": task_id,
                        "content": refined_analysis
                    })

            self.increment_iteration()

            # If we've done enough refinement, we're done
            if self.state.phase_iterations['refinement'] >= self.max_refinement_iterations:
                break

        # Prepare final results
        completed_tasks = task_queue.get_completed_tasks()
        pending_tasks = task_queue.get_pending_tasks()
        failed_tasks = task_queue.get_failed_tasks()

        # Convert task_results dictionary to a list for the final output
        all_results["tasks"] = list(task_results.values())
        all_results["status"] = "completed" if not pending_tasks else "partial"
        all_results["completed_tasks"] = len(completed_tasks)
        all_results["pending_tasks"] = len(pending_tasks)
        all_results["failed_tasks"] = len(failed_tasks)
        all_results["total_tasks"] = len(task_queue.get_all_tasks())
        all_results["phase_iterations"] = self.state.phase_iterations

        # Filter memory to remove duplicates
        filtered_memory = []
        memory_task_plans = {}

        for item in self.get_memory():
            # For plan items, only keep the latest plan for each task
            if item.get("type") == "plan" and "task_id" in item:
                task_id = item["task_id"]
                memory_task_plans[task_id] = item
            # For other items, keep them all
            else:
                filtered_memory.append(item)

        # Add the latest plan for each task to the filtered memory
        filtered_memory.extend(memory_task_plans.values())

        all_results["memory"] = filtered_memory

        return all_results

    async def _refine_financial_analysis(self, input: str, current_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine the current financial analysis to improve its quality.

        Args:
            input: The original input
            current_analysis: The current analysis to refine

        Returns:
            Refined financial analysis
        """
        try:
            # Extract the current analysis text and metrics
            analysis_text = current_analysis.get("analysis", "")
            metrics = current_analysis.get("metrics", {})
            # trends data is extracted for potential future use but not currently used
            _ = current_analysis.get("trends", [])
            insights = current_analysis.get("insights", [])

            # Generate a refinement prompt
            refinement_prompt = f"""
            I need to refine the following financial analysis for: "{input}"

            Current analysis: "{analysis_text}"

            Please improve this financial analysis by:
            1. Making it more concise and focused on key financial metrics
            2. Ensuring all major financial aspects are covered
            3. Providing more specific financial ratio interpretations
            4. Improving the clarity of trend analysis
            5. Ensuring numerical data is presented clearly

            Please provide only the refined analysis text.
            """

            # Generate the refined analysis using the LLM
            refined_text = await self.llm.generate(prompt=refinement_prompt)

            # Refine metrics
            metrics_prompt = f"""
            Based on the refined analysis, please improve the financial metrics for {input}.

            Current metrics: {json.dumps(metrics, indent=2)}

            Please provide a more focused set of metrics with clear values and interpretations.
            Format your response as a JSON object with metric names as keys and values as strings.
            """

            metrics_response = await self.llm.generate(prompt=metrics_prompt)

            # Try to parse metrics as JSON
            try:
                # Extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', metrics_response, re.DOTALL)
                refined_metrics = json.loads(json_match.group(0)) if json_match else metrics
            except:
                refined_metrics = metrics

            # Refine insights
            insights_prompt = f"""
            Based on the refined analysis, please improve the financial insights for {input}.

            Current insights: {json.dumps(insights, indent=2)}

            Please provide more specific, actionable insights based on the financial analysis.
            Format your response as a JSON array of insight strings.
            """

            insights_response = await self.llm.generate(prompt=insights_prompt)

            # Try to parse insights as JSON
            try:
                # Extract JSON from the response
                json_match = re.search(r'\[.*\]', insights_response, re.DOTALL)
                refined_insights = json.loads(json_match.group(0)) if json_match else insights
            except:
                refined_insights = insights

            # Create the refined analysis object
            refined_analysis = current_analysis.copy()
            refined_analysis["analysis"] = refined_text.strip()
            refined_analysis["metrics"] = refined_metrics
            refined_analysis["insights"] = refined_insights
            refined_analysis["refinement_iteration"] = refined_analysis.get("refinement_iteration", 0) + 1

            # Add confidence score if dynamic termination is enabled
            if self.enable_dynamic_termination:
                confidence_prompt = f"""
                On a scale of 0.0 to 1.0, how confident are you that the following financial analysis fully and accurately addresses the financial situation for: "{input}"

                Analysis: "{refined_text.strip()}"

                Please respond with only a number between 0.0 and 1.0.
                """

                confidence_response = await self.llm.generate(prompt=confidence_prompt)
                try:
                    confidence = float(confidence_response.strip())
                    refined_analysis["confidence"] = min(max(confidence, 0.0), 1.0)  # Ensure it's between 0 and 1
                except ValueError:
                    refined_analysis["confidence"] = 0.5  # Default if parsing fails

            return refined_analysis

        except Exception as e:
            self.logger.error(f"Error refining financial analysis: {str(e)}")
            # Return the original analysis with an error note
            current_analysis["refinement_error"] = str(e)
            return current_analysis

    async def _analyze_financials(self, input: str, current_step: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze financial data based on input.

        Args:
            input: Input to analyze (e.g., ticker symbol or company name)
            current_step: Optional current step from a plan

        Returns:
            Dictionary containing analysis results
        """
        try:
            # Debug: Print available tools
            available_tools = self.environment.get_available_tools()
            print(f"Available tools: {list(available_tools.keys()) if available_tools else 'None'}")

            # Initialize results containers
            financial_results = None
            semantic_results = None

            # If we have a current step with a specific tool, use that
            if current_step and "tool" in current_step:
                tool_name = current_step.get("tool")
                tool_params = current_step.get("parameters", {})

                print(f"Using tool specified in plan: {tool_name}")

                try:
                    # Execute the tool directly
                    result = await self.environment.execute_action({
                        "tool": tool_name,
                        "args": tool_params
                    })

                    # Store the result based on the tool type
                    if tool_name == "sec_financial_data":
                        financial_results = result
                    elif tool_name == "sec_semantic_search":
                        semantic_results = result

                except Exception as e:
                    print(f"Error executing planned tool {tool_name}: {str(e)}")
                    # Fall back to LLM-driven tool calling
                    tool_results = await self.process_with_llm_tools(input)
            else:
                # Process the input using LLM-driven tool calling
                tool_results = await self.process_with_llm_tools(input)

                # Extract results from tool calls
                for result in tool_results.get("results", []):
                    if result.get("success", False):
                        tool_name = result.get("tool")
                        tool_result = result.get("result", {})

                        if tool_name == "sec_financial_data":
                            financial_results = tool_result
                        elif tool_name == "sec_semantic_search":
                            semantic_results = tool_result

            # Process the analysis based on the current step if available
            focus_area = None
            if current_step:
                focus_area = current_step.get("description", "")

            # Parse the financial data
            financial_metrics = []
            if financial_results and financial_results.get("results"):
                for result in financial_results["results"]:
                    financial_metrics.append({
                        "metric": result.get("metric_name", ""),
                        "value": result.get("value", ""),
                        "period": result.get("period_end_date", ""),
                        "filing_type": result.get("filing_type", "")
                    })

            # Extract relevant text from semantic search
            financial_context = []
            if semantic_results and semantic_results.get("results"):
                for result in semantic_results["results"]:
                    financial_context.append({
                        "text": result.get("text", ""),
                        "company": result.get("metadata", {}).get("company", ""),
                        "filing_type": result.get("metadata", {}).get("filing_type", ""),
                        "filing_date": result.get("metadata", {}).get("filing_date", "")
                    })

            # Generate financial analysis using the LLM
            # If we have a focus area from the plan, include it in the prompt
            focus_instruction = ""
            if current_step:
                focus_area = current_step.get("description", "")
                if focus_area:
                    focus_instruction = f"\nFocus specifically on: {focus_area}"

            analysis_prompt = f"""
            Based on the following financial data for {input}, provide a comprehensive financial analysis:

            Financial Metrics:
            {json.dumps(financial_metrics, indent=2) if financial_metrics else "No financial metrics available."}

            Financial Context from SEC Filings:
            {json.dumps(financial_context, indent=2) if financial_context else "No financial context available."}{focus_instruction}

            Please provide:
            1. Key financial metrics analysis
            2. Financial ratio calculations and interpretations
            3. Trend analysis over time
            4. Strengths and weaknesses
            5. Overall financial health assessment
            """

            # Generate the analysis using the LLM
            analysis_response = await self.llm.generate(prompt=analysis_prompt)

            # Extract key metrics for structured output
            metrics_prompt = f"""
            Based on the financial data and your analysis, extract the key financial metrics and their values.
            Format your response as a JSON object with metric names as keys and values as strings.
            Include metrics like revenue growth, profit margin, debt ratio, etc.
            """

            metrics_response = await self.llm.generate(prompt=metrics_prompt)

            # Try to parse metrics as JSON
            try:
                # Extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', metrics_response, re.DOTALL)
                metrics_json = json.loads(json_match.group(0)) if json_match else {}
            except:
                metrics_json = {
                    "revenue_growth": "N/A",
                    "profit_margin": "N/A",
                    "debt_ratio": "N/A"
                }

            # Extract trends
            trends_prompt = f"""
            Based on the financial data and your analysis, list the key financial trends observed.
            Format your response as a JSON array of trend descriptions.
            """

            trends_response = await self.llm.generate(prompt=trends_prompt)

            # Try to parse trends as JSON
            try:
                # Extract JSON from the response
                json_match = re.search(r'\[.*\]', trends_response, re.DOTALL)
                trends_json = json.loads(json_match.group(0)) if json_match else []
            except:
                trends_json = [
                    "Insufficient data to determine trends"
                ]

            # Extract insights
            insights_prompt = f"""
            Based on the financial data and your analysis, provide key financial insights.
            Format your response as a JSON array of insight descriptions.
            """

            insights_response = await self.llm.generate(prompt=insights_prompt)

            # Try to parse insights as JSON
            try:
                # Extract JSON from the response
                json_match = re.search(r'\[.*\]', insights_response, re.DOTALL)
                insights_json = json.loads(json_match.group(0)) if json_match else []
            except:
                insights_json = [
                    "Insufficient data to provide meaningful insights"
                ]

            # Return the comprehensive analysis
            return {
                "input": input,
                "analysis": analysis_response.strip(),
                "metrics": metrics_json,
                "trends": trends_json,
                "insights": insights_json,
                "supporting_data": {
                    "financial_metrics": financial_metrics,
                    "financial_context": financial_context
                }
            }

        except Exception as e:
            # Return error information
            return {
                "input": input,
                "error": str(e),
                "analysis": "I encountered an error while analyzing the financial data. Please try again or provide more specific information.",
                "metrics": {},
                "trends": ["Error in analysis"],
                "insights": [f"Error details: {str(e)}"]
            }