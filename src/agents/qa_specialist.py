import json
import traceback
from typing import Dict, Any, List, Optional
from .base import Agent, Goal
from ..capabilities.base import Capability
from ..capabilities.time_awareness import TimeAwarenessCapability
from ..capabilities.planning import PlanningCapability
from ..environments.financial import FinancialEnvironment

class QASpecialistAgent(Agent):
    """Agent specialized in answering financial questions and providing detailed explanations."""

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
        Initialize the QA specialist agent.

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
                name="question_answering",
                description="Answer financial questions accurately and comprehensively"
            ),
            Goal(
                name="explanation_generation",
                description="Generate clear and detailed explanations of financial concepts"
            ),
            Goal(
                name="context_understanding",
                description="Understand and maintain context across multiple questions"
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
            agent_type="qa_specialist"
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
            self.capabilities.append(PlanningCapability(
                enable_dynamic_replanning=True,
                enable_step_reflection=True,
                min_steps_before_reflection=1,
                max_plan_steps=5,  # Limited planning depth
                plan_detail_level="medium",
                is_coordinator=False,  # This is not a coordinator agent
                respect_existing_plan=True,  # Respect plans from coordinator
                # Planning instructions focused on how to accomplish the task
                planning_instructions="""You are a QA specialist agent responsible for retrieving and providing information.
                Your goal is to create a detailed plan to accomplish the task assigned by the coordinator.

                For each task:
                1. Analyze the task objective to understand what information is needed
                2. Determine which tools would be most appropriate to retrieve this information
                3. Create a step-by-step plan to gather and process the information
                4. Include specific tool selections and parameters in your plan

                Available tools:
                - sec_financial_data: Retrieves financial data from SEC filings
                  - query_type="companies": Lists all available companies
                  - query_type="metrics": Retrieves financial metrics for a company
                  - query_type="filings": Lists filings for a specific company

                - sec_semantic_search: Searches for information in SEC filings using semantic search
                  - Useful for finding specific information in filing text

                - sec_graph_query: Queries the knowledge graph for relationships between entities
                  - Useful for finding connections between companies, filings, etc.

                - sec_data: Retrieves specific filing data
                  - Useful for getting complete filing content

                Your plan should be detailed enough to accomplish the task effectively while staying focused on the specific objective.
                """
            ))

    async def run(self, user_input: str, plan: Optional[Dict] = None, memory: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run the QA specialist agent with a provided plan.

        Args:
            user_input: The input to process (e.g., financial question or concept)
            plan: Optional plan provided by the coordinator with specific instructions
            memory: Optional memory to initialize with

        Returns:
            Dictionary containing answer and supporting information
        """
        # Initialize memory if provided
        if memory:
            for item in memory:
                self.state.add_memory_item(item)

        # Add plan to memory if provided
        if plan:
            self.logger.info(f"Received plan from coordinator: {plan}")
            self.add_to_memory({
                "type": "execution_plan",
                "content": plan
            })

        # Initialize capabilities
        for capability in self.capabilities:
            await capability.init(self, {"input": user_input})

        # Log the start of processing
        self.logger.info(f"Processing question: {user_input}")

        # Check if we have a plan from the coordinator
        memory_items = self.get_memory()
        execution_plans = [item for item in memory_items if item.get("type") == "execution_plan"]
        has_coordinator_plan = len(execution_plans) > 0

        # If we have a plan from the coordinator, skip planning phase
        if has_coordinator_plan:
            self.logger.info("Skipping planning phase as plan was provided by coordinator")
            self.state.set_phase('execution')

            # Parse the question to extract key information for reference
            question_analysis = self._parse_question(user_input)

            # Add analysis to memory
            self.add_to_memory({
                "type": "question_analysis",
                "content": question_analysis
            })
        else:
            # Set initial phase to planning if no plan was provided
            self.state.set_phase('planning')
            self.logger.info(f"Starting planning phase")

        # Execute the agent loop through all phases
        answer = None

        # Phase 1: Planning (only if no plan was provided)
        while not self.should_terminate() and self.state.current_phase == 'planning' and not has_coordinator_plan:
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break

            # In planning phase, we focus on understanding the question and creating a plan
            self.logger.info(f"Planning how to answer: {user_input}")

            # Parse the question to extract key information
            question_analysis = self._parse_question(user_input)

            # Add analysis to memory
            self.add_to_memory({
                "type": "question_analysis",
                "content": question_analysis
            })

            # Process with capabilities
            for capability in self.capabilities:
                await capability.process_result(
                    self,
                    {"input": user_input},
                    user_input,
                    {"type": "question_analysis"},
                    question_analysis
                )

            self.increment_iteration()

            # If we've done enough planning, move to execution phase
            if self.state.phase_iterations['planning'] >= self.max_planning_iterations:
                self.state.set_phase('execution')
                self.logger.info(f"Moving to execution phase")

        # Phase 2: Execution
        while not self.should_terminate() and self.state.current_phase == 'execution':
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break

            # In execution phase, we gather data and generate an initial answer
            self.logger.info(f"Executing data gathering for: {user_input}")

            # Check if we have a plan with specific tool instructions
            if has_coordinator_plan:
                plan_content = execution_plans[-1].get("content", {})
                self.logger.info(f"Executing plan from coordinator: {plan_content}")

                # Extract high-level task information from the plan
                task_objective = plan_content.get("task_objective", "")
                success_criteria = plan_content.get("success_criteria", [])

                self.logger.info(f"Task objective from coordinator: {task_objective}")
                self.logger.info(f"Success criteria from coordinator: {success_criteria}")

                # Add the task objective to the planning context
                if hasattr(self.state, 'update_context'):
                    self.state.update_context({
                        "planning": {
                            "task_objective": task_objective,
                            "success_criteria": success_criteria
                        }
                    })

                # Use LLM-driven tool selection based on the task objective
                self.logger.info(f"Using LLM-driven tool selection based on task objective: {task_objective}")

                # Create a prompt that includes the task objective and success criteria
                tool_selection_prompt = f"""
                You need to accomplish the following task: {task_objective}

                Success criteria:
                {', '.join(success_criteria) if success_criteria else 'Provide accurate and relevant information'}

                Based on this task, select the most appropriate tool and parameters to use.
                Available tools:
                - sec_financial_data: Retrieves financial data from SEC filings
                  - query_type="companies": Lists all available companies
                  - query_type="metrics": Retrieves financial metrics for a company
                  - query_type="filings": Lists filings for a specific company

                - sec_semantic_search: Searches for information in SEC filings using semantic search
                  - parameters: {{"query": "search text"}}

                - sec_graph_query: Queries the knowledge graph for relationships between entities
                  - query_type="related_companies": Finds companies related to a given company
                  - query_type="company_filings": Finds filings for a specific company

                - sec_data: Retrieves specific filing data
                  - parameters: {{"company": "TICKER", "filing_type": "10-K"}}

                Return your selection as a JSON object with the following structure:
                {{"tool": "tool_name", "parameters": {{"param1": "value1", "param2": "value2"}}}}
                """

                try:
                    # Define the tool selection function
                    tool_selection_function = {
                        "name": "select_tool",
                        "description": "Select the most appropriate tool to answer the user's question",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "tool": {
                                    "type": "string",
                                    "description": "The name of the tool to use",
                                    "enum": ["sec_financial_data", "sec_semantic_search", "sec_graph_query", "sec_data"]
                                },
                                "parameters": {
                                    "type": "object",
                                    "description": "Parameters for the selected tool"
                                }
                            },
                            "required": ["tool"]
                        }
                    }

                    # Use function calling to get tool selection
                    self.logger.info("Using function calling for tool selection")
                    response = await self.llm.generate_with_functions(
                        prompt=tool_selection_prompt,
                        functions=[tool_selection_function],
                        function_call={"name": "select_tool"},  # Force the model to call this function
                        temperature=0.2  # Lower temperature for more deterministic tool selection
                    )

                    self.logger.info(f"Function call response: {response}")

                    # Check if we have a function call in the response
                    if "function_call" not in response:
                        self.logger.warning("No function call in response, falling back to normal answer generation")
                        answer = await self._generate_answer(user_input)
                        continue

                    # Extract function call information
                    function_call = response["function_call"]
                    self.logger.info(f"Function call: {function_call}")

                    # Parse the arguments JSON
                    try:
                        tool_selection = json.loads(function_call["arguments"])
                        self.logger.info(f"Parsed tool selection: {tool_selection}")

                        # Extract tool name and parameters
                        tool_name = tool_selection.get("tool")
                        tool_params = tool_selection.get("parameters", {})

                        self.logger.info(f"Selected tool: {tool_name} with parameters: {tool_params}")

                        # Execute the selected tool
                        if tool_name:
                            self.logger.info(f"Executing selected tool: {tool_name}")
                            tool_result = await self.environment.execute_action({
                                "tool": tool_name,
                                "args": tool_params
                            })

                            # Create a tool results structure
                            tool_results = {
                                "input": user_input,
                                "tool_calls": [{
                                    "tool": tool_name,
                                    "args": tool_params
                                }],
                                "results": [{
                                    "success": True,
                                    "tool": tool_name,
                                    "result": tool_result
                                }],
                                "timing": {"total": 0.0, "tool_selection": 0.0, "tool_execution": 0.0}
                            }

                            # Add the result to memory
                            self.add_to_memory({
                                "type": "tool_result",
                                "tool": tool_name,
                                "args": tool_params,
                                "result": tool_result
                            })

                            # Generate answer using the tool results
                            answer = await self._generate_answer_from_results(user_input, tool_results)
                        else:
                            # No tool selected, use normal answer generation
                            self.logger.info("No tool selected, using normal answer generation")
                            answer = await self._generate_answer(user_input)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing tool selection JSON: {str(e)}")
                        # Fall back to normal answer generation
                        answer = await self._generate_answer(user_input)
                except Exception as e:
                    self.logger.error(f"Error in LLM-driven tool selection: {str(e)}")
                    # Fall back to normal answer generation
                    answer = await self._generate_answer(user_input)
            else:
                # No plan, use normal answer generation
                answer = await self._generate_answer(user_input)

            # Add result to memory
            self.add_to_memory({
                "type": "qa_response",
                "content": answer
            })

            # Process result with capabilities
            for capability in self.capabilities:
                answer = await capability.process_result(
                    self,
                    {"input": user_input},
                    user_input,
                    {"type": "qa_response"},
                    answer
                )

            self.increment_iteration()

            # If we've done enough execution, move to refinement phase
            if self.state.phase_iterations['execution'] >= self.max_execution_iterations:
                self.state.set_phase('refinement')
                self.logger.info(f"Moving to refinement phase")

        # Phase 3: Refinement
        while not self.should_terminate() and self.state.current_phase == 'refinement':
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break

            # In refinement phase, we improve the answer
            self.logger.info(f"Refining answer for: {user_input}")

            # Get the current answer from memory
            memory_items = self.get_memory()
            qa_responses = [item for item in memory_items if item.get("type") == "qa_response"]

            if qa_responses:
                current_answer = qa_responses[-1].get("content", {})

                # Refine the answer
                refined_answer = await self._refine_answer(user_input, current_answer)

                # Add refined result to memory
                self.add_to_memory({
                    "type": "qa_response",
                    "content": refined_answer
                })

                # Process result with capabilities
                for capability in self.capabilities:
                    refined_answer = await capability.process_result(
                        self,
                        {"input": user_input},
                        user_input,
                        {"type": "qa_response"},
                        refined_answer
                    )

                answer = refined_answer

            self.increment_iteration()

            # If we've done enough refinement, we're done
            if self.state.phase_iterations['refinement'] >= self.max_refinement_iterations:
                break

        # Log the completion of processing
        self.logger.info(f"Result: Keys: {list(answer.keys()) if answer else []}")

        result = {
            "status": "completed",
            "answer": answer,
            "memory": self.get_memory(),
            "phase_iterations": self.state.phase_iterations
        }

        return result

    async def _generate_answer(self, input: str) -> Dict[str, Any]:
        """
        Generate answer based on input.

        Args:
            input: Input to process (e.g., financial question)

        Returns:
            Dictionary containing answer and supporting information
        """
        try:
            # Debug: Print available tools
            available_tools = self.environment.get_available_tools()
            print(f"Available tools: {list(available_tools.keys()) if available_tools else 'None'}")

            # Initialize results containers
            semantic_results = None
            graph_results = None
            financial_results = None

            # Use LLM-driven tool calling
            print("Using LLM-driven tool calling")

            # Add detailed debugging
            self.logger.info(f"QA Agent state before processing: {self.state.__dict__}")
            self.logger.info(f"QA Agent state has update_context: {hasattr(self.state, 'update_context')}")

            # Update the agent's context with the input
            try:
                self.logger.info("Attempting to update context")
                self.state.context["input"] = input  # Direct assignment as fallback
                if hasattr(self.state, 'update_context'):
                    self.logger.info("Using update_context method")
                    self.state.update_context({"input": input})
                self.logger.info(f"Context updated successfully: {self.state.context}")
            except Exception as e:
                self.logger.error(f"Error updating context: {str(e)}")

            # Process the question using LLM-driven tool calling
            try:
                self.logger.info("Attempting to process with LLM tools")
                # Try to use the parent class method first
                try:
                    tool_results = await self.process_with_llm_tools(input)
                    self.logger.info(f"Successfully processed input with LLM tools")
                except AttributeError:
                    # If the parent method fails, use our own implementation
                    self.logger.info("Parent process_with_llm_tools failed, using local implementation")
                    tool_results = await self._local_process_with_llm_tools(input)
            except Exception as e:
                self.logger.error(f"Error in process_with_llm_tools: {str(e)}")
                self.logger.error(f"Error type: {type(e).__name__}")
                self.logger.error(f"Error traceback: {traceback.format_exc()}")
                # Fallback to direct tool calling if LLM-driven approach fails
                self.logger.info("Falling back to direct tool execution")

                # Check if this is a request for available companies
                if "companies" in input.lower() and ("available" in input.lower() or "database" in input.lower() or "data" in input.lower()):
                    self.logger.info("Detected request for available companies, using direct tool execution")
                    try:
                        # Use the financial data tool to query available companies
                        companies_result = await self.environment.execute_action({
                            "tool": "sec_financial_data",
                            "args": {
                                "query_type": "companies"
                            }
                        })

                        # Extract company information
                        companies = []
                        if isinstance(companies_result, dict):
                            if "companies" in companies_result:
                                companies = companies_result["companies"]
                            elif "results" in companies_result and isinstance(companies_result["results"], list):
                                for company in companies_result["results"]:
                                    if isinstance(company, dict) and "ticker" in company:
                                        companies.append(company["ticker"])

                        if not companies:
                            # Fallback to hardcoded list
                            companies = ["AAPL", "GOOG", "MSFT", "NVDA"]

                        # Create a successful result
                        tool_results = {
                            "results": [{
                                "success": True,
                                "tool": "sec_financial_data",
                                "result": {
                                    "companies": companies
                                }
                            }]
                        }
                    except Exception as e:
                        self.logger.error(f"Error in direct tool execution: {str(e)}")
                        tool_results = {"results": []}
                else:
                    # For other types of queries, return empty results
                    tool_results = {"results": []}

            # Extract results from tool calls
            for result in tool_results.get("results", []):
                if result.get("success", False):
                    tool_name = result.get("tool")
                    tool_result = result.get("result", {})

                    if tool_name == "sec_semantic_search":
                        semantic_results = tool_result
                    elif tool_name == "sec_financial_data":
                        financial_results = tool_result
                    elif tool_name == "sec_graph_query":
                        graph_results = tool_result
                    elif tool_name == "sec_data":
                        # Handle SEC data tool results
                        if "filing_content" in tool_result:
                            semantic_results = {
                                "results": [{
                                    "text": tool_result.get("filing_content", ""),
                                    "metadata": {
                                        "company": tool_result.get("company", ""),
                                        "filing_type": tool_result.get("filing_type", ""),
                                        "filing_date": tool_result.get("filing_date", "")
                                    }
                                }]
                            }

            # Generate question analysis for reference
            question_parts = self._parse_question(input)

            # 5. Generate comprehensive answer using retrieved information
            # Combine all results to create a comprehensive answer
            semantic_context = []
            if semantic_results and semantic_results.get("results"):
                for result in semantic_results["results"]:
                    semantic_context.append({
                        "text": result.get("text", ""),
                        "company": result.get("metadata", {}).get("company", ""),
                        "filing_type": result.get("metadata", {}).get("filing_type", ""),
                        "filing_date": result.get("metadata", {}).get("filing_date", "")
                    })

            # Format financial data if available
            financial_data = []
            if financial_results and financial_results.get("results"):
                for result in financial_results["results"]:
                    financial_data.append({
                        "metric": result.get("metric_name", ""),
                        "value": result.get("value", ""),
                        "period": result.get("period_end_date", ""),
                        "filing_type": result.get("filing_type", "")
                    })

            # Format filing information if available
            filing_info = []
            if graph_results and graph_results.get("results"):
                for result in graph_results["results"]:
                    filing_info.append({
                        "filing_type": result.get("filing_type", ""),
                        "filing_date": result.get("filing_date", ""),
                        "accession_number": result.get("accession_number", "")
                    })

            # Check if this is a company listing request with results
            is_company_listing = False
            companies_list = []

            for result in tool_results.get("results", []):
                if result.get("tool") == "sec_financial_data" and "companies" in result.get("result", {}):
                    companies_list = result["result"]["companies"]
                    is_company_listing = True

            # Generate a natural language answer using the LLM
            if is_company_listing and companies_list:
                self.logger.info(f"Generating answer for company listing request with {len(companies_list)} companies")
                answer_prompt = f"""
                The user asked: "{input}"

                The following companies are available in the database: {', '.join(companies_list)}

                Please provide a friendly and helpful response that lists these companies and explains that the user can ask questions about any of them.
                """
            else:
                self.logger.info("Generating answer for general question")
                answer_prompt = f"""
                Based on the following information, answer the question: "{input}"

                Financial Data:
                {json.dumps(financial_data, indent=2) if financial_data else "No financial data available."}

                Semantic Search Results:
                {json.dumps(semantic_context, indent=2) if semantic_context else "No semantic search results available."}

                Filing Information:
                {json.dumps(filing_info, indent=2) if filing_info else "No filing information available."}

                Please provide a comprehensive answer that directly addresses the question.
                """

            # Generate the answer using the LLM
            answer_response = await self.llm.generate(prompt=answer_prompt)

            # Use the LLM-generated text as the answer
            answer = answer_response.strip()

            # Generate an explanation of the sources
            explanation = "This answer is based on "
            if financial_data:
                explanation += f"financial data from SEC filings"
            if semantic_context:
                explanation += f"{', and ' if financial_data else ''}information found in SEC filing text"
            if filing_info:
                explanation += f"{', and ' if financial_data or semantic_context else ''}SEC filing metadata"
            explanation += "."

            # Return the comprehensive answer
            return {
                "input": input,
                "answer": answer,
                "explanation": explanation,
                "supporting_data": {
                    "semantic_context": semantic_context,
                    "financial_data": financial_data,
                    "filing_info": filing_info
                },
                "question_analysis": question_parts
            }

        except Exception as e:
            # Return error information
            return {
                "input": input,
                "error": str(e),
                "answer": "I encountered an error while processing your question. Please try again or rephrase your question.",
                "explanation": f"Error details: {str(e)}"
            }

    async def _local_process_with_llm_tools(self, input_text: str) -> Dict[str, Any]:
        """
        Local implementation of process_with_llm_tools to use as a fallback.

        Args:
            input_text: User's input text

        Returns:
            Dictionary containing tool call results and other information
        """
        self.logger.info(f"Using local implementation of process_with_llm_tools for: {input_text[:100]}{'...' if len(input_text) > 100 else ''}")

        # For company listing requests, directly call the sec_financial_data tool
        if "companies" in input_text.lower() and ("available" in input_text.lower() or "database" in input_text.lower() or "data" in input_text.lower()):
            self.logger.info("Detected company listing request, using sec_financial_data tool")
            try:
                # Execute the tool
                result = await self.environment.execute_action({
                    "tool": "sec_financial_data",
                    "args": {
                        "query_type": "companies"
                    }
                })

                return {
                    "input": input_text,
                    "tool_calls": [{
                        "tool": "sec_financial_data",
                        "args": {"query_type": "companies"}
                    }],
                    "results": [{
                        "success": True,
                        "tool": "sec_financial_data",
                        "result": result
                    }],
                    "timing": {
                        "total": 0.0,
                        "tool_selection": 0.0,
                        "tool_execution": 0.0
                    }
                }
            except Exception as e:
                self.logger.error(f"Error executing sec_financial_data tool: {str(e)}")
                return {
                    "input": input_text,
                    "tool_calls": [],
                    "results": [],
                    "timing": {"total": 0.0, "tool_selection": 0.0, "tool_execution": 0.0}
                }

        # For other requests, return empty results
        return {
            "input": input_text,
            "tool_calls": [],
            "results": [],
            "timing": {"total": 0.0, "tool_selection": 0.0, "tool_execution": 0.0}
        }

    async def _generate_answer_from_results(self, input: str, tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer based on provided tool results.

        Args:
            input: Input to process (e.g., financial question)
            tool_results: Results from tool execution

        Returns:
            Dictionary containing answer and supporting information
        """
        try:
            # Initialize results containers
            semantic_results = None
            graph_results = None
            financial_results = None

            # Extract results from tool calls
            for result in tool_results.get("results", []):
                if result.get("success", False):
                    tool_name = result.get("tool")
                    tool_result = result.get("result", {})

                    if tool_name == "sec_semantic_search":
                        semantic_results = tool_result
                    elif tool_name == "sec_financial_data":
                        financial_results = tool_result
                    elif tool_name == "sec_graph_query":
                        graph_results = tool_result
                    elif tool_name == "sec_data":
                        # Handle SEC data tool results
                        if "filing_content" in tool_result:
                            semantic_results = {
                                "results": [{
                                    "text": tool_result.get("filing_content", ""),
                                    "metadata": {
                                        "company": tool_result.get("company", ""),
                                        "filing_type": tool_result.get("filing_type", ""),
                                        "filing_date": tool_result.get("filing_date", "")
                                    }
                                }]
                            }

            # Generate question analysis for reference
            question_parts = self._parse_question(input)

            # Combine all results to create a comprehensive answer
            semantic_context = []
            if semantic_results and semantic_results.get("results"):
                for result in semantic_results["results"]:
                    semantic_context.append({
                        "text": result.get("text", ""),
                        "company": result.get("metadata", {}).get("company", ""),
                        "filing_type": result.get("metadata", {}).get("filing_type", ""),
                        "filing_date": result.get("metadata", {}).get("filing_date", "")
                    })

            # Format financial data if available
            financial_data = []
            if financial_results and financial_results.get("results"):
                for result in financial_results["results"]:
                    financial_data.append({
                        "metric": result.get("metric_name", ""),
                        "value": result.get("value", ""),
                        "period": result.get("period_end_date", ""),
                        "filing_type": result.get("filing_type", "")
                    })

            # Format filing information if available
            filing_info = []
            if graph_results and graph_results.get("results"):
                for result in graph_results["results"]:
                    filing_info.append({
                        "filing_type": result.get("filing_type", ""),
                        "filing_date": result.get("filing_date", ""),
                        "accession_number": result.get("accession_number", "")
                    })

            # Check if this is a company listing request with results
            is_company_listing = False
            companies_list = []

            for result in tool_results.get("results", []):
                if result.get("tool") == "sec_financial_data" and "companies" in result.get("result", {}):
                    companies_list = result["result"]["companies"]
                    is_company_listing = True

            # Generate a natural language answer using the LLM
            if is_company_listing and companies_list:
                self.logger.info(f"Generating answer for company listing request with {len(companies_list)} companies")
                answer_prompt = f"""
                The user asked: "{input}"

                The following companies are available in the database: {', '.join(companies_list)}

                Please provide a friendly and helpful response that lists these companies and explains that the user can ask questions about any of them.
                """
            else:
                self.logger.info("Generating answer for general question")
                answer_prompt = f"""
                Based on the following information, answer the question: "{input}"

                Financial Data:
                {json.dumps(financial_data, indent=2) if financial_data else "No financial data available."}

                Semantic Search Results:
                {json.dumps(semantic_context, indent=2) if semantic_context else "No semantic search results available."}

                Filing Information:
                {json.dumps(filing_info, indent=2) if filing_info else "No filing information available."}

                Please provide a comprehensive answer that directly addresses the question.
                """

            # Generate the answer using the LLM
            answer_response = await self.llm.generate(prompt=answer_prompt)

            # Use the LLM-generated text as the answer
            answer = answer_response.strip()

            # Generate an explanation of the sources
            explanation = "This answer is based on "
            if financial_data:
                explanation += f"financial data from SEC filings"
            if semantic_context:
                explanation += f"{', and ' if financial_data else ''}information found in SEC filing text"
            if filing_info:
                explanation += f"{', and ' if financial_data or semantic_context else ''}SEC filing metadata"
            explanation += "."

            # Return the comprehensive answer
            return {
                "input": input,
                "answer": answer,
                "explanation": explanation,
                "supporting_data": {
                    "semantic_context": semantic_context,
                    "financial_data": financial_data,
                    "filing_info": filing_info
                },
                "question_analysis": question_parts
            }

        except Exception as e:
            # Return error information
            return {
                "input": input,
                "error": str(e),
                "answer": "I encountered an error while processing your question. Please try again or rephrase your question.",
                "explanation": f"Error details: {str(e)}"
            }

    async def _refine_answer(self, question: str, current_answer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine the current answer to improve its quality.

        Args:
            question: The original question
            current_answer: The current answer to refine

        Returns:
            Refined answer
        """
        try:
            # Extract the current answer text and supporting data
            answer_text = current_answer.get("answer", "")
            # supporting_data and question_analysis are extracted for potential future use
            _ = current_answer.get("supporting_data", {})
            _ = current_answer.get("question_analysis", {})

            # Generate a refinement prompt
            refinement_prompt = f"""
            I need to refine the following answer to the question: "{question}"

            Current answer: "{answer_text}"

            Please improve this answer by:
            1. Making it more concise and direct
            2. Ensuring it fully addresses the question
            3. Adding any missing context or clarifications
            4. Improving the explanation of financial concepts if present
            5. Ensuring numerical data is presented clearly

            Please provide only the refined answer text.
            """

            # Generate the refined answer using the LLM
            refined_text = await self.llm.generate(prompt=refinement_prompt)

            # Create the refined answer object
            refined_answer = current_answer.copy()
            refined_answer["answer"] = refined_text.strip()
            refined_answer["refinement_iteration"] = refined_answer.get("refinement_iteration", 0) + 1

            # Add confidence score if dynamic termination is enabled
            if self.enable_dynamic_termination:
                confidence_prompt = f"""
                On a scale of 0.0 to 1.0, how confident are you that the following answer fully and accurately addresses the question: "{question}"

                Answer: "{refined_text.strip()}"

                Please respond with only a number between 0.0 and 1.0.
                """

                confidence_response = await self.llm.generate(prompt=confidence_prompt)
                try:
                    confidence = float(confidence_response.strip())
                    refined_answer["confidence"] = min(max(confidence, 0.0), 1.0)  # Ensure it's between 0 and 1
                except ValueError:
                    refined_answer["confidence"] = 0.5  # Default if parsing fails

            return refined_answer

        except Exception as e:
            self.logger.error(f"Error refining answer: {str(e)}")
            # Return the original answer with an error note
            current_answer["refinement_error"] = str(e)
            return current_answer

    def _parse_question(self, question: str) -> Dict[str, Any]:
        """
        Parse the question to extract key information.

        Args:
            question: The question to parse

        Returns:
            Dictionary containing extracted information
        """
        # This is a simplified implementation
        # In practice, this would use NLP techniques to extract entities and intents

        # Extract potential companies (simple implementation)
        companies = []
        common_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]
        for ticker in common_tickers:
            if ticker in question.upper():
                companies.append(ticker)

        # Extract potential filing types
        filing_types = []
        if "10-K" in question or "annual" in question.lower():
            filing_types.append("10-K")
        if "10-Q" in question or "quarterly" in question.lower():
            filing_types.append("10-Q")
        if "8-K" in question or "current report" in question.lower():
            filing_types.append("8-K")

        # Extract potential metrics
        metrics = []
        common_metrics = ["Revenue", "Net Income", "EPS", "Gross Margin", "Operating Income"]
        for metric in common_metrics:
            if metric.lower() in question.lower():
                metrics.append(metric)

        # Use TimeAwarenessCapability to extract temporal information
        time_capability = next((cap for cap in self.capabilities if isinstance(cap, TimeAwarenessCapability)), None)
        temporal_info = {}

        if time_capability:
            # Extract temporal references
            temporal_references = time_capability._extract_temporal_references(question)

            # Get date range from temporal references
            if "date_range" in temporal_references:
                date_range = list(temporal_references["date_range"])
                temporal_info["date_range"] = date_range

            # Get fiscal period information
            if "fiscal_period" in temporal_references:
                fiscal_year, fiscal_quarter = temporal_references["fiscal_period"]
                temporal_info["fiscal_year"] = fiscal_year
                temporal_info["fiscal_quarter"] = fiscal_quarter

            # Add all temporal references
            temporal_info["temporal_references"] = temporal_references
        else:
            # Fallback to simple date range extraction
            date_range = None
            if "2023" in question:
                date_range = ["2023-01-01", "2023-12-31"]
            elif "2022" in question:
                date_range = ["2022-01-01", "2022-12-31"]

            if date_range:
                temporal_info["date_range"] = date_range

        return {
            "companies": companies,
            "filing_types": filing_types,
            "metrics": metrics,
            **temporal_info
        }