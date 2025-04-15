import json
from typing import Dict, Any, List, Optional
from .base import Agent, Goal
from ..capabilities.base import Capability
from ..capabilities.time_awareness import TimeAwarenessCapability
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

    async def run(self, user_input: str, memory: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run the QA specialist agent.

        Args:
            user_input: The input to process (e.g., financial question or concept)
            memory: Optional memory to initialize with

        Returns:
            Dictionary containing answer and supporting information
        """
        # Initialize memory if provided
        if memory:
            for item in memory:
                self.state.add_memory_item(item)

        # Initialize capabilities
        for capability in self.capabilities:
            await capability.init(self, {"input": user_input})

        # Log the start of processing
        self.logger.info(f"Processing question: {user_input}")

        # Set initial phase to planning
        self.state.set_phase('planning')
        self.logger.info(f"Starting planning phase")

        # Execute the agent loop through all phases
        answer = None

        # Phase 1: Planning
        while not self.should_terminate() and self.state.current_phase == 'planning':
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

            # Process the input and generate answer
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

            # Process the question using LLM-driven tool calling
            tool_results = await self.process_with_llm_tools(input)

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

            # Generate a natural language answer using the LLM
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