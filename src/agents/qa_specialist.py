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
        max_iterations: int = 1,
        max_duration_seconds: int = 180,
        environment: Optional[FinancialEnvironment] = None,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 4000,
        max_tool_calls: int = 3
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

        # Initialize the base agent
        super().__init__(
            goals=goals,
            capabilities=capabilities,
            max_iterations=max_iterations,
            max_duration_seconds=max_duration_seconds,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            environment=environment,
            max_tool_calls=max_tool_calls
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

        while not self.should_terminate():
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break

            # Process the input and generate answer
            self.logger.info(f"Generating answer for: {user_input}")
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

        # Log the completion of processing
        self.logger.info(f"Result: Keys: {list(answer.keys())}")

        result = {
            "status": "completed",
            "answer": answer,
            "memory": self.get_memory()
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