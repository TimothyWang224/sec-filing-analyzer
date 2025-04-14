import json
import re
from typing import Dict, Any, List, Optional
from .base import Agent, Goal
from ..capabilities.base import Capability
from ..capabilities.time_awareness import TimeAwarenessCapability
from ..environments.financial import FinancialEnvironment

class FinancialAnalystAgent(Agent):
    """Agent specialized in analyzing financial statements and metrics."""

    def __init__(
        self,
        capabilities: Optional[List[Capability]] = None,
        max_iterations: int = 1,
        max_duration_seconds: int = 180,
        environment: Optional[FinancialEnvironment] = None,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 1000,
        max_tool_calls: int = 3
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

        # Initialize capabilities
        for capability in self.capabilities:
            await capability.init(self, {"input": user_input})

        while not self.should_terminate():
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break

            # Process the input and generate analysis
            analysis_result = await self._analyze_financials(user_input)

            # Add result to memory
            self.add_to_memory({
                "type": "financial_analysis",
                "content": analysis_result
            })

            # Process result with capabilities
            for capability in self.capabilities:
                analysis_result = await capability.process_result(
                    self,
                    {"input": user_input},
                    user_input,
                    {"type": "financial_analysis"},
                    analysis_result
                )

            self.increment_iteration()

        return {
            "status": "completed",
            "analysis": analysis_result,
            "memory": self.get_memory()
        }

    async def _analyze_financials(self, input: str) -> Dict[str, Any]:
        """
        Analyze financial data based on input.

        Args:
            input: Input to analyze (e.g., ticker symbol or company name)

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
            analysis_prompt = f"""
            Based on the following financial data for {input}, provide a comprehensive financial analysis:

            Financial Metrics:
            {json.dumps(financial_metrics, indent=2) if financial_metrics else "No financial metrics available."}

            Financial Context from SEC Filings:
            {json.dumps(financial_context, indent=2) if financial_context else "No financial context available."}

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