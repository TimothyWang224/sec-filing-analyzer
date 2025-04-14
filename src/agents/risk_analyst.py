import json
import re
from typing import Dict, Any, List, Optional
from .base import Agent, Goal
from ..capabilities.base import Capability
from ..capabilities.time_awareness import TimeAwarenessCapability
from ..environments.financial import FinancialEnvironment

class RiskAnalystAgent(Agent):
    """Agent specialized in identifying and analyzing financial and operational risks."""

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
        Initialize the risk analyst agent.

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
                name="risk_identification",
                description="Identify potential financial and operational risks"
            ),
            Goal(
                name="risk_assessment",
                description="Assess the severity and likelihood of identified risks"
            ),
            Goal(
                name="risk_monitoring",
                description="Monitor and track changes in risk factors"
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
        Run the risk analyst agent.

        Args:
            user_input: The input to process (e.g., company name or risk assessment request)
            memory: Optional memory to initialize with

        Returns:
            Dictionary containing risk analysis results and recommendations
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

            # Process the input and generate risk analysis
            risk_analysis = await self._analyze_risks(user_input)

            # Add result to memory
            self.add_to_memory({
                "type": "risk_analysis",
                "content": risk_analysis
            })

            # Process result with capabilities
            for capability in self.capabilities:
                risk_analysis = await capability.process_result(
                    self,
                    {"input": user_input},
                    user_input,
                    {"type": "risk_analysis"},
                    risk_analysis
                )

            self.increment_iteration()

        return {
            "status": "completed",
            "risk_analysis": risk_analysis,
            "memory": self.get_memory()
        }

    async def _analyze_risks(self, input: str) -> Dict[str, Any]:
        """
        Analyze risks based on input.

        Args:
            input: Input to analyze (e.g., company name)

        Returns:
            Dictionary containing risk analysis results
        """
        try:
            # Debug: Print available tools
            available_tools = self.environment.get_available_tools()
            print(f"Available tools: {list(available_tools.keys()) if available_tools else 'None'}")

            # Initialize results containers
            semantic_results = None
            graph_results = None

            # Process the input using LLM-driven tool calling
            tool_results = await self.process_with_llm_tools(input)

            # Extract results from tool calls
            for result in tool_results.get("results", []):
                if result.get("success", False):
                    tool_name = result.get("tool")
                    tool_result = result.get("result", {})

                    if tool_name == "sec_semantic_search":
                        semantic_results = tool_result
                    elif tool_name == "sec_graph_query":
                        graph_results = tool_result

            # Extract relevant text from semantic search
            risk_context = []
            if semantic_results and semantic_results.get("results"):
                for result in semantic_results["results"]:
                    risk_context.append({
                        "text": result.get("text", ""),
                        "company": result.get("metadata", {}).get("company", ""),
                        "filing_type": result.get("metadata", {}).get("filing_type", ""),
                        "filing_date": result.get("metadata", {}).get("filing_date", "")
                    })

            # Generate risk analysis using the LLM
            analysis_prompt = f"""
            Based on the following information for {input}, provide a comprehensive risk analysis:

            Risk Context from SEC Filings:
            {json.dumps(risk_context, indent=2) if risk_context else "No risk context available."}

            Please identify and analyze:
            1. Financial risks (market risk, credit risk, liquidity risk, etc.)
            2. Operational risks (supply chain, technology, regulatory, etc.)
            3. Strategic risks (competition, market changes, etc.)
            4. Risk severity and likelihood assessment
            5. Risk mitigation recommendations
            """

            # Generate the analysis using the LLM
            analysis_response = await self.llm.generate(prompt=analysis_prompt)

            # Extract financial risks
            financial_risks_prompt = f"""
            Based on the SEC filing information and your analysis, identify the key financial risks.
            For each risk, provide the name, severity (High/Medium/Low), likelihood (High/Medium/Low), and a brief description.
            Format your response as a JSON array of risk objects.
            """

            financial_risks_response = await self.llm.generate(prompt=financial_risks_prompt)

            # Try to parse financial risks as JSON
            try:
                # Extract JSON from the response
                import re
                json_match = re.search(r'\[.*\]', financial_risks_response, re.DOTALL)
                financial_risks_json = json.loads(json_match.group(0)) if json_match else []
            except:
                financial_risks_json = [
                    {
                        "name": "Unknown Financial Risk",
                        "severity": "Medium",
                        "likelihood": "Medium",
                        "description": "Insufficient data to identify specific financial risks"
                    }
                ]

            # Extract operational risks
            operational_risks_prompt = f"""
            Based on the SEC filing information and your analysis, identify the key operational risks.
            For each risk, provide the name, severity (High/Medium/Low), likelihood (High/Medium/Low), and a brief description.
            Format your response as a JSON array of risk objects.
            """

            operational_risks_response = await self.llm.generate(prompt=operational_risks_prompt)

            # Try to parse operational risks as JSON
            try:
                # Extract JSON from the response
                json_match = re.search(r'\[.*\]', operational_risks_response, re.DOTALL)
                operational_risks_json = json.loads(json_match.group(0)) if json_match else []
            except:
                operational_risks_json = [
                    {
                        "name": "Unknown Operational Risk",
                        "severity": "Medium",
                        "likelihood": "Medium",
                        "description": "Insufficient data to identify specific operational risks"
                    }
                ]

            # Extract risk trends
            trends_prompt = f"""
            Based on the SEC filing information and your analysis, identify key risk trends for {input}.
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
                    "Insufficient data to determine risk trends"
                ]

            # Extract recommendations
            recommendations_prompt = f"""
            Based on the identified risks and trends, provide specific risk mitigation recommendations for {input}.
            Format your response as a JSON array of recommendation descriptions.
            """

            recommendations_response = await self.llm.generate(prompt=recommendations_prompt)

            # Try to parse recommendations as JSON
            try:
                # Extract JSON from the response
                json_match = re.search(r'\[.*\]', recommendations_response, re.DOTALL)
                recommendations_json = json.loads(json_match.group(0)) if json_match else []
            except:
                recommendations_json = [
                    "Insufficient data to provide meaningful risk mitigation recommendations"
                ]

            # Return the comprehensive risk analysis
            return {
                "input": input,
                "analysis": analysis_response.strip(),
                "risk_factors": {
                    "financial_risks": financial_risks_json,
                    "operational_risks": operational_risks_json
                },
                "risk_trends": trends_json,
                "recommendations": recommendations_json,
                "supporting_data": {
                    "risk_context": risk_context
                }
            }

        except Exception as e:
            # Return error information
            return {
                "input": input,
                "error": str(e),
                "analysis": "I encountered an error while analyzing the risks. Please try again or provide more specific information.",
                "risk_factors": {
                    "financial_risks": [],
                    "operational_risks": []
                },
                "risk_trends": ["Error in analysis"],
                "recommendations": [f"Error details: {str(e)}"]
            }