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
        # Agent iteration parameters
        max_iterations: int = 1,  # Legacy parameter, still used for backward compatibility
        max_planning_iterations: int = 1,
        max_execution_iterations: int = 2,
        max_refinement_iterations: int = 1,
        # Tool execution parameters
        max_tool_retries: int = 2,
        tools_per_iteration: int = 1,
        # Runtime parameters
        max_duration_seconds: int = 180,
        # LLM parameters
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 1000,
        # Environment
        environment: Optional[FinancialEnvironment] = None,
        # Termination parameters
        enable_dynamic_termination: bool = False,
        min_confidence_threshold: float = 0.8
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
            min_confidence_threshold=min_confidence_threshold
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

        # Log the start of processing
        self.logger.info(f"Processing risk analysis request: {user_input}")

        # Set initial phase to planning
        self.state.set_phase('planning')
        self.logger.info(f"Starting planning phase")

        # Execute the agent loop through all phases
        risk_analysis = None

        # Phase 1: Planning
        while not self.should_terminate() and self.state.current_phase == 'planning':
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break

            # In planning phase, we focus on understanding the company and identifying risk areas
            self.logger.info(f"Planning risk analysis for: {user_input}")

            # Parse the input to extract key information
            company_info = self._parse_company_info(user_input)

            # Add analysis to memory
            self.add_to_memory({
                "type": "company_info",
                "content": company_info
            })

            # Process with capabilities
            for capability in self.capabilities:
                await capability.process_result(
                    self,
                    {"input": user_input},
                    user_input,
                    {"type": "company_info"},
                    company_info
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

            # In execution phase, we gather data and generate an initial risk analysis
            self.logger.info(f"Executing risk data gathering for: {user_input}")

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

            # In refinement phase, we improve the risk analysis
            self.logger.info(f"Refining risk analysis for: {user_input}")

            # Get the current analysis from memory
            memory_items = self.get_memory()
            risk_analyses = [item for item in memory_items if item.get("type") == "risk_analysis"]

            if risk_analyses:
                current_analysis = risk_analyses[-1].get("content", {})

                # Refine the analysis
                refined_analysis = await self._refine_risk_analysis(user_input, current_analysis)

                # Add refined result to memory
                self.add_to_memory({
                    "type": "risk_analysis",
                    "content": refined_analysis
                })

                # Process result with capabilities
                for capability in self.capabilities:
                    refined_analysis = await capability.process_result(
                        self,
                        {"input": user_input},
                        user_input,
                        {"type": "risk_analysis"},
                        refined_analysis
                    )

                risk_analysis = refined_analysis

            self.increment_iteration()

            # If we've done enough refinement, we're done
            if self.state.phase_iterations['refinement'] >= self.max_refinement_iterations:
                break

        # Log the completion of processing
        self.logger.info(f"Risk analysis completed for: {user_input}")

        return {
            "status": "completed",
            "risk_analysis": risk_analysis,
            "memory": self.get_memory(),
            "phase_iterations": self.state.phase_iterations
        }

    def _parse_company_info(self, input: str) -> Dict[str, Any]:
        """
        Parse the input to extract company information.

        Args:
            input: Input to parse (e.g., company name or ticker)

        Returns:
            Dictionary containing company information
        """
        # Extract potential company name or ticker
        company_name = input
        ticker = None

        # Check for ticker pattern (all caps, 1-5 letters)
        import re
        ticker_match = re.search(r'\b[A-Z]{1,5}\b', input)
        if ticker_match:
            ticker = ticker_match.group(0)

        # Extract potential industry or sector
        industry = None
        sector = None

        # Common industries and sectors
        industries = ["technology", "healthcare", "finance", "retail", "energy", "manufacturing"]
        sectors = ["tech", "health", "financial", "consumer", "energy", "industrial"]

        for ind in industries:
            if ind.lower() in input.lower():
                industry = ind
                break

        for sec in sectors:
            if sec.lower() in input.lower():
                sector = sec
                break

        # Extract potential time frame
        time_capability = next((cap for cap in self.capabilities if isinstance(cap, TimeAwarenessCapability)), None)
        temporal_info = {}

        if time_capability:
            temporal_references = time_capability.extract_temporal_references(input)
            temporal_info["temporal_references"] = temporal_references
        else:
            # Simple date extraction
            year_match = re.search(r'\b(20\d{2})\b', input)
            if year_match:
                year = year_match.group(1)
                temporal_info["year"] = year
                temporal_info["date_range"] = [f"{year}-01-01", f"{year}-12-31"]

        return {
            "company_name": company_name,
            "ticker": ticker,
            "industry": industry,
            "sector": sector,
            **temporal_info
        }

    async def _refine_risk_analysis(self, input: str, current_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine the current risk analysis to improve its quality.

        Args:
            input: The original input
            current_analysis: The current analysis to refine

        Returns:
            Refined risk analysis
        """
        try:
            # Extract the current analysis text and risk factors
            analysis_text = current_analysis.get("analysis", "")
            risk_factors = current_analysis.get("risk_factors", {})
            risk_trends = current_analysis.get("risk_trends", [])
            recommendations = current_analysis.get("recommendations", [])

            # Generate a refinement prompt
            refinement_prompt = f"""
            I need to refine the following risk analysis for: "{input}"

            Current analysis: "{analysis_text}"

            Please improve this risk analysis by:
            1. Making it more concise and focused on key risks
            2. Ensuring all major risk categories are covered
            3. Providing more specific risk assessments with clear severity and likelihood ratings
            4. Improving the actionability of recommendations
            5. Ensuring numerical data and trends are presented clearly

            Please provide only the refined analysis text.
            """

            # Generate the refined analysis using the LLM
            refined_text = await self.llm.generate(prompt=refinement_prompt)

            # Refine risk factors
            risk_factors_prompt = f"""
            Based on the refined analysis, please improve the risk factors for {input}.

            Current financial risks: {json.dumps(risk_factors.get('financial_risks', []), indent=2)}
            Current operational risks: {json.dumps(risk_factors.get('operational_risks', []), indent=2)}

            For each risk category, provide a more focused list with clear severity and likelihood ratings.
            Format your response as a JSON object with 'financial_risks' and 'operational_risks' arrays.
            Each risk should have 'name', 'severity', 'likelihood', and 'description' fields.
            """

            risk_factors_response = await self.llm.generate(prompt=risk_factors_prompt)

            # Try to parse risk factors as JSON
            try:
                # Extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', risk_factors_response, re.DOTALL)
                refined_risk_factors = json.loads(json_match.group(0)) if json_match else risk_factors
            except:
                refined_risk_factors = risk_factors

            # Refine recommendations
            recommendations_prompt = f"""
            Based on the refined analysis, please improve the risk mitigation recommendations for {input}.

            Current recommendations: {json.dumps(recommendations, indent=2)}

            Please provide more specific, actionable recommendations that address the key risks identified.
            Format your response as a JSON array of recommendation strings.
            """

            recommendations_response = await self.llm.generate(prompt=recommendations_prompt)

            # Try to parse recommendations as JSON
            try:
                # Extract JSON from the response
                json_match = re.search(r'\[.*\]', recommendations_response, re.DOTALL)
                refined_recommendations = json.loads(json_match.group(0)) if json_match else recommendations
            except:
                refined_recommendations = recommendations

            # Create the refined analysis object
            refined_analysis = current_analysis.copy()
            refined_analysis["analysis"] = refined_text.strip()
            refined_analysis["risk_factors"] = refined_risk_factors
            refined_analysis["recommendations"] = refined_recommendations
            refined_analysis["refinement_iteration"] = refined_analysis.get("refinement_iteration", 0) + 1

            # Add confidence score if dynamic termination is enabled
            if self.enable_dynamic_termination:
                confidence_prompt = f"""
                On a scale of 0.0 to 1.0, how confident are you that the following risk analysis fully and accurately addresses the risks for: "{input}"

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
            self.logger.error(f"Error refining risk analysis: {str(e)}")
            # Return the original analysis with an error note
            current_analysis["refinement_error"] = str(e)
            return current_analysis

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