import json
import re
from typing import Dict, Any, List, Optional
from .base import Agent, Goal
from .financial_analyst import FinancialAnalystAgent
from .risk_analyst import RiskAnalystAgent
from .qa_specialist import QASpecialistAgent
from ..capabilities.base import Capability
from ..capabilities.time_awareness import TimeAwarenessCapability
from ..capabilities.planning import PlanningCapability
from ..environments.financial import FinancialEnvironment

class FinancialDiligenceCoordinator(Agent):
    """Coordinates multiple agents for comprehensive financial diligence."""

    def __init__(
        self,
        capabilities: Optional[List[Capability]] = None,
        max_iterations: int = 1,
        max_duration_seconds: int = 300,
        environment: Optional[FinancialEnvironment] = None,
        llm_model: str = "gpt-4o",
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 2000,
        max_tool_calls: int = 3
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
            Goal(
                name="coordination",
                description="Coordinate multiple agents for comprehensive analysis"
            ),
            Goal(
                name="synthesis",
                description="Synthesize insights from multiple agents"
            ),
            Goal(
                name="reporting",
                description="Generate comprehensive diligence reports"
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

        # Add PlanningCapability if not already present
        has_planning = any(isinstance(cap, PlanningCapability) for cap in self.capabilities)
        if not has_planning:
            self.capabilities.append(PlanningCapability(
                enable_dynamic_replanning=True,
                enable_step_reflection=True,
                min_steps_before_reflection=2,
                max_plan_steps=10,
                plan_detail_level="high"
            ))

        # Initialize specialized agents
        self.financial_analyst = FinancialAnalystAgent(
            environment=self.environment,
            llm_model="gpt-4o-mini",
            llm_temperature=0.3
        )
        self.risk_analyst = RiskAnalystAgent(
            environment=self.environment,
            llm_model="gpt-4o-mini",
            llm_temperature=0.3
        )
        self.qa_specialist = QASpecialistAgent(
            environment=self.environment,
            llm_model="gpt-4o-mini",
            llm_temperature=0.5
        )

    async def run(self, user_input: str, memory: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run the financial diligence coordinator.

        Args:
            user_input: The input to process (e.g., company name or diligence request)
            memory: Optional memory to initialize with

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

        while not self.should_terminate():
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break

            # Check if we have a plan from the planning capability
            planning_context = self.state.get_context().get("planning", {})
            current_step = planning_context.get("current_step", {})

            # If we have a plan with a current step, follow it
            if current_step:
                print(f"Executing plan step: {current_step.get('description')}")

                # If the step specifies an agent, run that agent
                if "agent" in current_step:
                    agent_name = current_step.get("agent")
                    print(f"Running {agent_name} as specified in the plan...")

                    financial_analysis = None
                    risk_analysis = None
                    qa_response = None

                    if agent_name == "financial_analyst":
                        financial_analysis = await self.financial_analyst.run(user_input)
                    elif agent_name == "risk_analyst":
                        risk_analysis = await self.risk_analyst.run(user_input)
                    elif agent_name == "qa_specialist":
                        qa_response = await self.qa_specialist.run(user_input)
                else:
                    # If no specific agent is specified, determine which agents to run
                    agent_selection = await self._select_agents(user_input)

                    financial_analysis = None
                    risk_analysis = None
                    qa_response = None

                    # Run selected agents
                    if agent_selection.get("financial_analyst", False):
                        print("Running Financial Analyst Agent...")
                        financial_analysis = await self.financial_analyst.run(user_input)

                    if agent_selection.get("risk_analyst", False):
                        print("Running Risk Analyst Agent...")
                        risk_analysis = await self.risk_analyst.run(user_input)

                    if agent_selection.get("qa_specialist", False):
                        print("Running QA Specialist Agent...")
                        qa_response = await self.qa_specialist.run(user_input)
            else:
                # If we don't have a plan or current step, fall back to the original behavior
                agent_selection = await self._select_agents(user_input)

                financial_analysis = None
                risk_analysis = None
                qa_response = None

                # Run selected agents
                if agent_selection.get("financial_analyst", False):
                    print("Running Financial Analyst Agent...")
                    financial_analysis = await self.financial_analyst.run(user_input)

                if agent_selection.get("risk_analyst", False):
                    print("Running Risk Analyst Agent...")
                    risk_analysis = await self.risk_analyst.run(user_input)

                if agent_selection.get("qa_specialist", False):
                    print("Running QA Specialist Agent...")
                    qa_response = await self.qa_specialist.run(user_input)

            # Generate comprehensive report
            diligence_report = await self._generate_diligence_report(
                user_input,
                financial_analysis,
                risk_analysis,
                qa_response
            )

            # Add result to memory
            self.add_to_memory({
                "type": "diligence_report",
                "content": diligence_report
            })

            # Process result with capabilities
            for capability in self.capabilities:
                diligence_report = await capability.process_result(
                    self,
                    {"input": user_input},
                    user_input,
                    {"type": "diligence_report"},
                    diligence_report
                )

            self.increment_iteration()

        return {
            "status": "completed",
            "diligence_report": diligence_report,
            "memory": self.get_memory()
        }

    async def _select_agents(self, user_input: str) -> Dict[str, bool]:
        """
        Determine which specialized agents to run based on the input.

        Args:
            user_input: The user's input

        Returns:
            Dictionary mapping agent names to boolean indicating whether to run them
        """
        # Create a prompt for agent selection
        prompt = f"""
        Based on the following user input, determine which specialized agents should be run:

        User Input: "{user_input}"

        Available Agents:
        1. Financial Analyst Agent - Specializes in analyzing financial statements and metrics
        2. Risk Analyst Agent - Specializes in identifying and analyzing financial and operational risks
        3. QA Specialist Agent - Specializes in answering specific financial questions

        For each agent, determine if it should be run (true/false) based on the user input.
        Return your decision as a JSON object with the following structure:
        {{"financial_analyst": true/false, "risk_analyst": true/false, "qa_specialist": true/false}}
        """

        # Generate agent selection
        response = await self.llm.generate(prompt=prompt, temperature=0.3)

        # Parse agent selection from response
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                agent_selection = json.loads(json_match.group(0))
            else:
                # Default to running all agents if parsing fails
                agent_selection = {
                    "financial_analyst": True,
                    "risk_analyst": True,
                    "qa_specialist": True
                }
        except Exception as e:
            print(f"Error parsing agent selection: {str(e)}")
            # Default to running all agents if parsing fails
            agent_selection = {
                "financial_analyst": True,
                "risk_analyst": True,
                "qa_specialist": True
            }

        return agent_selection

    async def _generate_diligence_report(
        self,
        input: str,
        financial_analysis: Optional[Dict[str, Any]],
        risk_analysis: Optional[Dict[str, Any]],
        qa_response: Optional[Dict[str, Any]]
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
                    "insights": financial_analysis.get("analysis", {}).get("insights", [])
                }

            # Extract risk profile information
            risk_profile = {}
            if risk_analysis:
                risk_profile = {
                    "risk_factors": risk_analysis.get("risk_analysis", {}).get("risk_factors", {}),
                    "risk_trends": risk_analysis.get("risk_analysis", {}).get("risk_trends", []),
                    "recommendations": risk_analysis.get("risk_analysis", {}).get("recommendations", [])
                }

            # Extract QA information
            qa_info = {}
            if qa_response:
                qa_info = {
                    "question": input,
                    "answer": qa_response.get("answer", {}).get("answer", "")
                }

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
                json_match = re.search(r'\[.*\]', findings_response, re.DOTALL)
                key_findings = json.loads(json_match.group(0)) if json_match else []
            except:
                key_findings = [
                    "Insufficient data to determine key findings"
                ]

            # Generate recommendations
            recommendations_prompt = f"""
            Based on the financial analysis and risk assessment for {input}, provide strategic recommendations.
            Format your response as a JSON array of recommendation descriptions.
            """

            recommendations_response = await self.llm.generate(prompt=recommendations_prompt)

            # Try to parse recommendations as JSON
            try:
                # Extract JSON from the response
                json_match = re.search(r'\[.*\]', recommendations_response, re.DOTALL)
                recommendations = json.loads(json_match.group(0)) if json_match else []
            except:
                recommendations = [
                    "Insufficient data to provide meaningful recommendations"
                ]

            # Return the comprehensive diligence report
            return {
                "input": input,
                "executive_summary": executive_summary.strip(),
                "financial_health": financial_health,
                "risk_profile": risk_profile,
                "qa_response": qa_info.get("answer", "") if qa_info else "",
                "key_findings": key_findings,
                "recommendations": recommendations
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
                "recommendations": [f"Error details: {str(e)}"]
            }