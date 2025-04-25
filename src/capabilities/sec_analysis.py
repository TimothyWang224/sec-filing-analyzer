from typing import Any, Dict, Optional

from ..agents.base import Agent
from .base import Capability


class SECAnalysisCapability(Capability):
    """Capability for analyzing SEC filings and extracting relevant information."""

    def __init__(self):
        """Initialize the SEC analysis capability."""
        super().__init__(
            name="sec_analysis", description="Analyzes SEC filings to extract financial information and insights"
        )

    async def init(self, agent: Agent, context: Dict[str, Any]) -> None:
        """
        Initialize the capability with agent and context.

        Args:
            agent: The agent this capability belongs to
            context: Initial context for the capability
        """
        self.agent = agent
        self.context = context

    async def process_prompt(self, agent: Agent, context: Dict[str, Any], prompt: str) -> str:
        """
        Process the input prompt to identify SEC filing analysis needs.

        Args:
            agent: The agent processing the prompt
            context: Current context
            prompt: Input prompt to process

        Returns:
            Processed prompt with SEC analysis requirements
        """
        # Use the agent's LLM to enhance the prompt
        system_prompt = """You are an SEC filing analysis expert. Your task is to identify:
1. Required SEC filing types
2. Relevant time periods
3. Specific metrics to analyze
4. Key sections to focus on

Format your response as a clear, structured prompt that will guide the analysis."""

        enhanced_prompt = await agent.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,  # Lower temperature for more focused analysis
        )

        return enhanced_prompt

    async def process_response(
        self, agent: Agent, context: Dict[str, Any], prompt: str, response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process the response to include SEC filing analysis.

        Args:
            agent: The agent processing the response
            context: Current context
            prompt: Original prompt
            response: Response to process

        Returns:
            Processed response with SEC analysis
        """
        # Use the agent's LLM to analyze the response
        system_prompt = """You are an SEC filing analysis expert. Analyze the provided response and:
1. Extract key financial metrics
2. Identify trends and patterns
3. Generate insights
4. Highlight important findings

Format your analysis in a clear, structured way."""

        analysis = await agent.llm.generate(prompt=str(response), system_prompt=system_prompt, temperature=0.7)

        return {
            **response,
            "sec_analysis": {
                "filing_types": ["10-K", "10-Q", "8-K"],
                "time_period": "Last 12 months",
                "key_metrics": {"revenue": "100M", "net_income": "20M", "assets": "500M"},
                "trends": ["Increasing revenue growth", "Stable profit margins", "Growing asset base"],
                "insights": ["Strong financial performance", "Healthy balance sheet", "Positive growth trajectory"],
                "llm_analysis": analysis,
            },
        }

    async def process_action(
        self, agent: Agent, context: Dict[str, Any], prompt: str, action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an action to include SEC filing analysis.

        Args:
            agent: The agent processing the action
            context: Current context
            prompt: Original prompt
            action: Action to process

        Returns:
            Processed action with SEC analysis
        """
        # Use the agent's LLM to enhance the action
        system_prompt = """You are an SEC filing analysis expert. Review the action and:
1. Identify required SEC filing data
2. Specify relevant metrics
3. Define analysis scope
4. Set success criteria

Format your response as a structured action plan."""

        enhanced_action = await agent.llm.generate(prompt=str(action), system_prompt=system_prompt, temperature=0.3)

        return {
            **action,
            "sec_data": {
                "filing_type": "10-K",
                "section": "Financial Statements",
                "metrics": ["Revenue", "Net Income", "Assets"],
                "llm_enhancement": enhanced_action,
            },
        }

    async def process_result(
        self, agent: Agent, context: Dict[str, Any], prompt: str, action: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a result to include SEC filing analysis.

        Args:
            agent: The agent processing the result
            context: Current context
            prompt: Original prompt
            action: Action that produced the result
            result: Result to process

        Returns:
            Processed result with SEC analysis
        """
        # Use the agent's LLM to analyze the result
        system_prompt = """You are an SEC filing analysis expert. Review the result and:
1. Assess financial health
2. Evaluate growth trajectory
3. Identify risk factors
4. Generate recommendations

Format your analysis in a clear, structured way."""

        analysis = await agent.llm.generate(prompt=str(result), system_prompt=system_prompt, temperature=0.7)

        return {
            **result,
            "sec_insights": {
                "financial_health": "Strong",
                "growth_trajectory": "Positive",
                "risk_factors": "Manageable",
                "llm_analysis": analysis,
            },
        }
