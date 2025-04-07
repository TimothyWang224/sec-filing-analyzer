from typing import Dict, Any, List, Optional
from ..agents.base import Agent, Goal
from ..capabilities.base import Capability
from ..memory.base import Memory
from ..sec_filing_analyzer.llm import get_agent_config

class FinancialAnalystAgent(Agent):
    """Agent specialized in analyzing financial statements and metrics."""
    
    def __init__(
        self,
        capabilities: Optional[List[Capability]] = None,
        max_iterations: int = 10,
        max_duration_seconds: int = 180,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the financial analyst agent.
        
        Args:
            capabilities: List of capabilities to extend agent behavior
            max_iterations: Maximum number of action loops
            max_duration_seconds: Maximum runtime in seconds
            llm_config: Optional LLM configuration to override defaults
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
        
        # Use provided config or get default from centralized config
        llm_config = llm_config or get_agent_config("financial_analyst")
        
        super().__init__(
            goals=goals,
            capabilities=capabilities,
            max_iterations=max_iterations,
            max_duration_seconds=max_duration_seconds,
            llm_config=llm_config
        )
        
    async def run(self, user_input: str, memory: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run the financial analyst agent.
        
        Args:
            user_input: The input to process (e.g., ticker symbol or analysis request)
            memory: Optional memory to initialize with
            
        Returns:
            Dictionary containing analysis results and insights
        """
        if memory:
            self.memory = memory
            
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
                "type": "analysis",
                "content": analysis_result
            })
            
            # Process result with capabilities
            for capability in self.capabilities:
                analysis_result = await capability.process_result(
                    self,
                    {"input": user_input},
                    user_input,
                    {"type": "analysis"},
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
            input: Input to analyze (e.g., ticker symbol)
            
        Returns:
            Dictionary containing analysis results
        """
        # This is a placeholder for the actual analysis logic
        # In practice, this would use various tools to:
        # 1. Retrieve financial data
        # 2. Calculate ratios
        # 3. Identify trends
        # 4. Generate insights
        
        return {
            "input": input,
            "metrics": {
                "revenue_growth": "10%",
                "profit_margin": "25%",
                "debt_ratio": "0.4"
            },
            "trends": [
                "Increasing revenue growth",
                "Stable profit margins",
                "Declining debt ratio"
            ],
            "insights": [
                "Strong financial performance",
                "Healthy balance sheet",
                "Positive growth trajectory"
            ]
        } 