from typing import Dict, Any, List, Optional
from base import Agent, Goal
from financial_analyst import FinancialAnalystAgent
from risk_analyst import RiskAnalystAgent
from qa_specialist import QASpecialistAgent
from capabilities.base import Capability
from memory.base import Memory
from sec_filing_analyzer.llm import get_agent_config

class FinancialDiligenceCoordinator(Agent):
    """Coordinates multiple agents for comprehensive financial diligence."""
    
    def __init__(
        self,
        capabilities: Optional[List[Capability]] = None,
        max_iterations: int = 10,
        max_duration_seconds: int = 300
    ):
        """
        Initialize the financial diligence coordinator.
        
        Args:
            capabilities: List of capabilities to extend coordinator behavior
            max_iterations: Maximum number of action loops
            max_duration_seconds: Maximum runtime in seconds
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
        
        # Get coordinator-specific LLM configuration
        llm_config = get_agent_config("coordinator")
        
        super().__init__(
            goals=goals,
            capabilities=capabilities,
            max_iterations=max_iterations,
            max_duration_seconds=max_duration_seconds,
            llm_config=llm_config
        )
        
        # Initialize specialized agents with their configurations
        self.financial_analyst = FinancialAnalystAgent(
            llm_config=get_agent_config("financial_analyst")
        )
        self.risk_analyst = RiskAnalystAgent(
            llm_config=get_agent_config("risk_analyst")
        )
        self.qa_specialist = QASpecialistAgent(
            llm_config=get_agent_config("qa_specialist")
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
                    
            # Run financial analysis
            financial_analysis = await self.financial_analyst.run(user_input)
            
            # Run risk analysis
            risk_analysis = await self.risk_analyst.run(user_input)
            
            # Generate comprehensive report
            diligence_report = await self._generate_diligence_report(
                user_input,
                financial_analysis,
                risk_analysis
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
        
    async def _generate_diligence_report(
        self,
        input: str,
        financial_analysis: Dict[str, Any],
        risk_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive diligence report.
        
        Args:
            input: Input to analyze (e.g., company name)
            financial_analysis: Results from financial analyst
            risk_analysis: Results from risk analyst
            
        Returns:
            Dictionary containing comprehensive diligence report
        """
        # This is a placeholder for the actual report generation logic
        # In practice, this would:
        # 1. Combine insights from multiple agents
        # 2. Identify correlations and patterns
        # 3. Generate recommendations
        # 4. Create comprehensive report
        
        return {
            "input": input,
            "executive_summary": "Comprehensive analysis of company's financial health and risks...",
            "financial_health": {
                "metrics": financial_analysis.get("metrics", {}),
                "trends": financial_analysis.get("trends", []),
                "insights": financial_analysis.get("insights", [])
            },
            "risk_profile": {
                "risk_factors": risk_analysis.get("risk_factors", {}),
                "risk_trends": risk_analysis.get("risk_trends", []),
                "recommendations": risk_analysis.get("recommendations", [])
            },
            "key_findings": [
                "Strong financial performance with positive growth trajectory",
                "Moderate risk profile with manageable challenges",
                "Opportunities for improvement in risk management"
            ],
            "recommendations": [
                "Continue current growth strategy",
                "Implement suggested risk mitigation measures",
                "Monitor key performance indicators"
            ]
        } 