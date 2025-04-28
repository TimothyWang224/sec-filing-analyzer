from typing import Any, Dict, Optional

from agents import (
    FinancialAnalystAgent,
    FinancialDiligenceCoordinator,
    QASpecialistAgent,
    RiskAnalystAgent,
)
from capabilities import SECAnalysisCapability
from environments import FinancialEnvironment
from memory import FinancialMemory


class SECFilingAnalyzer:
    """Main API for SEC filing analysis."""

    def __init__(self):
        """Initialize the SEC filing analyzer."""
        self.environment = FinancialEnvironment()
        self.memory = FinancialMemory()

        # Initialize agents with capabilities
        self.financial_analyst = FinancialAnalystAgent(capabilities=[SECAnalysisCapability()])
        self.risk_analyst = RiskAnalystAgent(capabilities=[SECAnalysisCapability()])
        self.qa_specialist = QASpecialistAgent(capabilities=[SECAnalysisCapability()])
        self.coordinator = FinancialDiligenceCoordinator(capabilities=[SECAnalysisCapability()])

    async def analyze_financials(
        self,
        ticker: str,
        filing_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze financial statements for a company.

        Args:
            ticker: Company ticker symbol
            filing_type: Type of filing to analyze
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary containing financial analysis results
        """
        return await self.financial_analyst.run(ticker, memory=self.memory)

    async def assess_risks(
        self,
        ticker: str,
        filing_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assess risks for a company.

        Args:
            ticker: Company ticker symbol
            filing_type: Type of filing to analyze
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary containing risk assessment results
        """
        return await self.risk_analyst.run(ticker, memory=self.memory)

    async def answer_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Answer a financial question.

        Args:
            question: The question to answer
            context: Optional context for the question

        Returns:
            Dictionary containing answer and supporting information
        """
        return await self.qa_specialist.run(question, memory=self.memory)

    async def perform_diligence(
        self,
        ticker: str,
        filing_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive financial diligence.

        Args:
            ticker: Company ticker symbol
            filing_type: Type of filing to analyze
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary containing comprehensive diligence results
        """
        return await self.coordinator.run(ticker, memory=self.memory)

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis memory.

        Returns:
            Dictionary containing memory summary
        """
        return self.memory.get_summary()
