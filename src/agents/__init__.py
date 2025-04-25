from .base import Agent, Goal
from .coordinator import FinancialDiligenceCoordinator
from .core import AgentState, DynamicTermination, LLMToolCaller
from .financial_analyst import FinancialAnalystAgent
from .qa_specialist import QASpecialistAgent
from .registry import AgentRegistry
from .risk_analyst import RiskAnalystAgent

# Register all agents
AgentRegistry.register("financial_analyst", FinancialAnalystAgent)
AgentRegistry.register("risk_analyst", RiskAnalystAgent)
AgentRegistry.register("qa_specialist", QASpecialistAgent)
AgentRegistry.register("coordinator", FinancialDiligenceCoordinator)

__all__ = [
    "Agent",
    "Goal",
    "AgentRegistry",
    "FinancialAnalystAgent",
    "RiskAnalystAgent",
    "QASpecialistAgent",
    "FinancialDiligenceCoordinator",
    "AgentState",
    "DynamicTermination",
    "LLMToolCaller",
]
