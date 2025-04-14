from .base import Agent, Goal
from .financial_analyst import FinancialAnalystAgent
from .risk_analyst import RiskAnalystAgent
from .qa_specialist import QASpecialistAgent
from .coordinator import FinancialDiligenceCoordinator
from .registry import AgentRegistry
from .core import AgentState, DynamicTermination, LLMToolCaller

# Register all agents
AgentRegistry.register("financial_analyst", FinancialAnalystAgent)
AgentRegistry.register("risk_analyst", RiskAnalystAgent)
AgentRegistry.register("qa_specialist", QASpecialistAgent)
AgentRegistry.register("coordinator", FinancialDiligenceCoordinator)

__all__ = [
    'Agent',
    'Goal',
    'AgentRegistry',
    'FinancialAnalystAgent',
    'RiskAnalystAgent',
    'QASpecialistAgent',
    'FinancialDiligenceCoordinator',
    'AgentState',
    'DynamicTermination',
    'LLMToolCaller'
]