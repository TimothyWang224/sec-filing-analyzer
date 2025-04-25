"""
Smoke test for the agents module.
"""

# pytest is used as a test runner
from src.agents import (
    FinancialAnalystAgent,
    RiskAnalystAgent,
    QASpecialistAgent,
    FinancialDiligenceCoordinator,
    AgentRegistry
)

def test_agent_registry():
    """Test that the agent registry contains the expected agents."""
    agents = AgentRegistry.list_agents()
    assert "financial_analyst" in agents
    assert "risk_analyst" in agents
    assert "qa_specialist" in agents
    assert "coordinator" in agents

    # Test that we can get agents by name
    assert AgentRegistry.get("financial_analyst") == FinancialAnalystAgent
    assert AgentRegistry.get("risk_analyst") == RiskAnalystAgent
    assert AgentRegistry.get("qa_specialist") == QASpecialistAgent
    assert AgentRegistry.get("coordinator") == FinancialDiligenceCoordinator
