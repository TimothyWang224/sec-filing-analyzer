"""
Smoke test for the API module.
"""

# pytest is used as a test runner
from src.api import SECFilingAnalyzer

def test_api_initialization():
    """Test that the API can be initialized."""
    analyzer = SECFilingAnalyzer()
    assert analyzer is not None
    assert hasattr(analyzer, 'financial_analyst')
    assert hasattr(analyzer, 'risk_analyst')
    assert hasattr(analyzer, 'qa_specialist')
    assert hasattr(analyzer, 'coordinator')
