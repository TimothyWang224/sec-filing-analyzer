"""
Smoke test for the capabilities module.
"""

# pytest is used as a test runner
from src.capabilities import SECAnalysisCapability

def test_sec_analysis_capability():
    """Test that the SEC analysis capability can be initialized."""
    capability = SECAnalysisCapability()
    assert capability is not None
    assert hasattr(capability, 'name')
