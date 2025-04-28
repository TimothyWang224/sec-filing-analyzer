"""
Tests for the graphrag module initialization.
"""

from sec_filing_analyzer.graphrag import SECEntities, SECStructure


def test_imports():
    """Test that the module exports the expected classes."""
    # Verify that the expected classes are exported
    assert SECStructure is not None
    assert SECEntities is not None

    # Verify that the classes are the expected types
    assert isinstance(SECStructure(), SECStructure)
    assert isinstance(SECEntities(), SECEntities)
