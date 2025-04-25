"""
Smoke test for the tools module.
"""

# pytest is used as a test runner
from src.tools import (
    ToolRegistry,
    SchemaRegistry
)

def test_tool_registry():
    """Test that the tool registry is initialized."""
    assert ToolRegistry is not None

    # Test that we can get the list of tools
    tools = ToolRegistry.list_tools()
    assert isinstance(tools, dict)

def test_schema_registry():
    """Test that the schema registry is initialized."""
    assert SchemaRegistry is not None
