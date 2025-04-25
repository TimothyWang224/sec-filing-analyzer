"""
Unit tests for the ToolRegistry class.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.contracts import ToolSpec
from src.tools.base import Tool
from src.tools.registry import ToolRegistry


class TestTool(Tool):
    """Test tool for testing the registry."""

    async def _execute_abstract(self, **kwargs):
        """Execute the test tool."""
        return {"result": "test_result"}


class TestToolRegistry:
    """Test suite for the ToolRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear the registry before each test
        ToolRegistry._tools = {}
        ToolRegistry._schema_mappings = {}
        ToolRegistry._tool_specs = {}

    def test_register_tool(self):
        """Test registering a tool with the registry."""
        # Register a tool
        ToolRegistry._register_tool(TestTool, name="test_tool", tags=["test"])

        # Check that the tool was registered
        assert "test_tool" in ToolRegistry._tools
        assert ToolRegistry._tools["test_tool"]["class"] == TestTool
        assert ToolRegistry._tools["test_tool"]["tags"] == ["test"]

    def test_get_tool(self):
        """Test getting a tool from the registry."""
        # Register a tool
        ToolRegistry._register_tool(TestTool, name="test_tool", tags=["test"])

        # Get the tool
        tool = ToolRegistry.get("test_tool")

        # Check that the tool was retrieved
        assert tool is not None
        assert tool["class"] == TestTool
        assert tool["tags"] == ["test"]

    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        # Get a nonexistent tool
        tool = ToolRegistry.get("nonexistent_tool")

        # Check that None was returned
        assert tool is None

    def test_list_tools(self):
        """Test listing all tools in the registry."""
        # Register two tools
        ToolRegistry._register_tool(TestTool, name="test_tool1", tags=["test"])
        ToolRegistry._register_tool(TestTool, name="test_tool2", tags=["test"])

        # List all tools
        tools = ToolRegistry.list_tools()

        # Check that both tools are in the list
        assert "test_tool1" in tools
        assert "test_tool2" in tools

    def test_get_tool_spec(self):
        """Test getting a tool specification."""
        # Register a tool
        ToolRegistry._register_tool(TestTool, name="test_tool", tags=["test"])

        # Get the tool spec
        tool_spec = ToolRegistry.get_tool_spec("test_tool")

        # Check that the tool spec was retrieved
        assert tool_spec is not None
        assert tool_spec.name == "test_tool"

    def test_get_schema_mappings(self):
        """Test getting schema mappings for a tool."""
        # Register a tool with schema mappings
        TestTool._db_schema = "test_schema"
        TestTool._parameter_mappings = {"param1": "field1", "param2": "field2"}
        ToolRegistry._register_tool(TestTool, name="test_tool", tags=["test"])

        # Get the schema mappings
        mappings = ToolRegistry.get_schema_mappings("test_tool")

        # Check that the mappings were retrieved
        assert mappings == {"param1": "field1", "param2": "field2"}

    def test_validate_schema_mappings_valid(self):
        """Test validating valid schema mappings."""
        # Register a tool with schema mappings
        TestTool._db_schema = "test_schema"
        TestTool._parameter_mappings = {"param1": "field1", "param2": "field2"}
        ToolRegistry._register_tool(TestTool, name="test_tool", tags=["test"])

        # Mock the SchemaRegistry.get_field_info method
        with patch("src.tools.registry.SchemaRegistry.get_field_info", return_value={"type": "string"}):
            # Validate the schema mappings
            is_valid, errors = ToolRegistry.validate_schema_mappings("test_tool")

            # Check that the mappings are valid
            assert is_valid
            assert not errors

    def test_validate_schema_mappings_invalid(self):
        """Test validating invalid schema mappings."""
        # Register a tool with schema mappings
        TestTool._db_schema = "test_schema"
        TestTool._parameter_mappings = {"param1": "field1", "param2": "field2"}
        ToolRegistry._register_tool(TestTool, name="test_tool", tags=["test"])

        # Mock the SchemaRegistry.get_field_info method to return None for field2
        def mock_get_field_info(schema_name, field_name):
            if field_name == "field1":
                return {"type": "string"}
            return None

        with patch("src.tools.registry.SchemaRegistry.get_field_info", side_effect=mock_get_field_info):
            # Validate the schema mappings
            is_valid, errors = ToolRegistry.validate_schema_mappings("test_tool")

            # Check that the mappings are invalid
            assert not is_valid
            assert len(errors) == 1
            assert "field2" in errors[0]

    def test_validate_all_schema_mappings(self):
        """Test validating all schema mappings."""
        # Register two tools with schema mappings
        TestTool._db_schema = "test_schema"
        TestTool._parameter_mappings = {"param1": "field1", "param2": "field2"}
        ToolRegistry._register_tool(TestTool, name="test_tool1", tags=["test"])
        ToolRegistry._register_tool(TestTool, name="test_tool2", tags=["test"])

        # Mock the SchemaRegistry.get_field_info method
        with patch("src.tools.registry.SchemaRegistry.get_field_info", return_value={"type": "string"}):
            # Validate all schema mappings
            is_valid, all_errors = ToolRegistry.validate_all_schema_mappings()

            # Check that all mappings are valid
            assert is_valid
            assert not all_errors

    def test_get_tool_documentation(self):
        """Test getting tool documentation."""
        # Register a tool
        ToolRegistry._register_tool(TestTool, name="test_tool", tags=["test"])

        # Get the tool documentation
        doc = ToolRegistry.get_tool_documentation("test_tool")

        # Check that the documentation was retrieved
        assert "test_tool" in doc.lower()
        assert "test tool for testing the registry" in doc.lower()

    def test_get_compact_tool_documentation(self):
        """Test getting compact tool documentation."""
        # Register a tool with a compact description
        TestTool._compact_description = "A compact description"
        ToolRegistry._register_tool(TestTool, name="test_tool", tags=["test"])

        # Get the compact tool documentation
        doc = ToolRegistry.get_compact_tool_documentation()

        # Check that the documentation was retrieved
        assert "test_tool" in doc.lower()
        assert "a compact description" in doc.lower()
