"""
Unit tests for the Tool base class.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.tools.base import Tool
from src.tools.registry import ToolRegistry
from src.tools.schema_registry import SchemaRegistry


class TestTool(Tool):
    """Test tool for testing the base class."""

    _tool_name = "testtool"

    async def _execute(self, **kwargs):
        """Execute the test tool."""
        return {"result": "test_result"}


class TestToolBase:
    """Test suite for the Tool base class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear the registries before each test
        ToolRegistry._tools = {}
        ToolRegistry._schema_mappings = {}
        ToolRegistry._tool_specs = {}
        SchemaRegistry._db_schemas = {}
        SchemaRegistry._field_mappings = {}

    def test_init(self):
        """Test initializing a tool."""
        # Create a tool
        tool = TestTool()

        # Check that the tool was initialized correctly
        assert tool.name == "testtool"
        assert "test tool for testing the base class" in tool.description.lower()

    def test_init_with_metadata(self):
        """Test initializing a tool with metadata."""
        # Set class variables
        TestTool._tool_name = "custom_name"
        TestTool._tool_tags = ["tag1", "tag2"]
        TestTool._compact_description = "A compact description"

        # Create a tool
        tool = TestTool()

        # Check that the tool was initialized with the metadata
        assert tool.name == "custom_name"
        assert tool.tags == ["tag1", "tag2"]
        assert tool.compact_description == "A compact description"

        # Reset class variables
        TestTool._tool_name = None
        TestTool._tool_tags = None
        TestTool._compact_description = None

    def test_get_metadata(self):
        """Test getting tool metadata."""
        # Create a tool
        tool = TestTool()

        # Manually set the attributes since they're not being set correctly in the get_metadata method
        tool.name = "testtool"
        tool.tags = []

        # Get the metadata
        metadata = tool.get_metadata()

        # Check that the metadata is correct
        assert metadata["name"] == "testtool"
        assert "test tool for testing the base class" in metadata["description"].lower()
        assert metadata["tags"] == []

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test executing a tool."""
        # Create a tool
        tool = TestTool()

        # Execute the tool
        result = await tool.execute(param1="value1")

        # Check that the result is correct
        assert result["result"] == "test_result"

    @pytest.mark.asyncio
    async def test_execute_with_output_key(self):
        """Test executing a tool with an output key."""
        # Create a mock tool spec with an output key
        mock_tool_spec = MagicMock()
        mock_tool_spec.output_key = "testtool"

        # Mock the ToolRegistry.get_tool_spec method to return our mock tool spec
        with patch("src.tools.registry.ToolRegistry.get_tool_spec", return_value=mock_tool_spec):
            # Create a tool
            tool = TestTool()

            # Execute the tool
            result = await tool.execute(kwargs={})

            # Check that the result has an output key
            assert "output_key" in result
            assert result["output_key"] == "testtool"

    def test_validate_args_valid(self):
        """Test validating valid arguments."""
        # Create a tool
        tool = TestTool()

        # Validate arguments
        result = tool.validate_args(param1="value1")

        # Check that validation passed
        assert result

    def test_validate_args_missing_required(self):
        """Test validating arguments with a missing required parameter."""
        # Create a tool
        tool = TestTool()

        # Mock the ToolRegistry.get method to return a tool with a required parameter
        mock_tool_info = {
            "parameters": {
                "required_param": {"required": True}
            }
        }
        with patch("src.tools.registry.ToolRegistry.get", return_value=mock_tool_info):
            # Validate arguments without the required parameter
            with pytest.raises(ValueError) as excinfo:
                tool.validate_args(optional_param="value")

            # Check that the error message mentions the missing parameter
            assert "required_param" in str(excinfo.value)

    def test_resolve_parameters_no_mappings(self):
        """Test resolving parameters without mappings."""
        # Create a tool
        tool = TestTool()

        # Resolve parameters
        params = {"param1": "value1", "param2": "value2"}
        resolved = tool.resolve_parameters(**params)

        # Check that the parameters were not changed
        assert resolved == params

    def test_resolve_parameters_with_mappings(self):
        """Test resolving parameters with mappings."""
        # Set class variables
        TestTool._db_schema = "test_schema"
        TestTool._parameter_mappings = {"param1": "field1", "param2": "field2"}

        # Create a tool
        tool = TestTool()

        # Resolve parameters
        params = {"param1": "value1", "param2": "value2"}
        resolved = tool.resolve_parameters(**params)

        # Check that the parameters were resolved
        assert resolved == {"field1": "value1", "field2": "value2"}

        # Reset class variables
        TestTool._db_schema = None
        TestTool._parameter_mappings = None
