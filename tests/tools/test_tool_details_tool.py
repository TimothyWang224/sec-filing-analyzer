"""
Unit tests for the ToolDetailsTool.
"""

from unittest.mock import patch

import pytest

from src.tools.tool_details import SUPPORTED_QUERIES, ToolDetailsTool


class TestToolDetailsTool:
    """Test suite for the ToolDetailsTool."""

    @pytest.fixture
    def tool(self):
        """Create a ToolDetailsTool instance for testing."""
        return ToolDetailsTool()

    def test_init(self, tool):
        """Test initializing the tool."""
        assert tool.name == "tool_details"
        assert (
            "tool for getting detailed information about other tools"
            in tool.description.lower()
        )

    @pytest.mark.asyncio
    async def test_execute_valid_tool_name(self, tool):
        """Test executing the tool with a valid tool name."""
        # Mock the ToolRegistry.get and get_tool_documentation methods
        mock_tool_info = {
            "description": "A test tool",
            "parameters": {"param1": {"description": "A test parameter"}},
        }
        mock_doc = "TOOL: test_tool\nDESCRIPTION: A test tool\nPARAMETERS:\n  param1: A test parameter"

        with (
            patch("src.tools.registry.ToolRegistry.get", return_value=mock_tool_info),
            patch(
                "src.tools.registry.ToolRegistry.get_tool_documentation",
                return_value=mock_doc,
            ),
        ):
            # Execute the tool
            result = await tool.execute(
                query_type="tool_details", parameters={"tool_name": "test_tool"}
            )

            # Check that the result is correct
            assert result["success"] is True
            assert result["tool_name"] == "test_tool"
            assert result["description"] == "A test tool"
            assert "parameters" in result
            assert result["documentation"] == mock_doc
            assert "output_key" in result
            assert result["output_key"] == "tool_details"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, tool):
        """Test executing the tool with a nonexistent tool name."""
        # Mock the ToolRegistry.get method to return None
        with (
            patch("src.tools.registry.ToolRegistry.get", return_value=None),
            patch(
                "src.tools.registry.ToolRegistry.list_tools",
                return_value={"tool1": {}, "tool2": {}},
            ),
        ):
            # Execute the tool
            result = await tool.execute(
                query_type="tool_details", parameters={"tool_name": "nonexistent_tool"}
            )

            # Check that the result contains the error message
            assert result["success"] is False
            assert "error" in result
            assert "nonexistent_tool" in result["error"]
            assert "available_tools" in result
            assert len(result["available_tools"]) == 2

    @pytest.mark.asyncio
    async def test_execute_invalid_query_type(self, tool):
        """Test executing the tool with an invalid query type."""
        # Execute the tool with an invalid query type
        result = await tool.execute(
            query_type="invalid_query", parameters={"tool_name": "test_tool"}
        )

        # Check that the result contains an error message
        assert result["query_type"] == "invalid_query"
        assert result["success"] is False
        assert "error" in result
        assert "Unsupported query type" in result["error"]

    def test_validate_args_valid(self, tool):
        """Test validating valid arguments."""
        # Validate arguments
        result = tool.validate_args(
            query_type="tool_details", parameters={"tool_name": "test_tool"}
        )

        # Check that validation passed
        assert result is True

    def test_validate_args_invalid_query_type(self, tool):
        """Test validating arguments with an invalid query type."""
        # Validate arguments with an invalid query type
        result = tool.validate_args(
            query_type="invalid_query", parameters={"tool_name": "test_tool"}
        )

        # Check that validation failed
        assert result is False

    def test_validate_args_missing_parameters(self, tool):
        """Test validating arguments with missing parameters."""
        # Validate arguments with missing parameters
        result = tool.validate_args(query_type="tool_details", parameters=None)

        # Check that validation failed
        assert result is False

    def test_validate_args_invalid_tool_name(self, tool):
        """Test validating arguments with an invalid tool name."""
        # Validate arguments with an invalid tool name
        result = tool.validate_args(
            query_type="tool_details", parameters={"tool_name": ""}
        )

        # Check that validation failed
        assert result is False

    def test_supported_queries(self):
        """Test that the SUPPORTED_QUERIES dictionary is correctly defined."""
        assert "tool_details" in SUPPORTED_QUERIES
        assert SUPPORTED_QUERIES["tool_details"].__name__ == "ToolDetailsParams"
