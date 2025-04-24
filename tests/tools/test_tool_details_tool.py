"""
Unit tests for the ToolDetailsTool.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.tools.tool_details import ToolDetailsTool
from src.tools.registry import ToolRegistry
from src.errors import ParameterError


class TestToolDetailsTool:
    """Test suite for the ToolDetailsTool."""
    
    @pytest.fixture
    def tool(self):
        """Create a ToolDetailsTool instance for testing."""
        return ToolDetailsTool()
    
    def test_init(self, tool):
        """Test initializing the tool."""
        assert tool.name == "tool_details"
        assert "tool for getting detailed information about available tools" in tool.description.lower()
    
    @pytest.mark.asyncio
    async def test_execute_valid_tool_name(self, tool):
        """Test executing the tool with a valid tool name."""
        # Mock the ToolRegistry.get_tool_documentation method
        mock_doc = "TOOL: test_tool\nDESCRIPTION: A test tool\nPARAMETERS:\n  param1: A test parameter"
        with patch("src.tools.registry.ToolRegistry.get_tool_documentation", return_value=mock_doc):
            # Execute the tool
            result = await tool.execute(tool_name="test_tool")
            
            # Check that the result is correct
            assert result["tool_name"] == "test_tool"
            assert result["documentation"] == mock_doc
            assert "output_key" in result
            assert result["output_key"] == "tool_details"
    
    @pytest.mark.asyncio
    async def test_execute_all_tools(self, tool):
        """Test executing the tool without a tool name to get all tools."""
        # Mock the ToolRegistry.get_tool_documentation method
        mock_doc = "TOOL: test_tool1\nDESCRIPTION: A test tool\n\nTOOL: test_tool2\nDESCRIPTION: Another test tool"
        with patch("src.tools.registry.ToolRegistry.get_tool_documentation", return_value=mock_doc):
            # Execute the tool
            result = await tool.execute()
            
            # Check that the result is correct
            assert result["tool_name"] is None
            assert result["documentation"] == mock_doc
            assert "output_key" in result
            assert result["output_key"] == "tool_details"
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, tool):
        """Test executing the tool with a nonexistent tool name."""
        # Mock the ToolRegistry.get_tool_documentation method to return an error message
        mock_doc = "Tool 'nonexistent_tool' not found"
        with patch("src.tools.registry.ToolRegistry.get_tool_documentation", return_value=mock_doc):
            # Execute the tool
            result = await tool.execute(tool_name="nonexistent_tool")
            
            # Check that the result contains the error message
            assert result["tool_name"] == "nonexistent_tool"
            assert result["documentation"] == mock_doc
    
    @pytest.mark.asyncio
    async def test_execute_with_format(self, tool):
        """Test executing the tool with a specific format."""
        # Mock the ToolRegistry.get_tool_documentation method
        mock_doc = "# Test Tool\n\nA test tool\n\n## Parameters\n\n- param1: A test parameter"
        with patch("src.tools.registry.ToolRegistry.get_tool_documentation", return_value=mock_doc):
            # Execute the tool
            result = await tool.execute(tool_name="test_tool", format="markdown")
            
            # Check that the result is correct
            assert result["tool_name"] == "test_tool"
            assert result["documentation"] == mock_doc
            assert result["format"] == "markdown"
    
    def test_validate_args_valid(self, tool):
        """Test validating valid arguments."""
        # Validate arguments
        result = tool.validate_args(tool_name="test_tool")
        
        # Check that validation passed
        assert result
    
    def test_validate_args_no_tool_name(self, tool):
        """Test validating arguments without a tool name."""
        # Validate arguments without a tool name
        result = tool.validate_args()
        
        # Check that validation passed (tool_name is optional)
        assert result
    
    def test_validate_args_invalid_format(self, tool):
        """Test validating arguments with an invalid format."""
        # Validate arguments with an invalid format
        with pytest.raises(ParameterError):
            tool.validate_args(tool_name="test_tool", format="invalid_format")
