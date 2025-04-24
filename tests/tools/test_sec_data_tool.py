"""
Unit tests for the SECDataTool.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.tools.sec_data import SECDataTool
from src.errors import QueryTypeUnsupported, ParameterError


class TestSECDataTool:
    """Test suite for the SECDataTool."""
    
    @pytest.fixture
    def tool(self):
        """Create a SECDataTool instance for testing."""
        return SECDataTool()
    
    def test_init(self, tool):
        """Test initializing the tool."""
        assert tool.name == "sec_data"
        assert "tool for retrieving and processing sec filing data" in tool.description.lower()
    
    @pytest.mark.asyncio
    async def test_execute_valid_parameters(self, tool):
        """Test executing the tool with valid parameters."""
        # Execute the tool
        result = await tool.execute(
            query_type="sec_data",
            parameters={
                "ticker": "AAPL",
                "filing_type": "10-K",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "sections": ["Financial Statements"]
            }
        )
        
        # Check that the result is correct
        assert result["ticker"] == "AAPL"
        assert result["filing_type"] == "10-K"
        assert result["time_period"]["start"] == "2023-01-01"
        assert result["time_period"]["end"] == "2023-12-31"
        assert "Financial Statements" in result["sections"]
        assert "data" in result
        assert "output_key" in result
        assert result["output_key"] == "sec_data"
    
    @pytest.mark.asyncio
    async def test_execute_minimal_parameters(self, tool):
        """Test executing the tool with minimal parameters."""
        # Execute the tool
        result = await tool.execute(
            query_type="sec_data",
            parameters={
                "ticker": "AAPL"
            }
        )
        
        # Check that the result is correct
        assert result["ticker"] == "AAPL"
        assert result["filing_type"] == "10-K"  # Default
        assert "data" in result
    
    @pytest.mark.asyncio
    async def test_execute_invalid_query_type(self, tool):
        """Test executing the tool with an invalid query type."""
        # Execute the tool with an invalid query type
        with pytest.raises(QueryTypeUnsupported) as excinfo:
            await tool.execute(
                query_type="invalid_query",
                parameters={
                    "ticker": "AAPL"
                }
            )
        
        # Check that the error message is correct
        assert "invalid_query" in str(excinfo.value)
        assert "sec_data" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_execute_missing_ticker(self, tool):
        """Test executing the tool without a ticker."""
        # Execute the tool without a ticker
        with pytest.raises(ParameterError) as excinfo:
            await tool.execute(
                query_type="sec_data",
                parameters={}
            )
        
        # Check that the error message is correct
        assert "ticker" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_execute_invalid_filing_type(self, tool):
        """Test executing the tool with an invalid filing type."""
        # Execute the tool with an invalid filing type
        with pytest.raises(ParameterError) as excinfo:
            await tool.execute(
                query_type="sec_data",
                parameters={
                    "ticker": "AAPL",
                    "filing_type": "invalid"
                }
            )
        
        # Check that the error message is correct
        assert "filing_type" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_execute_invalid_date_format(self, tool):
        """Test executing the tool with an invalid date format."""
        # Execute the tool with an invalid date format
        with pytest.raises(ParameterError) as excinfo:
            await tool.execute(
                query_type="sec_data",
                parameters={
                    "ticker": "AAPL",
                    "start_date": "01/01/2023"  # Wrong format
                }
            )
        
        # Check that the error message is correct
        assert "date" in str(excinfo.value).lower()
    
    def test_validate_args_valid(self, tool):
        """Test validating valid arguments."""
        # Validate arguments
        result = tool.validate_args(
            query_type="sec_data",
            parameters={
                "ticker": "AAPL",
                "filing_type": "10-K",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "sections": ["Financial Statements"]
            }
        )
        
        # Check that validation passed
        assert result
    
    def test_validate_args_invalid_query_type(self, tool):
        """Test validating arguments with an invalid query type."""
        # Validate arguments with an invalid query type
        result = tool.validate_args(
            query_type="invalid_query",
            parameters={
                "ticker": "AAPL"
            }
        )
        
        # Check that validation failed
        assert not result
