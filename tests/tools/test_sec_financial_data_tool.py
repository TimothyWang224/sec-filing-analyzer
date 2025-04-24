"""
Unit tests for the SECFinancialDataTool.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.tools.sec_financial_data import SECFinancialDataTool
from src.errors import QueryTypeUnsupported, ParameterError, StorageUnavailable, DataNotFound


class TestSECFinancialDataTool:
    """Test suite for the SECFinancialDataTool."""
    
    @pytest.fixture
    def mock_db_store(self):
        """Create a mock DuckDB store."""
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {"ticker": "AAPL", "metric_name": "Revenue", "value": 100000000000}
        ]
        return mock_store
    
    @pytest.fixture
    def tool(self, mock_db_store):
        """Create a SECFinancialDataTool instance for testing."""
        with patch("src.tools.sec_financial_data.OptimizedDuckDBStore", return_value=mock_db_store):
            tool = SECFinancialDataTool(db_path="test.duckdb")
            tool.db_store = mock_db_store
            return tool
    
    def test_init(self):
        """Test initializing the tool."""
        with patch("src.tools.sec_financial_data.OptimizedDuckDBStore"):
            tool = SECFinancialDataTool(db_path="test.duckdb")
            assert tool.name == "sec_financial_data"
            assert "tool for querying financial data from sec filings" in tool.description.lower()
    
    def test_init_with_db_error(self):
        """Test initializing the tool with a database error."""
        with patch("src.tools.sec_financial_data.OptimizedDuckDBStore", side_effect=Exception("DB error")):
            tool = SECFinancialDataTool(db_path="test.duckdb")
            assert tool.db_store is None
            assert tool.db_error is not None
            assert "DB error" in str(tool.db_error)
    
    @pytest.mark.asyncio
    async def test_execute_financial_facts(self, tool, mock_db_store):
        """Test executing the tool with financial_facts query type."""
        # Execute the tool
        result = await tool.execute(
            query_type="financial_facts",
            parameters={
                "ticker": "AAPL",
                "metrics": ["Revenue"],
                "start_date": "2022-01-01",
                "end_date": "2022-12-31"
            }
        )
        
        # Check that the result is correct
        assert result["query_type"] == "financial_facts"
        assert result["parameters"]["ticker"] == "AAPL"
        assert "Revenue" in result["parameters"]["metrics"]
        assert "results" in result
        assert "output_key" in result
        assert result["output_key"] == "sec_financial_data"
        
        # Check that the database was queried
        mock_db_store.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_company_info(self, tool, mock_db_store):
        """Test executing the tool with company_info query type."""
        # Mock the query_company_info method
        mock_company_info = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "cik": "0000320193"
        }
        with patch.object(tool, "_query_company_info", return_value=mock_company_info):
            # Execute the tool
            result = await tool.execute(
                query_type="company_info",
                parameters={
                    "ticker": "AAPL"
                }
            )
            
            # Check that the result is correct
            assert result["ticker"] == "AAPL"
            assert result["name"] == "Apple Inc."
            assert result["cik"] == "0000320193"
    
    @pytest.mark.asyncio
    async def test_execute_metrics(self, tool, mock_db_store):
        """Test executing the tool with metrics query type."""
        # Mock the query_metrics method
        mock_metrics = {
            "query_type": "metrics",
            "parameters": {"ticker": "AAPL"},
            "results": [
                {"metric_name": "Revenue", "description": "Total revenue"}
            ]
        }
        with patch.object(tool, "_query_metrics", return_value=mock_metrics):
            # Execute the tool
            result = await tool.execute(
                query_type="metrics",
                parameters={
                    "ticker": "AAPL"
                }
            )
            
            # Check that the result is correct
            assert result["query_type"] == "metrics"
            assert result["parameters"]["ticker"] == "AAPL"
            assert len(result["results"]) == 1
            assert result["results"][0]["metric_name"] == "Revenue"
    
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
        assert "sec_financial_data" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_execute_db_unavailable(self, tool):
        """Test executing the tool when the database is unavailable."""
        # Set db_store to None to simulate unavailable database
        tool.db_store = None
        tool.db_error = Exception("DB error")
        
        # Execute the tool
        with pytest.raises(StorageUnavailable) as excinfo:
            await tool.execute(
                query_type="financial_facts",
                parameters={
                    "ticker": "AAPL",
                    "metrics": ["Revenue"]
                }
            )
        
        # Check that the error message is correct
        assert "database is unavailable" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_execute_data_not_found(self, tool, mock_db_store):
        """Test executing the tool when data is not found."""
        # Mock the query method to return an empty list
        mock_db_store.query.return_value = []
        
        # Execute the tool
        with pytest.raises(DataNotFound) as excinfo:
            await tool.execute(
                query_type="financial_facts",
                parameters={
                    "ticker": "UNKNOWN",
                    "metrics": ["Revenue"]
                }
            )
        
        # Check that the error message is correct
        assert "No data found" in str(excinfo.value)
    
    def test_validate_args_valid(self, tool):
        """Test validating valid arguments."""
        # Validate arguments
        result = tool.validate_args(
            query_type="financial_facts",
            parameters={
                "ticker": "AAPL",
                "metrics": ["Revenue"],
                "start_date": "2022-01-01",
                "end_date": "2022-12-31"
            }
        )
        
        # Check that validation passed
        assert result
    
    def test_validate_args_invalid_query_type(self, tool):
        """Test validating arguments with an invalid query type."""
        # Validate arguments with an invalid query type
        with pytest.raises(QueryTypeUnsupported):
            tool.validate_args(
                query_type="invalid_query",
                parameters={
                    "ticker": "AAPL"
                }
            )
    
    def test_validate_args_missing_required_parameter(self, tool):
        """Test validating arguments with a missing required parameter."""
        # Validate arguments without a required parameter
        with pytest.raises(ParameterError):
            tool.validate_args(
                query_type="financial_facts",
                parameters={}  # Missing ticker
            )
