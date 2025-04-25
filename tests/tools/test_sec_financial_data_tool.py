"""
Unit tests for the SECFinancialDataTool.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.errors import DataNotFound, ParameterError, QueryTypeUnsupported, StorageUnavailable
from src.tools.sec_financial_data import SUPPORTED_QUERIES, SECFinancialDataTool


class TestSECFinancialDataTool:
    """Test suite for the SECFinancialDataTool."""

    @pytest.fixture
    def mock_db_store(self):
        """Create a mock DuckDB store."""
        mock_store = MagicMock()
        # Add methods that the tool will call
        mock_store.get_company_info = MagicMock(return_value={"ticker": "AAPL", "name": "Apple Inc."})
        mock_store.get_all_companies = MagicMock(return_value=[{"ticker": "AAPL"}, {"ticker": "MSFT"}])
        mock_store.get_available_metrics = MagicMock(return_value=[{"metric_name": "Revenue"}])
        mock_store.query_time_series = MagicMock(return_value=[{"ticker": "AAPL", "metric": "Revenue", "value": 100}])
        mock_store.query_financial_ratios = MagicMock(return_value=[{"ticker": "AAPL", "ratio": "PE", "value": 20}])
        mock_store.execute_custom_query = MagicMock(return_value=[{"result": "test"}])
        # Add a method to handle the missing query_financial_facts
        mock_store.query = MagicMock(return_value=[{"ticker": "AAPL", "metric_name": "Revenue", "value": 100000000000}])
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
        # Patch the _query_financial_facts method to return a known result
        mock_result = {
            "query_type": "financial_facts",
            "parameters": {
                "ticker": "AAPL",
                "metrics": ["Revenue"],
                "start_date": "2022-01-01",
                "end_date": "2022-12-31",
            },
            "results": [{"ticker": "AAPL", "metric_name": "Revenue", "value": 100000000000}],
            "output_key": "sec_financial_data",
        }
        with patch.object(tool, "_query_financial_facts", return_value=mock_result):
            # Execute the tool
            result = await tool.execute(
                query_type="financial_facts",
                parameters={
                    "ticker": "AAPL",
                    "metrics": ["Revenue"],
                    "start_date": "2022-01-01",
                    "end_date": "2022-12-31",
                },
            )

            # Check that the result is correct
            assert result["query_type"] == "financial_facts"
            assert result["parameters"]["ticker"] == "AAPL"
            assert "Revenue" in result["parameters"]["metrics"]
            assert "results" in result
            assert len(result["results"]) == 1
            assert result["results"][0]["ticker"] == "AAPL"
            assert result["results"][0]["metric_name"] == "Revenue"
            assert "output_key" in result
            assert result["output_key"] == "sec_financial_data"

    @pytest.mark.asyncio
    async def test_execute_company_info(self, tool, mock_db_store):
        """Test executing the tool with company_info query type."""
        # Execute the tool
        result = await tool.execute(query_type="company_info", parameters={"ticker": "AAPL"})

        # Check that the result is correct
        assert result["query_type"] == "company_info"
        assert result["parameters"]["ticker"] == "AAPL"
        assert "results" in result

        # Check that the database method was called
        mock_db_store.get_company_info.assert_called_once_with(ticker="AAPL")

    @pytest.mark.asyncio
    async def test_execute_companies(self, tool, mock_db_store):
        """Test executing the tool with companies query type."""
        # Execute the tool
        result = await tool.execute(query_type="companies", parameters={})

        # Check that the result is correct
        assert result["query_type"] == "company_info"  # It's an alias for company_info
        assert "results" in result

        # Check that the database method was called
        mock_db_store.get_all_companies.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_metrics(self, tool, mock_db_store):
        """Test executing the tool with metrics query type."""
        # Mock the _query_metrics method to return a known result
        mock_result = {
            "query_type": "metrics",
            "parameters": {"ticker": "AAPL", "category": "Income Statement"},
            "results": [{"metric_name": "Revenue", "description": "Total revenue"}],
            "output_key": "sec_financial_data",
        }
        with patch.object(tool, "_query_metrics", return_value=mock_result):
            # Execute the tool
            result = await tool.execute(
                query_type="metrics",
                parameters={
                    "ticker": "AAPL",  # Required parameter
                    "category": "Income Statement",
                },
            )

            # Check that the result is correct
            assert result["query_type"] == "metrics"
            assert result["parameters"]["ticker"] == "AAPL"
            assert result["parameters"]["category"] == "Income Statement"
            assert "results" in result
            assert "output_key" in result
            assert result["output_key"] == "sec_financial_data"

    @pytest.mark.asyncio
    async def test_execute_time_series(self, tool, mock_db_store):
        """Test executing the tool with time_series query type."""
        # Execute the tool
        result = await tool.execute(
            query_type="time_series",
            parameters={"ticker": "AAPL", "metric": "Revenue", "start_date": "2022-01-01", "end_date": "2022-12-31"},
        )

        # Check that the result is correct
        assert result["query_type"] == "time_series"
        assert result["parameters"]["ticker"] == "AAPL"
        assert result["parameters"]["metric"] == "Revenue"
        assert "results" in result

        # Check that the database method was called
        mock_db_store.query_time_series.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_financial_ratios(self, tool, mock_db_store):
        """Test executing the tool with financial_ratios query type."""
        # Execute the tool
        result = await tool.execute(
            query_type="financial_ratios",
            parameters={"ticker": "AAPL", "ratios": ["PE"], "start_date": "2022-01-01", "end_date": "2022-12-31"},
        )

        # Check that the result is correct
        assert result["query_type"] == "financial_ratios"
        assert result["parameters"]["ticker"] == "AAPL"
        assert result["parameters"]["ratios"] == ["PE"]
        assert "results" in result

        # Check that the database method was called
        mock_db_store.query_financial_ratios.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_custom_sql(self, tool, mock_db_store):
        """Test executing the tool with custom_sql query type."""
        # Execute the tool
        result = await tool.execute(
            query_type="custom_sql", parameters={"sql_query": "SELECT * FROM companies WHERE ticker = 'AAPL'"}
        )

        # Check that the result is correct
        assert result["query_type"] == "custom_sql"
        assert "sql_query" in result["parameters"]
        assert "results" in result

        # Check that the database method was called
        mock_db_store.execute_custom_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_invalid_query_type(self, tool):
        """Test executing the tool with an invalid query type."""
        # Execute the tool with an invalid query type
        with pytest.raises(QueryTypeUnsupported) as excinfo:
            await tool.execute(query_type="invalid_query", parameters={"ticker": "AAPL"})

        # Check that the error message is correct
        assert "invalid_query" in str(excinfo.value)
        assert "sec_financial_data" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_execute_db_unavailable(self, tool):
        """Test executing the tool when the database is unavailable."""
        # Set db_store to None to simulate unavailable database
        tool.db_store = None
        tool.db_error = "DB error"

        # Execute the tool - this should return an error message, not raise an exception
        result = await tool.execute(
            query_type="financial_facts",
            parameters={
                "ticker": "AAPL",
                "metrics": ["Revenue"],
                "start_date": "2022-01-01",  # Required parameter
                "end_date": "2022-12-31",  # Required parameter
            },
        )

        # Check that the result contains an error message
        assert "error" in result
        assert "Database connection failed" in result["error"]
        assert "DB error" in result["error"]
        assert result["results"] == []

    def test_validate_args_valid(self, tool):
        """Test validating valid arguments."""
        # Validate arguments
        result = tool.validate_args(
            query_type="financial_facts",
            parameters={"ticker": "AAPL", "metrics": ["Revenue"], "start_date": "2022-01-01", "end_date": "2022-12-31"},
        )

        # Check that validation passed
        assert result is True

    def test_validate_args_invalid_query_type(self, tool):
        """Test validating arguments with an invalid query type."""
        # Validate arguments with an invalid query type
        result = tool.validate_args(query_type="invalid_query", parameters={"ticker": "AAPL"})

        # Check that validation failed
        assert result is False

    def test_validate_args_missing_required_parameter(self, tool):
        """Test validating arguments with a missing required parameter."""
        # Validate arguments without a required parameter
        result = tool.validate_args(
            query_type="financial_facts",
            parameters={},  # Missing ticker
        )

        # Check that validation failed
        assert result is False

    def test_supported_queries(self):
        """Test that the SUPPORTED_QUERIES dictionary is correctly defined."""
        assert "financial_facts" in SUPPORTED_QUERIES
        assert "company_info" in SUPPORTED_QUERIES
        assert "metrics" in SUPPORTED_QUERIES
        assert "time_series" in SUPPORTED_QUERIES
        assert "financial_ratios" in SUPPORTED_QUERIES
        assert "custom_sql" in SUPPORTED_QUERIES
