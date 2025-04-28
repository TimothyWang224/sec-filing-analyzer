"""
Unit tests for the SECGraphQueryTool.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.tools.sec_graph_query import SUPPORTED_QUERIES, SECGraphQueryTool


class TestSECGraphQueryTool:
    """Test suite for the SECGraphQueryTool."""

    @pytest.fixture
    def mock_graph_store(self):
        """Create a mock graph store."""
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "filing_type": "10-K",
                "filing_date": "2022-10-28",
                "accession_number": "0000320193-22-000108",
                "fiscal_year": "2022",
                "fiscal_period": "FY",
            }
        ]
        return mock_store

    @pytest.fixture
    def tool(self, mock_graph_store):
        """Create a SECGraphQueryTool instance for testing."""
        with patch(
            "src.tools.sec_graph_query.GraphStore", return_value=mock_graph_store
        ):
            tool = SECGraphQueryTool(use_neo4j=False)
            tool.graph_store = mock_graph_store
            return tool

    def test_init(self):
        """Test initializing the tool."""
        with patch("src.tools.sec_graph_query.GraphStore"):
            tool = SECGraphQueryTool(use_neo4j=False)
            assert tool.name == "sec_graph_query"
            assert (
                "tool for querying the sec filing graph database"
                in tool.description.lower()
            )

    @pytest.mark.asyncio
    async def test_execute_company_filings(self, tool, mock_graph_store):
        """Test executing the tool with company_filings query type."""
        # Execute the tool
        result = await tool.execute(
            query_type="company_filings",
            parameters={"ticker": "AAPL", "filing_types": ["10-K"], "limit": 5},
        )

        # Check that the result is correct
        assert result["query_type"] == "company_filings"
        assert result["parameters"]["ticker"] == "AAPL"
        assert "10-K" in result["parameters"]["filing_types"]
        assert result["parameters"]["limit"] == 5
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["filing_type"] == "10-K"
        assert "output_key" in result
        assert result["output_key"] == "sec_graph_query"

        # Check that the graph database was queried
        mock_graph_store.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_filing_sections(self, tool, mock_graph_store):
        """Test executing the tool with filing_sections query type."""
        # Set up mock return value for filing sections
        mock_graph_store.query.return_value = [
            {"title": "Risk Factors", "section_type": "risk_factors", "order": 1}
        ]

        # Execute the tool
        result = await tool.execute(
            query_type="filing_sections",
            parameters={
                "accession_number": "0000320193-22-000108",
                "section_types": ["risk_factors"],
            },
        )

        # Check that the result is correct
        assert result["query_type"] == "filing_sections"
        assert result["parameters"]["accession_number"] == "0000320193-22-000108"
        assert "section_types" in result["parameters"]
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Risk Factors"
        assert "output_key" in result
        assert result["output_key"] == "sec_graph_query"

    @pytest.mark.asyncio
    async def test_execute_related_companies(self, tool, mock_graph_store):
        """Test executing the tool with related_companies query type."""
        # Set up mock return value for related companies
        mock_graph_store.query.return_value = [
            {"ticker": "MSFT", "name": "Microsoft Corporation", "mention_count": 5}
        ]

        # Execute the tool
        result = await tool.execute(
            query_type="related_companies",
            parameters={"ticker": "AAPL", "relationship_type": "MENTIONS", "limit": 10},
        )

        # Check that the result is correct
        assert result["query_type"] == "related_companies"
        assert result["parameters"]["ticker"] == "AAPL"
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["ticker"] == "MSFT"
        assert "output_key" in result
        assert result["output_key"] == "sec_graph_query"

    @pytest.mark.asyncio
    async def test_execute_filing_timeline(self, tool, mock_graph_store):
        """Test executing the tool with filing_timeline query type."""
        # Set up mock return value for filing timeline
        mock_graph_store.query.return_value = [
            {
                "filing_date": "2022-10-28",
                "accession_number": "0000320193-22-000108",
                "fiscal_year": "2022",
                "fiscal_period": "FY",
            },
            {
                "filing_date": "2021-10-29",
                "accession_number": "0000320193-21-000105",
                "fiscal_year": "2021",
                "fiscal_period": "FY",
            },
        ]

        # Execute the tool
        result = await tool.execute(
            query_type="filing_timeline",
            parameters={"ticker": "AAPL", "filing_type": "10-K", "limit": 10},
        )

        # Check that the result is correct
        assert result["query_type"] == "filing_timeline"
        assert result["parameters"]["ticker"] == "AAPL"
        assert result["parameters"]["filing_type"] == "10-K"
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["filing_date"] == "2022-10-28"
        assert "output_key" in result
        assert result["output_key"] == "sec_graph_query"

    @pytest.mark.asyncio
    async def test_execute_section_types(self, tool, mock_graph_store):
        """Test executing the tool with section_types query type."""
        # Set up mock return value for section types
        mock_graph_store.query.return_value = [
            {"section_type": "risk_factors", "count": 100},
            {"section_type": "md_and_a", "count": 95},
        ]

        # Execute the tool
        result = await tool.execute(query_type="section_types", parameters={})

        # Check that the result is correct
        assert result["query_type"] == "section_types"
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["section_type"] == "risk_factors"
        assert "output_key" in result
        assert result["output_key"] == "sec_graph_query"

    @pytest.mark.asyncio
    async def test_execute_custom_cypher(self, tool, mock_graph_store):
        """Test executing the tool with custom_cypher query type."""
        # Set up mock return value for custom cypher
        mock_graph_store.query.return_value = [{"ticker": "AAPL", "count": 5}]

        # Execute the tool
        result = await tool.execute(
            query_type="custom_cypher",
            parameters={
                "cypher_query": "MATCH (c:Company) RETURN c.ticker as ticker, count(*) as count",
                "query_params": {},
            },
        )

        # Check that the result is correct
        assert result["query_type"] == "custom_cypher"
        assert "cypher_query" in result["parameters"]
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["ticker"] == "AAPL"
        assert "output_key" in result
        assert result["output_key"] == "sec_graph_query"

    @pytest.mark.asyncio
    async def test_execute_invalid_query_type(self, tool):
        """Test executing the tool with an invalid query type."""
        # Execute the tool with an invalid query type
        result = await tool.execute(
            query_type="invalid_query", parameters={"ticker": "AAPL"}
        )

        # Check that the result contains an error message
        assert result["query_type"] == "invalid_query"
        assert "error" in result
        assert "Unsupported query type" in result["error"]
        assert "invalid_query" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_graph_store_none(self, tool):
        """Test executing the tool when the graph store is None."""
        # Set graph_store to None to simulate unavailable database
        tool.graph_store = None

        # Execute the tool
        result = await tool.execute(
            query_type="company_filings", parameters={"ticker": "AAPL"}
        )

        # Check that the result contains an error message
        assert result["query_type"] == "company_filings"
        assert "error" in result
        assert "Graph store is not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_data_not_found(self, tool, mock_graph_store):
        """Test executing the tool when data is not found."""
        # Mock the query method to return an empty list
        mock_graph_store.query.return_value = []

        # Execute the tool - this should return a result with empty results list
        result = await tool.execute(
            query_type="company_filings", parameters={"ticker": "UNKNOWN"}
        )

        # Check that the result contains an empty results list
        assert result["query_type"] == "company_filings"
        assert result["parameters"]["ticker"] == "UNKNOWN"
        assert "results" in result
        assert len(result["results"]) == 0  # Empty results list
        assert "output_key" in result
        assert result["output_key"] == "sec_graph_query"

    def test_validate_args_valid(self, tool):
        """Test validating valid arguments."""
        # Validate arguments
        result = tool.validate_args(
            query_type="company_filings",
            parameters={"ticker": "AAPL", "filing_types": ["10-K"], "limit": 5},
        )

        # Check that validation passed
        assert result is True

    def test_validate_args_invalid_query_type(self, tool):
        """Test validating arguments with an invalid query type."""
        # Validate arguments with an invalid query type
        result = tool.validate_args(
            query_type="invalid_query", parameters={"ticker": "AAPL"}
        )

        # Check that validation failed
        assert result is False

    def test_validate_args_missing_required_parameter(self, tool):
        """Test validating arguments with a missing required parameter."""
        # Validate arguments without a required parameter
        result = tool.validate_args(
            query_type="company_filings",
            parameters={},  # Missing ticker
        )

        # Check that validation failed
        assert result is False

    def test_supported_queries(self):
        """Test that the SUPPORTED_QUERIES dictionary is correctly defined."""
        assert "company_filings" in SUPPORTED_QUERIES
        assert "filing_sections" in SUPPORTED_QUERIES
        assert "related_companies" in SUPPORTED_QUERIES
        assert "filing_timeline" in SUPPORTED_QUERIES
        assert "section_types" in SUPPORTED_QUERIES
        assert "custom_cypher" in SUPPORTED_QUERIES
