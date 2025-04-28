"""
Unit tests for the SECSemanticSearchTool.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.tools.sec_semantic_search import SUPPORTED_QUERIES, SECSemanticSearchTool


class TestSECSemanticSearchTool:
    """Test suite for the SECSemanticSearchTool."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_store = MagicMock()
        mock_store.search_vectors.return_value = [
            {
                "id": "doc1",
                "text": "Apple's revenue increased by 10% in 2022.",
                "metadata": {
                    "company": "Apple Inc.",
                    "ticker": "AAPL",
                    "filing_type": "10-K",
                    "filing_date": "2022-10-28",
                    "section": "Management Discussion",
                    "section_type": "MD&A",
                },
                "score": 0.95,
            }
        ]
        return mock_store

    @pytest.fixture
    def tool(self, mock_vector_store):
        """Create a SECSemanticSearchTool instance for testing."""
        with patch(
            "src.tools.sec_semantic_search.OptimizedVectorStore",
            return_value=mock_vector_store,
        ):
            tool = SECSemanticSearchTool(vector_store_path="test_vector_store")
            tool.vector_store = mock_vector_store
            return tool

    def test_init(self):
        """Test initializing the tool."""
        with patch("src.tools.sec_semantic_search.OptimizedVectorStore"):
            tool = SECSemanticSearchTool(vector_store_path="test_vector_store")
            assert tool.name == "sec_semantic_search"
            assert (
                "tool for performing semantic search on sec filings"
                in tool.description.lower()
            )

    def test_init_with_store_error(self):
        """Test initializing the tool with a vector store error."""
        # The tool doesn't handle initialization errors gracefully, so we need to catch the exception
        with pytest.raises(Exception) as excinfo:
            with patch(
                "src.tools.sec_semantic_search.OptimizedVectorStore",
                side_effect=Exception("Store error"),
            ):
                SECSemanticSearchTool(vector_store_path="test_vector_store")

        # Check that the error message is correct
        assert "Store error" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_execute_valid_parameters(self, tool, mock_vector_store):
        """Test executing the tool with valid parameters."""
        # Execute the tool
        result = await tool.execute(
            query_type="semantic_search",
            parameters={
                "query": "What was Apple's revenue in 2022?",
                "companies": ["AAPL"],
                "top_k": 3,
                "filing_types": ["10-K"],
                "date_range": ["2022-01-01", "2022-12-31"],
            },
        )

        # Check that the result is correct
        assert result["query"] == "What was Apple's revenue in 2022?"
        assert "AAPL" in result["companies"]
        assert "10-K" in result["filing_types"]
        assert result["date_range"][0] == "2022-01-01"
        assert "results" in result
        assert len(result["results"]) == 1
        assert (
            result["results"][0]["text"] == "Apple's revenue increased by 10% in 2022."
        )
        assert "output_key" in result
        assert result["output_key"] == "sec_semantic_search"

        # Check that the vector store was searched
        mock_vector_store.search_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_minimal_parameters(self, tool, mock_vector_store):
        """Test executing the tool with minimal parameters."""
        # Execute the tool
        result = await tool.execute(
            query_type="semantic_search",
            parameters={"query": "What was Apple's revenue in 2022?"},
        )

        # Check that the result is correct
        assert result["query"] == "What was Apple's revenue in 2022?"
        assert "results" in result

        # Check that the vector store was searched
        mock_vector_store.search_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_store_unavailable(self, tool):
        """Test executing the tool when the vector store is unavailable."""
        # Set vector_store to None to simulate unavailable store
        tool.vector_store = None

        # Execute the tool
        result = await tool.execute(
            query_type="semantic_search",
            parameters={"query": "What was Apple's revenue in 2022?"},
        )

        # Check that the result contains an error message
        assert result["query_type"] == "semantic_search"
        assert "error" in result
        assert "Vector store is not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_no_results(self, tool, mock_vector_store):
        """Test executing the tool when no results are found."""
        # Mock the search method to return an empty list
        mock_vector_store.search_vectors.return_value = []

        # Execute the tool
        result = await tool.execute(
            query_type="semantic_search",
            parameters={
                "query": "What was Apple's revenue in 2022?",
                "companies": ["AAPL"],
            },
        )

        # Check that the result contains the expected fields
        assert result["query_type"] == "semantic_search"
        assert "results" in result
        assert len(result["results"]) == 0  # Empty results list

    @pytest.mark.asyncio
    async def test_execute_search_error(self, tool, mock_vector_store):
        """Test executing the tool when the search raises an error."""
        # Mock the search method to raise an exception
        mock_vector_store.search_vectors.side_effect = Exception("Search error")

        # Execute the tool
        result = await tool.execute(
            query_type="semantic_search",
            parameters={"query": "What was Apple's revenue in 2022?"},
        )

        # Check that the result contains an error message
        assert result["query_type"] == "semantic_search"
        assert "error" in result
        assert "Error executing semantic search" in result["error"]

    def test_validate_args_valid(self, tool):
        """Test validating valid arguments."""
        # Validate arguments
        result = tool.validate_args(
            query_type="semantic_search",
            parameters={
                "query": "What was Apple's revenue in 2022?",
                "companies": ["AAPL"],
                "top_k": 3,
                "filing_types": ["10-K"],
                "date_range": ["2022-01-01", "2022-12-31"],
            },
        )

        # Check that validation passed
        assert result is True

    def test_validate_args_invalid_query_type(self, tool):
        """Test validating arguments with an invalid query type."""
        # Validate arguments with an invalid query type
        result = tool.validate_args(
            query_type="invalid_query",
            parameters={"query": "What was Apple's revenue in 2022?"},
        )

        # Check that validation failed
        assert result is False

    def test_validate_args_missing_query(self, tool):
        """Test validating arguments without a query."""
        # Validate arguments without a query
        result = tool.validate_args(
            query_type="semantic_search", parameters={"companies": ["AAPL"]}
        )

        # Check that validation failed
        assert result is False

    def test_supported_queries(self):
        """Test that the SUPPORTED_QUERIES dictionary is correctly defined."""
        assert "semantic_search" in SUPPORTED_QUERIES
