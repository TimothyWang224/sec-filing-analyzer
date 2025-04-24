"""
Unit tests for the SECSemanticSearchTool.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.tools.sec_semantic_search import SECSemanticSearchTool
from src.errors import QueryTypeUnsupported, ParameterError, StorageUnavailable, DataNotFound


class TestSECSemanticSearchTool:
    """Test suite for the SECSemanticSearchTool."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            {
                "text": "Apple's revenue increased by 10% in 2022.",
                "metadata": {
                    "ticker": "AAPL",
                    "filing_type": "10-K",
                    "filing_date": "2022-10-28"
                },
                "score": 0.95
            }
        ]
        return mock_store
    
    @pytest.fixture
    def tool(self, mock_vector_store):
        """Create a SECSemanticSearchTool instance for testing."""
        with patch("src.tools.sec_semantic_search.OptimizedVectorStore", return_value=mock_vector_store):
            tool = SECSemanticSearchTool(vector_store_path="test_vector_store")
            tool.vector_store = mock_vector_store
            return tool
    
    def test_init(self):
        """Test initializing the tool."""
        with patch("src.tools.sec_semantic_search.OptimizedVectorStore"):
            tool = SECSemanticSearchTool(vector_store_path="test_vector_store")
            assert tool.name == "sec_semantic_search"
            assert "tool for semantic search on sec filings" in tool.description.lower()
    
    def test_init_with_store_error(self):
        """Test initializing the tool with a vector store error."""
        with patch("src.tools.sec_semantic_search.OptimizedVectorStore", side_effect=Exception("Store error")):
            tool = SECSemanticSearchTool(vector_store_path="test_vector_store")
            assert tool.vector_store is None
            assert tool.store_error is not None
            assert "Store error" in str(tool.store_error)
    
    @pytest.mark.asyncio
    async def test_execute_valid_parameters(self, tool, mock_vector_store):
        """Test executing the tool with valid parameters."""
        # Execute the tool
        result = await tool.execute(
            query="What was Apple's revenue in 2022?",
            companies=["AAPL"],
            top_k=3,
            filing_types=["10-K"],
            date_range=["2022-01-01", "2022-12-31"]
        )
        
        # Check that the result is correct
        assert result["query"] == "What was Apple's revenue in 2022?"
        assert "AAPL" in result["parameters"]["companies"]
        assert result["parameters"]["top_k"] == 3
        assert "10-K" in result["parameters"]["filing_types"]
        assert result["parameters"]["date_range"][0] == "2022-01-01"
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["text"] == "Apple's revenue increased by 10% in 2022."
        assert "output_key" in result
        assert result["output_key"] == "sec_semantic_search"
        
        # Check that the vector store was searched
        mock_vector_store.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_minimal_parameters(self, tool, mock_vector_store):
        """Test executing the tool with minimal parameters."""
        # Execute the tool
        result = await tool.execute(
            query="What was Apple's revenue in 2022?"
        )
        
        # Check that the result is correct
        assert result["query"] == "What was Apple's revenue in 2022?"
        assert "results" in result
        
        # Check that the vector store was searched
        mock_vector_store.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_store_unavailable(self, tool):
        """Test executing the tool when the vector store is unavailable."""
        # Set vector_store to None to simulate unavailable store
        tool.vector_store = None
        tool.store_error = Exception("Store error")
        
        # Execute the tool
        with pytest.raises(StorageUnavailable) as excinfo:
            await tool.execute(
                query="What was Apple's revenue in 2022?"
            )
        
        # Check that the error message is correct
        assert "vector store is unavailable" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_execute_no_results(self, tool, mock_vector_store):
        """Test executing the tool when no results are found."""
        # Mock the search method to return an empty list
        mock_vector_store.search.return_value = []
        
        # Execute the tool
        result = await tool.execute(
            query="What was Apple's revenue in 2022?",
            companies=["AAPL"]
        )
        
        # Check that the result has an empty results list
        assert result["query"] == "What was Apple's revenue in 2022?"
        assert "results" in result
        assert len(result["results"]) == 0
    
    @pytest.mark.asyncio
    async def test_execute_search_error(self, tool, mock_vector_store):
        """Test executing the tool when the search raises an error."""
        # Mock the search method to raise an exception
        mock_vector_store.search.side_effect = Exception("Search error")
        
        # Execute the tool
        with pytest.raises(StorageUnavailable) as excinfo:
            await tool.execute(
                query="What was Apple's revenue in 2022?"
            )
        
        # Check that the error message is correct
        assert "Error performing semantic search" in str(excinfo.value)
    
    def test_resolve_parameters(self, tool):
        """Test resolving parameters."""
        # Resolve parameters
        params = {
            "query": "What was Apple's revenue in 2022?",
            "companies": ["AAPL"],
            "top_k": 3,
            "filing_types": ["10-K"],
            "date_range": ["2022-01-01", "2022-12-31"]
        }
        resolved = tool.resolve_parameters(**params)
        
        # Check that the parameters were resolved correctly
        assert resolved["query"] == "What was Apple's revenue in 2022?"
        assert resolved["companies"] == ["AAPL"]
        assert resolved["top_k"] == 3
        assert resolved["filing_types"] == ["10-K"]
        assert resolved["date_range"] == ["2022-01-01", "2022-12-31"]
    
    def test_resolve_parameters_defaults(self, tool):
        """Test resolving parameters with defaults."""
        # Resolve parameters with minimal input
        params = {
            "query": "What was Apple's revenue in 2022?"
        }
        resolved = tool.resolve_parameters(**params)
        
        # Check that the parameters were resolved with defaults
        assert resolved["query"] == "What was Apple's revenue in 2022?"
        assert resolved["companies"] == []  # Default
        assert resolved["top_k"] == 5  # Default
        assert resolved["filing_types"] == ["10-K", "10-Q"]  # Default
        assert len(resolved["date_range"]) == 2  # Default date range
    
    def test_validate_args_valid(self, tool):
        """Test validating valid arguments."""
        # Validate arguments
        result = tool.validate_args(
            query="What was Apple's revenue in 2022?",
            companies=["AAPL"],
            top_k=3,
            filing_types=["10-K"],
            date_range=["2022-01-01", "2022-12-31"]
        )
        
        # Check that validation passed
        assert result
    
    def test_validate_args_missing_query(self, tool):
        """Test validating arguments without a query."""
        # Validate arguments without a query
        with pytest.raises(ValueError) as excinfo:
            tool.validate_args(
                companies=["AAPL"]
            )
        
        # Check that the error message is correct
        assert "query" in str(excinfo.value)
    
    def test_validate_args_invalid_top_k(self, tool):
        """Test validating arguments with an invalid top_k."""
        # Validate arguments with an invalid top_k
        with pytest.raises(ValueError) as excinfo:
            tool.validate_args(
                query="What was Apple's revenue in 2022?",
                top_k=0  # Invalid
            )
        
        # Check that the error message is correct
        assert "top_k" in str(excinfo.value)
