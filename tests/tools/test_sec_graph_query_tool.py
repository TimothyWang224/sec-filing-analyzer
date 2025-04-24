"""
Unit tests for the SECGraphQueryTool.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.tools.sec_graph_query import SECGraphQueryTool
from src.errors import QueryTypeUnsupported, ParameterError, StorageUnavailable, DataNotFound


class TestSECGraphQueryTool:
    """Test suite for the SECGraphQueryTool."""
    
    @pytest.fixture
    def mock_graph_db(self):
        """Create a mock graph database."""
        mock_db = MagicMock()
        mock_db.query.return_value = [
            {
                "ticker": "AAPL",
                "filing_type": "10-K",
                "filing_date": "2022-10-28",
                "accession_number": "0000320193-22-000108"
            }
        ]
        return mock_db
    
    @pytest.fixture
    def tool(self, mock_graph_db):
        """Create a SECGraphQueryTool instance for testing."""
        with patch("src.tools.sec_graph_query.SECStructure", return_value=mock_graph_db):
            tool = SECGraphQueryTool(use_neo4j=False)
            tool.graph_db = mock_graph_db
            return tool
    
    def test_init(self):
        """Test initializing the tool."""
        with patch("src.tools.sec_graph_query.SECStructure"):
            tool = SECGraphQueryTool(use_neo4j=False)
            assert tool.name == "sec_graph_query"
            assert "tool for querying the sec filing graph database" in tool.description.lower()
    
    def test_init_with_db_error(self):
        """Test initializing the tool with a database error."""
        with patch("src.tools.sec_graph_query.SECStructure", side_effect=Exception("DB error")):
            tool = SECGraphQueryTool(use_neo4j=False)
            assert tool.graph_db is None
            assert tool.db_error is not None
            assert "DB error" in str(tool.db_error)
    
    @pytest.mark.asyncio
    async def test_execute_company_filings(self, tool, mock_graph_db):
        """Test executing the tool with company_filings query type."""
        # Execute the tool
        result = await tool.execute(
            query_type="company_filings",
            parameters={
                "ticker": "AAPL",
                "filing_types": ["10-K"],
                "limit": 5
            }
        )
        
        # Check that the result is correct
        assert result["query_type"] == "company_filings"
        assert result["parameters"]["ticker"] == "AAPL"
        assert "10-K" in result["parameters"]["filing_types"]
        assert result["parameters"]["limit"] == 5
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["ticker"] == "AAPL"
        assert "output_key" in result
        assert result["output_key"] == "sec_graph_query"
        
        # Check that the graph database was queried
        mock_graph_db.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_filing_entities(self, tool, mock_graph_db):
        """Test executing the tool with filing_entities query type."""
        # Mock the query_filing_entities method
        mock_entities = {
            "query_type": "filing_entities",
            "parameters": {"accession_number": "0000320193-22-000108"},
            "results": [
                {"entity_type": "Person", "name": "Tim Cook", "role": "CEO"}
            ]
        }
        with patch.object(tool, "_query_filing_entities", return_value=mock_entities):
            # Execute the tool
            result = await tool.execute(
                query_type="filing_entities",
                parameters={
                    "accession_number": "0000320193-22-000108"
                }
            )
            
            # Check that the result is correct
            assert result["query_type"] == "filing_entities"
            assert result["parameters"]["accession_number"] == "0000320193-22-000108"
            assert len(result["results"]) == 1
            assert result["results"][0]["name"] == "Tim Cook"
    
    @pytest.mark.asyncio
    async def test_execute_entity_relationships(self, tool, mock_graph_db):
        """Test executing the tool with entity_relationships query type."""
        # Mock the query_entity_relationships method
        mock_relationships = {
            "query_type": "entity_relationships",
            "parameters": {"entity_name": "Tim Cook"},
            "results": [
                {"relationship_type": "WORKS_FOR", "target_entity": "Apple Inc."}
            ]
        }
        with patch.object(tool, "_query_entity_relationships", return_value=mock_relationships):
            # Execute the tool
            result = await tool.execute(
                query_type="entity_relationships",
                parameters={
                    "entity_name": "Tim Cook"
                }
            )
            
            # Check that the result is correct
            assert result["query_type"] == "entity_relationships"
            assert result["parameters"]["entity_name"] == "Tim Cook"
            assert len(result["results"]) == 1
            assert result["results"][0]["target_entity"] == "Apple Inc."
    
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
        assert "sec_graph_query" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_execute_db_unavailable(self, tool):
        """Test executing the tool when the database is unavailable."""
        # Set graph_db to None to simulate unavailable database
        tool.graph_db = None
        tool.db_error = Exception("DB error")
        
        # Execute the tool
        with pytest.raises(StorageUnavailable) as excinfo:
            await tool.execute(
                query_type="company_filings",
                parameters={
                    "ticker": "AAPL"
                }
            )
        
        # Check that the error message is correct
        assert "graph database is unavailable" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_execute_data_not_found(self, tool, mock_graph_db):
        """Test executing the tool when data is not found."""
        # Mock the query method to return an empty list
        mock_graph_db.query.return_value = []
        
        # Execute the tool
        with pytest.raises(DataNotFound) as excinfo:
            await tool.execute(
                query_type="company_filings",
                parameters={
                    "ticker": "UNKNOWN"
                }
            )
        
        # Check that the error message is correct
        assert "No data found" in str(excinfo.value)
    
    def test_validate_args_valid(self, tool):
        """Test validating valid arguments."""
        # Validate arguments
        result = tool.validate_args(
            query_type="company_filings",
            parameters={
                "ticker": "AAPL",
                "filing_types": ["10-K"],
                "limit": 5
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
                query_type="company_filings",
                parameters={}  # Missing ticker
            )
