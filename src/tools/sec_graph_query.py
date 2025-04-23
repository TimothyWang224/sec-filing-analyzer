"""
SEC Graph Query Tool

This module provides a tool for agents to query the Neo4j graph database
containing SEC filing relationships and structure.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Type

from ..tools.base import Tool
from ..tools.decorator import tool
from ..contracts import BaseModel, ToolSpec, field_validator
from ..errors import ParameterError, QueryTypeUnsupported, StorageUnavailable, DataNotFound
from sec_filing_analyzer.storage import GraphStore
from sec_filing_analyzer.config import StorageConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define parameter models
class CompanyFilingsParams(BaseModel):
    """Parameters for company filings queries."""
    ticker: str
    filing_types: Optional[List[str]] = None
    limit: int = 10

class FilingSectionsParams(BaseModel):
    """Parameters for filing sections queries."""
    accession_number: str
    section_types: Optional[List[str]] = None
    limit: int = 50

class RelatedCompaniesParams(BaseModel):
    """Parameters for related companies queries."""
    ticker: str
    relationship_type: str = "MENTIONS"
    limit: int = 10

class FilingTimelineParams(BaseModel):
    """Parameters for filing timeline queries."""
    ticker: str
    filing_type: str = "10-K"
    limit: int = 10

class SectionTypesParams(BaseModel):
    """Parameters for section types queries."""
    pass

class CustomCypherParams(BaseModel):
    """Parameters for custom Cypher queries."""
    cypher_query: str
    query_params: Dict[str, Any] = {}

# Map query types to parameter models
SUPPORTED_QUERIES: Dict[str, Type[BaseModel]] = {
    "company_filings": CompanyFilingsParams,
    "filing_sections": FilingSectionsParams,
    "related_companies": RelatedCompaniesParams,
    "filing_timeline": FilingTimelineParams,
    "section_types": SectionTypesParams,
    "custom_cypher": CustomCypherParams
}

# Register tool specification
from .registry import ToolRegistry

# The tool registration is handled by the @tool decorator
# The ToolSpec will be created automatically by the ToolRegistry._register_tool method

@tool(
    name="sec_graph_query",
    tags=["sec", "graph", "query"],
    compact_description="Query relationships between companies, filings, and entities"
    # Not using schema mappings for this tool since it has a nested parameter structure
)
class SECGraphQueryTool(Tool):
    """Tool for querying the SEC filing graph database.

    Queries the graph database for structural information about SEC filings, companies, and their relationships.
    Use this tool to find connections between entities or to retrieve structured information about filings.
    """

    def __init__(
        self,
        graph_store_dir: Optional[str] = None,
        use_neo4j: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
        database: Optional[str] = None
    ):
        """Initialize the SEC graph query tool.

        Args:
            graph_store_dir: Optional directory for the graph store
            use_neo4j: Whether to use Neo4j (True) or in-memory graph (False)
            username: Neo4j username
            password: Neo4j password
            url: Neo4j URL
            database: Neo4j database name
        """
        super().__init__()

        # Initialize graph store
        config = StorageConfig()
        self.graph_store_dir = graph_store_dir or config.graph_store_path
        self.graph_store = GraphStore(
            store_dir=self.graph_store_dir,
            use_neo4j=use_neo4j,
            username=username,
            password=password,
            url=url,
            database=database
        )

    async def _execute(
        self,
        query_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a graph query on the SEC filing database.

        Args:
            query_type: Type of query to execute (e.g., "company_filings", "filing_sections", "related_companies")
            parameters: Optional parameters for the query

        Returns:
            Dictionary containing query results

        Raises:
            QueryTypeUnsupported: If the query type is not supported
            ParameterError: If the parameters are invalid
            StorageUnavailable: If the graph store is unavailable
            DataNotFound: If no results are found
        """
        try:
            # Validate query type
            if query_type not in SUPPORTED_QUERIES:
                supported_types = list(SUPPORTED_QUERIES.keys())
                raise QueryTypeUnsupported(query_type, "sec_graph_query", supported_types)

            # Validate parameters using the appropriate model
            param_model = SUPPORTED_QUERIES[query_type]
            if parameters is None:
                parameters = {}

            try:
                # Validate parameters
                params = param_model(**parameters)
            except Exception as e:
                raise ParameterError(str(e))

            logger.info(f"Executing graph query: {query_type}")

            # Check if graph store is available
            if self.graph_store is None:
                raise StorageUnavailable("graph_store", "Graph store is not initialized")

            # Execute the appropriate query based on query_type
            result = None
            if query_type == "company_filings":
                result = self._query_company_filings(params.model_dump())
            elif query_type == "filing_sections":
                result = self._query_filing_sections(params.model_dump())
            elif query_type == "related_companies":
                result = self._query_related_companies(params.model_dump())
            elif query_type == "filing_timeline":
                result = self._query_filing_timeline(params.model_dump())
            elif query_type == "section_types":
                result = self._query_section_types(params.model_dump())
            elif query_type == "custom_cypher":
                result = self._execute_custom_cypher(params.model_dump())

            # Add output key to result
            if result and isinstance(result, dict):
                result["output_key"] = "sec_graph_query"
                return result
            else:
                raise DataNotFound("graph_query_results", {
                    "query_type": query_type,
                    "parameters": parameters
                })

        except (QueryTypeUnsupported, ParameterError, StorageUnavailable, DataNotFound) as e:
            # Re-raise known errors
            raise
        except Exception as e:
            logger.error(f"Error executing graph query: {str(e)}")
            raise StorageUnavailable("graph_store", f"Error executing graph query: {str(e)}")

    def _query_company_filings(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query filings for a specific company."""
        ticker = parameters.get("ticker")
        filing_types = parameters.get("filing_types")
        limit = parameters.get("limit", 10)

        if not ticker:
            return {"error": "Missing required parameter: ticker", "results": []}

        # Build Cypher query
        query = """
        MATCH (c:Company {ticker: $ticker})-[:FILED]->(f:Filing)
        """

        if filing_types:
            query += "WHERE f.filing_type IN $filing_types "

        query += """
        RETURN f.filing_type as filing_type,
               f.filing_date as filing_date,
               f.accession_number as accession_number,
               f.fiscal_year as fiscal_year,
               f.fiscal_period as fiscal_period
        ORDER BY f.filing_date DESC
        LIMIT $limit
        """

        # Execute query
        results = self.graph_store.query(query, {
            "ticker": ticker,
            "filing_types": filing_types,
            "limit": limit
        })

        return {
            "query_type": "company_filings",
            "parameters": parameters,
            "results": results
        }

    def _query_filing_sections(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query sections for a specific filing."""
        accession_number = parameters.get("accession_number")
        section_types = parameters.get("section_types")
        limit = parameters.get("limit", 50)

        if not accession_number:
            return {"error": "Missing required parameter: accession_number", "results": []}

        # Build Cypher query
        query = """
        MATCH (f:Filing {accession_number: $accession_number})-[:CONTAINS]->(s:Section)
        """

        if section_types:
            query += "WHERE s.section_type IN $section_types "

        query += """
        RETURN s.title as title,
               s.section_type as section_type,
               s.order as order
        ORDER BY s.order ASC
        LIMIT $limit
        """

        # Execute query
        results = self.graph_store.query(query, {
            "accession_number": accession_number,
            "section_types": section_types,
            "limit": limit
        })

        return {
            "query_type": "filing_sections",
            "parameters": parameters,
            "results": results
        }

    def _query_related_companies(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query companies related to a specific company."""
        ticker = parameters.get("ticker")
        relationship_type = parameters.get("relationship_type", "MENTIONS")
        limit = parameters.get("limit", 10)

        if not ticker:
            return {"error": "Missing required parameter: ticker", "results": []}

        # Build Cypher query
        query = f"""
        MATCH (c1:Company {{ticker: $ticker}})-[:FILED]->(:Filing)-[:CONTAINS]->(:Section)-[:MENTIONS]->(c2:Company)
        WHERE c1 <> c2
        RETURN c2.ticker as ticker,
               c2.name as name,
               count(*) as mention_count
        ORDER BY mention_count DESC
        LIMIT $limit
        """

        # Execute query
        results = self.graph_store.query(query, {
            "ticker": ticker,
            "limit": limit
        })

        return {
            "query_type": "related_companies",
            "parameters": parameters,
            "results": results
        }

    def _query_filing_timeline(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query the timeline of filings for a company."""
        ticker = parameters.get("ticker")
        filing_type = parameters.get("filing_type", "10-K")
        limit = parameters.get("limit", 10)

        if not ticker:
            return {"error": "Missing required parameter: ticker", "results": []}

        # Build Cypher query
        query = """
        MATCH (c:Company {ticker: $ticker})-[:FILED]->(f:Filing)
        WHERE f.filing_type = $filing_type
        RETURN f.filing_date as filing_date,
               f.accession_number as accession_number,
               f.fiscal_year as fiscal_year,
               f.fiscal_period as fiscal_period
        ORDER BY f.filing_date DESC
        LIMIT $limit
        """

        # Execute query
        results = self.graph_store.query(query, {
            "ticker": ticker,
            "filing_type": filing_type,
            "limit": limit
        })

        return {
            "query_type": "filing_timeline",
            "parameters": parameters,
            "results": results
        }

    def _query_section_types(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query the available section types in the database."""
        # Build Cypher query
        query = """
        MATCH (s:Section)
        WHERE s.section_type IS NOT NULL
        RETURN DISTINCT s.section_type as section_type,
               count(*) as count
        ORDER BY count DESC
        """

        # Execute query
        results = self.graph_store.query(query)

        return {
            "query_type": "section_types",
            "parameters": parameters,
            "results": results
        }

    def _execute_custom_cypher(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom Cypher query."""
        cypher_query = parameters.get("cypher_query")
        query_params = parameters.get("query_params", {})

        if not cypher_query:
            return {"error": "Missing required parameter: cypher_query", "results": []}

        # Execute query
        results = self.graph_store.query(cypher_query, query_params)

        return {
            "query_type": "custom_cypher",
            "parameters": {
                "cypher_query": cypher_query,
                # Don't return potentially sensitive query_params
            },
            "results": results
        }

    def validate_args(
        self,
        query_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate the tool arguments.

        Args:
            query_type: Type of query to execute
            parameters: Optional parameters for the query

        Returns:
            True if arguments are valid, False otherwise
        """
        try:
            # Validate query type
            if query_type not in SUPPORTED_QUERIES:
                logger.error(f"Invalid query_type: must be one of {list(SUPPORTED_QUERIES.keys())}")
                return False

            # Validate parameters using the appropriate model
            param_model = SUPPORTED_QUERIES[query_type]
            if parameters is None:
                parameters = {}

            try:
                # Validate parameters
                param_model(**parameters)
                return True
            except Exception as e:
                logger.error(f"Parameter validation error: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False
