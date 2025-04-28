"""
SEC Graph Query Tool

This module provides a tool for agents to query the Neo4j graph database
containing SEC filing relationships and structure.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from sec_filing_analyzer.config import StorageConfig
from sec_filing_analyzer.storage import GraphStore

from ..contracts import BaseModel
from ..tools.base import Tool
from ..tools.decorator import tool

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
    "custom_cypher": CustomCypherParams,
}

# The tool registration is handled by the @tool decorator


@tool(
    name="sec_graph_query",
    tags=["sec", "graph", "query"],
    compact_description="Query relationships between companies, filings, and entities",
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
        database: Optional[str] = None,
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
            database=database,
        )

    async def _execute_abstract(self, query_type: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a graph query on the SEC filing database.

        Args:
            query_type: Type of query to execute (e.g., "company_filings", "filing_sections", "related_companies")
            parameters: Optional parameters for the query

        Returns:
            A standardized response dictionary with the following fields:
            - query_type: The type of query that was executed
            - parameters: The parameters that were used
            - results: The results of the query (empty list for errors)
            - output_key: The tool's name
            - success: Boolean indicating whether the operation was successful

            Error responses will additionally have:
            - error or warning: The error message (depending on error_type)
        """
        # Ensure parameters is a dictionary
        if parameters is None:
            parameters = {}

        try:
            # Validate query type
            if query_type not in SUPPORTED_QUERIES:
                supported_types = list(SUPPORTED_QUERIES.keys())
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message=f"Unsupported query type: {query_type}. Supported types: {supported_types}",
                )

            # Validate parameters using the appropriate model
            param_model = SUPPORTED_QUERIES[query_type]

            try:
                # Validate parameters
                params = param_model(**parameters)
            except Exception as e:
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message=f"Parameter validation error: {str(e)}",
                )

            logger.info(f"Executing graph query: {query_type}")

            # Check if graph store is available
            if self.graph_store is None:
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message="Graph store is not initialized",
                )

            # Execute the appropriate query based on query_type
            result = None
            try:
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
                else:
                    return self.format_error_response(
                        query_type=query_type,
                        parameters=parameters,
                        error_message=f"Unknown query type: {query_type}",
                    )
            except Exception as e:
                logger.error(f"Error executing graph query: {str(e)}")
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message=f"Error executing graph query: {str(e)}",
                )

            # Check if we have a valid result
            if result and isinstance(result, dict):
                # Ensure the result has the output_key
                if "output_key" not in result:
                    result["output_key"] = self.name
                return result
            else:
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message="No results found",
                    error_type="warning",
                )

        except Exception as e:
            logger.error(f"Unexpected error executing graph query: {str(e)}")
            return self.format_error_response(
                query_type=query_type,
                parameters=parameters,
                error_message=f"Unexpected error: {str(e)}",
            )

    def _query_company_filings(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query filings for a specific company."""
        ticker = parameters.get("ticker")
        filing_types = parameters.get("filing_types")
        limit = parameters.get("limit", 10)

        if not ticker:
            return self.format_error_response(
                query_type="company_filings",
                parameters=parameters,
                error_message="Missing required parameter: ticker",
            )

        try:
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
            results = self.graph_store.query(query, {"ticker": ticker, "filing_types": filing_types, "limit": limit})

            if not results:
                return self.format_error_response(
                    query_type="company_filings",
                    parameters=parameters,
                    error_message=f"No filings found for ticker: {ticker}",
                    error_type="warning",
                )

            return self.format_success_response(query_type="company_filings", parameters=parameters, results=results)
        except Exception as e:
            logger.error(f"Error querying company filings: {str(e)}")
            return self.format_error_response(
                query_type="company_filings",
                parameters=parameters,
                error_message=f"Error querying company filings: {str(e)}",
            )

    def _query_filing_sections(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query sections for a specific filing."""
        accession_number = parameters.get("accession_number")
        section_types = parameters.get("section_types")
        limit = parameters.get("limit", 50)

        if not accession_number:
            return self.format_error_response(
                query_type="filing_sections",
                parameters=parameters,
                error_message="Missing required parameter: accession_number",
            )

        try:
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
            results = self.graph_store.query(
                query,
                {
                    "accession_number": accession_number,
                    "section_types": section_types,
                    "limit": limit,
                },
            )

            if not results:
                return self.format_error_response(
                    query_type="filing_sections",
                    parameters=parameters,
                    error_message=f"No sections found for accession number: {accession_number}",
                    error_type="warning",
                )

            return self.format_success_response(query_type="filing_sections", parameters=parameters, results=results)
        except Exception as e:
            logger.error(f"Error querying filing sections: {str(e)}")
            return self.format_error_response(
                query_type="filing_sections",
                parameters=parameters,
                error_message=f"Error querying filing sections: {str(e)}",
            )

    def _query_related_companies(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query companies related to a specific company."""
        ticker = parameters.get("ticker")
        # Note: relationship_type is not currently used in the query
        # but kept for future extensibility
        # relationship_type = parameters.get("relationship_type", "MENTIONS")
        limit = parameters.get("limit", 10)

        if not ticker:
            return self.format_error_response(
                query_type="related_companies",
                parameters=parameters,
                error_message="Missing required parameter: ticker",
            )

        try:
            # Build Cypher query
            query = """
            MATCH (c1:Company {ticker: $ticker})-[:FILED]->(:Filing)-[:CONTAINS]->(:Section)-[:MENTIONS]->(c2:Company)
            WHERE c1 <> c2
            RETURN c2.ticker as ticker,
                   c2.name as name,
                   count(*) as mention_count
            ORDER BY mention_count DESC
            LIMIT $limit
            """

            # Execute query
            results = self.graph_store.query(query, {"ticker": ticker, "limit": limit})

            if not results:
                return self.format_error_response(
                    query_type="related_companies",
                    parameters=parameters,
                    error_message=f"No related companies found for ticker: {ticker}",
                    error_type="warning",
                )

            return self.format_success_response(query_type="related_companies", parameters=parameters, results=results)
        except Exception as e:
            logger.error(f"Error querying related companies: {str(e)}")
            return self.format_error_response(
                query_type="related_companies",
                parameters=parameters,
                error_message=f"Error querying related companies: {str(e)}",
            )

    def _query_filing_timeline(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query the timeline of filings for a company."""
        ticker = parameters.get("ticker")
        filing_type = parameters.get("filing_type", "10-K")
        limit = parameters.get("limit", 10)

        if not ticker:
            return self.format_error_response(
                query_type="filing_timeline",
                parameters=parameters,
                error_message="Missing required parameter: ticker",
            )

        try:
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
            results = self.graph_store.query(query, {"ticker": ticker, "filing_type": filing_type, "limit": limit})

            if not results:
                return self.format_error_response(
                    query_type="filing_timeline",
                    parameters=parameters,
                    error_message=f"No filing timeline found for ticker: {ticker} and filing type: {filing_type}",
                    error_type="warning",
                )

            return self.format_success_response(query_type="filing_timeline", parameters=parameters, results=results)
        except Exception as e:
            logger.error(f"Error querying filing timeline: {str(e)}")
            return self.format_error_response(
                query_type="filing_timeline",
                parameters=parameters,
                error_message=f"Error querying filing timeline: {str(e)}",
            )

    def _query_section_types(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query the available section types in the database."""
        try:
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

            if not results:
                return self.format_error_response(
                    query_type="section_types",
                    parameters=parameters,
                    error_message="No section types found in the database",
                    error_type="warning",
                )

            return self.format_success_response(query_type="section_types", parameters=parameters, results=results)
        except Exception as e:
            logger.error(f"Error querying section types: {str(e)}")
            return self.format_error_response(
                query_type="section_types",
                parameters=parameters,
                error_message=f"Error querying section types: {str(e)}",
            )

    def _execute_custom_cypher(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom Cypher query."""
        cypher_query = parameters.get("cypher_query")
        query_params = parameters.get("query_params", {})

        if not cypher_query:
            return self.format_error_response(
                query_type="custom_cypher",
                parameters={"cypher_query": cypher_query},
                error_message="Missing required parameter: cypher_query",
            )

        try:
            # Execute query
            results = self.graph_store.query(cypher_query, query_params)

            if not results:
                return self.format_error_response(
                    query_type="custom_cypher",
                    parameters={"cypher_query": cypher_query},
                    error_message="No results found for the custom Cypher query",
                    error_type="warning",
                )

            return self.format_success_response(
                query_type="custom_cypher",
                parameters={"cypher_query": cypher_query},  # Don't return potentially sensitive query_params
                results=results,
            )
        except Exception as e:
            logger.error(f"Error executing custom Cypher query: {str(e)}")
            return self.format_error_response(
                query_type="custom_cypher",
                parameters={"cypher_query": cypher_query},
                error_message=f"Error executing custom Cypher query: {str(e)}",
            )

    def validate_args(self, query_type: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
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
