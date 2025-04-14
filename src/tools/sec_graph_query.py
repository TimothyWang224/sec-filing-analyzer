"""
SEC Graph Query Tool

This module provides a tool for agents to query the Neo4j graph database
containing SEC filing relationships and structure.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from ..tools.base import Tool
from sec_filing_analyzer.storage import GraphStore
from sec_filing_analyzer.config import StorageConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECGraphQueryTool(Tool):
    """Tool for querying the SEC filing graph database.

    Queries the graph database for structural information about SEC filings, companies, and their relationships.
    Use this tool to find connections between entities or to retrieve structured information about filings.
    """

    _tool_name = "sec_graph_query"
    _tool_tags = ["sec", "graph", "query"]
    _compact_description = "Query relationships between companies, filings, and entities"

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

    async def execute(
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
        """
        try:
            logger.info(f"Executing graph query: {query_type}")

            if parameters is None:
                parameters = {}

            # Execute the appropriate query based on query_type
            if query_type == "company_filings":
                return self._query_company_filings(parameters)
            elif query_type == "filing_sections":
                return self._query_filing_sections(parameters)
            elif query_type == "related_companies":
                return self._query_related_companies(parameters)
            elif query_type == "filing_timeline":
                return self._query_filing_timeline(parameters)
            elif query_type == "section_types":
                return self._query_section_types(parameters)
            elif query_type == "custom_cypher":
                return self._execute_custom_cypher(parameters)
            else:
                return {
                    "error": f"Unknown query type: {query_type}",
                    "results": []
                }

        except Exception as e:
            logger.error(f"Error executing graph query: {str(e)}")
            return {
                "error": str(e),
                "query_type": query_type,
                "parameters": parameters,
                "results": []
            }

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
        # Validate query_type
        valid_query_types = [
            "company_filings",
            "filing_sections",
            "related_companies",
            "filing_timeline",
            "section_types",
            "custom_cypher"
        ]

        if not query_type or query_type not in valid_query_types:
            logger.error(f"Invalid query_type: must be one of {valid_query_types}")
            return False

        # Validate parameters based on query_type
        if parameters is None:
            parameters = {}

        if query_type == "company_filings" and "ticker" not in parameters:
            logger.error("Missing required parameter for company_filings: ticker")
            return False

        if query_type == "filing_sections" and "accession_number" not in parameters:
            logger.error("Missing required parameter for filing_sections: accession_number")
            return False

        if query_type == "related_companies" and "ticker" not in parameters:
            logger.error("Missing required parameter for related_companies: ticker")
            return False

        if query_type == "filing_timeline" and "ticker" not in parameters:
            logger.error("Missing required parameter for filing_timeline: ticker")
            return False

        if query_type == "custom_cypher" and "cypher_query" not in parameters:
            logger.error("Missing required parameter for custom_cypher: cypher_query")
            return False

        return True
