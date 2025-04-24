"""
SEC Semantic Search Tool

This module provides a tool for agents to perform semantic search on SEC filings
using the optimized vector store.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Type
from datetime import datetime

from ..tools.base import Tool
from ..tools.decorator import tool
from ..contracts import BaseModel, field_validator
from sec_filing_analyzer.storage import OptimizedVectorStore
from sec_filing_analyzer.config import StorageConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define parameter models
class SemanticSearchParams(BaseModel):
    """Parameters for semantic search queries."""
    query: str
    companies: Optional[List[str]] = None
    top_k: int = 5
    filing_types: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None
    sections: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    hybrid_search_weight: float = 0.5

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Query must be a non-empty string")
        return v

    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v

    @field_validator('companies')
    @classmethod
    def validate_companies(cls, v):
        if v is not None and not all(isinstance(company, str) for company in v):
            raise ValueError("Companies must be a list of strings")
        return v

    @field_validator('filing_types')
    @classmethod
    def validate_filing_types(cls, v):
        valid_filing_types = ["10-K", "10-Q", "8-K", "S-1", "S-4", "20-F", "40-F", "6-K"]
        if v is not None and not all(filing_type in valid_filing_types for filing_type in v):
            raise ValueError(f"Filing types must be in {valid_filing_types}")
        return v

    @field_validator('date_range')
    @classmethod
    def validate_date_range(cls, v):
        if v is not None:
            if len(v) != 2:
                raise ValueError("Date range must be a tuple of (start_date, end_date)")

            start_date, end_date = v
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Dates must be in format 'YYYY-MM-DD'")

            if start_date > end_date:
                raise ValueError("Start date must be before end date")
        return v

    @field_validator('hybrid_search_weight')
    @classmethod
    def validate_hybrid_search_weight(cls, v):
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("Hybrid search weight must be a float between 0 and 1")
        return v

# Map query types to parameter models
SUPPORTED_QUERIES: Dict[str, Type[BaseModel]] = {
    "semantic_search": SemanticSearchParams
}

# The tool registration is handled by the @tool decorator

@tool(
    name="sec_semantic_search",
    tags=["sec", "semantic", "search"],
    compact_description="Search SEC filings using natural language queries"
    # Not using schema mappings for this tool since it has a complex parameter structure
)
class SECSemanticSearchTool(Tool):
    """Tool for performing semantic search on SEC filings.

    Performs semantic search on SEC filings to find relevant information based on natural language queries.
    Use this tool to search for specific topics, concepts, or information within SEC filings.
    """

    def __init__(self, vector_store_path: Optional[str] = None):
        """Initialize the SEC semantic search tool.

        Args:
            vector_store_path: Optional path to the vector store
        """
        super().__init__()

        # Initialize vector store
        config = StorageConfig()
        self.vector_store_path = vector_store_path or config.vector_store_path
        self.vector_store = OptimizedVectorStore(store_path=self.vector_store_path)

    async def _execute(
        self,
        query_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute semantic search on SEC filings.

        Args:
            query_type: Type of query to execute (e.g., "semantic_search")
            parameters: Parameters for the query

        Returns:
            A standardized response dictionary with the following fields:
            - query_type: The type of query that was executed
            - parameters: The parameters that were used
            - results: The search results with text and metadata
            - output_key: The tool's name
            - success: Boolean indicating whether the operation was successful
            - query: The search query text
            - total_results: The number of results found
            - companies: The companies that were searched
            - filing_types: The filing types that were searched
            - date_range: The date range that was searched
            - sections: The sections that were searched

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
                    error_message=f"Unsupported query type: {query_type}. Supported types: {supported_types}"
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
                    error_message=f"Parameter validation error: {str(e)}"
                )

            # Extract parameters
            query = params.query
            companies = params.companies
            top_k = params.top_k
            filing_types = params.filing_types
            date_range = params.date_range
            sections = params.sections
            keywords = params.keywords
            hybrid_search_weight = params.hybrid_search_weight

            logger.info(f"Executing semantic search: {query}")
            logger.info(f"Companies: {companies}")
            logger.info(f"Filing types: {filing_types}")
            logger.info(f"Date range: {date_range}")

            # Check if vector store is available
            if self.vector_store is None:
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message="Vector store is not initialized"
                )

            try:
                # Perform search
                search_results = self.vector_store.search_vectors(
                    query_text=query,
                    companies=companies,
                    top_k=top_k,
                    filing_types=filing_types,
                    date_range=date_range,
                    sections=sections,
                    keywords=keywords,
                    hybrid_search_weight=hybrid_search_weight
                )
            except Exception as e:
                logger.error(f"Error executing semantic search: {str(e)}")
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message=f"Error executing semantic search: {str(e)}"
                )

            # Check if we have results
            if not search_results:
                return self.format_error_response(
                    query_type=query_type,
                    parameters=parameters,
                    error_message=f"No results found for query: {query}",
                    error_type="warning"
                )

            # Format results
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "id": result.get("id", ""),
                    "score": result.get("score", 0.0),
                    "text": result.get("text", ""),
                    "metadata": {
                        "company": result.get("metadata", {}).get("company", ""),
                        "ticker": result.get("metadata", {}).get("ticker", ""),
                        "filing_type": result.get("metadata", {}).get("filing_type", ""),
                        "filing_date": result.get("metadata", {}).get("filing_date", ""),
                        "section": result.get("metadata", {}).get("section", ""),
                        "section_type": result.get("metadata", {}).get("section_type", "")
                    }
                }
                formatted_results.append(formatted_result)

            # Create a custom result with additional fields
            result = self.format_success_response(
                query_type=query_type,
                parameters=parameters,
                results=formatted_results
            )

            # Add additional fields
            result["query"] = query
            result["total_results"] = len(formatted_results)
            result["companies"] = companies
            result["filing_types"] = filing_types
            result["date_range"] = date_range
            result["sections"] = sections

            return result

        except Exception as e:
            logger.error(f"Unexpected error executing semantic search: {str(e)}")
            return self.format_error_response(
                query_type=query_type,
                parameters=parameters,
                error_message=f"Unexpected error: {str(e)}"
            )

    def validate_args(
        self,
        query_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate the tool arguments.

        Args:
            query_type: Type of query to execute
            parameters: Parameters for the query

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
