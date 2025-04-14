"""
SEC Semantic Search Tool

This module provides a tool for agents to perform semantic search on SEC filings
using the optimized vector store.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..tools.base import Tool
from sec_filing_analyzer.storage import OptimizedVectorStore
from sec_filing_analyzer.config import StorageConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECSemanticSearchTool(Tool):
    """Tool for performing semantic search on SEC filings.

    Performs semantic search on SEC filings to find relevant information based on natural language queries.
    Use this tool to search for specific topics, concepts, or information within SEC filings.
    """

    _tool_name = "sec_semantic_search"
    _tool_tags = ["sec", "semantic", "search"]
    _compact_description = "Search SEC filings using natural language queries"

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

    async def execute(
        self,
        query: str,
        companies: Optional[List[str]] = None,
        top_k: int = 5,
        filing_types: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
        sections: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        hybrid_search_weight: float = 0.5
    ) -> Dict[str, Any]:
        """
        Execute semantic search on SEC filings.

        Args:
            query: The search query text
            companies: Optional list of company tickers to search within
            top_k: Number of results to return
            filing_types: Optional list of filing types to filter by (e.g., ['10-K', '10-Q'])
            date_range: Optional tuple of (start_date, end_date) in format 'YYYY-MM-DD'
            sections: Optional list of document sections to filter by
            keywords: Optional list of keywords to search for
            hybrid_search_weight: Weight for hybrid search (0.0 = pure vector, 1.0 = pure keyword)

        Returns:
            Dictionary containing search results
        """
        try:
            print(f"SECSemanticSearchTool.execute: Executing semantic search: {query}")
            print(f"SECSemanticSearchTool.execute: Companies: {companies}")
            print(f"SECSemanticSearchTool.execute: Filing types: {filing_types}")
            print(f"SECSemanticSearchTool.execute: Date range: {date_range}")

            # Perform search
            print(f"SECSemanticSearchTool.execute: Vector store path: {self.vector_store_path}")
            print(f"SECSemanticSearchTool.execute: Vector store initialized: {self.vector_store is not None}")

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

            print(f"SECSemanticSearchTool.execute: Search results: {len(search_results) if search_results else 'None'}")

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

            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "companies": companies,
                "filing_types": filing_types,
                "date_range": date_range,
                "sections": sections
            }

        except Exception as e:
            logger.error(f"Error executing semantic search: {str(e)}")
            return {
                "error": str(e),
                "query": query,
                "results": [],
                "total_results": 0
            }

    def validate_args(
        self,
        query: str,
        companies: Optional[List[str]] = None,
        top_k: int = 5,
        filing_types: Optional[List[str]] = None,
        date_range: Optional[Tuple[str, str]] = None,
        sections: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        hybrid_search_weight: float = 0.5
    ) -> bool:
        """
        Validate the tool arguments.

        Args:
            query: The search query text
            companies: Optional list of company tickers to search within
            top_k: Number of results to return
            filing_types: Optional list of filing types to filter by
            date_range: Optional tuple of (start_date, end_date)
            sections: Optional list of document sections to filter by
            keywords: Optional list of keywords to search for
            hybrid_search_weight: Weight for hybrid search

        Returns:
            True if arguments are valid, False otherwise
        """
        # Validate query
        if not query or not isinstance(query, str):
            logger.error("Invalid query: must be a non-empty string")
            return False

        # Validate top_k
        if not isinstance(top_k, int) or top_k <= 0:
            logger.error("Invalid top_k: must be a positive integer")
            return False

        # Validate companies if provided
        if companies and not all(isinstance(company, str) for company in companies):
            logger.error("Invalid companies: must be a list of strings")
            return False

        # Validate filing_types if provided
        valid_filing_types = ["10-K", "10-Q", "8-K", "S-1", "S-4", "20-F", "40-F", "6-K"]
        if filing_types and not all(filing_type in valid_filing_types for filing_type in filing_types):
            logger.error(f"Invalid filing_types: must be in {valid_filing_types}")
            return False

        # Validate date_range if provided
        if date_range:
            try:
                if len(date_range) != 2:
                    logger.error("Invalid date_range: must be a tuple of (start_date, end_date)")
                    return False

                start_date, end_date = date_range
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")

                if start_date > end_date:
                    logger.error("Invalid date_range: start_date must be before end_date")
                    return False
            except ValueError:
                logger.error("Invalid date_range: dates must be in format 'YYYY-MM-DD'")
                return False

        # Validate hybrid_search_weight
        if not isinstance(hybrid_search_weight, (int, float)) or hybrid_search_weight < 0 or hybrid_search_weight > 1:
            logger.error("Invalid hybrid_search_weight: must be a float between 0 and 1")
            return False

        return True
