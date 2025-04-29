"""
Vector Search Tool for SEC Filing Analyzer.

This module provides a tool for searching SEC filings using vector embeddings.
"""

import logging
from typing import Any, Dict, List, Optional

from src.tools.sec_semantic_search import SECSemanticSearchTool

logger = logging.getLogger(__name__)


class VectorSearchTool(SECSemanticSearchTool):
    """
    Tool for searching SEC filings using vector embeddings.

    This tool allows searching for information in SEC filings using natural language queries.
    It uses vector embeddings to find semantically similar content.
    """

    def __init__(self, vector_store_path: Optional[str] = None):
        """
        Initialize the VectorSearchTool.

        Args:
            vector_store_path: Optional path to the vector store
        """
        super().__init__(vector_store_path=vector_store_path)

    def search(self, query: str, companies: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for information in SEC filings.

        Args:
            query: The search query
            companies: Optional list of company tickers to filter by
            top_k: Number of results to return

        Returns:
            List of search results
        """
        logger.info(f"Searching for: {query}")

        # Use the underlying SECSemanticSearchTool to perform the search
        parameters = {
            "query_type": "semantic_search",
            "parameters": {
                "query": query,
                "companies": companies,
                "top_k": top_k,
            },
        }

        results = self.execute(**parameters)

        return results
