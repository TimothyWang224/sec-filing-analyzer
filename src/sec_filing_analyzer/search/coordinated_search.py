"""
Coordinated Search Module

This module provides coordinated search functionality that combines the optimized
vector store and graph store for more comprehensive search results.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import time

from ..storage import OptimizedVectorStore, GraphStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoordinatedSearch:
    """
    Coordinated search class that combines vector and graph search capabilities.
    """

    def __init__(
        self,
        vector_store: Optional[OptimizedVectorStore] = None,
        graph_store: Optional[GraphStore] = None
    ):
        """Initialize the coordinated search.

        Args:
            vector_store: Vector store for semantic search
            graph_store: Graph store for relationship search
        """
        self.vector_store = vector_store or OptimizedVectorStore()
        self.graph_store = graph_store or GraphStore()
        logger.info("Initialized coordinated search")

    def search(
        self,
        query_text: str,
        companies: Optional[List[str]] = None,
        filing_types: Optional[List[str]] = None,
        top_k: int = 5,
        include_related: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform a coordinated search across vector and graph stores.

        Args:
            query_text: Query text to search for
            companies: Optional list of company tickers to filter by
            filing_types: Optional list of filing types to filter by
            top_k: Number of results to return
            include_related: Whether to include related documents from graph
            metadata_filter: Optional additional metadata filter

        Returns:
            Dictionary containing search results and performance metrics
        """
        start_time = time.time()
        
        # 1. Vector search with company filter
        vector_start_time = time.time()
        vector_results = self.vector_store.search_vectors(
            query_text=query_text,
            companies=companies,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        vector_time = time.time() - vector_start_time
        
        # 2. Enhance results with graph information if requested
        enhanced_results = []
        graph_time = 0
        
        if include_related:
            graph_start_time = time.time()
            
            for result in vector_results:
                filing_id = result['id']
                
                # Get related documents from graph with same company filter
                related_docs = self.graph_store.get_filing_relationships(
                    filing_id=filing_id,
                    companies=companies
                )
                
                # Add to enhanced results
                enhanced_results.append({
                    **result,
                    "related_documents": related_docs
                })
                
            graph_time = time.time() - graph_start_time
        else:
            enhanced_results = vector_results
        
        # 3. Prepare final results
        total_time = time.time() - start_time
        
        return {
            "results": enhanced_results,
            "performance": {
                "total_time": total_time,
                "vector_search_time": vector_time,
                "graph_search_time": graph_time,
                "result_count": len(enhanced_results)
            },
            "filters": {
                "companies": companies,
                "filing_types": filing_types,
                "metadata": metadata_filter
            }
        }
    
    def get_company_filings(
        self,
        companies: List[str],
        filing_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get all filings for specified companies.

        Args:
            companies: List of company tickers
            filing_types: Optional list of filing types to filter by

        Returns:
            List of filing dictionaries
        """
        return self.graph_store.get_filings_by_companies(
            companies=companies,
            filing_types=filing_types
        )
    
    def search_within_filing(
        self,
        filing_id: str,
        query_text: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Search within a specific filing.

        Args:
            filing_id: Filing ID to search within
            query_text: Query text to search for
            top_k: Number of results to return

        Returns:
            Dictionary containing search results
        """
        # Get filing metadata from graph store
        filing_metadata = None
        try:
            # Try to get company information for the filing
            if self.graph_store.use_neo4j:
                with self.graph_store.driver.session(database=self.graph_store.database) as session:
                    query = """
                    MATCH (c:Company)-[:FILED]->(f:Filing {accession_number: $filing_id})
                    RETURN c.ticker as ticker, f.filing_type as filing_type
                    """
                    result = session.run(query, filing_id=filing_id)
                    records = list(result)
                    if records:
                        filing_metadata = dict(records[0])
            else:
                # For in-memory graph
                for node, attrs in self.graph_store.graph.nodes(data=True):
                    if attrs.get("type") == "filing" and node == filing_id:
                        filing_metadata = {
                            "ticker": attrs.get("ticker"),
                            "filing_type": attrs.get("filing_type")
                        }
                        break
        except Exception as e:
            logger.warning(f"Error getting filing metadata: {e}")
        
        # If we found company information, use it to filter the search
        companies = None
        if filing_metadata and "ticker" in filing_metadata:
            companies = [filing_metadata["ticker"]]
        
        # Search for chunks within this filing
        chunk_results = self.vector_store.search_vectors(
            query_text=query_text,
            companies=companies,
            top_k=top_k,
            metadata_filter={"original_doc_id": filing_id}
        )
        
        return {
            "filing_id": filing_id,
            "filing_metadata": filing_metadata,
            "results": chunk_results,
            "result_count": len(chunk_results)
        }
