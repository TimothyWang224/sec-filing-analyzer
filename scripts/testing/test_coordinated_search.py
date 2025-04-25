"""
Test script for coordinated search.

This script demonstrates the coordinated search functionality that combines
the optimized vector store and graph store.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from sec_filing_analyzer.config import StorageConfig
from sec_filing_analyzer.search import CoordinatedSearch
from sec_filing_analyzer.storage import GraphStore, OptimizedVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_company_search(search: CoordinatedSearch, companies: List[str], query: str) -> None:
    """Test search for specific companies.

    Args:
        search: CoordinatedSearch instance
        companies: List of company tickers to search
        query: Query text
    """
    logger.info(f"Testing search for companies {companies} with query: '{query}'")

    # Perform search
    results = search.search(query_text=query, companies=companies, top_k=5, include_related=True)

    # Print results
    logger.info(f"Found {len(results['results'])} results in {results['performance']['total_time']:.4f} seconds")
    logger.info(f"Vector search time: {results['performance']['vector_search_time']:.4f} seconds")
    logger.info(f"Graph search time: {results['performance']['graph_search_time']:.4f} seconds")

    for i, result in enumerate(results["results"]):
        logger.info(f"Result {i + 1}:")
        logger.info(f"  ID: {result['id']}")
        logger.info(f"  Score: {result['score']:.4f}")
        logger.info(f"  Company: {result['metadata'].get('ticker', 'Unknown')}")
        logger.info(f"  Form: {result['metadata'].get('form', 'Unknown')}")
        logger.info(f"  Text: {result['text'][:100]}...")

        # Print related documents
        if "related_documents" in result and result["related_documents"]:
            logger.info(f"  Related documents: {len(result['related_documents'])}")
            for j, related in enumerate(result["related_documents"][:3]):  # Show first 3
                logger.info(
                    f"    Related {j + 1}: {related.get('type', 'Unknown')} - {related.get('to_id', 'Unknown')}"
                )

        logger.info("")


def test_filing_search(search: CoordinatedSearch, filing_id: str, query: str) -> None:
    """Test search within a specific filing.

    Args:
        search: CoordinatedSearch instance
        filing_id: Filing ID to search within
        query: Query text
    """
    logger.info(f"Testing search within filing {filing_id} with query: '{query}'")

    # Perform search
    results = search.search_within_filing(filing_id=filing_id, query_text=query, top_k=5)

    # Print results
    logger.info(f"Filing metadata: {results['filing_metadata']}")
    logger.info(f"Found {len(results['results'])} results")

    for i, result in enumerate(results["results"]):
        logger.info(f"Result {i + 1}:")
        logger.info(f"  ID: {result['id']}")
        logger.info(f"  Score: {result['score']:.4f}")
        logger.info(f"  Text: {result['text'][:100]}...")
        logger.info("")


def get_available_companies(search: CoordinatedSearch) -> List[str]:
    """Get available companies in the graph store.

    Args:
        search: CoordinatedSearch instance

    Returns:
        List of company tickers
    """
    # Query Neo4j for companies
    if search.graph_store.use_neo4j:
        try:
            with search.graph_store.driver.session(database=search.graph_store.database) as session:
                query = """
                MATCH (c:Company)
                RETURN c.ticker as ticker
                """
                result = session.run(query)
                return [record["ticker"] for record in result]
        except Exception as e:
            logger.error(f"Error getting companies from Neo4j: {e}")
            return []
    else:
        # For in-memory graph
        companies = []
        for node, attrs in search.graph_store.graph.nodes(data=True):
            if attrs.get("type") == "company" and "ticker" in attrs:
                companies.append(attrs["ticker"])
        return companies


def get_sample_filing_id(search: CoordinatedSearch, company: str = None) -> str:
    """Get a sample filing ID for testing.

    Args:
        search: CoordinatedSearch instance
        company: Optional company ticker

    Returns:
        Filing ID
    """
    companies = [company] if company else get_available_companies(search)
    if not companies:
        return None

    # Get filings for the first company
    filings = search.get_company_filings(companies=[companies[0]])
    if not filings:
        return None

    # Return the first filing ID
    return filings[0]["id"]


def main():
    """Main function to test coordinated search."""
    # Initialize vector and graph stores
    vector_store = OptimizedVectorStore(store_path=StorageConfig().vector_store_path)
    graph_store = GraphStore(use_neo4j=True)

    # Initialize coordinated search
    search = CoordinatedSearch(vector_store=vector_store, graph_store=graph_store)

    # Get available companies
    companies = get_available_companies(search)
    logger.info(f"Available companies: {companies}")

    if not companies:
        logger.error("No companies found in the graph store")
        return

    # Test company search
    test_company_search(
        search=search,
        companies=[companies[0]],  # Use the first company
        query="revenue growth and profitability",
    )

    # Test filing search
    filing_id = get_sample_filing_id(search, companies[0])
    if filing_id:
        test_filing_search(search=search, filing_id=filing_id, query="risk factors")
    else:
        logger.error("No filings found for testing")


if __name__ == "__main__":
    main()
