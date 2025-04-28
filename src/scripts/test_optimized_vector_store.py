"""
Test script for the optimized vector store.

This script demonstrates the optimized vector store with NumPy binary storage and FAISS.
It shows how to search for documents by company and measure performance.
"""

import logging
import time
from typing import List


from sec_filing_analyzer.config import StorageConfig
from sec_filing_analyzer.storage import OptimizedVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_search_performance(
    vector_store: OptimizedVectorStore, query: str, companies: List[str] = None
) -> None:
    """Test search performance for a query.

    Args:
        vector_store: The vector store to search
        query: The query text
        companies: Optional list of companies to search within
    """
    logger.info(f"Testing search performance for query: '{query}'")
    logger.info(f"Companies filter: {companies if companies else 'All'}")

    # Measure search time
    start_time = time.time()
    results = vector_store.search_vectors(query, companies=companies, top_k=5)
    end_time = time.time()

    search_time = end_time - start_time

    # Print results
    logger.info(f"Search completed in {search_time:.4f} seconds")
    logger.info(f"Found {len(results)} results")

    for i, result in enumerate(results):
        logger.info(f"Result {i + 1}:")
        logger.info(f"  ID: {result['id']}")
        logger.info(f"  Score: {result['score']:.4f}")
        logger.info(f"  Company: {result['metadata'].get('ticker', 'Unknown')}")
        logger.info(f"  Form: {result['metadata'].get('form', 'Unknown')}")
        logger.info(f"  Text: {result['text'][:100]}...")
        logger.info("")


def main():
    """Main function to test the optimized vector store."""
    # Initialize vector store
    vector_store = OptimizedVectorStore(store_path=StorageConfig().vector_store_path)

    # Get vector store statistics
    stats = vector_store.get_stats()
    logger.info(f"Vector store statistics: {stats}")

    # List available companies
    companies = vector_store.list_companies()
    logger.info(f"Available companies: {companies}")

    # Test search performance for different scenarios

    # 1. Search across all companies
    test_search_performance(vector_store, "revenue growth and profitability")

    # 2. Search within a specific company
    if companies:
        test_search_performance(
            vector_store, "revenue growth and profitability", [companies[0]]
        )

    # 3. Search with a more specific query
    test_search_performance(vector_store, "impact of inflation on operating expenses")

    # 4. Search for risk factors
    test_search_performance(vector_store, "risk factors related to supply chain")

    # 5. Search within multiple companies
    if len(companies) >= 2:
        test_search_performance(vector_store, "competitive landscape", companies[:2])


if __name__ == "__main__":
    main()
