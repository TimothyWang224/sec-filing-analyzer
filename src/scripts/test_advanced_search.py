"""
Test script for advanced search capabilities.

This script demonstrates the advanced search capabilities of the optimized vector store,
including different FAISS index types and advanced filtering options.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sec_filing_analyzer.config import StorageConfig
from sec_filing_analyzer.search import CoordinatedSearch
from sec_filing_analyzer.storage import GraphStore, OptimizedVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_faiss_index_types(store_path: str, query: str = "revenue growth and profitability"):
    """Test different FAISS index types.

    Args:
        store_path: Path to the vector store
        query: Query text to search for
    """
    logger.info("Testing different FAISS index types")

    # Test parameters
    index_types = ["flat", "ivf", "hnsw", "ivfpq"]
    companies = ["NVDA"]  # Use a single company for consistent comparison

    results = {}

    for index_type in index_types:
        logger.info(f"Testing {index_type} index")

        # Initialize vector store with specific index type
        vector_store = OptimizedVectorStore(store_path=store_path, index_type=index_type)

        # Perform search and measure time
        start_time = time.time()
        search_results = vector_store.search_vectors(query_text=query, companies=companies, top_k=5)
        search_time = time.time() - start_time

        # Store results
        results[index_type] = {
            "time": search_time,
            "count": len(search_results),
            "index_params": vector_store.index_params,
        }

        # Print results
        logger.info(f"  Search time: {search_time:.4f} seconds")
        logger.info(f"  Results: {len(search_results)}")
        logger.info(f"  Index parameters: {vector_store.index_params}")

        # Print top result
        if search_results:
            logger.info(f"  Top result: {search_results[0]['id']} (score: {search_results[0]['score']:.4f})")

        logger.info("")

    # Print summary
    logger.info("Summary of FAISS index types:")
    for index_type, data in results.items():
        logger.info(f"  {index_type}: {data['time']:.4f} seconds, {data['count']} results")


def test_advanced_filtering(store_path: str):
    """Test advanced filtering options.

    Args:
        store_path: Path to the vector store
    """
    logger.info("Testing advanced filtering options")

    # Initialize vector store
    vector_store = OptimizedVectorStore(
        store_path=store_path,
        index_type="flat",  # Use flat index for exact results
    )

    # Get available companies
    companies = list(vector_store.company_to_docs.keys())
    if "unknown" in companies:
        companies.remove("unknown")

    if not companies:
        logger.error("No companies found in the vector store")
        return

    logger.info(f"Available companies: {companies}")

    # 1. Test filing type filtering
    logger.info("\n1. Testing filing type filtering")
    filing_types = ["10-K", "10-Q"]

    for filing_type in filing_types:
        results = vector_store.search_vectors(
            query_text="financial performance",
            companies=companies[:1],  # Use first company
            top_k=5,
            filing_types=[filing_type],
        )

        logger.info(f"  {filing_type} results: {len(results)}")
        if results:
            for i, result in enumerate(results[:2]):  # Show top 2
                logger.info(f"    Result {i + 1}: {result['id']} - {result['metadata'].get('form', 'Unknown')}")

    # 2. Test date range filtering
    logger.info("\n2. Testing date range filtering")
    # Use a wide date range to ensure results
    date_ranges = [("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31")]

    for date_range in date_ranges:
        results = vector_store.search_vectors(
            query_text="financial performance",
            companies=companies[:1],  # Use first company
            top_k=5,
            date_range=date_range,
        )

        logger.info(f"  Date range {date_range} results: {len(results)}")
        if results:
            for i, result in enumerate(results[:2]):  # Show top 2
                logger.info(f"    Result {i + 1}: {result['id']} - {result['metadata'].get('filing_date', 'Unknown')}")

    # 3. Test keyword filtering
    logger.info("\n3. Testing keyword filtering")
    keywords_tests = [(["revenue", "growth"], "any"), (["revenue", "growth"], "all"), (["revenue growth"], "exact")]

    for keywords, match_type in keywords_tests:
        results = vector_store.search_vectors(
            query_text="financial performance",
            companies=companies[:1],  # Use first company
            top_k=5,
            keywords=keywords,
            keyword_match_type=match_type,
            hybrid_search_weight=0.5,  # Equal weight to vector and keyword
        )

        logger.info(f"  Keywords {keywords} ({match_type}) results: {len(results)}")
        if results:
            for i, result in enumerate(results[:2]):  # Show top 2
                logger.info(
                    f"    Result {i + 1}: {result['id']} - Vector: {result.get('vector_score', 0):.4f}, Keyword: {result.get('keyword_score', 0):.4f}"
                )

    # 4. Test sorting options
    logger.info("\n4. Testing sorting options")
    sort_options = ["relevance", "date", "company"]

    for sort_by in sort_options:
        results = vector_store.search_vectors(
            query_text="financial performance",
            companies=companies,  # Use all companies
            top_k=5,
            sort_by=sort_by,
        )

        logger.info(f"  Sorting by {sort_by} results: {len(results)}")
        if results:
            for i, result in enumerate(results[:3]):  # Show top 3
                if sort_by == "date":
                    sort_value = result["metadata"].get("filing_date", "Unknown")
                elif sort_by == "company":
                    sort_value = result["metadata"].get("ticker", "Unknown")
                else:
                    sort_value = f"{result['score']:.4f}"

                logger.info(f"    Result {i + 1}: {result['id']} - {sort_value}")


def test_coordinated_search(store_path: str):
    """Test coordinated search with graph integration.

    Args:
        store_path: Path to the vector store
    """
    logger.info("Testing coordinated search with graph integration")

    # Initialize vector store and graph store
    vector_store = OptimizedVectorStore(
        store_path=store_path,
        index_type="flat",  # Use flat index for exact results
    )

    graph_store = GraphStore(use_neo4j=True)

    # Initialize coordinated search
    search = CoordinatedSearch(vector_store=vector_store, graph_store=graph_store)

    # Get available companies
    companies = list(vector_store.company_to_docs.keys())
    if "unknown" in companies:
        companies.remove("unknown")

    if not companies:
        logger.error("No companies found in the vector store")
        return

    logger.info(f"Available companies: {companies}")

    # Test coordinated search
    results = search.search(
        query_text="risk factors related to supply chain",
        companies=companies[:1],  # Use first company
        top_k=3,
        include_related=True,
        filing_types=["10-K"],
        date_range=("2022-01-01", "2023-12-31"),
    )

    logger.info(f"Found {len(results['results'])} results in {results['performance']['total_time']:.4f} seconds")
    logger.info(f"Vector search time: {results['performance']['vector_search_time']:.4f} seconds")
    logger.info(f"Graph search time: {results['performance']['graph_search_time']:.4f} seconds")

    for i, result in enumerate(results["results"]):
        logger.info(f"Result {i + 1}:")
        logger.info(f"  ID: {result['id']}")
        logger.info(f"  Score: {result['score']:.4f}")
        logger.info(f"  Company: {result['metadata'].get('ticker', 'Unknown')}")
        logger.info(f"  Form: {result['metadata'].get('form', 'Unknown')}")
        logger.info(f"  Date: {result['metadata'].get('filing_date', 'Unknown')}")
        logger.info(f"  Text: {result['text'][:100]}...")

        # Print related documents
        if "related_documents" in result and result["related_documents"]:
            logger.info(f"  Related documents: {len(result['related_documents'])}")
            for j, related in enumerate(result["related_documents"][:3]):  # Show first 3
                logger.info(
                    f"    Related {j + 1}: {related.get('type', 'Unknown')} - {related.get('to_id', 'Unknown')}"
                )

        logger.info("")


def main():
    """Main function to test advanced search capabilities."""
    parser = argparse.ArgumentParser(description="Test advanced search capabilities")
    parser.add_argument(
        "--test", choices=["faiss", "filtering", "coordinated", "all"], default="all", help="Test to run (default: all)"
    )
    args = parser.parse_args()

    # Get vector store path from config
    store_path = StorageConfig().vector_store_path

    if args.test == "faiss" or args.test == "all":
        test_faiss_index_types(store_path)

    if args.test == "filtering" or args.test == "all":
        test_advanced_filtering(store_path)

    if args.test == "coordinated" or args.test == "all":
        test_coordinated_search(store_path)


if __name__ == "__main__":
    main()
