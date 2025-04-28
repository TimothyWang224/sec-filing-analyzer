"""
Test script for delta index functionality.

This script demonstrates the delta index functionality of the optimized vector store,
including incremental updates and merging.
"""

import logging
import time
from typing import Any, Dict, List

from sec_filing_analyzer.config import StorageConfig
from sec_filing_analyzer.embeddings import EmbeddingGenerator
from sec_filing_analyzer.storage import OptimizedVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_documents(num_docs: int = 10, company: str = "TEST") -> List[Dict[str, Any]]:
    """Create test documents for delta index testing.

    Args:
        num_docs: Number of documents to create
        company: Company ticker for the documents

    Returns:
        List of document dictionaries
    """
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()

    # Create test documents
    documents = []

    for i in range(num_docs):
        # Create document text
        text = f"This is a test document {i} for company {company}. It contains information about revenue growth and profitability."

        # Generate embedding
        embedding = embedding_generator.generate_embeddings([text])[0]

        # Create document
        doc = {
            "id": f"{company}_DOC_{i}",
            "text": text,
            "embedding": embedding,
            "metadata": {
                "ticker": company,
                "form": "10-K" if i % 2 == 0 else "10-Q",
                "filing_date": f"2023-{(i % 12) + 1:02d}-01",
                "section": "MD&A" if i % 3 == 0 else "Risk Factors",
            },
        }

        documents.append(doc)

    return documents


def test_delta_index(store_path: str, index_type: str = "flat"):
    """Test delta index functionality.

    Args:
        store_path: Path to the vector store
        index_type: Type of FAISS index to use
    """
    logger.info(f"Testing delta index functionality with {index_type} index")

    # Initialize vector store with specific index type
    vector_store = OptimizedVectorStore(store_path=store_path, index_type=index_type)

    # Test 1: Create initial index
    logger.info("\nTest 1: Create initial index")

    # Create test documents for company A
    company_a_docs = create_test_documents(10, "TESTA")

    # Measure time to create initial index
    start_time = time.time()

    # Add documents to index
    success = vector_store.add_documents_to_index(company_a_docs, ["TESTA"])

    creation_time = time.time() - start_time

    logger.info(f"Initial index creation time: {creation_time:.2f} seconds")
    logger.info(f"Success: {success}")

    # Test 2: Search with delta index
    logger.info("\nTest 2: Search with delta index")

    # Perform a search
    results = vector_store.search_vectors(query_text="revenue growth", companies=["TESTA"], top_k=5)

    logger.info(f"Found {len(results)} results")
    for i, result in enumerate(results[:3]):  # Show top 3
        logger.info(f"  Result {i + 1}: {result['id']} - Score: {result['score']:.4f}")

    # Test 3: Add more documents to delta index
    logger.info("\nTest 3: Add more documents to delta index")

    # Create test documents for company B
    company_b_docs = create_test_documents(10, "TESTB")

    # Measure time to add documents
    start_time = time.time()

    # Add documents to index
    success = vector_store.add_documents_to_index(company_b_docs, ["TESTB"])

    update_time = time.time() - start_time

    logger.info(f"Delta index update time: {update_time:.2f} seconds")
    logger.info(f"Success: {success}")

    # Test 4: Search across both companies
    logger.info("\nTest 4: Search across both companies")

    # Perform a search
    results = vector_store.search_vectors(query_text="revenue growth", companies=["TESTA", "TESTB"], top_k=5)

    logger.info(f"Found {len(results)} results")
    for i, result in enumerate(results[:3]):  # Show top 3
        logger.info(
            f"  Result {i + 1}: {result['id']} - Score: {result['score']:.4f} - Company: {result['metadata'].get('ticker', 'Unknown')}"
        )

    # Test 5: Merge delta index
    logger.info("\nTest 5: Merge delta index")

    # Measure time to merge
    start_time = time.time()

    # Merge delta index
    success = vector_store.merge_delta_index()

    merge_time = time.time() - start_time

    logger.info(f"Delta index merge time: {merge_time:.2f} seconds")
    logger.info(f"Success: {success}")

    # Test 6: Search after merge
    logger.info("\nTest 6: Search after merge")

    # Perform a search
    results = vector_store.search_vectors(query_text="revenue growth", companies=["TESTA", "TESTB"], top_k=5)

    logger.info(f"Found {len(results)} results")
    for i, result in enumerate(results[:3]):  # Show top 3
        logger.info(
            f"  Result {i + 1}: {result['id']} - Score: {result['score']:.4f} - Company: {result['metadata'].get('ticker', 'Unknown')}"
        )

    # Test 7: Add more documents after merge
    logger.info("\nTest 7: Add more documents after merge")

    # Create test documents for company C
    company_c_docs = create_test_documents(10, "TESTC")

    # Measure time to add documents
    start_time = time.time()

    # Add documents to index
    success = vector_store.add_documents_to_index(company_c_docs, ["TESTC"])

    update_time = time.time() - start_time

    logger.info(f"Delta index update time: {update_time:.2f} seconds")
    logger.info(f"Success: {success}")

    # Test 8: Search across all companies
    logger.info("\nTest 8: Search across all companies")

    # Perform a search
    results = vector_store.search_vectors(query_text="revenue growth", companies=["TESTA", "TESTB", "TESTC"], top_k=5)

    logger.info(f"Found {len(results)} results")
    for i, result in enumerate(results[:3]):  # Show top 3
        logger.info(
            f"  Result {i + 1}: {result['id']} - Score: {result['score']:.4f} - Company: {result['metadata'].get('ticker', 'Unknown')}"
        )

    # Summary
    logger.info("\nSummary:")
    logger.info(f"Index type: {index_type}")
    logger.info(f"Initial index creation time: {creation_time:.2f} seconds")
    logger.info(f"Delta index update time: {update_time:.2f} seconds")
    logger.info(f"Delta index merge time: {merge_time:.2f} seconds")

    # Clean up test data
    logger.info("\nCleaning up test data...")

    # Remove test documents from vector store
    for company in ["TESTA", "TESTB", "TESTC"]:
        for i in range(10):
            doc_id = f"{company}_DOC_{i}"

            # Remove embedding file
            embedding_path = vector_store.embeddings_dir / f"{doc_id}.npy"
            if embedding_path.exists():
                embedding_path.unlink()

            # Remove text file
            text_path = vector_store.text_dir / f"{doc_id}.txt"
            if text_path.exists():
                text_path.unlink()

            # Remove metadata file
            metadata_path = vector_store.metadata_dir / f"{doc_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()

    # Remove index files
    for company in ["TESTA", "TESTB", "TESTC"]:
        for suffix in ["main", "delta"]:
            index_path = vector_store.index_dir / f"{company}_{index_type}_{suffix}.index"
            if index_path.exists():
                index_path.unlink()

            mapping_path = vector_store.index_dir / f"{company}_{index_type}_{suffix}.mapping.json"
            if mapping_path.exists():
                mapping_path.unlink()

            params_path = vector_store.index_dir / f"{company}_{index_type}_{suffix}.params.json"
            if params_path.exists():
                params_path.unlink()

    logger.info("Cleanup complete")


def main():
    """Main function to test delta index functionality."""
    # Get vector store path from config
    store_path = StorageConfig().vector_store_path

    # Test with different index types
    index_types = ["flat", "hnsw"]

    for index_type in index_types:
        test_delta_index(store_path, index_type)


if __name__ == "__main__":
    main()
