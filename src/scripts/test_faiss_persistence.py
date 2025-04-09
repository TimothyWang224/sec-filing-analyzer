"""
Test script for FAISS index persistence.

This script demonstrates the persistence capabilities of the optimized vector store,
including saving and loading FAISS indexes from disk.
"""

import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from sec_filing_analyzer.storage import OptimizedVectorStore
from sec_filing_analyzer.config import StorageConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_index_persistence(store_path: str, companies: List[str], index_type: str = "flat"):
    """Test FAISS index persistence.
    
    Args:
        store_path: Path to the vector store
        companies: List of company tickers to test with
        index_type: Type of FAISS index to use
    """
    logger.info(f"Testing FAISS index persistence with {index_type} index for companies: {companies}")
    
    # Initialize vector store with specific index type
    vector_store = OptimizedVectorStore(
        store_path=store_path,
        index_type=index_type
    )
    
    # Check if index files exist
    index_path = vector_store._get_index_path(companies)
    mapping_path = vector_store._get_mapping_path(companies)
    
    logger.info(f"Index path: {index_path}")
    logger.info(f"Mapping path: {mapping_path}")
    
    if index_path.exists() and mapping_path.exists():
        logger.info("Index files already exist. Removing them for clean test.")
        os.remove(index_path)
        os.remove(mapping_path)
        
        # Also remove params file if it exists
        params_path = Path(str(index_path).replace(".index", ".params.json"))
        if params_path.exists():
            os.remove(params_path)
    
    # Test 1: First-time index creation and persistence
    logger.info("\nTest 1: First-time index creation and persistence")
    
    # Measure time to create and save index
    start_time = time.time()
    
    # Perform a search to trigger index creation
    results = vector_store.search_vectors(
        query_text="revenue growth",
        companies=companies,
        top_k=5
    )
    
    creation_time = time.time() - start_time
    
    logger.info(f"Index creation time: {creation_time:.2f} seconds")
    logger.info(f"Found {len(results)} results")
    
    # Verify index files were created
    if index_path.exists() and mapping_path.exists():
        logger.info("Index files were successfully created and saved to disk")
        
        # Get file sizes
        index_size = index_path.stat().st_size / (1024 * 1024)  # MB
        mapping_size = mapping_path.stat().st_size / 1024  # KB
        
        logger.info(f"Index file size: {index_size:.2f} MB")
        logger.info(f"Mapping file size: {mapping_size:.2f} KB")
    else:
        logger.error("Index files were not created")
    
    # Test 2: Loading existing index
    logger.info("\nTest 2: Loading existing index")
    
    # Create a new vector store instance
    new_vector_store = OptimizedVectorStore(
        store_path=store_path,
        index_type=index_type
    )
    
    # Measure time to load index
    start_time = time.time()
    
    # Perform a search to trigger index loading
    results = new_vector_store.search_vectors(
        query_text="revenue growth",
        companies=companies,
        top_k=5
    )
    
    load_time = time.time() - start_time
    
    logger.info(f"Index load time: {load_time:.2f} seconds")
    logger.info(f"Found {len(results)} results")
    logger.info(f"Speed improvement: {creation_time / load_time:.2f}x faster")
    
    # Test 3: Force rebuilding index
    logger.info("\nTest 3: Force rebuilding index")
    
    # Measure time to rebuild index
    start_time = time.time()
    
    # Perform a search with force_rebuild=True
    results = new_vector_store.search_vectors(
        query_text="revenue growth",
        companies=companies,
        top_k=5,
        force_rebuild=True
    )
    
    rebuild_time = time.time() - start_time
    
    logger.info(f"Index rebuild time: {rebuild_time:.2f} seconds")
    logger.info(f"Found {len(results)} results")
    
    # Test 4: Using rebuild_index method
    logger.info("\nTest 4: Using rebuild_index method")
    
    # Measure time to rebuild index
    start_time = time.time()
    
    # Rebuild index
    success = new_vector_store.rebuild_index(companies)
    
    rebuild_time = time.time() - start_time
    
    logger.info(f"Index rebuild time: {rebuild_time:.2f} seconds")
    logger.info(f"Rebuild success: {success}")
    
    # Summary
    logger.info("\nSummary:")
    logger.info(f"Index type: {index_type}")
    logger.info(f"Companies: {companies}")
    logger.info(f"Creation time: {creation_time:.2f} seconds")
    logger.info(f"Load time: {load_time:.2f} seconds")
    logger.info(f"Rebuild time: {rebuild_time:.2f} seconds")
    logger.info(f"Speed improvement (load vs create): {creation_time / load_time:.2f}x")

def main():
    """Main function to test FAISS index persistence."""
    # Get vector store path from config
    store_path = StorageConfig().vector_store_path
    
    # Test with different index types
    index_types = ["flat", "hnsw"]
    
    for index_type in index_types:
        # Test with a single company
        test_index_persistence(
            store_path=store_path,
            companies=["AAPL"],
            index_type=index_type
        )
        
        # Test with multiple companies
        test_index_persistence(
            store_path=store_path,
            companies=["AAPL", "NVDA"],
            index_type=index_type
        )

if __name__ == "__main__":
    main()
