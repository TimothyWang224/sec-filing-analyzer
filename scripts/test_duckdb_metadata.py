#!/usr/bin/env python
"""
Test the DuckDB metadata store.

This script tests the DuckDB metadata store by loading it and performing some basic operations.
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sec_filing_analyzer.storage.duckdb_metadata_store import DuckDBMetadataStore

# Path to the DuckDB database
db_path = "data/vector_store/metadata.duckdb"


def test_duckdb_metadata_store():
    """Test the DuckDB metadata store."""
    print(f"Testing DuckDB metadata store at {db_path}")

    # Start timer
    start_time = time.time()

    # Initialize DuckDB store
    db_store = DuckDBMetadataStore(db_path, read_only=True)

    # Get document count
    doc_count = db_store.get_document_count()
    print(f"Document count: {doc_count}")

    # Get company count
    company_count = db_store.get_company_count()
    print(f"Company count: {company_count}")

    # Build company-to-documents mapping
    mapping_start_time = time.time()
    company_to_docs = db_store.build_company_to_docs_mapping()
    mapping_time = time.time() - mapping_start_time
    print(f"Built company-to-documents mapping in {mapping_time:.2f} seconds")
    print(f"Companies: {list(company_to_docs.keys())}")

    # Get a sample document
    sample_doc_id = next(iter(next(iter(company_to_docs.values()))))
    print(f"Sample document ID: {sample_doc_id}")

    # Get metadata for the sample document
    metadata_start_time = time.time()
    metadata = db_store.get_metadata(sample_doc_id)
    metadata_time = time.time() - metadata_start_time
    print(f"Got metadata for {sample_doc_id} in {metadata_time:.2f} seconds")
    print(f"Metadata keys: {list(metadata.keys())}")

    # Close the connection
    db_store.close()

    # Total time
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    test_duckdb_metadata_store()
