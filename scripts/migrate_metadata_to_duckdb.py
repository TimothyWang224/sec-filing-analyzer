#!/usr/bin/env python
"""
Migrate metadata from JSON files to DuckDB.

This script migrates document metadata from individual JSON files to a DuckDB database,
providing significant performance improvements for metadata access.

Usage:
    python migrate_metadata_to_duckdb.py --metadata-dir <metadata_dir> --db-path <db_path>

Example:
    python migrate_metadata_to_duckdb.py --metadata-dir data/vector_store/metadata --db-path data/vector_store/metadata.duckdb
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sec_filing_analyzer.storage.duckdb_metadata_store import DuckDBMetadataStore

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def migrate_metadata_to_duckdb(metadata_dir: str, db_path: str) -> None:
    """Migrate metadata from JSON files to DuckDB.

    Args:
        metadata_dir: Directory containing metadata JSON files
        db_path: Path to the DuckDB database file
    """
    start_time = time.time()

    # Initialize DuckDB store in write mode
    db_store = DuckDBMetadataStore(db_path, read_only=False, create_if_missing=True)

    try:
        # Import metadata
        count = db_store.import_metadata_from_files(Path(metadata_dir))

        # Verify migration
        doc_count = db_store.get_document_count()
        company_count = db_store.get_company_count()

        duration = time.time() - start_time
        logger.info(
            f"Migration completed: {count} documents migrated in {duration:.2f} seconds"
        )
        logger.info(
            f"Database contains {doc_count} documents from {company_count} companies"
        )
    finally:
        # Close connection
        db_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate metadata from JSON files to DuckDB"
    )
    parser.add_argument(
        "--metadata-dir", required=True, help="Directory containing metadata JSON files"
    )
    parser.add_argument(
        "--db-path", required=True, help="Path to the DuckDB database file"
    )

    args = parser.parse_args()

    migrate_metadata_to_duckdb(args.metadata_dir, args.db_path)
