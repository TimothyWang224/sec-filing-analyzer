"""
DuckDB-based metadata store for efficient document metadata storage and retrieval.

This module provides a DuckDB-based implementation for storing and retrieving
document metadata, offering significant performance improvements over file-based
approaches, especially for large collections of documents.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import duckdb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DuckDBMetadataStore:
    """DuckDB-based metadata store for efficient document metadata storage and retrieval."""

    def __init__(
        self,
        db_path: Union[str, Path],
        read_only: bool = True,
        create_if_missing: bool = True,
    ):
        """Initialize the DuckDB metadata store.

        Args:
            db_path: Path to the DuckDB database file
            read_only: Whether to open the database in read-only mode
            create_if_missing: Whether to create the database if it doesn't exist
        """
        self.db_path = Path(db_path)
        self.read_only = read_only
        self.create_if_missing = create_if_missing
        self.conn = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the DuckDB database connection and schema."""
        # Create parent directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if database exists
        db_exists = self.db_path.exists()

        if not db_exists and not self.create_if_missing:
            raise FileNotFoundError(f"Database file {self.db_path} does not exist")

        # Connect to the database
        try:
            self.conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
            logger.info(f"Connected to DuckDB at {self.db_path}")

            # Create schema if needed
            if not db_exists or not self.read_only:
                self._create_schema()
        except Exception as e:
            logger.error(f"Error connecting to DuckDB: {e}")
            raise

    def _create_schema(self) -> None:
        """Create the database schema if it doesn't exist."""
        if self.read_only:
            logger.warning("Cannot create schema in read-only mode")
            return

        try:
            # Create document metadata table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    doc_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    filing_type TEXT,
                    filing_date DATE,
                    accession_number TEXT,
                    company_name TEXT,
                    title TEXT,
                    content_type TEXT,
                    section TEXT,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for common query patterns
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_metadata_ticker
                ON document_metadata(ticker)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_metadata_filing_type
                ON document_metadata(filing_type)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_metadata_filing_date
                ON document_metadata(filing_date)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_metadata_accession_number
                ON document_metadata(accession_number)
            """)

            logger.info("Created DuckDB schema for document metadata")
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Closed DuckDB connection")

    def __del__(self) -> None:
        """Destructor to ensure the database connection is closed."""
        self.close()

    def import_metadata_from_files(self, metadata_dir: Path) -> int:
        """Import metadata from JSON files into the DuckDB database.

        Args:
            metadata_dir: Directory containing metadata JSON files

        Returns:
            Number of documents imported
        """
        if self.read_only:
            raise ValueError("Cannot import metadata in read-only mode")

        if not metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory {metadata_dir} does not exist")

        start_time = time.time()
        count = 0

        try:
            # Begin transaction
            self.conn.execute("BEGIN TRANSACTION")

            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    doc_id = metadata_file.stem

                    # Try to use orjson for faster parsing if available
                    try:
                        import orjson

                        with open(metadata_file, "rb") as f:
                            metadata = orjson.loads(f.read())
                    except ImportError:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)

                    # Extract common fields
                    ticker = metadata.get("ticker", "unknown")
                    filing_type = metadata.get("filing_type")
                    filing_date = metadata.get("filing_date")
                    accession_number = metadata.get("accession_number")
                    company_name = metadata.get("company_name")
                    title = metadata.get("title")
                    content_type = metadata.get("content_type")
                    section = metadata.get("section")

                    # Store the full metadata as JSON
                    metadata_json = json.dumps(metadata)

                    # Insert into database
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO document_metadata (
                            doc_id, ticker, filing_type, filing_date, accession_number,
                            company_name, title, content_type, section, metadata_json,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            doc_id,
                            ticker,
                            filing_type,
                            filing_date,
                            accession_number,
                            company_name,
                            title,
                            content_type,
                            section,
                            metadata_json,
                            datetime.now(),
                        ),
                    )

                    count += 1

                    # Log progress periodically
                    if count % 1000 == 0:
                        logger.info(f"Imported {count} documents...")

                except Exception as e:
                    logger.warning(
                        f"Error importing metadata from {metadata_file}: {e}"
                    )

            # Commit transaction
            self.conn.execute("COMMIT")

            duration = time.time() - start_time
            logger.info(f"Imported {count} documents in {duration:.2f} seconds")

            return count
        except Exception as e:
            # Rollback transaction on error
            self.conn.execute("ROLLBACK")
            logger.error(f"Error importing metadata: {e}")
            raise

    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a document.

        Args:
            doc_id: Document ID

        Returns:
            Metadata dictionary or None if not found
        """
        try:
            result = self.conn.execute(
                """
                SELECT metadata_json FROM document_metadata WHERE doc_id = ?
            """,
                (doc_id,),
            ).fetchone()

            if result and result[0]:
                return json.loads(result[0])

            return None
        except Exception as e:
            logger.warning(f"Error getting metadata for {doc_id}: {e}")
            return None

    def get_company_documents(self, ticker: str) -> Set[str]:
        """Get all document IDs for a company.

        Args:
            ticker: Company ticker symbol

        Returns:
            Set of document IDs
        """
        try:
            if ticker == "all":
                results = self.conn.execute("""
                    SELECT doc_id FROM document_metadata
                """).fetchall()
            else:
                results = self.conn.execute(
                    """
                    SELECT doc_id FROM document_metadata WHERE ticker = ?
                """,
                    (ticker,),
                ).fetchall()

            return {row[0] for row in results}
        except Exception as e:
            logger.warning(f"Error getting documents for company {ticker}: {e}")
            return set()

    def build_company_to_docs_mapping(self) -> Dict[str, Set[str]]:
        """Build a mapping from companies to document IDs.

        Returns:
            Dictionary mapping company tickers to sets of document IDs
        """
        start_time = time.time()

        try:
            # Get all unique tickers
            tickers = self.conn.execute("""
                SELECT DISTINCT ticker FROM document_metadata
            """).fetchall()

            # Build mapping
            company_to_docs = {}

            # Add "all" category
            all_docs = self.get_company_documents("all")
            company_to_docs["all"] = all_docs

            # Add company-specific mappings
            for row in tickers:
                ticker = row[0]
                if ticker != "unknown":
                    company_to_docs[ticker] = self.get_company_documents(ticker)

            logger.info(
                f"Built company-to-documents mapping for {len(company_to_docs)} companies in {time.time() - start_time:.2f} seconds"
            )
            return company_to_docs
        except Exception as e:
            logger.warning(f"Error building company-to-documents mapping: {e}")
            return {"all": set()}

    def get_document_count(self) -> int:
        """Get the total number of documents in the database.

        Returns:
            Number of documents
        """
        try:
            result = self.conn.execute("""
                SELECT COUNT(*) FROM document_metadata
            """).fetchone()

            return result[0] if result else 0
        except Exception as e:
            logger.warning(f"Error getting document count: {e}")
            return 0

    def get_company_count(self) -> int:
        """Get the number of unique companies in the database.

        Returns:
            Number of companies
        """
        try:
            result = self.conn.execute("""
                SELECT COUNT(DISTINCT ticker) FROM document_metadata WHERE ticker != 'unknown'
            """).fetchone()

            return result[0] if result else 0
        except Exception as e:
            logger.warning(f"Error getting company count: {e}")
            return 0

    def get_documents_by_query(
        self, query: str, params: Tuple = ()
    ) -> List[Dict[str, Any]]:
        """Get documents matching a custom SQL query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of document metadata dictionaries
        """
        try:
            results = self.conn.execute(query, params).fetchall()

            documents = []
            for row in results:
                if row[0]:  # metadata_json column
                    documents.append(json.loads(row[0]))

            return documents
        except Exception as e:
            logger.warning(f"Error executing query: {e}")
            return []

    def get_metadata_migration_script(self) -> str:
        """Get a script for migrating from file-based metadata to DuckDB.

        Returns:
            Python script as a string
        """
        script = """
import json
import logging
import time
from pathlib import Path

from sec_filing_analyzer.storage.duckdb_metadata_store import DuckDBMetadataStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_metadata_to_duckdb(metadata_dir, db_path):
    \"\"\"Migrate metadata from JSON files to DuckDB.\"\"\"
    start_time = time.time()

    # Initialize DuckDB store in write mode
    db_store = DuckDBMetadataStore(db_path, read_only=False, create_if_missing=True)

    try:
        # Import metadata
        count = db_store.import_metadata_from_files(Path(metadata_dir))

        duration = time.time() - start_time
        logger.info(f"Migration completed: {count} documents migrated in {duration:.2f} seconds")
    finally:
        # Close connection
        db_store.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate metadata from JSON files to DuckDB")
    parser.add_argument("--metadata-dir", required=True, help="Directory containing metadata JSON files")
    parser.add_argument("--db-path", required=True, help="Path to the DuckDB database file")

    args = parser.parse_args()

    migrate_metadata_to_duckdb(args.metadata_dir, args.db_path)
"""
        return script
