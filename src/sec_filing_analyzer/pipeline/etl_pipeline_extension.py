"""
Extension for the ETL pipeline to ensure proper tracking in DuckDB
"""

import logging
import os

# Import the DuckDB manager
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb

from ..config import ConfigProvider, ETLConfig, StorageConfig
from ..storage.sync_manager import StorageSyncManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.sec_filing_analyzer.utils.duckdb_manager import duckdb_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETLPipelineExtension:
    """
    Extension for the ETL pipeline to ensure proper tracking in DuckDB.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        filings_dir: Optional[str] = None,
        read_only: bool = True,
    ):
        """
        Initialize the ETL pipeline extension.

        Args:
            db_path: Path to the DuckDB database
            vector_store_path: Path to the vector store
            filings_dir: Path to the filings directory
            read_only: Whether to open the database in read-only mode
        """
        # Get DuckDB path from ETLConfig, not StorageConfig
        etl_config = ConfigProvider.get_config(ETLConfig)
        self.db_path = db_path or etl_config.db_path or "data/financial_data.duckdb"
        self.vector_store_path = vector_store_path or StorageConfig().vector_store_path or "data/vector_store"
        self.filings_dir = filings_dir or ETLConfig().filings_dir or "data/filings"
        self.read_only = read_only

        # Initialize sync manager with read-only mode
        self.sync_manager = StorageSyncManager(
            db_path=self.db_path,
            vector_store_path=self.vector_store_path,
            filings_dir=self.filings_dir,
            read_only=read_only,
        )

    def register_filing(
        self,
        ticker: str,
        filing_type: str,
        filing_date: str,
        accession_number: str,
        file_path: Optional[str] = None,
        processing_status: str = "downloaded",
        document_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a filing in DuckDB.

        Args:
            ticker: Company ticker
            filing_type: Filing type (e.g., 10-K, 10-Q)
            filing_date: Filing date (YYYY-MM-DD)
            accession_number: SEC accession number
            file_path: Path to the filing file
            processing_status: Processing status
            document_url: URL to the filing document

        Returns:
            Dictionary with registration results
        """
        try:
            # Ensure company exists
            self.sync_manager._ensure_company_exists(ticker)

            # Connect to DuckDB in read-write mode (required for registration)
            conn = duckdb_manager.get_read_write_connection(self.db_path)

            # Generate filing ID
            filing_id = f"{ticker}_{accession_number}"

            # Check if filing exists
            existing = conn.execute("SELECT id FROM filings WHERE accession_number = ?", [accession_number]).fetchone()

            if existing:
                # Update filing
                conn.execute(
                    """
                    UPDATE filings
                    SET ticker = ?, filing_type = ?, filing_date = ?,
                        local_file_path = ?, processing_status = ?, document_url = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE accession_number = ?
                    """,
                    [ticker, filing_type, filing_date, file_path, processing_status, document_url, accession_number],
                )
                logger.info(f"Updated filing {filing_id} in database")
                result = {"status": "updated", "filing_id": filing_id}
            else:
                # Add filing
                conn.execute(
                    """
                    INSERT INTO filings (
                        id, ticker, accession_number, filing_type, filing_date,
                        local_file_path, processing_status, document_url, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    [
                        filing_id,
                        ticker,
                        accession_number,
                        filing_type,
                        filing_date,
                        file_path,
                        processing_status,
                        document_url,
                    ],
                )
                logger.info(f"Added filing {filing_id} to database")
                result = {"status": "added", "filing_id": filing_id}

            # Close connection
            conn.close()

            return result

        except Exception as e:
            logger.error(f"Error registering filing {accession_number}: {e}")
            return {"status": "error", "error": str(e)}

    def update_filing_status(
        self, accession_number: str, processing_status: str, file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update the processing status of a filing.

        Args:
            accession_number: SEC accession number
            processing_status: New processing status
            file_path: Path to the filing file

        Returns:
            Dictionary with update results
        """
        try:
            # Connect to DuckDB in read-write mode (required for status updates)
            conn = duckdb_manager.get_read_write_connection(self.db_path)

            # Check if filing exists
            existing = conn.execute("SELECT id FROM filings WHERE accession_number = ?", [accession_number]).fetchone()

            if existing:
                # Update filing
                if file_path:
                    conn.execute(
                        """
                        UPDATE filings
                        SET processing_status = ?, local_file_path = ?, last_updated = CURRENT_TIMESTAMP
                        WHERE accession_number = ?
                        """,
                        [processing_status, file_path, accession_number],
                    )
                else:
                    conn.execute(
                        """
                        UPDATE filings
                        SET processing_status = ?, last_updated = CURRENT_TIMESTAMP
                        WHERE accession_number = ?
                        """,
                        [processing_status, accession_number],
                    )

                logger.info(f"Updated filing {existing[0]} status to {processing_status}")
                result = {"status": "updated", "filing_id": existing[0]}
            else:
                logger.warning(f"Filing {accession_number} not found in database")
                result = {"status": "not_found", "accession_number": accession_number}

            # Close connection
            conn.close()

            return result

        except Exception as e:
            logger.error(f"Error updating filing status {accession_number}: {e}")
            return {"status": "error", "error": str(e)}

    def sync_storage(self) -> Dict[str, Any]:
        """
        Synchronize storage systems.

        Returns:
            Dictionary with synchronization results
        """
        return self.sync_manager.sync_all()

    def close(self):
        """Close connections."""
        self.sync_manager.close()
