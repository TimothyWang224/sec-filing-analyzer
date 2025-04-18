"""
Data Lifecycle Manager for SEC Filing Analyzer

This module provides functionality to manage the lifecycle of data across different storage systems:
- DuckDB (relational database)
- Vector Store (embeddings)
- File System (raw filings, processed filings, etc.)
- Neo4j (graph database)
"""

import os
import logging
import duckdb
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set

import numpy as np

from .sync_manager import StorageSyncManager
from ..config import ConfigProvider, StorageConfig, ETLConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLifecycleManager:
    """
    Manager for the lifecycle of data across different storage systems.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        filings_dir: Optional[str] = None,
        graph_store_dir: Optional[str] = None
    ):
        """
        Initialize the data lifecycle manager.

        Args:
            db_path: Path to the DuckDB database
            vector_store_path: Path to the vector store
            filings_dir: Path to the filings directory
            graph_store_dir: Path to the graph store directory
        """
        # Initialize configuration
        ConfigProvider.initialize()
        storage_config = ConfigProvider.get_config(StorageConfig)
        etl_config = ConfigProvider.get_config(ETLConfig)

        # Get DuckDB path from ETLConfig, not StorageConfig
        self.db_path = db_path or etl_config.db_path or "data/financial_data.duckdb"
        self.vector_store_path = Path(vector_store_path or storage_config.vector_store_path or "data/vector_store")
        self.filings_dir = Path(filings_dir or etl_config.filings_dir or "data/filings")
        self.graph_store_dir = Path(graph_store_dir or "data/graph_store")

        # Initialize sync manager
        self.sync_manager = StorageSyncManager(
            db_path=self.db_path,
            vector_store_path=str(self.vector_store_path),
            filings_dir=str(self.filings_dir),
            graph_store_dir=str(self.graph_store_dir)
        )

        # Initialize DuckDB connection
        self._init_db_connection()

    def _init_db_connection(self):
        """Initialize the DuckDB connection."""
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to DuckDB at {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to DuckDB: {e}")
            raise

    def close(self):
        """Close the DuckDB connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("Closed DuckDB connection")

        # Close sync manager
        self.sync_manager.close()

    def get_filing_info(self, accession_number: str) -> Dict[str, Any]:
        """
        Get information about a filing from DuckDB.

        Args:
            accession_number: SEC accession number

        Returns:
            Dictionary with filing information
        """
        try:
            # Get filing from DuckDB
            filing = self.conn.execute(
                "SELECT * FROM filings WHERE accession_number = ?",
                [accession_number]
            ).fetchone()

            if not filing:
                logger.warning(f"Filing {accession_number} not found in DuckDB")
                return {"error": f"Filing {accession_number} not found in DuckDB"}

            # Convert to dictionary
            filing_dict = {
                "id": filing[0],
                "ticker": filing[1],
                "accession_number": filing[2],
                "filing_type": filing[3],
                "filing_date": str(filing[4]),
                "document_url": filing[5],
                "local_file_path": filing[6],
                "processing_status": filing[7],
                "last_updated": str(filing[8])
            }

            # Get vector store information
            vector_store_info = self._get_vector_store_info(filing_dict["ticker"], accession_number)
            filing_dict["vector_store"] = vector_store_info

            # Get file system information
            file_system_info = self._get_file_system_info(filing_dict["ticker"], accession_number)
            filing_dict["file_system"] = file_system_info

            return filing_dict

        except Exception as e:
            logger.error(f"Error getting filing info for {accession_number}: {e}")
            return {"error": str(e)}

    def _get_vector_store_info(self, ticker: str, accession_number: str) -> Dict[str, Any]:
        """
        Get information about a filing in the vector store.

        Args:
            ticker: Company ticker
            accession_number: SEC accession number

        Returns:
            Dictionary with vector store information
        """
        try:
            # Check if document embedding exists
            doc_embedding_path = self.vector_store_path / "by_company" / ticker / f"{accession_number}.npy"
            doc_embedding_exists = doc_embedding_path.exists()

            # Check if chunk embeddings exist
            chunk_embedding_pattern = f"{accession_number}_chunk_*.npy"
            chunk_embedding_dir = self.vector_store_path / "by_company" / ticker
            chunk_embeddings = list(chunk_embedding_dir.glob(chunk_embedding_pattern))

            # Check if metadata exists
            metadata_path = self.vector_store_path / "metadata" / f"{accession_number}.json"
            metadata_exists = metadata_path.exists()

            # Get metadata if it exists
            metadata = None
            if metadata_exists:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

            return {
                "document_embedding_exists": doc_embedding_exists,
                "document_embedding_path": str(doc_embedding_path) if doc_embedding_exists else None,
                "chunk_embeddings_count": len(chunk_embeddings),
                "chunk_embeddings_paths": [str(path) for path in chunk_embeddings[:5]],  # First 5 for brevity
                "metadata_exists": metadata_exists,
                "metadata_path": str(metadata_path) if metadata_exists else None,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error getting vector store info for {ticker} {accession_number}: {e}")
            return {"error": str(e)}

    def _get_file_system_info(self, ticker: str, accession_number: str) -> Dict[str, Any]:
        """
        Get information about a filing in the file system.

        Args:
            ticker: Company ticker
            accession_number: SEC accession number

        Returns:
            Dictionary with file system information
        """
        try:
            # Check different filing directories
            file_info = {}

            for subdir in ["raw", "html", "processed", "xbrl"]:
                subdir_path = self.filings_dir / subdir
                if not subdir_path.exists():
                    file_info[subdir] = {"exists": False}
                    continue

                # Check company directory
                company_path = subdir_path / ticker
                if not company_path.exists():
                    file_info[subdir] = {"exists": False}
                    continue

                # Search for files
                files = []

                # Try direct match
                for ext in ["html", "txt", "xml", "json"]:
                    file_path = company_path / f"{accession_number}.{ext}"
                    if file_path.exists():
                        files.append(str(file_path))

                # Try in year subdirectories
                for year_dir in company_path.iterdir():
                    if year_dir.is_dir():
                        for ext in ["html", "txt", "xml", "json"]:
                            file_path = year_dir / f"{accession_number}.{ext}"
                            if file_path.exists():
                                files.append(str(file_path))

                # Try with ticker prefix
                for ext in ["html", "txt", "xml", "json"]:
                    file_path = company_path / f"{ticker}_{accession_number}.{ext}"
                    if file_path.exists():
                        files.append(str(file_path))

                # Try in year subdirectories with ticker prefix
                for year_dir in company_path.iterdir():
                    if year_dir.is_dir():
                        for ext in ["html", "txt", "xml", "json"]:
                            file_path = year_dir / f"{ticker}_{accession_number}.{ext}"
                            if file_path.exists():
                                files.append(str(file_path))

                file_info[subdir] = {
                    "exists": len(files) > 0,
                    "files": files
                }

            return file_info

        except Exception as e:
            logger.error(f"Error getting file system info for {ticker} {accession_number}: {e}")
            return {"error": str(e)}

    def delete_filing(self, accession_number: str, dry_run: bool = True) -> Dict[str, Any]:
        """
        Delete a filing from all storage systems.

        Args:
            accession_number: SEC accession number
            dry_run: If True, only simulate deletion

        Returns:
            Dictionary with deletion results
        """
        try:
            # Get filing info
            filing_info = self.get_filing_info(accession_number)

            if "error" in filing_info:
                return filing_info

            # Initialize results
            results = {
                "duckdb": {"status": "pending", "details": {}},
                "vector_store": {"status": "pending", "details": {}},
                "file_system": {"status": "pending", "details": {}},
                "dry_run": dry_run
            }

            # Delete from DuckDB
            duckdb_result = self._delete_from_duckdb(accession_number, dry_run)
            results["duckdb"] = duckdb_result

            # Delete from vector store
            vector_store_result = self._delete_from_vector_store(
                filing_info["ticker"],
                accession_number,
                dry_run
            )
            results["vector_store"] = vector_store_result

            # Delete from file system
            file_system_result = self._delete_from_file_system(
                filing_info["ticker"],
                accession_number,
                dry_run
            )
            results["file_system"] = file_system_result

            # Overall status
            if all(r["status"] == "success" for r in [duckdb_result, vector_store_result, file_system_result]):
                results["status"] = "success"
            elif any(r["status"] == "error" for r in [duckdb_result, vector_store_result, file_system_result]):
                results["status"] = "error"
            else:
                results["status"] = "partial"

            return results

        except Exception as e:
            logger.error(f"Error deleting filing {accession_number}: {e}")
            return {"status": "error", "error": str(e), "dry_run": dry_run}

    def _delete_from_duckdb(self, accession_number: str, dry_run: bool = True) -> Dict[str, Any]:
        """
        Delete a filing from DuckDB.

        Args:
            accession_number: SEC accession number
            dry_run: If True, only simulate deletion

        Returns:
            Dictionary with deletion results
        """
        try:
            # Get filing from DuckDB
            filing = self.conn.execute(
                "SELECT id, ticker FROM filings WHERE accession_number = ?",
                [accession_number]
            ).fetchone()

            if not filing:
                logger.warning(f"Filing {accession_number} not found in DuckDB")
                return {"status": "not_found", "details": {}}

            filing_id = filing[0]
            ticker = filing[1]

            # Check for related data
            financial_facts_count = self.conn.execute(
                "SELECT COUNT(*) FROM financial_facts WHERE filing_id = ?",
                [filing_id]
            ).fetchone()[0]

            # If not a dry run, delete the filing
            if not dry_run:
                # Delete related data first
                if financial_facts_count > 0:
                    self.conn.execute(
                        "DELETE FROM financial_facts WHERE filing_id = ?",
                        [filing_id]
                    )

                # Delete the filing
                self.conn.execute(
                    "DELETE FROM filings WHERE accession_number = ?",
                    [accession_number]
                )

                logger.info(f"Deleted filing {accession_number} from DuckDB")

            return {
                "status": "success",
                "details": {
                    "filing_id": filing_id,
                    "ticker": ticker,
                    "financial_facts_count": financial_facts_count,
                    "deleted": not dry_run
                }
            }

        except Exception as e:
            logger.error(f"Error deleting filing {accession_number} from DuckDB: {e}")
            return {"status": "error", "error": str(e)}

    def _delete_from_vector_store(self, ticker: str, accession_number: str, dry_run: bool = True) -> Dict[str, Any]:
        """
        Delete a filing from the vector store.

        Args:
            ticker: Company ticker
            accession_number: SEC accession number
            dry_run: If True, only simulate deletion

        Returns:
            Dictionary with deletion results
        """
        try:
            # Get vector store info
            vector_store_info = self._get_vector_store_info(ticker, accession_number)

            if "error" in vector_store_info:
                return {"status": "error", "error": vector_store_info["error"]}

            # Files to delete
            files_to_delete = []

            # Document embedding
            if vector_store_info["document_embedding_exists"]:
                files_to_delete.append(vector_store_info["document_embedding_path"])

            # Chunk embeddings
            files_to_delete.extend(vector_store_info["chunk_embeddings_paths"])

            # Metadata
            if vector_store_info["metadata_exists"]:
                files_to_delete.append(vector_store_info["metadata_path"])

            # If not a dry run, delete the files
            if not dry_run:
                for file_path in files_to_delete:
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Deleted file {file_path}")

            return {
                "status": "success",
                "details": {
                    "files_deleted": len(files_to_delete),
                    "files": files_to_delete,
                    "deleted": not dry_run
                }
            }

        except Exception as e:
            logger.error(f"Error deleting filing {accession_number} from vector store: {e}")
            return {"status": "error", "error": str(e)}

    def _delete_from_file_system(self, ticker: str, accession_number: str, dry_run: bool = True) -> Dict[str, Any]:
        """
        Delete a filing from the file system.

        Args:
            ticker: Company ticker
            accession_number: SEC accession number
            dry_run: If True, only simulate deletion

        Returns:
            Dictionary with deletion results
        """
        try:
            # Get file system info
            file_system_info = self._get_file_system_info(ticker, accession_number)

            if "error" in file_system_info:
                return {"status": "error", "error": file_system_info["error"]}

            # Files to delete
            files_to_delete = []

            # Collect files from each subdirectory
            for subdir, info in file_system_info.items():
                if info.get("exists", False):
                    files_to_delete.extend(info.get("files", []))

            # If not a dry run, delete the files
            if not dry_run:
                for file_path in files_to_delete:
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Deleted file {file_path}")

            return {
                "status": "success",
                "details": {
                    "files_deleted": len(files_to_delete),
                    "files": files_to_delete,
                    "deleted": not dry_run
                }
            }

        except Exception as e:
            logger.error(f"Error deleting filing {accession_number} from file system: {e}")
            return {"status": "error", "error": str(e)}

    def get_company_filings(self, ticker: str) -> Dict[str, Any]:
        """
        Get all filings for a company.

        Args:
            ticker: Company ticker

        Returns:
            Dictionary with company filings
        """
        try:
            # Get filings from DuckDB
            filings = self.conn.execute(
                "SELECT * FROM filings WHERE ticker = ? ORDER BY filing_date DESC",
                [ticker]
            ).fetchdf()

            if filings.empty:
                logger.warning(f"No filings found for {ticker}")
                return {"ticker": ticker, "filings": []}

            # Convert to list of dictionaries
            filings_list = filings.to_dict(orient="records")

            return {"ticker": ticker, "filings": filings_list}

        except Exception as e:
            logger.error(f"Error getting filings for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    def get_filing_types(self, ticker: str) -> Dict[str, Any]:
        """
        Get all filing types for a company.

        Args:
            ticker: Company ticker

        Returns:
            Dictionary with filing types
        """
        try:
            # Get filing types from DuckDB
            filing_types = self.conn.execute(
                """
                SELECT filing_type, COUNT(*) as count
                FROM filings
                WHERE ticker = ?
                GROUP BY filing_type
                ORDER BY count DESC
                """,
                [ticker]
            ).fetchdf()

            if filing_types.empty:
                logger.warning(f"No filing types found for {ticker}")
                return {"ticker": ticker, "filing_types": []}

            # Convert to list of dictionaries
            filing_types_list = filing_types.to_dict(orient="records")

            return {"ticker": ticker, "filing_types": filing_types_list}

        except Exception as e:
            logger.error(f"Error getting filing types for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    def get_filing_dates(self, ticker: str, filing_type: str) -> Dict[str, Any]:
        """
        Get all filing dates for a company and filing type.

        Args:
            ticker: Company ticker
            filing_type: Filing type

        Returns:
            Dictionary with filing dates
        """
        try:
            # Get filing dates from DuckDB
            filing_dates = self.conn.execute(
                """
                SELECT filing_date, accession_number
                FROM filings
                WHERE ticker = ? AND filing_type = ?
                ORDER BY filing_date DESC
                """,
                [ticker, filing_type]
            ).fetchdf()

            if filing_dates.empty:
                logger.warning(f"No filing dates found for {ticker} {filing_type}")
                return {"ticker": ticker, "filing_type": filing_type, "filing_dates": []}

            # Convert to list of dictionaries
            filing_dates_list = filing_dates.to_dict(orient="records")

            return {"ticker": ticker, "filing_type": filing_type, "filing_dates": filing_dates_list}

        except Exception as e:
            logger.error(f"Error getting filing dates for {ticker} {filing_type}: {e}")
            return {"ticker": ticker, "filing_type": filing_type, "error": str(e)}


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Lifecycle Manager")
    parser.add_argument("--db-path", default="data/financial_data.duckdb", help="Path to DuckDB database")
    parser.add_argument("--vector-store-path", default="data/vector_store", help="Path to vector store")
    parser.add_argument("--filings-dir", default="data/filings", help="Path to filings directory")
    parser.add_argument("--graph-store-dir", default="data/graph_store", help="Path to graph store directory")
    parser.add_argument("--action", choices=["info", "delete"], required=True, help="Action to perform")
    parser.add_argument("--accession-number", help="SEC accession number")
    parser.add_argument("--ticker", help="Company ticker")
    parser.add_argument("--dry-run", action="store_true", help="Simulate deletion without actually deleting files")

    args = parser.parse_args()

    # Create lifecycle manager
    lifecycle_manager = DataLifecycleManager(
        db_path=args.db_path,
        vector_store_path=args.vector_store_path,
        filings_dir=args.filings_dir,
        graph_store_dir=args.graph_store_dir
    )

    try:
        # Perform action
        if args.action == "info":
            if args.accession_number:
                # Get filing info
                result = lifecycle_manager.get_filing_info(args.accession_number)
                print(json.dumps(result, indent=2))
            elif args.ticker:
                # Get company filings
                result = lifecycle_manager.get_company_filings(args.ticker)
                print(json.dumps(result, indent=2))
            else:
                print("Error: Either --accession-number or --ticker is required for info action")
        elif args.action == "delete":
            if args.accession_number:
                # Delete filing
                result = lifecycle_manager.delete_filing(args.accession_number, dry_run=args.dry_run)
                print(json.dumps(result, indent=2))
            else:
                print("Error: --accession-number is required for delete action")
    finally:
        # Close connection
        lifecycle_manager.close()
