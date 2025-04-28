"""
Enhanced Synchronization Manager for SEC Filing Analyzer

This module provides an enhanced version of the StorageSyncManager with better error handling
and the ability to continue synchronization even if some parts fail.
"""

import logging
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedStorageSyncManager:
    """
    Enhanced manager for synchronizing data between different storage systems.
    This class extends the functionality of StorageSyncManager with better error handling.
    """

    def __init__(
        self,
        db_path: str = "data/financial_data.duckdb",
        vector_store_path: str = "data/vector_store",
        filings_dir: str = "data/filings",
        graph_store_dir: str = "data/graph_store",
        read_only: bool = True,
    ):
        """
        Initialize the enhanced storage synchronization manager.

        Args:
            db_path: Path to the DuckDB database
            vector_store_path: Path to the vector store
            filings_dir: Path to the filings directory
            graph_store_dir: Path to the graph store directory
            read_only: Whether to open the database in read-only mode
        """
        from sec_filing_analyzer.storage.sync_manager import StorageSyncManager

        # Create the original sync manager
        self.sync_manager = StorageSyncManager(
            db_path=db_path,
            vector_store_path=vector_store_path,
            filings_dir=filings_dir,
            graph_store_dir=graph_store_dir,
            read_only=read_only,
        )

        # Store paths for reference
        self.db_path = db_path
        self.vector_store_path = Path(vector_store_path)
        self.filings_dir = Path(filings_dir)
        self.graph_store_dir = Path(graph_store_dir)

        # Get a reference to the database connection
        self.conn = self.sync_manager.conn

    def sync_all(self) -> Dict[str, Any]:
        """
        Synchronize all storage systems with enhanced error handling.
        If one part fails, the others will still be processed.

        Returns:
            Dictionary with synchronization results
        """
        results = {
            "vector_store": {"found": 0, "added": 0, "updated": 0, "errors": 0},
            "file_system": {"found": 0, "added": 0, "updated": 0, "errors": 0},
            "path_update": {"updated": 0, "not_found": 0, "errors": 0},
            "status_update": {"updated": 0, "errors": 0},
            "total_filings": 0,
            "overall_status": "success",
            "failed_components": [],
        }

        # Sync vector store
        try:
            vs_results = self.sync_manager.sync_vector_store()
            results["vector_store"] = vs_results
        except Exception as e:
            logger.error(f"Error synchronizing vector store: {e}")
            results["vector_store"]["errors"] += 1
            results["failed_components"].append("vector_store")

        # Sync file system
        try:
            fs_results = self.sync_manager.sync_file_system()
            results["file_system"] = fs_results
        except Exception as e:
            logger.error(f"Error synchronizing file system: {e}")
            results["file_system"]["errors"] += 1
            results["failed_components"].append("file_system")

        # Update filing paths
        try:
            path_results = self.sync_manager.update_filing_paths()
            results["path_update"] = path_results
        except Exception as e:
            logger.error(f"Error updating filing paths: {e}")
            results["path_update"]["errors"] += 1
            results["failed_components"].append("path_update")

        # Update processing status
        try:
            status_results = self.sync_manager.update_processing_status()
            results["status_update"] = status_results
        except Exception as e:
            logger.error(f"Error updating processing status: {e}")
            results["status_update"]["errors"] += 1
            results["failed_components"].append("status_update")

        # Get total filings count
        try:
            total = self.conn.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
            results["total_filings"] = total
        except Exception as e:
            logger.error(f"Error getting total filings count: {e}")
            results["failed_components"].append("total_count")

        # Set overall status
        if results["failed_components"]:
            results["overall_status"] = "partial_success"
            logger.warning(f"Sync completed with some failures: {results['failed_components']}")
        else:
            logger.info("Sync completed successfully")

        return results

    def sync_vector_store(self) -> Dict[str, int]:
        """
        Synchronize vector store with DuckDB.

        Returns:
            Dictionary with synchronization results
        """
        return self.sync_manager.sync_vector_store()

    def sync_file_system(self) -> Dict[str, int]:
        """
        Synchronize file system with DuckDB.

        Returns:
            Dictionary with synchronization results
        """
        return self.sync_manager.sync_file_system()

    def update_filing_paths(self) -> Dict[str, int]:
        """
        Update file paths for filings in DuckDB.

        Returns:
            Dictionary with update results
        """
        return self.sync_manager.update_filing_paths()

    def update_processing_status(self) -> Dict[str, int]:
        """
        Update processing status for filings in DuckDB.

        Returns:
            Dictionary with update results
        """
        return self.sync_manager.update_processing_status()

    def get_inventory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the filing inventory.

        Returns:
            Dictionary with inventory summary
        """
        return self.sync_manager.get_inventory_summary()

    def close(self):
        """Close connections."""
        self.sync_manager.close()
