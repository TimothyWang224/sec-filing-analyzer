"""
Script to synchronize storage systems
"""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sec_filing_analyzer.storage.sync_manager import StorageSyncManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run the storage synchronization process."""
    try:
        # Create sync manager
        sync_manager = StorageSyncManager()

        # Sync all storage systems
        logger.info("Starting storage synchronization...")
        results = sync_manager.sync_all()

        # Update filing paths
        logger.info("Updating filing paths...")
        path_results = sync_manager.update_filing_paths()

        # Update processing status
        logger.info("Updating processing status...")
        status_results = sync_manager.update_processing_status()

        # Get inventory summary
        logger.info("Getting inventory summary...")
        summary = sync_manager.get_inventory_summary()

        # Print results
        logger.info("Synchronization completed successfully")
        logger.info(
            f"Vector store: Found {results['vector_store']['found']}, Added {results['vector_store']['added']}, Updated {results['vector_store']['updated']}, Errors {results['vector_store']['errors']}"
        )
        logger.info(
            f"File system: Found {results['file_system']['found']}, Added {results['file_system']['added']}, Updated {results['file_system']['updated']}, Errors {results['file_system']['errors']}"
        )
        logger.info(
            f"Filing paths: Updated {path_results['updated']}, Not found {path_results['not_found']}, Errors {path_results['errors']}"
        )
        logger.info(f"Processing status: Updated {status_results['updated']}, Errors {status_results['errors']}")
        logger.info(f"Total filings: {summary['total_filings']}")

        # Print status counts
        logger.info("Filing status counts:")
        for status in summary["status_counts"]:
            logger.info(f"  {status['processing_status']}: {status['count']}")

        # Print company counts
        logger.info("Top companies:")
        for company in summary["company_counts"][:5]:
            logger.info(f"  {company['ticker']}: {company['count']}")

        # Print filing type counts
        logger.info("Filing types:")
        for filing_type in summary["type_counts"]:
            logger.info(f"  {filing_type['filing_type']}: {filing_type['count']}")

        # Close connection
        sync_manager.close()

        return 0

    except Exception as e:
        logger.error(f"Error running storage synchronization: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
