"""
Test script for the Data Lifecycle Manager
"""

import json
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sec_filing_analyzer.config import ConfigProvider, ETLConfig, StorageConfig
from src.sec_filing_analyzer.storage.lifecycle_manager import DataLifecycleManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_lifecycle_manager():
    """Test the Data Lifecycle Manager."""
    try:
        # Initialize configuration
        ConfigProvider.initialize()
        storage_config = ConfigProvider.get_config(StorageConfig)
        etl_config = ConfigProvider.get_config(ETLConfig)

        # Create lifecycle manager
        lifecycle_manager = DataLifecycleManager(
            db_path=etl_config.db_path,
            vector_store_path=storage_config.vector_store_path,
            filings_dir=etl_config.filings_dir,
        )

        # Get company filings
        ticker = "AAPL"  # Change to a ticker in your database
        logger.info(f"Getting filings for {ticker}...")
        company_filings = lifecycle_manager.get_company_filings(ticker)

        if "error" in company_filings:
            logger.error(f"Error getting filings for {ticker}: {company_filings['error']}")
            return False

        # Print company filings
        logger.info(f"Found {len(company_filings['filings'])} filings for {ticker}")

        # Get filing info for the first filing
        if company_filings["filings"]:
            accession_number = company_filings["filings"][0]["accession_number"]
            logger.info(f"Getting info for filing {accession_number}...")
            filing_info = lifecycle_manager.get_filing_info(accession_number)

            if "error" in filing_info:
                logger.error(f"Error getting info for filing {accession_number}: {filing_info['error']}")
            else:
                # Print filing info
                logger.info(f"Filing info for {accession_number}:")
                logger.info(f"  Ticker: {filing_info['ticker']}")
                logger.info(f"  Filing Type: {filing_info['filing_type']}")
                logger.info(f"  Filing Date: {filing_info['filing_date']}")
                logger.info(f"  Processing Status: {filing_info['processing_status']}")

                # Print vector store info
                vector_store_info = filing_info["vector_store"]
                logger.info(f"  Vector Store Info:")
                logger.info(f"    Document Embedding Exists: {vector_store_info['document_embedding_exists']}")
                logger.info(f"    Chunk Embeddings Count: {vector_store_info['chunk_embeddings_count']}")

                # Print file system info
                file_system_info = filing_info["file_system"]
                logger.info(f"  File System Info:")
                for subdir, info in file_system_info.items():
                    if isinstance(info, dict) and info.get("exists", False):
                        logger.info(f"    {subdir}: {len(info.get('files', []))} files")

                # Test deletion (dry run)
                logger.info(f"Testing deletion of filing {accession_number} (dry run)...")
                deletion_result = lifecycle_manager.delete_filing(accession_number, dry_run=True)

                if "error" in deletion_result:
                    logger.error(f"Error deleting filing {accession_number}: {deletion_result['error']}")
                else:
                    # Print deletion result
                    logger.info(f"Deletion result for {accession_number} (dry run):")
                    logger.info(f"  Status: {deletion_result['status']}")
                    logger.info(f"  DuckDB: {deletion_result['duckdb']['status']}")
                    logger.info(f"  Vector Store: {deletion_result['vector_store']['status']}")
                    logger.info(f"  File System: {deletion_result['file_system']['status']}")

        # Close lifecycle manager
        lifecycle_manager.close()

        return True

    except Exception as e:
        logger.error(f"Error testing lifecycle manager: {e}")
        return False


if __name__ == "__main__":
    test_lifecycle_manager()
