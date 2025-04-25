"""
Script to run the ETL pipeline
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sec_filing_analyzer.config import ConfigProvider, ETLConfig, StorageConfig
from src.sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
from src.sec_filing_analyzer.storage.sync_manager import StorageSyncManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_etl_pipeline():
    """Run the ETL pipeline."""
    try:
        # Initialize configuration
        ConfigProvider.initialize()
        etl_config = ConfigProvider.get_config(ETLConfig)
        storage_config = ConfigProvider.get_config(StorageConfig)

        # Initialize pipeline
        logger.info("Initializing ETL pipeline...")
        pipeline = SECFilingETLPipeline(
            max_workers=etl_config.max_workers,
            batch_size=etl_config.batch_size,
            rate_limit=etl_config.rate_limit,
            use_parallel=etl_config.use_parallel,
            process_semantic=True,
            process_quantitative=True,
            db_path=storage_config.duckdb_path,
        )

        # Process companies
        companies = ["AAPL", "MSFT", "NVDA"]
        filing_types = ["10-K", "10-Q"]
        start_date = "2023-01-01"
        end_date = "2023-12-31"

        for ticker in companies:
            logger.info(f"Processing {ticker}...")
            result = pipeline.process_company(
                ticker=ticker, filing_types=filing_types, start_date=start_date, end_date=end_date
            )

            logger.info(f"Result for {ticker}: {result}")

        # Sync storage
        logger.info("Synchronizing storage...")
        sync_manager = StorageSyncManager(
            db_path=storage_config.duckdb_path,
            vector_store_path=storage_config.vector_store_path,
            filings_dir=etl_config.filings_dir,
        )

        sync_results = sync_manager.sync_all()
        logger.info(f"Sync results: {sync_results}")

        # Update filing paths
        logger.info("Updating filing paths...")
        path_results = sync_manager.update_filing_paths()
        logger.info(f"Path update results: {path_results}")

        # Update processing status
        logger.info("Updating processing status...")
        status_results = sync_manager.update_processing_status()
        logger.info(f"Status update results: {status_results}")

        # Get inventory summary
        logger.info("Getting inventory summary...")
        summary = sync_manager.get_inventory_summary()
        logger.info(f"Total filings: {summary['total_filings']}")

        # Close connection
        sync_manager.close()

        return True

    except Exception as e:
        logger.error(f"Error running ETL pipeline: {e}")
        return False


if __name__ == "__main__":
    run_etl_pipeline()
