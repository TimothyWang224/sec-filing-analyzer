"""
Simple test script for the Data Lifecycle Manager
"""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sec_filing_analyzer.config import ConfigProvider, ETLConfig, StorageConfig
from src.sec_filing_analyzer.storage.lifecycle_manager import DataLifecycleManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_lifecycle_manager_init():
    """Test initializing the Data Lifecycle Manager."""
    try:
        # Initialize configuration
        logger.info("Initializing configuration...")
        ConfigProvider.initialize()

        # Get configurations
        logger.info("Getting configurations...")
        etl_config = ConfigProvider.get_config(ETLConfig)
        storage_config = ConfigProvider.get_config(StorageConfig)

        # Log configuration values
        logger.info(f"ETL Config - db_path: {etl_config.db_path}")
        logger.info(f"ETL Config - filings_dir: {etl_config.filings_dir}")
        logger.info(
            f"Storage Config - vector_store_path: {storage_config.vector_store_path}"
        )

        # Create lifecycle manager
        logger.info("Creating lifecycle manager...")
        lifecycle_manager = DataLifecycleManager(
            db_path=etl_config.db_path,
            vector_store_path=storage_config.vector_store_path,
            filings_dir=etl_config.filings_dir,
        )

        # Log success
        logger.info("Successfully initialized lifecycle manager")

        # Close lifecycle manager
        lifecycle_manager.close()
        logger.info("Closed lifecycle manager")

        return True

    except Exception as e:
        logger.error(f"Error initializing lifecycle manager: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = test_lifecycle_manager_init()
    print(f"Test {'succeeded' if success else 'failed'}")
