"""
Clean All Data Script

This script removes all data from all storage systems (DuckDB, vector store, file system).
Use with caution as this will delete all data!
"""

import logging
import os
import shutil
import sys
import time
from pathlib import Path

import duckdb

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sec_filing_analyzer.config import ConfigProvider, ETLConfig, StorageConfig
from src.sec_filing_analyzer.storage.lifecycle_manager import DataLifecycleManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_all_data(dry_run=False):
    """
    Clean all data from all storage systems.

    Args:
        dry_run: If True, only simulate deletion without actually deleting files
    """
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

        # Get all filings from DuckDB
        logger.info("Getting all filings from DuckDB...")
        try:
            filings = lifecycle_manager.conn.execute(
                "SELECT accession_number FROM filings"
            ).fetchdf()

            filing_count = len(filings)
            logger.info(f"Found {filing_count} filings in DuckDB")

            if filing_count > 0:
                # Ask for confirmation
                if not dry_run:
                    logger.info(
                        f"About to delete {filing_count} filings from all storage systems."
                    )
                    confirm = input(
                        f"Are you sure you want to delete {filing_count} filings? This cannot be undone! (yes/no): "
                    )

                    if confirm.lower() != "yes":
                        logger.info("Deletion cancelled by user")
                        lifecycle_manager.close()
                        return False

                # Delete each filing
                logger.info(
                    f"{'Simulating deletion' if dry_run else 'Deleting'} of {filing_count} filings..."
                )

                for i, row in enumerate(filings.itertuples()):
                    accession_number = row.accession_number
                    logger.info(
                        f"{'Simulating deletion' if dry_run else 'Deleting'} filing {i + 1}/{filing_count}: {accession_number}"
                    )

                    # Delete filing
                    result = lifecycle_manager.delete_filing(
                        accession_number, dry_run=dry_run
                    )

                    if result["status"] == "success":
                        logger.info(
                            f"Successfully {'simulated deletion' if dry_run else 'deleted'} filing {accession_number}"
                        )
                    else:
                        logger.warning(
                            f"Issue {'simulating deletion' if dry_run else 'deleting'} filing {accession_number}: {result['status']}"
                        )

                    # Sleep briefly to avoid overwhelming the system
                    time.sleep(0.1)

            # Close lifecycle manager
            lifecycle_manager.close()

            # If not a dry run, clean up storage directories
            if not dry_run:
                # Clean up vector store
                vector_store_path = Path(storage_config.vector_store_path)
                if vector_store_path.exists():
                    logger.info(f"Cleaning up vector store at {vector_store_path}...")

                    # Remove by_company directory
                    by_company_dir = vector_store_path / "by_company"
                    if by_company_dir.exists():
                        logger.info(f"Removing {by_company_dir}...")
                        shutil.rmtree(by_company_dir)

                    # Remove metadata directory
                    metadata_dir = vector_store_path / "metadata"
                    if metadata_dir.exists():
                        logger.info(f"Removing {metadata_dir}...")
                        shutil.rmtree(metadata_dir)

                    # Remove index files
                    for index_file in vector_store_path.glob("*.index"):
                        logger.info(f"Removing {index_file}...")
                        os.remove(index_file)

                # Clean up filings directory
                filings_dir = Path(etl_config.filings_dir)
                if filings_dir.exists():
                    logger.info(f"Cleaning up filings directory at {filings_dir}...")

                    # Remove subdirectories
                    for subdir in ["raw", "html", "processed", "xbrl"]:
                        subdir_path = filings_dir / subdir
                        if subdir_path.exists():
                            logger.info(f"Removing {subdir_path}...")
                            shutil.rmtree(subdir_path)

                # Recreate DuckDB database
                db_path = etl_config.db_path
                logger.info(f"Recreating DuckDB database at {db_path}...")

                # Remove existing database
                if os.path.exists(db_path):
                    os.remove(db_path)

                # Create new database with schema
                conn = duckdb.connect(db_path)

                # Create companies table
                conn.execute("""
                CREATE TABLE companies (
                    ticker VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    sector VARCHAR,
                    industry VARCHAR,
                    description TEXT,
                    cik VARCHAR,
                    exchange VARCHAR,
                    last_updated TIMESTAMP
                )
                """)

                # Create filings table with enhanced schema
                conn.execute("""
                CREATE TABLE filings (
                    id INTEGER PRIMARY KEY,
                    ticker VARCHAR,
                    accession_number VARCHAR UNIQUE,
                    filing_type VARCHAR,
                    filing_date DATE,
                    document_url VARCHAR,
                    local_file_path VARCHAR,
                    processing_status VARCHAR,
                    last_updated TIMESTAMP,
                    has_xbrl BOOLEAN DEFAULT FALSE,
                    has_html BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (ticker) REFERENCES companies(ticker)
                )
                """)

                # Create financial_facts table
                conn.execute("""
                CREATE TABLE financial_facts (
                    id INTEGER PRIMARY KEY,
                    filing_id INTEGER,
                    ticker VARCHAR,
                    metric VARCHAR,
                    value DOUBLE,
                    unit VARCHAR,
                    start_date DATE,
                    end_date DATE,
                    context_ref VARCHAR,
                    filing_type VARCHAR,
                    FOREIGN KEY (filing_id) REFERENCES filings(id),
                    FOREIGN KEY (ticker) REFERENCES companies(ticker)
                )
                """)

                # Create indexes
                conn.execute("CREATE INDEX idx_filings_ticker ON filings(ticker)")
                conn.execute(
                    "CREATE INDEX idx_filings_filing_type ON filings(filing_type)"
                )
                conn.execute(
                    "CREATE INDEX idx_filings_filing_date ON filings(filing_date)"
                )
                conn.execute(
                    "CREATE INDEX idx_filings_accession_number ON filings(accession_number)"
                )
                conn.execute(
                    "CREATE INDEX idx_filings_processing_status ON filings(processing_status)"
                )

                conn.execute(
                    "CREATE INDEX idx_facts_filing_id ON financial_facts(filing_id)"
                )
                conn.execute("CREATE INDEX idx_facts_ticker ON financial_facts(ticker)")
                conn.execute("CREATE INDEX idx_facts_metric ON financial_facts(metric)")
                conn.execute(
                    "CREATE INDEX idx_facts_dates ON financial_facts(start_date, end_date)"
                )

                conn.close()

                logger.info("DuckDB database recreated with schema")

            logger.info(f"All data {'would be' if dry_run else 'has been'} cleaned")
            return True

        except Exception as e:
            logger.error(f"Error getting filings from DuckDB: {e}")
            lifecycle_manager.close()
            return False

    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean all data from all storage systems"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate deletion without actually deleting files",
    )

    args = parser.parse_args()

    success = clean_all_data(dry_run=args.dry_run)
    print(
        f"Data cleaning {'simulation' if args.dry_run else 'operation'} {'succeeded' if success else 'failed'}"
    )
