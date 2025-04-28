"""
ETL Service for Streamlit Application

This service handles ETL pipeline execution and tracking for the Streamlit application.
"""

import inspect
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sec_filing_analyzer.config import ConfigProvider, ETLConfig, StorageConfig
from sec_filing_analyzer.data_retrieval import SECFilingsDownloader
from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline
from sec_filing_analyzer.storage import GraphStore, LlamaIndexVectorStore
from sec_filing_analyzer.storage.sync_manager_enhanced import EnhancedStorageSyncManager

# Import the edgar identity initialization function
from .etl_service_init import initialize_edgar_identity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETLJob:
    """Class representing an ETL job."""

    def __init__(
        self,
        job_id: str,
        tickers: List[str],
        filing_types: List[str],
        start_date: str,
        end_date: str,
        estimated_filings: int,
        submitted_at: datetime,
        force_reprocessing: bool = False,
    ):
        """Initialize an ETL job."""
        self.job_id = job_id
        self.tickers = tickers
        self.filing_types = filing_types
        self.start_date = start_date
        self.end_date = end_date
        self.estimated_filings = estimated_filings
        self.submitted_at = submitted_at
        self.force_reprocessing = force_reprocessing
        self.status = "Pending"
        self.completed_at = None
        self.results = {}
        self.logs = []
        self.progress = 0
        self.current_stage = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "id": self.job_id,
            "status": self.status,
            "companies": ", ".join(self.tickers[:3]) + ("..." if len(self.tickers) > 3 else ""),
            "filing_types": ", ".join(self.filing_types),
            "date_range": f"{self.start_date} to {self.end_date}",
            "filings": self.estimated_filings,
            "force_reprocessing": self.force_reprocessing,
            "submitted": self.submitted_at.strftime("%Y-%m-%d %H:%M"),
            "completed": self.completed_at.strftime("%Y-%m-%d %H:%M") if self.completed_at else None,
            "progress": self.progress,
        }

    def add_log(self, message: str) -> None:
        """Add a log message to the job."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        logger.info(f"Job {self.job_id}: {message}")

    def update_progress(self, progress: float, stage: str) -> None:
        """Update job progress."""
        self.progress = min(int(progress * 100), 100)
        self.current_stage = stage

    def complete(self, results: Dict[str, Any]) -> None:
        """Mark job as completed."""
        self.status = "Completed"
        self.completed_at = datetime.now()
        self.results = results
        self.progress = 100
        self.add_log("Job completed successfully")

    def fail(self, error: str) -> None:
        """Mark job as failed."""
        self.status = "Failed"
        self.completed_at = datetime.now()
        self.results = {"error": error}
        self.add_log(f"Job failed: {error}")


class ETLService:
    """Service for handling ETL pipeline execution and tracking."""

    def __init__(self):
        """Initialize the ETL service."""
        self.jobs: Dict[str, ETLJob] = {}
        self.active_job_id: Optional[str] = None
        self.pipeline: Optional[SECFilingETLPipeline] = None
        self.sync_manager: Optional[EnhancedStorageSyncManager] = None
        self._initialize_pipeline()
        self._initialize_sync_manager()

    def _initialize_pipeline(self) -> None:
        """Initialize the ETL pipeline."""
        try:
            # Initialize edgar identity first
            if not initialize_edgar_identity():
                logger.error("Failed to initialize edgar identity. ETL pipeline will not work properly.")
                # Continue anyway to allow other functionality to work

            # Initialize configuration
            ConfigProvider.initialize()
            etl_config = ConfigProvider.get_config(ETLConfig)
            storage_config = ConfigProvider.get_config(StorageConfig)

            # Initialize components
            graph_store = GraphStore()
            vector_store = LlamaIndexVectorStore(
                store_path=storage_config.vector_store_path,
                force_rebuild=False,  # Don't force rebuild the index on startup
                lazy_load=True,  # Use lazy loading to avoid rebuilding the index on startup
            )
            sec_downloader = SECFilingsDownloader()

            # Initialize pipeline
            self.pipeline = SECFilingETLPipeline(
                graph_store=graph_store,
                vector_store=vector_store,
                sec_downloader=sec_downloader,
                max_workers=etl_config.max_workers,
                batch_size=etl_config.batch_size,
                rate_limit=etl_config.rate_limit,
                use_parallel=etl_config.use_parallel,
                process_semantic=True,
                process_quantitative=True,
                db_path=etl_config.db_path,  # Use db_path from ETLConfig instead of StorageConfig
            )

            logger.info("ETL pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ETL pipeline: {e}")
            self.pipeline = None

    def _initialize_sync_manager(self) -> None:
        """Initialize the storage sync manager."""
        try:
            # Initialize configuration
            logger.info("Initializing configuration for sync manager")
            ConfigProvider.initialize()
            etl_config = ConfigProvider.get_config(ETLConfig)
            storage_config = ConfigProvider.get_config(StorageConfig)

            logger.info(f"ETL config db_path: {etl_config.db_path}")
            logger.info(f"ETL config db_read_only: {etl_config.db_read_only}")
            logger.info(f"Storage config vector_store_path: {storage_config.vector_store_path}")
            logger.info(f"Storage config filings_dir: {etl_config.filings_dir}")

            # Check if we already have a read-write connection to the database
            logger.info("Checking for existing database connections")
            from src.sec_filing_analyzer.utils.duckdb_manager import duckdb_manager

            self.duckdb_manager = duckdb_manager  # Store reference to duckdb_manager for later use
            read_write_key = f"{etl_config.db_path}:False"
            read_only = etl_config.db_read_only

            logger.info(f"Active connections: {list(duckdb_manager._active_connections.keys())}")

            # If we're trying to open in read-only mode but already have a read-write connection,
            # we need to use read-write mode to avoid connection conflicts
            if read_only and read_write_key in duckdb_manager._active_connections:
                logger.info("Using read-write mode for sync manager because a read-write connection already exists")
                read_only = False

            # Make sure the database directory exists
            import os

            db_dir = os.path.dirname(etl_config.db_path)
            logger.info(f"Ensuring database directory exists: {db_dir}")
            os.makedirs(db_dir, exist_ok=True)

            # Check if the database file exists
            db_exists = os.path.exists(etl_config.db_path)
            logger.info(f"Database file exists: {db_exists}")

            # Force close any existing connections to the database
            logger.info(f"Forcing close of all connections to {etl_config.db_path} before initializing sync manager")
            duckdb_manager.force_close_all_connections(etl_config.db_path)

            # Initialize enhanced sync manager
            logger.info("Creating EnhancedStorageSyncManager instance")
            self.sync_manager = EnhancedStorageSyncManager(
                db_path=etl_config.db_path,  # Use db_path from ETLConfig instead of StorageConfig
                vector_store_path=storage_config.vector_store_path,
                filings_dir=etl_config.filings_dir,
                read_only=read_only,  # Use adjusted read_only value
            )

            # Store the database path for later use
            self.db_path = etl_config.db_path

            logger.info("Storage sync manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing storage sync manager: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            self.sync_manager = None
            self.db_path = None
            self.duckdb_manager = None

    def _close_write_connection(self) -> None:
        """Close the write connection to the database."""
        if hasattr(self, "duckdb_manager") and self.duckdb_manager and hasattr(self, "db_path") and self.db_path:
            try:
                # Close the read-write connection to the database
                self.duckdb_manager.close_connection(self.db_path, read_only=False)
                logger.info(f"Closed read-write connection to {self.db_path}")
            except Exception as e:
                logger.error(f"Error closing read-write connection to {self.db_path}: {e}")

    def create_job(
        self,
        tickers: List[str],
        filing_types: List[str],
        start_date: str,
        end_date: str,
        estimated_filings: int,
        force_reprocessing: bool = False,
    ) -> str:
        """
        Create a new ETL job.

        Args:
            tickers: List of company ticker symbols
            filing_types: List of filing types to retrieve
            start_date: Start date for filings
            end_date: End date for filings
            estimated_filings: Estimated number of filings to retrieve
            force_reprocessing: Whether to force reprocessing of existing filings

        Returns:
            Job ID
        """
        # Generate a unique job ID
        job_id = f"job-{uuid.uuid4().hex[:8]}"

        # Create job
        job = ETLJob(
            job_id=job_id,
            tickers=tickers,
            filing_types=filing_types,
            start_date=start_date,
            end_date=end_date,
            estimated_filings=estimated_filings,
            submitted_at=datetime.now(),
            force_reprocessing=force_reprocessing,
        )

        # Add job to jobs dictionary
        self.jobs[job_id] = job

        # Log job creation
        job.add_log(f"Created ETL job for {len(tickers)} companies, {len(filing_types)} filing types")

        return job_id

    def start_job(self, job_id: str, log_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Start an ETL job.

        Args:
            job_id: Job ID
            log_callback: Optional callback function for logging

        Returns:
            True if job started successfully, False otherwise
        """
        # Check if pipeline is initialized
        if not self.pipeline:
            logger.warning("ETL pipeline not initialized, attempting to initialize...")
            self._initialize_pipeline()

            # Check again after initialization attempt
            if not self.pipeline:
                logger.error("Failed to initialize ETL pipeline")
                return False

        # Check if job exists
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return False

        # Check if another job is already running
        if self.active_job_id:
            logger.error(f"Another job is already running: {self.active_job_id}")
            return False

        # Get job
        job = self.jobs[job_id]

        # Update job status
        job.status = "Running"
        self.active_job_id = job_id

        # Log job start
        job.add_log("Starting ETL job")

        # Start job in a separate thread
        thread = threading.Thread(target=self._run_job, args=(job, log_callback))
        thread.daemon = True
        thread.start()

        return True

    def _run_job(self, job: ETLJob, log_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Run an ETL job.

        Args:
            job: ETL job
            log_callback: Optional callback function for logging
        """
        try:
            # Log job start
            job.add_log(f"Processing {len(job.tickers)} companies")
            if log_callback:
                log_callback(f"Processing {len(job.tickers)} companies")

            # Process each company
            results = {}
            for i, ticker in enumerate(job.tickers):
                # Update progress
                progress = i / len(job.tickers)
                stage = f"Processing {ticker}"
                job.update_progress(progress, stage)

                # Log company processing
                job.add_log(f"Processing {ticker}")
                if log_callback:
                    log_callback(f"Processing {ticker}")

                # Process company filings
                try:
                    # Check if the pipeline supports force_download parameter
                    if (
                        hasattr(self.pipeline, "process_company")
                        and "force_download" in inspect.signature(self.pipeline.process_company).parameters
                    ):
                        company_result = self.pipeline.process_company(
                            ticker=ticker,
                            filing_types=job.filing_types,
                            start_date=job.start_date,
                            end_date=job.end_date,
                            force_download=job.force_reprocessing,
                        )
                    else:
                        # Use the standard parameters without force_download
                        company_result = self.pipeline.process_company(
                            ticker=ticker,
                            filing_types=job.filing_types,
                            start_date=job.start_date,
                            end_date=job.end_date,
                        )

                    # Add result to results dictionary
                    results[ticker] = company_result

                    # Log company processing result
                    if "error" in company_result:
                        job.add_log(f"Error processing {ticker}: {company_result['error']}")
                        if log_callback:
                            log_callback(f"Error processing {ticker}: {company_result['error']}")
                    else:
                        num_filings = len(company_result.get("results", []))
                        job.add_log(f"Processed {num_filings} filings for {ticker}")
                        if log_callback:
                            log_callback(f"Processed {num_filings} filings for {ticker}")
                except Exception as e:
                    # Log error
                    error_msg = f"Error processing {ticker}: {str(e)}"
                    job.add_log(error_msg)
                    if log_callback:
                        log_callback(error_msg)

                    # Add error to results
                    results[ticker] = {"error": str(e)}

            # Sync storage after processing
            job.add_log("Synchronizing storage...")
            if log_callback:
                log_callback("Synchronizing storage...")

            if self.sync_manager:
                sync_results = self.sync_manager.sync_all()

                # Check sync status
                if sync_results.get("overall_status") == "success":
                    job.add_log(
                        f"Storage synchronization completed successfully: {sync_results['total_filings']} total filings"
                    )
                    if log_callback:
                        log_callback(
                            f"Storage synchronization completed successfully: {sync_results['total_filings']} total filings"
                        )
                elif sync_results.get("overall_status") == "partial_success":
                    failed_components = ", ".join(sync_results.get("failed_components", []))
                    job.add_log(
                        f"Storage synchronization partially completed: {sync_results['total_filings']} total filings. Failed components: {failed_components}"
                    )
                    if log_callback:
                        log_callback(
                            f"Storage synchronization partially completed. Failed components: {failed_components}"
                        )
                        log_callback("The successful parts have been synchronized and are available in the inventory.")
                else:
                    job.add_log("Storage synchronization failed")
                    if log_callback:
                        log_callback("Storage synchronization failed, but processed filings are still available.")
            else:
                job.add_log("Storage sync manager not initialized, skipping synchronization")
                if log_callback:
                    log_callback("Storage sync manager not initialized, skipping synchronization")

            # Set flag that index needs rebuilding
            from src.streamlit_app.utils import app_state

            app_state.set("index_needs_rebuild", True)
            job.add_log("Set flag that vector index needs rebuilding")
            if log_callback:
                log_callback(
                    "Vector index needs rebuilding. Please use the 'Rebuild Vector Index' button in the ETL Data Inventory page."
                )

            # Complete job
            job.complete(results)

            # Close the write connection to free up the database for other tools
            self._close_write_connection()

        except Exception as e:
            # Log error
            error_msg = f"Error running ETL job: {str(e)}"
            job.add_log(error_msg)
            if log_callback:
                log_callback(error_msg)

            # Fail job
            job.fail(str(e))

        finally:
            # Close the write connection to free up the database for other tools
            self._close_write_connection()

            # Clear active job
            self.active_job_id = None

    def get_job(self, job_id: str) -> Optional[ETLJob]:
        """
        Get an ETL job.

        Args:
            job_id: Job ID

        Returns:
            ETL job or None if not found
        """
        return self.jobs.get(job_id)

    def get_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all ETL jobs.

        Returns:
            List of ETL jobs as dictionaries
        """
        return [job.to_dict() for job in self.jobs.values()]

    def get_job_logs(self, job_id: str) -> List[str]:
        """
        Get logs for an ETL job.

        Args:
            job_id: Job ID

        Returns:
            List of log messages
        """
        job = self.jobs.get(job_id)
        if not job:
            return []

        return job.logs

    def get_active_job(self) -> Optional[ETLJob]:
        """
        Get the active ETL job.

        Returns:
            Active ETL job or None if no job is active
        """
        if not self.active_job_id:
            return None

        return self.jobs.get(self.active_job_id)

    def force_initialize_sync_manager(self) -> bool:
        """
        Force initialization of the storage sync manager.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Log the current state
            logger.info(f"Current sync_manager state: {self.sync_manager is not None}")

            # Close any existing sync manager
            if self.sync_manager:
                try:
                    logger.info("Closing existing sync manager")
                    self.sync_manager.close()
                except Exception as e:
                    logger.warning(f"Error closing existing sync manager: {e}")

            # Initialize the sync manager
            logger.info("Calling _initialize_sync_manager()")
            self._initialize_sync_manager()

            # Check if initialization was successful
            if self.sync_manager:
                logger.info("Successfully forced initialization of storage sync manager")
                return True
            else:
                logger.error("Failed to force initialize storage sync manager")
                return False
        except Exception as e:
            logger.error(f"Error forcing initialization of storage sync manager: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def sync_storage(self) -> Dict[str, Any]:
        """
        Synchronize storage systems.

        Returns:
            Dictionary with synchronization results
        """
        logger.info("sync_storage method called")

        # Check if sync manager is initialized
        if not self.sync_manager:
            logger.warning("Storage sync manager not initialized, attempting to initialize...")
            if not self.force_initialize_sync_manager():
                logger.error("Failed to initialize storage sync manager")
                return {"error": "Storage sync manager not initialized"}
            else:
                logger.info("Successfully initialized sync manager")
        else:
            logger.info("Sync manager already initialized")

        try:
            # Sync all storage systems using the enhanced sync manager
            # This already includes path and status updates
            logger.info("Calling sync_manager.sync_all()")
            results = self.sync_manager.sync_all()
            logger.info(f"sync_all() returned: {results}")

            # Add a user-friendly message based on the sync status
            if results.get("overall_status") == "success":
                results["message"] = "Storage synchronization completed successfully."
            elif results.get("overall_status") == "partial_success":
                failed_components = ", ".join(results.get("failed_components", []))
                results["message"] = (
                    f"Storage synchronization partially completed. Failed components: {failed_components}"
                )
                results["warning"] = "Some components failed to synchronize, but the successful parts are available."
            else:
                results["message"] = "Storage synchronization failed."
                results["error"] = "Failed to synchronize storage systems."

            # Close the write connection to free up the database for other tools
            self._close_write_connection()

            return results
        except Exception as e:
            logger.error(f"Error synchronizing storage: {e}")
            # Make sure to close the connection even if there's an error
            self._close_write_connection()
            return {"error": str(e)}

    def get_inventory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the filing inventory.

        Returns:
            Dictionary with inventory summary
        """
        if not self.sync_manager:
            logger.error("Storage sync manager not initialized")
            return {"error": "Storage sync manager not initialized"}

        try:
            # Get the inventory summary
            summary = self.sync_manager.get_inventory_summary()

            # Close the write connection to free up the database for other tools
            # This is a read operation, but we'll close any write connections that might exist
            self._close_write_connection()

            return summary
        except Exception as e:
            logger.error(f"Error getting inventory summary: {e}")
            # Make sure to close the connection even if there's an error
            self._close_write_connection()
            return {"error": str(e)}

    def estimate_filings_count(
        self,
        tickers: List[str],
        filing_types: List[str],
        start_date: str,
        end_date: str,
    ) -> int:
        """
        Estimate the number of filings to retrieve.

        Args:
            tickers: List of company ticker symbols
            filing_types: List of filing types to retrieve
            start_date: Start date for filings
            end_date: End date for filings

        Returns:
            Estimated number of filings
        """
        # In a real implementation, this would query the SEC API to get an accurate count
        # For now, we'll use a simple estimation formula

        # Calculate date range in years
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            years = (end - start).days / 365.25
        except:
            years = 1  # Default to 1 year if date parsing fails

        # Estimate filings per company per year
        filings_per_company_per_year = {
            "10-K": 1,
            "10-Q": 4,
            "8-K": 8,
            "DEF 14A": 1,
            "S-1": 0.2,
            "S-3": 0.2,
            "S-4": 0.1,
            "S-8": 0.1,
            "S-11": 0.1,
            "20-F": 1,
            "40-F": 1,
            "6-K": 4,
            "F-1": 0.2,
            "F-3": 0.2,
            "F-4": 0.1,
            "F-10": 0.1,
        }

        # Calculate estimated filings
        estimated_filings = 0
        for _ in tickers:
            for filing_type in filing_types:
                filings_per_year = filings_per_company_per_year.get(filing_type, 1)
                estimated_filings += filings_per_year * years

        return int(estimated_filings)


# Create a singleton instance
_instance = None


def get_etl_service() -> ETLService:
    """Get the ETL service singleton instance."""
    global _instance
    if _instance is None:
        _instance = ETLService()
    return _instance
