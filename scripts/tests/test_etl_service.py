"""
Test script for the ETL service.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the ETL service
from src.streamlit_app.services import get_etl_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_etl_service')

def test_etl_service():
    """Test the ETL service."""
    logger.info("Testing ETL service...")
    
    # Get the ETL service
    etl_service = get_etl_service()
    
    # Test estimating filings count
    tickers = ["AAPL", "MSFT"]
    filing_types = ["10-K", "10-Q"]
    start_date = "2022-01-01"
    end_date = "2022-12-31"
    
    estimated_filings = etl_service.estimate_filings_count(
        tickers=tickers,
        filing_types=filing_types,
        start_date=start_date,
        end_date=end_date
    )
    
    logger.info(f"Estimated filings: {estimated_filings}")
    
    # Create a job
    job_id = etl_service.create_job(
        tickers=tickers,
        filing_types=filing_types,
        start_date=start_date,
        end_date=end_date,
        estimated_filings=estimated_filings
    )
    
    logger.info(f"Created job: {job_id}")
    
    # Get the job
    job = etl_service.get_job(job_id)
    
    logger.info(f"Job status: {job.status}")
    
    # Start the job
    def log_callback(message):
        logger.info(f"Job log: {message}")
    
    etl_service.start_job(job_id, log_callback=log_callback)
    
    logger.info("Job started. Check the logs for progress.")
    
    # Wait for the job to complete
    import time
    while True:
        job = etl_service.get_job(job_id)
        
        if job.status in ["Completed", "Failed"]:
            break
        
        logger.info(f"Job progress: {job.progress}% - {job.current_stage}")
        time.sleep(5)
    
    # Get the final job status
    logger.info(f"Job final status: {job.status}")
    
    if job.status == "Completed":
        logger.info("Job completed successfully!")
    else:
        logger.error(f"Job failed: {job.results.get('error', 'Unknown error')}")
    
    # Get all jobs
    jobs = etl_service.get_jobs()
    
    logger.info(f"Total jobs: {len(jobs)}")
    
    return True

if __name__ == "__main__":
    test_etl_service()
