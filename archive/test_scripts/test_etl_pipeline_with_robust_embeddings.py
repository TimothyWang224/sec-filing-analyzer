"""
Test script for the ETL pipeline with robust embeddings.
"""

import logging
from datetime import datetime, timedelta
from src.sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_etl_pipeline_with_robust_embeddings():
    """Test the ETL pipeline with robust embeddings."""
    # Set up date range for 1 month
    today = datetime.now()
    one_month_ago = today - timedelta(days=30)
    start_date = one_month_ago.strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    print(f"Testing ETL pipeline with date range: {start_date} to {end_date}")

    # Initialize pipeline with semantic processing enabled
    pipeline = SECFilingETLPipeline(
        process_semantic=True,
        process_quantitative=False  # Disable quantitative processing to focus on embeddings
    )

    # Process NVDA filings with limit of 2 to keep the test short
    result = pipeline.process_company_filings(
        ticker="NVDA",
        filing_types=["8-K"],  # Use 8-K filings as they are typically shorter
        start_date=start_date,
        end_date=end_date,
        limit=2  # Limit to 2 filings for testing
    )

    # Print result
    print(f"Result: {result}")
    if "filings_processed" in result:
        print(f"Filings processed: {result['filings_processed']}")
    if "error" in result:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    test_etl_pipeline_with_robust_embeddings()
