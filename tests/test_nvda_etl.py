"""
Test script to run the ETL pipeline for NVDA filings.

This script processes NVDA filings from 2023 as a test case.
"""

import logging
from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Initialize pipeline
    pipeline = SECFilingETLPipeline()
    
    try:
        logger.info("Starting ETL process for NVDA")
        logger.info("Date range: 2023-01-01 to 2023-12-31")
        logger.info("Filing types: 10-K, 10-Q, 8-K")
        
        # Process company filings
        pipeline.process_company(
            ticker="NVDA",
            filing_types=["10-K", "10-Q", "8-K"],
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        logger.info("ETL process completed successfully")
        
    except Exception as e:
        logger.error(f"Error running ETL process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 