#!/usr/bin/env python
"""
Run ETL pipeline for NVIDIA Corporation (NVDA).

This script processes SEC filings for NVIDIA Corporation (NVDA) for the specified years.
It downloads the filings, extracts the data, and stores it in the database.

Usage:
    python scripts/run_nvda_etl.py --ticker NVDA --years 2023 2024
"""

import argparse
import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from sec_filing_analyzer.config import ConfigProvider, ETLConfig, StorageConfig
from sec_filing_analyzer.data_retrieval import SECFilingsDownloader
from sec_filing_analyzer.pipeline.etl_pipeline import SECFilingETLPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process SEC filings for NVIDIA Corporation")
    parser.add_argument(
        "--ticker", 
        type=str, 
        default="NVDA",
        help="Company ticker symbol (default: NVDA)"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        required=True,
        help="Years to process (e.g., 2023 2024)"
    )
    parser.add_argument(
        "--filing-types",
        nargs="+",
        default=["10-K", "10-Q"],
        help="Filing types to process (default: 10-K 10-Q)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/db_backup/financial_data.duckdb",
        help="Path to DuckDB database"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of worker threads"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for parallel processing"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.1,
        help="Rate limit for SEC API requests (seconds)"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing"
    )
    
    return parser.parse_args()


def main():
    """Run the ETL pipeline for NVIDIA Corporation."""
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Ensure EDGAR_IDENTITY is set
    if not os.environ.get("EDGAR_IDENTITY"):
        logger.error("EDGAR_IDENTITY environment variable is not set")
        logger.info("Please set EDGAR_IDENTITY in your .env file or environment")
        logger.info("Example: EDGAR_IDENTITY='Your Name (your.email@example.com)'")
        return 1
    
    # Initialize configuration
    ConfigProvider.initialize()
    
    # Create date ranges for each year
    date_ranges = []
    for year in args.years:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        date_ranges.append((start_date, end_date))
    
    # Initialize SEC downloader
    downloader = SECFilingsDownloader()
    
    # Initialize pipeline
    logger.info("Initializing ETL pipeline...")
    pipeline = SECFilingETLPipeline(
        sec_downloader=downloader,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        rate_limit=args.rate_limit,
        use_parallel=not args.no_parallel,
        process_semantic=True,
        process_quantitative=True,
        db_path=args.db_path,
    )
    
    logger.info(f"Processing {args.ticker} filings for years: {', '.join(map(str, args.years))}")
    logger.info(f"Filing types: {', '.join(args.filing_types)}")
    logger.info(f"Database path: {args.db_path}")
    logger.info(f"Parallel processing: {not args.no_parallel}")
    
    if not args.no_parallel:
        logger.info(f"Using {args.max_workers} worker threads")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Rate limit: {args.rate_limit} seconds")
    
    # Process filings for each date range
    results = []
    for start_date, end_date in date_ranges:
        logger.info(f"Processing filings from {start_date} to {end_date}")
        
        try:
            # Process company filings
            result = pipeline.process_company(
                ticker=args.ticker,
                filing_types=args.filing_types,
                start_date=start_date,
                end_date=end_date,
            )
            
            results.append(result)
            
            if result.get("status") == "success":
                logger.info(f"Successfully processed {args.ticker} filings from {start_date} to {end_date}")
            else:
                logger.warning(f"Partially processed {args.ticker} filings from {start_date} to {end_date}")
                logger.warning(f"Errors: {result.get('errors', [])}")
        
        except Exception as e:
            logger.error(f"Error processing {args.ticker} filings from {start_date} to {end_date}: {str(e)}")
            results.append({"status": "error", "error": str(e)})
    
    # Summarize results
    success_count = sum(1 for r in results if r.get("status") == "success")
    partial_count = sum(1 for r in results if r.get("status") == "partial")
    error_count = sum(1 for r in results if r.get("status") == "error")
    
    logger.info(f"ETL process completed for {args.ticker}")
    logger.info(f"Success: {success_count}, Partial: {partial_count}, Error: {error_count}")
    
    return 0


if __name__ == "__main__":
    exit(main())
