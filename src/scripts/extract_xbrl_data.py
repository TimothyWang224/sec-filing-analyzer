"""
Extract XBRL Data Script

This script extracts financial data from XBRL filings and stores it in DuckDB.
It can process a single filing or all filings for a company.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sec_filing_analyzer.data_processing.xbrl_extractor import XBRLExtractor
from sec_filing_analyzer.storage.financial_data_store import FinancialDataStore
import edgar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_filing(ticker: str, accession_number: str,
                  xbrl_extractor: XBRLExtractor,
                  financial_store: FinancialDataStore) -> bool:
    """Process a single filing.

    Args:
        ticker: Company ticker symbol
        accession_number: SEC accession number
        xbrl_extractor: XBRL extractor instance
        financial_store: Financial data store instance

    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate a filing ID
        filing_id = f"{ticker}_{accession_number.replace('-', '_')}"

        # Extract XBRL data
        xbrl_data = xbrl_extractor.extract_financials(
            ticker=ticker,
            filing_id=filing_id,
            accession_number=accession_number
        )

        # Store XBRL data
        success = financial_store.store_xbrl_data(xbrl_data)

        if success:
            logger.info(f"Successfully processed filing {ticker} {accession_number}")
        else:
            logger.warning(f"Failed to process filing {ticker} {accession_number}")

        return success
    except Exception as e:
        logger.error(f"Error processing filing {ticker} {accession_number}: {e}")
        return False

def process_company(ticker: str, filing_type: str, limit: int,
                   xbrl_extractor: XBRLExtractor,
                   financial_store: FinancialDataStore) -> int:
    """Process all filings for a company.

    Args:
        ticker: Company ticker symbol
        filing_type: Filing type (10-K, 10-Q, etc.)
        limit: Maximum number of filings to process
        xbrl_extractor: XBRL extractor instance
        financial_store: Financial data store instance

    Returns:
        Number of filings processed
    """
    try:
        # Get entity data
        entity = edgar.get_entity(ticker)

        # Get filings
        filings = entity.get_filings(form=filing_type, limit=limit)

        if not filings:
            logger.warning(f"No {filing_type} filings found for {ticker}")
            return 0

        logger.info(f"Found {len(filings)} {filing_type} filings for {ticker}")

        # Process each filing
        count = 0
        for filing in filings:
            accession_number = filing.accession_number
            if not accession_number:
                continue

            success = process_filing(
                ticker=ticker,
                accession_number=accession_number,
                xbrl_extractor=xbrl_extractor,
                financial_store=financial_store
            )

            if success:
                count += 1

        logger.info(f"Processed {count} filings for {ticker}")
        return count
    except Exception as e:
        logger.error(f"Error processing company {ticker}: {e}")
        return 0

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Extract XBRL data from SEC filings")
    parser.add_argument("--ticker", type=str, help="Company ticker symbol")
    parser.add_argument("--accession", type=str, help="SEC accession number")
    parser.add_argument("--filing-type", type=str, default="10-K", help="Filing type (10-K, 10-Q, etc.)")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of filings to process")
    parser.add_argument("--db-path", type=str, default="data/financial_data.duckdb", help="Path to DuckDB database")
    parser.add_argument("--cache-dir", type=str, default="data/xbrl_cache", help="Path to XBRL cache directory")

    args = parser.parse_args()

    # Initialize components
    xbrl_extractor = XBRLExtractor(cache_dir=args.cache_dir)
    financial_store = FinancialDataStore(db_path=args.db_path)

    # Process filings
    if args.ticker and args.accession:
        # Process a single filing
        process_filing(
            ticker=args.ticker,
            accession_number=args.accession,
            xbrl_extractor=xbrl_extractor,
            financial_store=financial_store
        )
    elif args.ticker:
        # Process all filings for a company
        process_company(
            ticker=args.ticker,
            filing_type=args.filing_type,
            limit=args.limit,
            xbrl_extractor=xbrl_extractor,
            financial_store=financial_store
        )
    else:
        logger.error("Please provide a ticker symbol")
        parser.print_help()

if __name__ == "__main__":
    main()
