"""
Test XBRL Extraction Script

This script tests the XBRL extraction and financial data storage functionality.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import json
import pandas as pd

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sec_filing_analyzer.data_processing.xbrl_extractor import XBRLExtractor
from sec_filing_analyzer.storage.financial_data_store import FinancialDataStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_xbrl_extraction(ticker: str, accession_number: str) -> None:
    """Test XBRL extraction for a specific filing.

    Args:
        ticker: Company ticker symbol
        accession_number: SEC accession number
    """
    # Initialize components
    xbrl_extractor = XBRLExtractor(cache_dir="data/xbrl_cache")

    # Extract XBRL data
    filing_id = f"{ticker}_{accession_number.replace('-', '_')}"
    xbrl_data = xbrl_extractor.extract_financials(
        ticker=ticker,
        filing_id=filing_id,
        accession_number=accession_number
    )

    # Print results
    if "error" in xbrl_data:
        print(f"Error extracting XBRL data: {xbrl_data['error']}")
        return

    print(f"\n=== XBRL Data for {ticker} {accession_number} ===")
    print(f"Filing ID: {xbrl_data['filing_id']}")
    print(f"Filing Date: {xbrl_data['filing_date']}")
    print(f"Fiscal Year: {xbrl_data['fiscal_year']}")
    print(f"Fiscal Quarter: {xbrl_data['fiscal_quarter']}")
    print(f"Filing Type: {xbrl_data['filing_type']}")

    print("\nKey Metrics:")
    for name, value in xbrl_data.get("metrics", {}).items():
        print(f"  {name}: {value}")

    print("\nFinancial Ratios:")
    for name, value in xbrl_data.get("ratios", {}).items():
        print(f"  {name}: {value:.4f}")

    print(f"\nTotal Facts: {len(xbrl_data.get('facts', []))}")

    # Save to file for inspection
    output_file = f"data/{ticker}_{accession_number.replace('-', '_')}_xbrl.json"
    with open(output_file, "w") as f:
        json.dump(xbrl_data, f, indent=2)

    print(f"\nFull data saved to {output_file}")

def test_financial_storage(ticker: str, accession_number: str) -> None:
    """Test financial data storage for a specific filing.

    Args:
        ticker: Company ticker symbol
        accession_number: SEC accession number
    """
    # Initialize components
    xbrl_extractor = XBRLExtractor(cache_dir="data/xbrl_cache")
    financial_store = FinancialDataStore(db_path="data/financial_data_test.duckdb")

    # Extract XBRL data
    filing_id = f"{ticker}_{accession_number.replace('-', '_')}"
    xbrl_data = xbrl_extractor.extract_financials(
        ticker=ticker,
        filing_id=filing_id,
        accession_number=accession_number
    )

    # Store XBRL data
    success = financial_store.store_xbrl_data(xbrl_data)

    if success:
        print(f"\nSuccessfully stored XBRL data for {ticker} {accession_number}")
    else:
        print(f"\nFailed to store XBRL data for {ticker} {accession_number}")

    # Get database stats
    stats = financial_store.get_database_stats()

    print("\n=== Database Statistics ===")
    print(f"Companies: {stats.get('company_count', 0)}")
    print(f"Filings: {stats.get('filing_count', 0)}")
    print(f"Financial Facts: {stats.get('fact_count', 0)}")
    print(f"Time Series Metrics: {stats.get('time_series_count', 0)}")
    print(f"Financial Ratios: {stats.get('ratio_count', 0)}")

    # Get company metrics
    metrics = financial_store.get_company_metrics(ticker=ticker)

    print(f"\n=== Financial Metrics for {ticker} ===")
    print(metrics.to_string(index=False))

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test XBRL extraction and financial data storage")
    parser.add_argument("--ticker", type=str, required=True, help="Company ticker symbol")
    parser.add_argument("--accession", type=str, required=True, help="SEC accession number")
    parser.add_argument("--extract-only", action="store_true", help="Only test extraction, not storage")

    args = parser.parse_args()

    # Create data directories if they don't exist
    os.makedirs("data/xbrl_cache", exist_ok=True)

    # Test XBRL extraction
    test_xbrl_extraction(args.ticker, args.accession)

    # Test financial data storage
    if not args.extract_only:
        test_financial_storage(args.ticker, args.accession)

if __name__ == "__main__":
    main()
