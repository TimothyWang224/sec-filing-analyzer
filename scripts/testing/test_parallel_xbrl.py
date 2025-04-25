"""
Test Parallel XBRL Extractor and Optimized DuckDB Store

This script tests the parallel XBRL extractor and optimized DuckDB store.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sec_filing_analyzer.data_processing.parallel_xbrl_extractor import ParallelXBRLExtractor
from sec_filing_analyzer.storage.optimized_duckdb_store import OptimizedDuckDBStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_parallel_extraction(tickers, accession_numbers, max_workers=4):
    """Test parallel XBRL extraction.

    Args:
        tickers: List of company ticker symbols
        accession_numbers: List of SEC accession numbers (same length as tickers)
        max_workers: Maximum number of worker threads
    """
    # Initialize the extractor
    extractor = ParallelXBRLExtractor(cache_dir="data/xbrl_cache", max_workers=max_workers, rate_limit=0.2)

    # Prepare companies data
    companies = []
    for i, ticker in enumerate(tickers):
        if i < len(accession_numbers):
            accession_number = accession_numbers[i]
            filing_id = f"{ticker}_{accession_number.replace('-', '_')}"

            company = {"ticker": ticker, "filings": [{"filing_id": filing_id, "accession_number": accession_number}]}
            companies.append(company)

    # Start timer
    start_time = time.time()

    # Extract financials in parallel
    results = extractor.extract_financials_for_companies(companies)

    # End timer
    end_time = time.time()
    elapsed = end_time - start_time

    # Print results
    print(f"\n=== Parallel XBRL Extraction Results ===")
    print(f"Processed {len(companies)} companies with {max_workers} workers")
    print(f"Total time: {elapsed:.2f} seconds")

    for ticker, filings in results.items():
        print(f"\nCompany: {ticker}")
        print(f"Filings: {len(filings)}")

        for filing in filings:
            if "error" in filing:
                print(f"  Error: {filing['error']}")
            else:
                print(f"  Filing: {filing['accession_number']}")
                print(f"  Facts: {len(filing.get('facts', []))}")
                print(f"  Metrics: {len(filing.get('metrics', {}))}")
                print(f"  Ratios: {len(filing.get('ratios', {}))}")

    return results


def test_optimized_duckdb_store(financial_data):
    """Test optimized DuckDB store.

    Args:
        financial_data: Dictionary mapping tickers to lists of financial data
    """
    # Initialize the store
    store = OptimizedDuckDBStore(db_path="data/test_optimized_financial_data.duckdb")

    # Get database stats before
    stats_before = store.get_database_stats()

    print("\n=== Database Statistics (Before) ===")
    print(f"Companies: {stats_before.get('company_count', 0)}")
    print(f"Filings: {stats_before.get('filing_count', 0)}")
    print(f"Financial Facts: {stats_before.get('fact_count', 0)}")
    print(f"Time Series Metrics: {stats_before.get('time_series_count', 0)}")
    print(f"Financial Ratios: {stats_before.get('ratio_count', 0)}")

    # Flatten financial data for batch storage
    all_financial_data = []
    for ticker, filings in financial_data.items():
        all_financial_data.extend(filings)

    # Start timer
    start_time = time.time()

    # Store financial data in batch
    stored_count = store.store_financial_data_batch(all_financial_data)

    # End timer
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"\n=== Batch Storage Results ===")
    print(f"Stored {stored_count} financial data records")
    print(f"Total time: {elapsed:.2f} seconds")

    # Get database stats after
    stats_after = store.get_database_stats()

    print("\n=== Database Statistics (After) ===")
    print(f"Companies: {stats_after.get('company_count', 0)}")
    print(f"Filings: {stats_after.get('filing_count', 0)}")
    print(f"Financial Facts: {stats_after.get('fact_count', 0)}")
    print(f"Time Series Metrics: {stats_after.get('time_series_count', 0)}")
    print(f"Financial Ratios: {stats_after.get('ratio_count', 0)}")

    # Test company metrics query
    for ticker in financial_data.keys():
        metrics_df = store.get_company_metrics(ticker=ticker)

        print(f"\n=== Company Metrics for {ticker} ===")
        print(metrics_df)

    # Test company comparison if multiple companies
    if len(financial_data.keys()) > 1:
        tickers = list(financial_data.keys())
        comparison_df = store.compare_companies(tickers=tickers, metric="revenue")

        print(f"\n=== Company Comparison (Revenue) ===")
        print(comparison_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test parallel XBRL extractor and optimized DuckDB store")
    parser.add_argument("--tickers", type=str, nargs="+", default=["AAPL", "MSFT"], help="Company ticker symbols")
    parser.add_argument(
        "--accessions",
        type=str,
        nargs="+",
        default=["0000320193-23-000077", "0000789019-23-001517"],
        help="SEC accession numbers",
    )
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of worker threads")

    args = parser.parse_args()

    # Create data directories if they don't exist
    os.makedirs("data/xbrl_cache", exist_ok=True)

    # Test parallel extraction
    results = test_parallel_extraction(
        tickers=args.tickers, accession_numbers=args.accessions, max_workers=args.workers
    )

    # Test optimized DuckDB store
    test_optimized_duckdb_store(results)
