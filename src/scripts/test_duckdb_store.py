"""
Test DuckDB Financial Store

This script tests the DuckDB financial store.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sec_filing_analyzer.storage.duckdb_financial_store import DuckDBFinancialStore
from sec_filing_analyzer.data_processing.simplified_xbrl_extractor import SimplifiedXBRLExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_duckdb_store():
    """Test the DuckDB financial store."""
    # Initialize the store
    store = DuckDBFinancialStore(db_path="data/test_financial_data.duckdb")
    
    # Get database stats
    stats = store.get_database_stats()
    
    print("\n=== Database Statistics ===")
    print(f"Companies: {stats.get('company_count', 0)}")
    print(f"Filings: {stats.get('filing_count', 0)}")
    print(f"Financial Facts: {stats.get('fact_count', 0)}")
    print(f"Time Series Metrics: {stats.get('time_series_count', 0)}")
    print(f"Financial Ratios: {stats.get('ratio_count', 0)}")
    
    # Store a company
    store.store_company(
        ticker="AAPL",
        name="Apple Inc.",
        cik="0000320193",
        sector="Technology",
        industry="Consumer Electronics"
    )
    
    # Store a filing
    store.store_filing(
        filing_id="AAPL_10K_2023",
        ticker="AAPL",
        accession_number="0000320193-23-000077",
        filing_type="10-K",
        filing_date="2023-10-27",
        fiscal_year=2023,
        fiscal_quarter=4
    )
    
    # Store some financial facts
    facts = [
        {
            "xbrl_tag": "us-gaap:Revenue",
            "metric_name": "revenue",
            "value": 394328000000,
            "unit": "USD",
            "period_type": "duration",
            "start_date": "2022-09-25",
            "end_date": "2023-09-30"
        },
        {
            "xbrl_tag": "us-gaap:NetIncomeLoss",
            "metric_name": "net_income",
            "value": 96995000000,
            "unit": "USD",
            "period_type": "duration",
            "start_date": "2022-09-25",
            "end_date": "2023-09-30"
        },
        {
            "xbrl_tag": "us-gaap:Assets",
            "metric_name": "total_assets",
            "value": 352583000000,
            "unit": "USD",
            "period_type": "instant",
            "end_date": "2023-09-30"
        }
    ]
    
    store.store_financial_facts("AAPL_10K_2023", facts)
    
    # Store time series metrics
    metrics = {
        "revenue": 394328000000,
        "net_income": 96995000000,
        "total_assets": 352583000000
    }
    
    store.store_time_series_metrics(
        ticker="AAPL",
        filing_id="AAPL_10K_2023",
        fiscal_year=2023,
        fiscal_quarter=4,
        metrics=metrics
    )
    
    # Store financial ratios
    ratios = {
        "net_margin": 0.246,
        "return_on_assets": 0.275,
        "return_on_equity": 0.789
    }
    
    store.store_financial_ratios(
        ticker="AAPL",
        filing_id="AAPL_10K_2023",
        fiscal_year=2023,
        fiscal_quarter=4,
        ratios=ratios
    )
    
    # Get updated database stats
    stats = store.get_database_stats()
    
    print("\n=== Updated Database Statistics ===")
    print(f"Companies: {stats.get('company_count', 0)}")
    print(f"Filings: {stats.get('filing_count', 0)}")
    print(f"Financial Facts: {stats.get('fact_count', 0)}")
    print(f"Time Series Metrics: {stats.get('time_series_count', 0)}")
    print(f"Financial Ratios: {stats.get('ratio_count', 0)}")
    
    # Get company metrics
    metrics_df = store.get_company_metrics(
        ticker="AAPL",
        metrics=["revenue", "net_income", "total_assets"]
    )
    
    print("\n=== Company Metrics ===")
    print(metrics_df)
    
    # Get financial ratios
    ratios_df = store.get_financial_ratios(
        ticker="AAPL"
    )
    
    print("\n=== Financial Ratios ===")
    print(ratios_df)

def test_with_extracted_data(ticker: str, accession_number: str):
    """Test the DuckDB financial store with extracted data.
    
    Args:
        ticker: Company ticker symbol
        accession_number: SEC accession number
    """
    # Initialize the extractor
    extractor = SimplifiedXBRLExtractor(cache_dir="data/xbrl_cache")
    
    # Initialize the store
    store = DuckDBFinancialStore(db_path="data/test_financial_data.duckdb")
    
    # Generate a filing ID
    filing_id = f"{ticker}_{accession_number.replace('-', '_')}"
    
    # Extract financials
    financials = extractor.extract_financials(
        ticker=ticker,
        filing_id=filing_id,
        accession_number=accession_number
    )
    
    # Check for errors
    if "error" in financials:
        logger.error(f"Error extracting financials: {financials['error']}")
        return
    
    # Store financial data
    success = store.store_financial_data(financials)
    
    if success:
        logger.info(f"Successfully stored financial data for {ticker} {accession_number}")
    else:
        logger.error(f"Failed to store financial data for {ticker} {accession_number}")
    
    # Get database stats
    stats = store.get_database_stats()
    
    print("\n=== Database Statistics ===")
    print(f"Companies: {stats.get('company_count', 0)}")
    print(f"Filings: {stats.get('filing_count', 0)}")
    print(f"Financial Facts: {stats.get('fact_count', 0)}")
    print(f"Time Series Metrics: {stats.get('time_series_count', 0)}")
    print(f"Financial Ratios: {stats.get('ratio_count', 0)}")
    
    # Get company metrics
    metrics_df = store.get_company_metrics(ticker=ticker)
    
    print("\n=== Company Metrics ===")
    print(metrics_df)
    
    # Get financial ratios
    ratios_df = store.get_financial_ratios(ticker=ticker)
    
    print("\n=== Financial Ratios ===")
    print(ratios_df)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the DuckDB financial store")
    parser.add_argument("--ticker", type=str, help="Company ticker symbol")
    parser.add_argument("--accession", type=str, help="SEC accession number")
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    if args.ticker and args.accession:
        # Test with extracted data
        test_with_extracted_data(args.ticker, args.accession)
    else:
        # Test with sample data
        test_duckdb_store()
