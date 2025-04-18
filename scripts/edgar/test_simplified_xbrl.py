"""
Test Simplified XBRL Extractor

This script tests the simplified XBRL extractor.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sec_filing_analyzer.data_processing.simplified_xbrl_extractor import SimplifiedXBRLExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_simplified_xbrl_extractor(ticker: str, accession_number: str):
    """Test the simplified XBRL extractor.

    Args:
        ticker: Company ticker symbol
        accession_number: SEC accession number
    """
    # Initialize the extractor
    extractor = SimplifiedXBRLExtractor(cache_dir="data/xbrl_cache")

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

    # Print basic information
    print(f"\n=== Financial Data for {ticker} {accession_number} ===")
    print(f"Filing Date: {financials.get('filing_date')}")
    print(f"Fiscal Year: {financials.get('fiscal_year')}")
    print(f"Fiscal Quarter: {financials.get('fiscal_quarter')}")
    print(f"Filing Type: {financials.get('filing_type')}")

    # Print statements
    statements = financials.get("statements", {})
    print(f"\nStatements: {len(statements)}")
    for statement_name in statements:
        print(f"  {statement_name}")

    # Print key metrics
    metrics = financials.get("metrics", {})
    print(f"\nKey Metrics: {len(metrics)}")
    for name, value in list(metrics.items())[:10]:  # Show first 10 metrics
        print(f"  {name}: {value}")

    # Print ratios
    ratios = financials.get("ratios", {})
    print(f"\nFinancial Ratios: {len(ratios)}")
    for name, value in ratios.items():
        print(f"  {name}: {value:.4f}")

    # Print facts count
    facts = financials.get("facts", [])
    print(f"\nTotal Facts: {len(facts)}")

    # Save to file for inspection
    output_file = f"data/{ticker}_{accession_number.replace('-', '_')}_financials.json"
    with open(output_file, "w") as f:
        json.dump(financials, f, indent=2)

    print(f"\nFull data saved to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the simplified XBRL extractor")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Company ticker symbol")
    parser.add_argument("--accession", type=str, default="0000320193-23-000077", help="SEC accession number")

    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs("data/xbrl_cache", exist_ok=True)

    # Test the extractor
    test_simplified_xbrl_extractor(args.ticker, args.accession)
