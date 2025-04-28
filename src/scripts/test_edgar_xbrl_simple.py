"""
Test script for the simplified EdgarXBRLExtractor.
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

# Import the simplified XBRL extractor
from sec_filing_analyzer.data_processing.edgar_xbrl_extractor_simple import (
    EdgarXBRLExtractorSimple,
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_edgar_xbrl_simple():
    """Test the simplified EdgarXBRLExtractor with a known filing."""
    try:
        # Create output directory
        output_dir = Path("data/test_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create the extractor
        extractor = EdgarXBRLExtractorSimple(cache_dir="data/xbrl_cache")

        # Use a known 10-K filing for Microsoft
        ticker = "MSFT"
        filing_id = "MSFT_10K_2022"
        accession_number = "0001564590-22-026876"  # Microsoft's 10-K from July 2022

        # Extract XBRL data
        logger.info(f"Extracting XBRL data for {ticker} {accession_number}...")
        xbrl_data = extractor.extract_financials(ticker=ticker, filing_id=filing_id, accession_number=accession_number)

        # Check if extraction was successful
        if "error" in xbrl_data:
            logger.error(f"Error extracting XBRL data: {xbrl_data['error']}")
            return None

        # Save the extracted data to a file for inspection
        output_file = output_dir / f"{ticker}_{accession_number}_xbrl_simple.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(xbrl_data, f, indent=2, default=str)

        logger.info(f"Saved XBRL data to {output_file}")

        # Print some basic information about the extracted data
        logger.info("XBRL data summary:")
        logger.info(f"  Filing ID: {xbrl_data.get('filing_id')}")
        logger.info(f"  Ticker: {xbrl_data.get('ticker')}")
        logger.info(f"  Filing Date: {xbrl_data.get('filing_date')}")
        logger.info(f"  Filing Type: {xbrl_data.get('filing_type')}")

        # Print statements
        if "statements" in xbrl_data:
            statements = xbrl_data["statements"]
            logger.info(f"  Number of Statements: {len(statements)}")
            for statement_name, statement_data in statements.items():
                logger.info(f"    {statement_name}: {len(statement_data.get('line_items', []))} line items")

        # Print metrics
        if "metrics" in xbrl_data:
            metrics = xbrl_data["metrics"]
            logger.info(f"  Number of Metrics: {len(metrics)}")
            for metric_name, metric_value in metrics.items():
                logger.info(f"    {metric_name}: {metric_value}")

        return xbrl_data

    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_edgar_xbrl_simple()
