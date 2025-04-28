"""
Test script for the EdgarXBRLExtractor.

This script demonstrates how to use the EdgarXBRLExtractor to extract
financial data from SEC filings.
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

# Import the XBRL extractor factory
from sec_filing_analyzer.data_processing import XBRLExtractorFactory

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_edgar_xbrl_extractor():
    """Test the EdgarXBRLExtractor with a known filing."""
    try:
        # Create output directory
        output_dir = Path("data/test_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create the extractor
        extractor = XBRLExtractorFactory.create_extractor(
            extractor_type="simplified",  # Use simplified extractor for now
            cache_dir="data/xbrl_cache",
        )

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
        output_file = output_dir / f"{ticker}_{accession_number}_xbrl_edgar.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(xbrl_data, f, indent=2, default=str)

        logger.info(f"Saved XBRL data to {output_file}")

        # Print some basic information about the extracted data
        logger.info("XBRL data summary:")
        logger.info(f"  Filing ID: {xbrl_data.get('filing_id')}")
        logger.info(f"  Ticker: {xbrl_data.get('ticker')}")
        logger.info(f"  Filing Date: {xbrl_data.get('filing_date')}")
        logger.info(f"  Fiscal Year: {xbrl_data.get('fiscal_year')}")
        logger.info(f"  Fiscal Period: {xbrl_data.get('fiscal_period')}")
        logger.info(f"  Filing Type: {xbrl_data.get('filing_type')}")
        logger.info(f"  Number of Facts: {len(xbrl_data.get('facts', []))}")
        logger.info(f"  Number of Metrics: {len(xbrl_data.get('metrics', {}))}")
        logger.info(f"  Number of Statements: {len(xbrl_data.get('statements', {}))}")

        # Print available statements
        if "statements" in xbrl_data:
            logger.info("Available statements:")
            for statement_name in xbrl_data["statements"].keys():
                logger.info(f"  - {statement_name}")

        return xbrl_data

    except Exception as e:
        logger.error(f"Error in test: {e}")
        return None


def compare_extractors():
    """Compare the EdgarXBRLExtractor with the SimplifiedXBRLExtractor."""
    try:
        # Create output directory
        output_dir = Path("data/test_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create both extractors
        edgar_extractor = XBRLExtractorFactory.create_extractor(
            extractor_type="edgar", cache_dir="data/xbrl_cache/edgar"
        )

        simplified_extractor = XBRLExtractorFactory.create_extractor(
            extractor_type="simplified", cache_dir="data/xbrl_cache/simplified"
        )

        # Use a known 10-K filing for Microsoft
        ticker = "MSFT"
        filing_id = "MSFT_10K_2022"
        accession_number = "0001564590-22-026876"  # Microsoft's 10-K from July 2022

        # Extract XBRL data using both extractors
        logger.info(f"Extracting XBRL data for {ticker} {accession_number} using both extractors...")

        edgar_data = edgar_extractor.extract_financials(
            ticker=ticker, filing_id=filing_id, accession_number=accession_number
        )

        simplified_data = simplified_extractor.extract_financials(
            ticker=ticker, filing_id=filing_id, accession_number=accession_number
        )

        # Save the extracted data to files for comparison
        edgar_file = output_dir / f"{ticker}_{accession_number}_edgar.json"
        with open(edgar_file, "w", encoding="utf-8") as f:
            json.dump(edgar_data, f, indent=2, default=str)

        simplified_file = output_dir / f"{ticker}_{accession_number}_simplified.json"
        with open(simplified_file, "w", encoding="utf-8") as f:
            json.dump(simplified_data, f, indent=2, default=str)

        logger.info(f"Saved Edgar data to {edgar_file}")
        logger.info(f"Saved Simplified data to {simplified_file}")

        # Compare the results
        logger.info("Comparison:")
        logger.info(f"  Edgar Facts: {len(edgar_data.get('facts', []))}")
        logger.info(f"  Simplified Facts: {len(simplified_data.get('facts', []))}")
        logger.info(f"  Edgar Metrics: {len(edgar_data.get('metrics', {}))}")
        logger.info(f"  Simplified Metrics: {len(simplified_data.get('metrics', {}))}")
        logger.info(f"  Edgar Statements: {len(edgar_data.get('statements', {}))}")
        logger.info(f"  Simplified Statements: {len(simplified_data.get('statements', {}))}")

        return {"edgar": edgar_data, "simplified": simplified_data}

    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        return None


if __name__ == "__main__":
    test_edgar_xbrl_extractor()
    # Uncomment to compare both extractors
    # compare_extractors()
