"""
Test script for the SEC downloader with XBRL extraction.
"""

import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Import SEC downloader
from sec_filing_analyzer.data_retrieval.sec_downloader import SECFilingsDownloader
from sec_filing_analyzer.data_retrieval.file_storage import FileStorage

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_xbrl_extraction():
    """Test XBRL extraction using the SEC downloader."""
    try:
        # Create output directory
        output_dir = Path("data/test_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize file storage and SEC downloader
        file_storage = FileStorage()
        sec_downloader = SECFilingsDownloader(
            file_storage=file_storage,
            xbrl_cache_dir="data/xbrl_cache",
            use_edgar_xbrl=True  # Use edgar extractor
        )

        # Use a known 10-K filing for Microsoft
        ticker = "MSFT"
        filing_id = "MSFT_10K_2022"
        accession_number = "0001564590-22-026876"  # Microsoft's 10-K from July 2022

        # Extract XBRL data
        logger.info(f"Extracting XBRL data for {ticker} {accession_number}...")
        xbrl_data = sec_downloader.extract_xbrl_data(
            ticker=ticker,
            filing_id=filing_id,
            accession_number=accession_number
        )

        # Check if extraction was successful
        if "error" in xbrl_data:
            logger.error(f"Error extracting XBRL data: {xbrl_data['error']}")
            return None

        # Save the extracted data to a file for inspection
        output_file = output_dir / f"{ticker}_{accession_number}_xbrl_sec.json"
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

        return xbrl_data

    except Exception as e:
        logger.error(f"Error in test: {e}")
        return None

if __name__ == "__main__":
    test_xbrl_extraction()
