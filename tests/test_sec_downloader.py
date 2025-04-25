"""
Test script for the SEC Filings Downloader.

This script tests downloading NVDA filings from 2023.
"""

import logging

from dotenv import load_dotenv

from sec_filing_analyzer.config import ETLConfig
from sec_filing_analyzer.data_retrieval.file_storage import FileStorage
from sec_filing_analyzer.data_retrieval.sec_downloader import SECFilingsDownloader

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_download_nvda_filings():
    """Test downloading NVDA filings."""
    try:
        # Initialize file storage
        file_storage = FileStorage(base_dir=ETLConfig().filings_dir)

        # Initialize SEC downloader
        sec_downloader = SECFilingsDownloader(file_storage=file_storage)

        # Download NVDA filings
        logger.info("Downloading NVDA filings from 2023")
        filings = sec_downloader.download_company_filings(
            ticker="NVDA", filing_types=["10-K"], start_date="2023-01-01", end_date="2023-12-31"
        )

        # Log results
        logger.info(f"Downloaded {len(filings)} filings")
        for i, filing in enumerate(filings):
            logger.info(
                f"Filing {i + 1}: {filing.get('form', 'Unknown')} from {filing.get('filing_date', 'Unknown date')}"
            )
            logger.info(f"Accession number: {filing.get('accession_number', 'Unknown')}")
            logger.info(f"Company: {filing.get('company', 'Unknown')} ({filing.get('ticker', 'Unknown')})")

    except Exception as e:
        logger.error(f"Error downloading filings: {str(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    test_download_nvda_filings()
