"""
Test script for the SEC downloader with standardized edgar utilities.
"""

import logging

from dotenv import load_dotenv

from sec_filing_analyzer.data_retrieval.file_storage import FileStorage

# Import SEC downloader
from sec_filing_analyzer.data_retrieval.sec_downloader import SECFilingsDownloader

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_download_filings():
    """Test downloading filings for Microsoft."""
    try:
        # Initialize file storage and SEC downloader
        file_storage = FileStorage()
        sec_downloader = SECFilingsDownloader(file_storage=file_storage)

        # Download filings for Microsoft
        logger.info("Downloading filings for Microsoft...")
        filings = sec_downloader.download_company_filings(
            ticker="MSFT",
            filing_types=["8-K"],
            start_date="2022-01-01",
            end_date="2022-12-31",
        )

        logger.info(f"Downloaded {len(filings)} filings")

        # Print details of the first few filings
        for i, filing in enumerate(filings[:3]):
            logger.info(f"Filing {i + 1}:")
            logger.info(f"  Accession: {filing.get('accession_number')}")
            logger.info(f"  Form: {filing.get('form')}")
            logger.info(f"  Date: {filing.get('filing_date')}")

        return filings
    except Exception as e:
        logger.error(f"Error downloading filings: {e}")
        return []


if __name__ == "__main__":
    test_download_filings()
