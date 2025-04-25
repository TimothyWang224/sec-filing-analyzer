"""
Test script for retrieving a specific SEC filing using the standardized edgar utilities.
"""

import logging

from dotenv import load_dotenv

# Import standardized edgar utilities
from sec_filing_analyzer.utils import edgar_utils

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_specific_filing():
    """Test retrieving a specific filing for Microsoft."""
    try:
        # Get entity for Microsoft
        logger.info("Getting Microsoft entity...")
        entity = edgar_utils.get_entity("MSFT")
        logger.info(f"Found entity: {entity.name} (CIK: {entity.cik})")

        # Get a specific 8-K filing from 2022
        logger.info("Getting a specific 8-K filing from 2022...")
        filings = edgar_utils.get_filings(
            ticker="MSFT", form_type="8-K", start_date="2022-01-01", end_date="2022-12-31", limit=1
        )

        if not filings:
            logger.error("No 8-K filings found for Microsoft in 2022")
            return

        filing = filings[0]
        logger.info(f"Found filing: {filing.form} from {filing.filing_date}")
        logger.info(f"Accession number: {filing.accession_number}")
        logger.info(f"Filing URL: {filing.filing_url}")

        # Get filing content
        logger.info("Getting filing content...")
        content = edgar_utils.get_filing_content(filing)

        # Check what content we got
        logger.info("Filing content:")
        logger.info(f"  Text: {'Available' if content.get('text') else 'Not available'}")
        logger.info(f"  HTML: {'Available' if content.get('html') else 'Not available'}")
        logger.info(f"  XML: {'Available' if content.get('xml') else 'Not available'}")
        logger.info(f"  XBRL: {'Available' if content.get('xbrl') else 'Not available'}")

        # Get filing metadata
        logger.info("Getting filing metadata...")
        metadata = edgar_utils.get_filing_metadata(filing, "MSFT")
        logger.info(f"Filing metadata: {metadata}")

        return filing
    except Exception as e:
        logger.error(f"Error in test: {e}")
        return None


if __name__ == "__main__":
    test_specific_filing()
