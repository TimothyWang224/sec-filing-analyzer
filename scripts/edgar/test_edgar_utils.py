"""
Test script for the standardized edgar utilities.

This script demonstrates how to use the standardized edgar utilities
to retrieve SEC filings in a consistent way.
"""

import logging

from dotenv import load_dotenv

# Import standardized edgar utilities
from sec_filing_analyzer.utils import edgar_utils

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set edgar logger to DEBUG level
edgar_logger = logging.getLogger("edgar")
edgar_logger.setLevel(logging.DEBUG)

# Set utils logger to DEBUG level
utils_logger = logging.getLogger("sec_filing_analyzer.utils")
utils_logger.setLevel(logging.DEBUG)


def test_get_entity():
    """Test getting an entity by ticker."""
    try:
        # Get entity for Microsoft
        entity = edgar_utils.get_entity("MSFT")
        logger.info(f"Found entity: {entity.name} (CIK: {entity.cik})")
        return entity
    except Exception as e:
        logger.error(f"Error getting entity: {e}")
        return None


def test_get_filings(ticker="MSFT", form_type="8-K", limit=5):
    """Test getting filings for a company."""
    try:
        # Get filings for Microsoft
        filings = edgar_utils.get_filings(
            ticker=ticker, form_type=form_type, limit=limit
        )

        logger.info(f"Found {len(filings)} {form_type} filings for {ticker}")

        # Print filing details
        for i, filing in enumerate(filings[:limit]):
            logger.info(f"Filing {i + 1}:")
            logger.info(f"  Accession: {filing.accession_number}")
            logger.info(f"  Form: {filing.form}")
            logger.info(f"  Date: {filing.filing_date}")
            logger.info(f"  URL: {filing.filing_url}")

        return filings
    except Exception as e:
        logger.error(f"Error getting filings: {e}")
        return []


def test_get_filing_by_accession(ticker="MSFT", accession_number=None):
    """Test getting a filing by accession number."""
    try:
        # If no accession number provided, get the first filing and use its accession number
        if not accession_number:
            filings = test_get_filings(ticker, limit=1)
            if not filings:
                logger.error("No filings found to test with")
                return None
            accession_number = filings[0].accession_number

        # Get filing by accession number
        filing = edgar_utils.get_filing_by_accession(ticker, accession_number)

        if filing:
            logger.info(f"Found filing by accession number: {accession_number}")
            logger.info(f"  Form: {filing.form}")
            logger.info(f"  Date: {filing.filing_date}")
            logger.info(f"  URL: {filing.filing_url}")

            # Test getting filing content
            content = edgar_utils.get_filing_content(filing)

            # Check what content we got
            logger.info("Filing content:")
            logger.info(
                f"  Text: {'Available' if content.get('text') else 'Not available'}"
            )
            logger.info(
                f"  HTML: {'Available' if content.get('html') else 'Not available'}"
            )
            logger.info(
                f"  XML: {'Available' if content.get('xml') else 'Not available'}"
            )
            logger.info(
                f"  XBRL: {'Available' if content.get('xbrl') else 'Not available'}"
            )

            # Test getting filing metadata
            metadata = edgar_utils.get_filing_metadata(filing, ticker)
            logger.info(f"Filing metadata: {metadata}")

            return filing
        else:
            logger.error(f"Filing with accession number {accession_number} not found")
            return None
    except Exception as e:
        logger.error(f"Error getting filing by accession number: {e}")
        return None


def main():
    """Main function to run the tests."""
    logger.info("Testing edgar utilities...")

    # Test getting an entity
    entity = test_get_entity()
    if not entity:
        logger.error("Failed to get entity, aborting tests")
        return

    # Test getting filings
    filings = test_get_filings()
    if not filings:
        logger.error("Failed to get filings, aborting tests")
        return

    # Test getting a filing by accession number
    filing = test_get_filing_by_accession()
    if not filing:
        logger.error("Failed to get filing by accession number")

    logger.info("Tests completed")


if __name__ == "__main__":
    main()
