"""
Basic test script for the edgar library's XBRL capabilities.
"""

import json
import logging
from pathlib import Path

import edgar
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_edgar_xbrl():
    """Test the edgar library's XBRL capabilities."""
    try:
        # Create output directory
        output_dir = Path("data/test_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set edgar identity from environment variables
        import os

        edgar_identity = os.getenv("EDGAR_IDENTITY")
        if edgar_identity:
            edgar.set_identity(edgar_identity)
            logger.info(f"Set edgar identity to: {edgar_identity}")

        # Get Microsoft entity
        logger.info("Getting Microsoft entity...")
        msft = edgar.get_entity("MSFT")
        logger.info(f"Found Microsoft entity with CIK: {msft.cik}")

        # Get a specific filing
        logger.info("Getting a specific filing...")
        accession_number = "0001564590-22-026876"  # Microsoft's 10-K from July 2022

        # Get all filings
        filings = msft.get_filings()
        logger.info(f"Retrieved {len(filings)} filings")

        # Find the filing with the matching accession number
        filing = None
        for f in filings:
            if f.accession_number == accession_number:
                filing = f
                break

        if not filing:
            logger.error(f"Filing with accession number {accession_number} not found")
            return None

        logger.info(f"Found filing: {filing.form} filed on {filing.filing_date}")
        logger.info(f"Filing URL: {filing.filing_url}")

        # Check if the filing has XBRL data
        logger.info("Checking if filing has XBRL data...")
        has_xbrl = hasattr(filing, "is_xbrl") and filing.is_xbrl
        logger.info(f"Filing has XBRL data: {has_xbrl}")

        if has_xbrl:
            # Get XBRL data
            logger.info("Getting XBRL data...")
            xbrl_data = filing.xbrl

            # Save XBRL data to file
            output_file = output_dir / "msft_xbrl_basic.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(xbrl_data, f, indent=2, default=str)

            logger.info(f"Saved XBRL data to {output_file}")

            # Print some basic information about the XBRL data
            logger.info("XBRL data summary:")
            logger.info(f"  Type: {type(xbrl_data)}")
            logger.info(f"  Attributes: {dir(xbrl_data)}")

            # Try to access some common XBRL attributes
            if hasattr(xbrl_data, "instance"):
                logger.info("  Has instance")
                if hasattr(xbrl_data.instance, "facts"):
                    logger.info(f"  Number of facts: {len(xbrl_data.instance.facts)}")

            if hasattr(xbrl_data, "get_balance_sheet"):
                balance_sheet = xbrl_data.get_balance_sheet()
                logger.info(f"  Has balance sheet: {balance_sheet is not None}")

            if hasattr(xbrl_data, "get_income_statement"):
                income_statement = xbrl_data.get_income_statement()
                logger.info(f"  Has income statement: {income_statement is not None}")

            return xbrl_data
        else:
            logger.warning("Filing does not have XBRL data")
            return None

    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_edgar_xbrl()
