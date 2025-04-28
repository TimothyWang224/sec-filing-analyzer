"""
ETL Pipeline Integration Module

This module integrates the XBRL extraction and financial data storage
with the existing ETL pipeline.
"""

import logging
from typing import Any, Dict, Optional

from sec_filing_analyzer.data_processing.xbrl_extractor import XBRLExtractor
from sec_filing_analyzer.storage.financial_data_store import FinancialDataStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialDataProcessor:
    """
    Processes financial data from SEC filings and integrates with the ETL pipeline.
    """

    def __init__(
        self, xbrl_cache_dir: Optional[str] = None, db_path: Optional[str] = None
    ):
        """Initialize the financial data processor.

        Args:
            xbrl_cache_dir: Directory to cache XBRL data
            db_path: Path to the DuckDB database
        """
        self.xbrl_extractor = XBRLExtractor(cache_dir=xbrl_cache_dir)
        self.financial_store = FinancialDataStore(db_path=db_path)
        logger.info("Initialized financial data processor")

    def process_filing(self, filing_id: str, filing_metadata: Dict[str, Any]) -> bool:
        """Process a filing and extract financial data.

        Args:
            filing_id: Internal filing ID
            filing_metadata: Filing metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract required metadata
            ticker = filing_metadata.get("ticker")
            accession_number = filing_metadata.get("accession_number")
            filing_url = filing_metadata.get("filing_url")

            if not ticker or not accession_number:
                logger.warning(f"Missing required metadata for filing {filing_id}")
                return False

            # Check if filing has XBRL data
            has_xbrl = filing_metadata.get("has_xbrl", True)
            if not has_xbrl:
                logger.info(f"Filing {filing_id} does not have XBRL data")
                return False

            # Extract XBRL data
            xbrl_data = self.xbrl_extractor.extract_financials(
                ticker=ticker,
                filing_id=filing_id,
                accession_number=accession_number,
                filing_url=filing_url,
            )

            # Store XBRL data
            success = self.financial_store.store_xbrl_data(xbrl_data)

            if success:
                logger.info(
                    f"Successfully processed financial data for filing {filing_id}"
                )
            else:
                logger.warning(
                    f"Failed to process financial data for filing {filing_id}"
                )

            return success
        except Exception as e:
            logger.error(f"Error processing financial data for filing {filing_id}: {e}")
            return False


# Integration function for the ETL pipeline
def process_filing_financials(
    filing_id: str,
    filing_metadata: Dict[str, Any],
    processor: Optional[FinancialDataProcessor] = None,
) -> bool:
    """Process financial data for a filing.

    This function can be called from the ETL pipeline to process financial data.

    Args:
        filing_id: Internal filing ID
        filing_metadata: Filing metadata
        processor: Optional FinancialDataProcessor instance

    Returns:
        True if successful, False otherwise
    """
    # Create processor if not provided
    if processor is None:
        processor = FinancialDataProcessor()

    # Process filing
    return processor.process_filing(filing_id, filing_metadata)
