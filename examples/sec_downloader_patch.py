"""
Patch for SECFilingsDownloader to support synthetic data mode.

This module patches the SECFilingsDownloader class to support synthetic data mode
for the demo. It's imported by run_nvda_etl.py when running in test mode.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from sec_filing_analyzer.data_retrieval.sec_downloader import SECFilingsDownloader

# Save the original get_filings method
original_get_filings = SECFilingsDownloader.get_filings

# Configure logging
logger = logging.getLogger(__name__)


def patched_get_filings(
    self,
    ticker: str,
    form_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    """
    Patched version of get_filings that returns synthetic data when in test mode.
    """
    # Check if we're in synthetic data mode
    if os.environ.get("SEC_USE_SYNTHETIC_DATA", "False").lower() in ("true", "1", "t"):
        logger.info(f"Using synthetic filings for {ticker}")

        # Get the synthetic data path
        synthetic_data_path = os.environ.get("SEC_SYNTHETIC_DATA_PATH", "data/synthetic/nvda_stub.txt")

        # Create a synthetic filing object
        filing = {
            "accession_number": "0001045810-24-000010",
            "form": form_type or "10-K",
            "filing_date": "2024-02-21",
            "document_url": synthetic_data_path,
            "ticker": ticker,
            "company_name": "NVIDIA CORPORATION",
            "cik": "0001045810",
            "fiscal_year": 2023,
            "fiscal_period": "FY",
        }

        # Return a list with the synthetic filing
        return [filing]

    # Otherwise, call the original method
    return original_get_filings(self, ticker, form_type, start_date, end_date, limit)


# Apply the patch
SECFilingsDownloader.get_filings = patched_get_filings
