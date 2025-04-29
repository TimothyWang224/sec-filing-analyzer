"""
SEC Filings Tool for SEC Filing Analyzer.

This module provides a tool for retrieving information about SEC filings.
"""

import logging
from typing import Any, Dict, List, Optional

from src.tools.sec_data import SECDataTool

logger = logging.getLogger(__name__)


class SECFilingsTool(SECDataTool):
    """
    Tool for retrieving information about SEC filings.

    This tool allows retrieving metadata about SEC filings, such as filing dates,
    available filings for a company, and other filing-related information.
    """

    def __init__(self):
        """Initialize the SECFilingsTool."""
        super().__init__()

    def get_filings(
        self,
        ticker: str,
        filing_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get filings for a company.

        Args:
            ticker: Company ticker symbol
            filing_type: Optional filing type (e.g., 10-K, 10-Q)
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            Dictionary containing filing information
        """
        logger.info(f"Getting filings for {ticker}")

        # Use the underlying SECDataTool to retrieve the data
        parameters = {
            "query_type": "filings",
            "parameters": {
                "ticker": ticker,
                "filing_type": filing_type,
                "start_date": start_date,
                "end_date": end_date,
            },
        }

        results = self.execute(**parameters)

        return results
