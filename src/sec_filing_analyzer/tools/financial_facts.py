"""
Financial Facts Tool for SEC Filing Analyzer.

This module provides a tool for retrieving financial facts from SEC filings.
"""

import logging
from typing import Any, Dict, List, Optional

from src.tools.sec_financial_data import SECFinancialDataTool

logger = logging.getLogger(__name__)


class FinancialFactsTool(SECFinancialDataTool):
    """
    Tool for retrieving financial facts from SEC filings.

    This tool allows retrieving structured financial data such as revenue, profit,
    and other financial metrics from SEC filings.
    """

    def __init__(self):
        """Initialize the FinancialFactsTool."""
        super().__init__()

    def get_financial_facts(
        self,
        ticker: str,
        metrics: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        filing_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get financial facts for a company.

        Args:
            ticker: Company ticker symbol
            metrics: List of financial metrics to retrieve
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            filing_type: Optional filing type (e.g., 10-K, 10-Q)

        Returns:
            Dictionary containing financial facts
        """
        logger.info(f"Getting financial facts for {ticker}: {metrics}")

        # Use the underlying SECFinancialDataTool to retrieve the data
        parameters = {
            "query_type": "financial_facts",
            "parameters": {
                "ticker": ticker,
                "metrics": metrics,
                "start_date": start_date,
                "end_date": end_date,
                "filing_type": filing_type,
            },
        }

        results = self.execute(**parameters)

        return results
