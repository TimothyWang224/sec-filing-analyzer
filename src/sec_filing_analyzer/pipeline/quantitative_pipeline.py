"""
Quantitative ETL Pipeline Module

This module provides a pipeline for extracting, transforming, and loading
quantitative data from SEC filings.
"""

import logging
from typing import Any, Dict, List, Optional

from ..data_retrieval.sec_downloader import SECFilingsDownloader
from ..quantitative.processing.edgar_xbrl_to_duckdb import EdgarXBRLToDuckDBExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantitativeETLPipeline:
    """
    A pipeline for extracting, transforming, and loading quantitative data from SEC filings.
    """

    def __init__(
        self,
        downloader: Optional[SECFilingsDownloader] = None,
        xbrl_extractor: Optional[EdgarXBRLToDuckDBExtractor] = None,
        db_path: Optional[str] = None,
        read_only: bool = True,
    ):
        """
        Initialize the quantitative ETL pipeline.

        Args:
            downloader: SEC filings downloader
            xbrl_extractor: XBRL to DuckDB extractor
            db_path: Path to the DuckDB database file
            read_only: Whether to open the database in read-only mode
        """
        self.downloader = downloader or SECFilingsDownloader()
        # Pass read_only parameter to the XBRL extractor
        self.xbrl_extractor = xbrl_extractor or EdgarXBRLToDuckDBExtractor(
            db_path=db_path, read_only=read_only
        )

        logger.info("Initialized quantitative ETL pipeline")

    def process_filing(
        self,
        ticker: str,
        filing_type: str,
        filing_date: Optional[str] = None,
        accession_number: Optional[str] = None,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a single filing through the quantitative ETL pipeline.

        Args:
            ticker: Company ticker symbol
            filing_type: Type of filing (e.g., '10-K', '10-Q')
            filing_date: Date of filing (optional)
            accession_number: SEC accession number (optional)
            force_download: Whether to force download even if cached

        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing {filing_type} filing for {ticker}")

            # Step 1: Download the filing if not already downloaded
            if not accession_number:
                # Get the filing object
                filings = self.downloader.get_filings(
                    ticker=ticker,
                    filing_types=[filing_type],
                    start_date=filing_date,
                    end_date=filing_date,
                    limit=1,
                )

                if not filings:
                    logger.error(f"Failed to find {filing_type} filing for {ticker}")
                    return {
                        "error": f"Failed to find {filing_type} filing for {ticker}"
                    }

                # Get the first filing
                filing = filings[0]

                # Download the filing
                filing_data = self.downloader.download_filing(filing, ticker)

                if not filing_data:
                    logger.error(
                        f"Failed to download {filing_type} filing for {ticker}"
                    )
                    return {
                        "error": f"Failed to download {filing_type} filing for {ticker}"
                    }

                accession_number = filing_data.get("accession_number")

            # Step 2: Extract XBRL data and store in DuckDB
            result = self.xbrl_extractor.process_filing(
                ticker=ticker, accession_number=accession_number
            )

            if "error" in result:
                logger.error(
                    f"Failed to extract XBRL data for {filing_type} filing for {ticker}: {result['error']}"
                )
                return {"error": result["error"]}

            logger.info(f"Successfully processed {filing_type} filing for {ticker}")

            # Get the filing ID from the result or generate one
            filing_id = result.get("filing_id") or f"{ticker}_{accession_number}"

            return {
                "status": "success",
                "filing_id": filing_id,
                "ticker": ticker,
                "filing_type": filing_type,
                "accession_number": accession_number,
                "has_xbrl": result.get("has_xbrl", False),
                "fiscal_info": result.get("fiscal_info", {}),
            }

        except Exception as e:
            logger.error(f"Error processing {filing_type} filing for {ticker}: {e}")
            return {"error": str(e)}

    def process_company_filings(
        self,
        ticker: str,
        filing_types: List[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Process multiple filings for a company through the quantitative ETL pipeline.

        Args:
            ticker: Company ticker symbol
            filing_types: List of filing types to process (e.g., ['10-K', '10-Q'])
            start_date: Start date for filings (optional)
            end_date: End date for filings (optional)
            limit: Maximum number of filings to process
            force_download: Whether to force download even if cached

        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing filings for {ticker}")

            # Default to 10-K and 10-Q filings if not specified
            if filing_types is None:
                filing_types = ["10-K", "10-Q"]

            # Step 1: Get the list of filings
            filings = self.downloader.get_filings(
                ticker=ticker,
                filing_types=filing_types,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )

            if not filings:
                logger.error(f"No filings found for {ticker}")
                return {"error": f"No filings found for {ticker}"}

            # Step 2: Process each filing
            results = []
            for filing in filings:
                result = self.process_filing(
                    ticker=ticker,
                    filing_type=filing.get("filing_type"),
                    accession_number=filing.get("accession_number"),
                    force_download=force_download,
                )
                results.append(result)

            logger.info(f"Successfully processed {len(results)} filings for {ticker}")

            return {
                "status": "success",
                "ticker": ticker,
                "num_filings": len(results),
                "results": results,
            }

        except Exception as e:
            logger.error(f"Error processing filings for {ticker}: {e}")
            return {"error": str(e)}

    def process_multiple_companies(
        self,
        tickers: List[str],
        filing_types: List[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit_per_company: int = 5,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Process filings for multiple companies through the quantitative ETL pipeline.

        Args:
            tickers: List of company ticker symbols
            filing_types: List of filing types to process (e.g., ['10-K', '10-Q'])
            start_date: Start date for filings (optional)
            end_date: End date for filings (optional)
            limit_per_company: Maximum number of filings to process per company
            force_download: Whether to force download even if cached

        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing filings for {len(tickers)} companies")

            # Process each company
            results = {}
            for ticker in tickers:
                result = self.process_company_filings(
                    ticker=ticker,
                    filing_types=filing_types,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit_per_company,
                    force_download=force_download,
                )
                results[ticker] = result

            logger.info(f"Successfully processed filings for {len(tickers)} companies")

            return {
                "status": "success",
                "num_companies": len(tickers),
                "results": results,
            }

        except Exception as e:
            logger.error(f"Error processing multiple companies: {e}")
            return {"error": str(e)}
