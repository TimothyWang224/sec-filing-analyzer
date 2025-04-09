"""
SEC Filings Downloader

This module provides functionality for downloading SEC filings.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import date

from edgar import Filing
from .file_storage import FileStorage
from ..utils import edgar_utils
from ..data_processing import XBRLExtractorFactory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECFilingsDownloader:
    """
    Downloader for SEC filings.
    """

    def __init__(
        self,
        file_storage: Optional[FileStorage] = None,
        xbrl_cache_dir: Optional[str] = None,
        use_edgar_xbrl: bool = True
    ):
        """
        Initialize the SEC filings downloader.

        Args:
            file_storage: Optional FileStorage instance for storing files
            xbrl_cache_dir: Optional directory to cache XBRL data
            use_edgar_xbrl: Whether to use the edgar library for XBRL extraction
        """
        self.file_storage = file_storage or FileStorage()
        self.xbrl_cache_dir = xbrl_cache_dir
        self.xbrl_extractor = XBRLExtractorFactory.create_extractor(
            extractor_type="edgar" if use_edgar_xbrl else "simplified",
            cache_dir=xbrl_cache_dir
        )

    def download_company_filings(
        self,
        ticker: str,
        filing_types: Optional[List[str]] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Download all filings for a company.

        Args:
            ticker: Company ticker symbol
            filing_types: List of filing types to download
            start_date: Start date for filing range (YYYY-MM-DD string or date object)
            end_date: End date for filing range (YYYY-MM-DD string or date object)

        Returns:
            List of downloaded filing metadata
        """
        # Use standardized utility to get filings
        form_type = filing_types[0] if filing_types else None
        try:
            filings = edgar_utils.get_filings(
                ticker=ticker,
                form_type=form_type,
                start_date=start_date,
                end_date=end_date
            )

            logger.info(f"Retrieved {len(filings)} filings for {ticker}")
        except Exception as e:
            logger.error(f"Error retrieving filings for {ticker}: {e}")
            return []

        # Download each filing
        downloaded_filings = []
        for filing in filings:
            try:
                filing_data = self.download_filing(filing, ticker)
                if filing_data:
                    downloaded_filings.append(filing_data)
            except Exception as e:
                logger.error(f"Error downloading filing {filing.accession_number}: {e}")

        return downloaded_filings

    def download_filing(self, filing: Filing, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Download a single filing.

        Args:
            filing: The filing to download
            ticker: Company ticker symbol

        Returns:
            Dict containing filing data if successful, None otherwise
        """
        # Check if filing is already downloaded
        cached_data = self.file_storage.load_cached_filing(filing.accession_number)
        if cached_data:
            logger.info(f"Using cached data for filing {filing.accession_number}")
            # Extract metadata from cached data if available
            if isinstance(cached_data, dict) and 'metadata' in cached_data:
                return cached_data['metadata']
            # If the cached data doesn't have the expected structure, create metadata from the filing object
            return edgar_utils.get_filing_metadata(filing, ticker)

        # Download filing
        try:
            # Use standardized utility to get filing content
            content = edgar_utils.get_filing_content(filing)

            # Extract content components
            text_content = content.get("text")
            html_content = content.get("html")
            xml_content = content.get("xml")

            # Check if we have text content
            if not text_content:
                logger.warning(f"No text content available for filing {filing.accession_number}")
                return None

            # Log what content we found
            if html_content:
                logger.info(f"Downloaded HTML content for filing {filing.accession_number}")
            if xml_content:
                logger.info(f"Downloaded XML content for filing {filing.accession_number}")

            # Create metadata using standardized utility
            metadata = edgar_utils.get_filing_metadata(filing, ticker)

            # Add content availability flags
            metadata.update({
                "has_html": html_content is not None,
                "has_xml": xml_content is not None,
                "has_xbrl": content.get("xbrl") is not None
            })

            # Save raw filing to disk
            self.file_storage.save_raw_filing(
                filing_id=filing.accession_number,
                content=text_content,
                metadata=metadata
            )

            # Save HTML filing to disk if available
            if html_content:
                self.file_storage.save_html_filing(
                    filing_id=filing.accession_number,
                    html_content=html_content,
                    metadata=metadata
                )

            # Save XML filing to disk if available
            if xml_content:
                self.file_storage.save_xml_filing(
                    filing_id=filing.accession_number,
                    xml_content=xml_content,
                    metadata=metadata
                )

            return metadata

        except Exception as e:
            logger.error(f"Error downloading filing {filing.accession_number}: {e}")
            return None

    def get_filing(self, filing_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a filing by ID.

        Args:
            filing_id: The filing ID (accession number)

        Returns:
            Dict containing filing data if found, None otherwise
        """
        return self.file_storage.load_raw_filing(filing_id)

    def get_filing_by_accession(self, ticker: str, accession_number: str) -> Optional[Dict[str, Any]]:
        """
        Get a filing by accession number.

        Args:
            ticker: Company ticker symbol
            accession_number: SEC accession number

        Returns:
            Dict containing filing data if found, None otherwise
        """
        # Check if filing is already downloaded
        cached_data = self.file_storage.load_cached_filing(accession_number)
        if cached_data:
            logger.info(f"Using cached data for filing {accession_number}")
            return cached_data

        # Try to get from raw files
        raw_data = self.file_storage.load_raw_filing(accession_number)
        if raw_data:
            return raw_data

        # If not found locally, try to download it
        try:
            # Use standardized utility to get filing by accession number
            filing = edgar_utils.get_filing_by_accession(ticker, accession_number)
            if filing:
                # Download the filing
                filing_data = self.download_filing(filing, ticker)
                return filing_data
        except Exception as e:
            logger.error(f"Error getting filing by accession number {accession_number}: {e}")

        return None

    def extract_xbrl_data(self, ticker: str, filing_id: str, accession_number: str) -> Dict[str, Any]:
        """
        Extract XBRL data from a filing using the configured XBRL extractor.

        Args:
            ticker: Company ticker symbol
            filing_id: Unique identifier for the filing
            accession_number: SEC accession number

        Returns:
            Dictionary containing extracted XBRL data
        """
        try:
            # Get the filing URL if available
            filing_url = None
            filing_data = self.get_filing_by_accession(ticker, accession_number)
            if filing_data and 'filing_url' in filing_data:
                filing_url = filing_data['filing_url']

            # Use the XBRL extractor to extract data
            xbrl_data = self.xbrl_extractor.extract_financials(
                ticker=ticker,
                filing_id=filing_id,
                accession_number=accession_number,
                filing_url=filing_url
            )

            # Cache the XBRL data
            if xbrl_data and self.file_storage:
                self.file_storage.save_xbrl_data(accession_number, xbrl_data)

            return xbrl_data
        except Exception as e:
            logger.error(f"Error extracting XBRL data for {ticker} {accession_number}: {e}")
            return {
                "filing_id": filing_id,
                "ticker": ticker,
                "accession_number": accession_number,
                "error": str(e)
            }

    def list_filings(
        self,
        ticker: Optional[str] = None,
        year: Optional[str] = None,
        filing_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available filings.

        Args:
            ticker: Filter by company ticker
            year: Filter by filing year
            filing_type: Filter by filing type

        Returns:
            List of filing metadata
        """
        return self.file_storage.list_filings(
            ticker=ticker,
            year=year,
            filing_type=filing_type
        )