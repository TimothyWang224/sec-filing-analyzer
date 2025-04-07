"""
SEC Filings Downloader

This module provides functionality for downloading SEC filings.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from edgar import Company, Filing
from .file_storage import FileStorage

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
    ):
        """Initialize the SEC filings downloader."""
        self.file_storage = file_storage or FileStorage()
    
    def download_company_filings(
        self,
        ticker: str,
        filing_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Download all filings for a company.
        
        Args:
            ticker: Company ticker symbol
            filing_types: List of filing types to download
            start_date: Start date for filing range
            end_date: End date for filing range
            
        Returns:
            List of downloaded filing metadata
        """
        # Get company
        company = Company(ticker)
        
        # Get filings
        filings = company.get_filings(
            form_types=filing_types,
            start_date=start_date,
            end_date=end_date
        )
        
        # Download each filing
        downloaded_filings = []
        for filing in filings:
            try:
                filing_data = self.download_filing(filing)
                if filing_data:
                    downloaded_filings.append(filing_data)
            except Exception as e:
                logger.error(f"Error downloading filing {filing.accession_number}: {e}")
        
        return downloaded_filings
    
    def download_filing(self, filing: Filing) -> Optional[Dict[str, Any]]:
        """
        Download a single filing.
        
        Args:
            filing: The filing to download
            
        Returns:
            Dict containing filing data if successful, None otherwise
        """
        # Check if filing is already downloaded
        cached_data = self.file_storage.load_cached_filing(filing.accession_number)
        if cached_data:
            logger.info(f"Using cached data for filing {filing.accession_number}")
            return cached_data
        
        # Download filing
        try:
            # Download text content
            filing.download()
            
            # Download HTML content if available
            html_content = None
            try:
                html_content = filing.download_html()
                logger.info(f"Downloaded HTML content for filing {filing.accession_number}")
            except Exception as e:
                logger.warning(f"Could not download HTML content for filing {filing.accession_number}: {e}")
            
            # Create metadata
            metadata = {
                "accession_number": filing.accession_number,
                "form": filing.form,
                "filing_date": filing.filing_date,
                "company": filing.company,
                "ticker": filing.ticker,
                "description": filing.description,
                "url": filing.url,
                "has_html": html_content is not None
            }
            
            # Save raw filing to disk
            self.file_storage.save_raw_filing(
                filing_id=filing.accession_number,
                content=filing.text,
                metadata=metadata
            )
            
            # Save HTML filing to disk if available
            if html_content:
                self.file_storage.save_html_filing(
                    filing_id=filing.accession_number,
                    html_content=html_content,
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
            filing_id: The filing ID
            
        Returns:
            Dict containing filing data if found, None otherwise
        """
        return self.file_storage.load_raw_filing(filing_id)
    
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