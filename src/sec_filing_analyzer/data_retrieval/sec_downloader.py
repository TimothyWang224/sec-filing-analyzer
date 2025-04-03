"""
SEC Filings Downloader

Handles downloading and caching of SEC filings.
"""

import os
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from rich.console import Console

from edgar.entities import Company
from edgar.core import set_identity
from edgar.httpclient import http_client, async_http_client
from edgar.httprequests import download_file, download_file_async, throttle_requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class SECFilingsDownloader:
    """Downloads and caches SEC filings."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the downloader.
        
        Args:
            cache_dir: Optional directory for caching filings
        """
        self.cache_dir = cache_dir or Path("data/cache/sec_filings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up SEC API identity
        self._set_identity()
        
    def _set_identity(self):
        """Set up the user agent info required by SEC EDGAR API."""
        user_agent = os.getenv("EDGAR_USER_AGENT")
        if not user_agent:
            raise ValueError("EDGAR_USER_AGENT must be set in .env file")
        set_identity(user_agent)
    
    async def get_filings(
        self,
        ticker: str,
        years: Optional[List[int]] = None,
        filing_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get SEC filings for a company.
        
        Args:
            ticker: Company ticker symbol
            years: Optional list of years to retrieve
            filing_types: Optional list of filing types to retrieve
            
        Returns:
            List of filing dictionaries with metadata and content
        """
        # Get company object
        company = Company(ticker)
        
        # Set date range
        if years:
            start_date = f"{min(years)}-01-01"
            end_date = f"{max(years)}-12-31"
        else:
            # Default to last year
            current_year = datetime.now().year
            start_date = f"{current_year-1}-01-01"
            end_date = f"{current_year-1}-12-31"
        
        # Get filings
        filings = company.get_filings(filing_date=f"{start_date}:{end_date}")
        
        # Filter by type if specified
        if filing_types:
            filings = [f for f in filings if f.form in filing_types]
        
        # Process filings
        processed_filings = []
        for filing in filings:
            # Check cache first
            cache_path = self._get_cache_path(filing)
            if cache_path.exists():
                processed_filings.append(self._load_from_cache(cache_path))
                continue
            
            # Download and process filing
            try:
                processed_filing = await self._process_filing(filing)
                processed_filings.append(processed_filing)
                
                # Cache the result
                self._save_to_cache(processed_filing, cache_path)
                
            except Exception as e:
                logger.error(f"Error processing filing {filing.accession_number}: {e}")
                continue
        
        return processed_filings
    
    def _get_cache_path(self, filing) -> Path:
        """Get cache path for a filing."""
        return self.cache_dir / f"{filing.accession_number}.json"
    
    def _load_from_cache(self, cache_path: Path) -> Dict[str, Any]:
        """Load filing from cache."""
        import json
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    def _save_to_cache(self, filing_data: Dict[str, Any], cache_path: Path):
        """Save filing to cache."""
        import json
        with open(cache_path, 'w') as f:
            json.dump(filing_data, f)
    
    async def _process_filing(self, filing) -> Dict[str, Any]:
        """Process a single filing."""
        # Get filing content
        content = await download_file_async(filing)
        
        # Extract metadata
        metadata = {
            "accession_number": filing.accession_number,
            "form": filing.form,
            "filing_date": filing.filing_date,
            "company": filing.company_name,
            "ticker": filing.ticker,
            "description": filing.description
        }
        
        return {
            **metadata,
            "content": content
        } 