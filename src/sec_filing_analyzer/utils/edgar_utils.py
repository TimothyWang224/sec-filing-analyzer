"""
Edgar Utilities

This module provides standardized utility functions for interacting with the edgar library.
It ensures consistent usage of edgar's capabilities throughout the codebase.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
import edgar
from edgar import Filing

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_entity(ticker: str) -> Any:
    """
    Get an entity (company) by ticker symbol using edgar's built-in capabilities.

    Args:
        ticker: Company ticker symbol

    Returns:
        Entity object from edgar library

    Raises:
        ValueError: If entity cannot be found
    """
    try:
        entity = edgar.get_entity(ticker)
        if not entity:
            raise ValueError(f"Entity not found for ticker: {ticker}")
        return entity
    except Exception as e:
        logger.error(f"Error getting entity for ticker {ticker}: {e}")
        raise ValueError(f"Failed to get entity for ticker {ticker}: {str(e)}")

def get_filings(
    ticker: str,
    form_type: Optional[str] = None,
    start_date: Optional[Union[str, date]] = None,
    end_date: Optional[Union[str, date]] = None,
    limit: Optional[int] = None
) -> List[Filing]:
    """
    Get filings for a company using edgar's built-in capabilities.

    Args:
        ticker: Company ticker symbol
        form_type: Optional filing type (e.g., "10-K", "10-Q", "8-K")
        start_date: Optional start date for filtering (YYYY-MM-DD string or date object)
        end_date: Optional end date for filtering (YYYY-MM-DD string or date object)
        limit: Optional maximum number of filings to return

    Returns:
        List of Filing objects from edgar library

    Raises:
        ValueError: If filings cannot be retrieved
    """
    try:
        # Get entity
        entity = get_entity(ticker)

        # Prepare date parameter if dates are provided
        date_param = None
        if start_date or end_date:
            # Convert string dates to date objects if needed
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

            # Format date parameter as expected by edgar library
            start_str = start_date.strftime("%Y-%m-%d") if start_date else ""
            end_str = end_date.strftime("%Y-%m-%d") if end_date else ""
            date_param = f"{start_str}:{end_str}"

        # Get filings using edgar's built-in filtering
        # Note: The edgar library doesn't support the limit parameter directly
        filings = entity.get_filings(form=form_type, date=date_param)

        # If no filings found, log a warning
        if not filings:
            logger.warning(f"No filings found for {ticker} with form {form_type} in date range")
            return []

        # Apply limit if specified
        if limit and len(filings) > limit:
            filings = filings[:limit]

        return filings
    except Exception as e:
        logger.error(f"Error getting filings for ticker {ticker}: {e}")
        raise ValueError(f"Failed to get filings for ticker {ticker}: {str(e)}")

def get_filing_by_accession(ticker: str, accession_number: str) -> Optional[Filing]:
    """
    Get a specific filing by accession number using edgar's built-in capabilities.

    Args:
        ticker: Company ticker symbol
        accession_number: SEC accession number

    Returns:
        Filing object if found, None otherwise
    """
    try:
        # Get all filings for the company
        filings = get_filings(ticker)

        # Find the filing with the matching accession number
        for filing in filings:
            if filing.accession_number == accession_number:
                return filing

        # If no matching filing found, log a warning
        logger.warning(f"No filing found for {ticker} with accession number {accession_number}")
        return None
    except Exception as e:
        logger.error(f"Error getting filing by accession number for ticker {ticker}: {e}")
        return None

def get_filing_content(filing: Filing) -> Dict[str, Any]:
    """
    Get content from a filing using edgar's built-in capabilities.

    Args:
        filing: Filing object from edgar library

    Returns:
        Dictionary containing text, html, and xml content if available
    """
    content = {
        "text": None,
        "html": None,
        "xml": None,
        "xbrl": None
    }

    try:
        # Get text content
        content["text"] = filing.text()
    except Exception as e:
        logger.warning(f"Error getting text content for filing {filing.accession_number}: {e}")

    try:
        # Get HTML content
        content["html"] = filing.html()
    except Exception as e:
        logger.warning(f"Error getting HTML content for filing {filing.accession_number}: {e}")

    try:
        # Get XML content
        content["xml"] = filing.xml()
    except Exception as e:
        logger.warning(f"Error getting XML content for filing {filing.accession_number}: {e}")

    try:
        # Get XBRL data if available
        if hasattr(filing, 'is_xbrl') and filing.is_xbrl:
            content["xbrl"] = filing.xbrl
    except Exception as e:
        logger.warning(f"Error getting XBRL data for filing {filing.accession_number}: {e}")

    return content

def get_filing_metadata(filing: Filing, ticker: str) -> Dict[str, Any]:
    """
    Get metadata from a filing using edgar's built-in capabilities.

    Args:
        filing: Filing object from edgar library
        ticker: Company ticker symbol

    Returns:
        Dictionary containing filing metadata
    """
    try:
        # Convert filing date to ISO format string if it's a date object
        filing_date = filing.filing_date
        if isinstance(filing_date, (datetime, date)):
            filing_date = filing_date.isoformat()

        metadata = {
            "accession_number": filing.accession_number,
            "id": filing.accession_number,  # Add id field as a duplicate of accession_number
            "form": filing.form,
            "filing_type": filing.form,  # Add filing_type as a duplicate of form for consistency
            "filing_date": filing_date,
            "company": filing.company,
            "ticker": ticker,
            "cik": filing.cik,
            "filing_url": filing.filing_url
        }
        return metadata
    except Exception as e:
        logger.error(f"Error getting metadata for filing {filing.accession_number}: {e}")
        # Return basic metadata if full metadata cannot be retrieved
        return {
            "accession_number": filing.accession_number,
            "id": filing.accession_number,  # Add id field as a duplicate of accession_number
            "form": filing.form,
            "ticker": ticker
        }
