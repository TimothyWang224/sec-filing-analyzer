"""
SEC Filing Entity Extractor

This module provides functionality for extracting and analyzing entities from SEC filings,
leveraging edgartools for entity handling and maintaining graph-specific functionality.
"""

import logging
from typing import Any, Dict, List, Optional

# Import from the installed edgar package
from edgar import Company, Document, EntityData, find_company, get_entity_submissions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SECEntities:
    """
    Class for extracting and analyzing entities from SEC filings.
    Combines edgartools entity handling with graph-specific functionality.
    """

    def __init__(self):
        """Initialize the SEC entities extractor."""
        pass

    def get_company_data(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get company data using edgartools.

        Args:
            identifier: Company identifier (CIK or ticker)

        Returns:
            Dict containing company data if found, None otherwise
        """
        try:
            company = find_company(identifier)
            if company:
                return self._format_company_data(company)
        except Exception as e:
            logger.error(f"Error getting company data: {e}")
        return None

    def _format_company_data(self, company: Company) -> Dict[str, Any]:
        """
        Format company data into a standardized structure.

        Args:
            company: Company object from edgartools

        Returns:
            Dict containing formatted company data
        """
        return {
            "cik": company.cik,
            "name": company.name,
            "tickers": company.tickers,
            "sic": company.sic,
            "industry": company.industry,
            "address": company.address,
            "phone": company.phone,
            "website": company.website,
            "fiscal_year_end": company.fiscal_year_end,
            "entity_data": self._get_entity_data(company.cik),
        }

    def _get_entity_data(self, cik: str) -> Optional[EntityData]:
        """
        Get entity data using edgartools.

        Args:
            cik: Company CIK

        Returns:
            EntityData object if found, None otherwise
        """
        try:
            return get_entity_submissions(cik)
        except Exception as e:
            logger.error(f"Error getting entity data: {e}")
        return None

    def get_filing_metadata(self, filing_content: str) -> Dict[str, Any]:
        """
        Extract metadata from filing content.

        Args:
            filing_content: The content of the SEC filing

        Returns:
            Dict containing filing metadata
        """
        metadata = {}

        try:
            # Create document
            doc = Document.parse(filing_content)
            if doc and hasattr(doc, "metadata") and doc.metadata is not None:
                metadata = doc.metadata
        except Exception as e:
            logger.error(f"Error extracting filing metadata: {e}")

        return metadata

    def extract_entities(self, filing_content: str) -> List[Dict[str, Any]]:
        """
        Extract entities from filing content.

        Args:
            filing_content: The content of the SEC filing

        Returns:
            List of extracted entities
        """
        entities = []

        # First try to extract from metadata
        metadata = self.get_filing_metadata(filing_content)
        if metadata.get("cik"):
            company_data = self.get_company_data(metadata["cik"])
            if company_data:
                entities.append(company_data)

        # If no entities found from metadata, use NLP techniques
        if not entities:
            # TODO: Implement NLP-based entity extraction
            pass

        return entities
