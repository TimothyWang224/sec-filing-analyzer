"""
SEC Filing ETL Pipeline

This module provides the main ETL pipeline for processing SEC filings.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from edgar import Company, Filing
from edgar.files.htmltools import chunk

from ..config import ETL_CONFIG, STORAGE_CONFIG
from ..storage import GraphStore, LlamaIndexVectorStore
from ..data_retrieval.filing_processor import FilingProcessor
from ..data_retrieval.file_storage import FileStorage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECFilingETLPipeline:
    """
    ETL pipeline for processing SEC filings.
    """
    
    def __init__(
        self,
        graph_store: Optional[GraphStore] = None,
        vector_store: Optional[LlamaIndexVectorStore] = None,
        filing_processor: Optional[FilingProcessor] = None,
        file_storage: Optional[FileStorage] = None,
    ):
        """Initialize the ETL pipeline."""
        self.graph_store = graph_store or GraphStore()
        self.vector_store = vector_store or LlamaIndexVectorStore(
            store_dir=STORAGE_CONFIG["vector_store_path"]
        )
        self.filing_processor = filing_processor or FilingProcessor(
            graph_store=self.graph_store,
            vector_store=self.vector_store
        )
        self.file_storage = file_storage or FileStorage(
            base_dir=ETL_CONFIG["cache_dir"].parent / "filings"
        )
    
    def process_company(
        self,
        ticker: str,
        filing_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """
        Process all filings for a company.
        
        Args:
            ticker: Company ticker symbol
            filing_types: List of filing types to process
            start_date: Start date for filing range
            end_date: End date for filing range
        """
        # Get company
        company = Company(ticker)
        
        # Get filings
        filings = company.get_filings(
            form_types=filing_types or ETL_CONFIG["default_filing_types"],
            start_date=start_date,
            end_date=end_date
        )
        
        # Process each filing
        for filing in filings:
            try:
                self.process_filing(filing)
            except Exception as e:
                logger.error(f"Error processing filing {filing.accession_number}: {e}")
    
    def process_filing(self, filing: Filing) -> None:
        """
        Process a single filing.
        
        Args:
            filing: The filing to process
        """
        # Download filing
        filing.download()
        
        # Download HTML content if available
        html_content = None
        try:
            html_content = filing.download_html()
            logger.info(f"Downloaded HTML content for filing {filing.accession_number}")
        except Exception as e:
            logger.warning(f"Could not download HTML content for filing {filing.accession_number}: {e}")
        
        # Extract year from filing date
        year = filing.filing_date.split("-")[0]
        
        # Create metadata with all available information
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
        
        # Process filing
        processed_data = self.filing_processor.process_filing({
            "id": filing.accession_number,
            "text": filing.text,
            "embedding": filing.embedding,
            "metadata": metadata
        })
        
        # Save processed filing to disk
        self.file_storage.save_processed_filing(
            filing_id=filing.accession_number,
            processed_data=processed_data,
            metadata=metadata
        )
        
        # Cache filing for quick access
        self.file_storage.cache_filing(
            filing_id=filing.accession_number,
            filing_data={
                "metadata": metadata,
                "processed_data": processed_data
            }
        ) 