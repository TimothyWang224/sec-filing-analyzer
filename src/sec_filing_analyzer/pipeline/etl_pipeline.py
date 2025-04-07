"""
SEC Filing ETL Pipeline

This module provides the main ETL pipeline for processing SEC filings.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from llama_index.embeddings.openai import OpenAIEmbedding

from sec_filing_analyzer.config import ETLConfig, StorageConfig
from sec_filing_analyzer.storage import GraphStore, LlamaIndexVectorStore
from sec_filing_analyzer.data_retrieval import SECFilingsDownloader, FilingProcessor
from sec_filing_analyzer.data_retrieval.file_storage import FileStorage
from sec_filing_analyzer.data_processing.chunking import FilingChunker

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
        sec_downloader: Optional[SECFilingsDownloader] = None,
    ):
        """Initialize the ETL pipeline."""
        self.graph_store = graph_store or GraphStore()
        self.vector_store = vector_store or LlamaIndexVectorStore(
            store_dir=StorageConfig().vector_store_path
        )
        self.file_storage = file_storage or FileStorage(
            base_dir=ETLConfig().cache_dir.parent / "filings"
        )
        self.filing_processor = filing_processor or FilingProcessor(
            graph_store=self.graph_store,
            vector_store=self.vector_store
        )
        self.sec_downloader = sec_downloader or SECFilingsDownloader(
            file_storage=self.file_storage
        )
        
        # Initialize embedding model
        self.embedding_model = OpenAIEmbedding()
        
        # Initialize filing chunker
        self.filing_chunker = FilingChunker(
            max_tokens=4000,  # Maximum tokens per chunk
            chunk_overlap=200  # Overlap between chunks
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
        # Download filings
        downloaded_filings = self.sec_downloader.download_company_filings(
            ticker=ticker,
            filing_types=filing_types or ETLConfig().filing_types,
            start_date=start_date,
            end_date=end_date
        )
        
        # Process each filing
        for filing_data in downloaded_filings:
            try:
                self.process_filing_data(filing_data)
            except Exception as e:
                logger.error(f"Error processing filing {filing_data['accession_number']}: {e}")
    
    def process_filing_data(self, filing_data: Dict[str, Any]) -> None:
        """
        Process filing data.
        
        Args:
            filing_data: The filing data to process
        """
        # Get filing content
        filing_content = self.file_storage.load_raw_filing(filing_data['accession_number'])
        if not filing_content:
            logger.error(f"Could not load content for filing {filing_data['accession_number']}")
            return
            
        # Get HTML content if available
        if filing_data.get('has_html'):
            try:
                html_content = self.file_storage.load_html_filing(filing_data['accession_number'])
                if html_content:
                    filing_content['html'] = html_content
            except Exception as e:
                logger.warning(f"Could not load HTML content for filing {filing_data['accession_number']}: {e}")
        
        # Process filing with chunker
        processed_data = self.filing_chunker.process_filing(filing_data, filing_content)
        
        # Process filing with embeddings
        self.filing_processor.process_filing(processed_data)
        
        # Save processed filing to disk
        self.file_storage.save_processed_filing(
            filing_id=filing_data['accession_number'],
            processed_data=processed_data,
            metadata=filing_data
        )
        
        # Cache filing for quick access
        self.file_storage.cache_filing(
            filing_id=filing_data['accession_number'],
            filing_data={
                "metadata": filing_data,
                "processed_data": processed_data
            }
        ) 