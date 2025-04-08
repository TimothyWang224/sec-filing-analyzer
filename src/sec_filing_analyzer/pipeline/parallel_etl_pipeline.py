"""
Parallel SEC Filing ETL Pipeline

This module provides a parallelized ETL pipeline for processing SEC filings.
"""

import logging
import concurrent.futures
from typing import Dict, List, Any, Optional
from pathlib import Path

from sec_filing_analyzer.config import ETLConfig, StorageConfig
from sec_filing_analyzer.storage import GraphStore, LlamaIndexVectorStore
from sec_filing_analyzer.data_retrieval import SECFilingsDownloader
from sec_filing_analyzer.data_retrieval.parallel_filing_processor import ParallelFilingProcessor
from sec_filing_analyzer.data_retrieval.file_storage import FileStorage
from sec_filing_analyzer.data_processing.chunking import FilingChunker
from sec_filing_analyzer.embeddings.parallel_embeddings import ParallelEmbeddingGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelSECFilingETLPipeline:
    """
    Parallel ETL pipeline for processing SEC filings.
    """

    def __init__(
        self,
        graph_store: Optional[GraphStore] = None,
        vector_store: Optional[LlamaIndexVectorStore] = None,
        filing_processor: Optional[ParallelFilingProcessor] = None,
        file_storage: Optional[FileStorage] = None,
        sec_downloader: Optional[SECFilingsDownloader] = None,
        max_workers: int = 4,
        batch_size: int = 100,
        rate_limit: float = 0.1
    ):
        """Initialize the parallel ETL pipeline."""
        self.graph_store = graph_store or GraphStore()
        self.vector_store = vector_store or LlamaIndexVectorStore(
            store_path=StorageConfig().vector_store_path
        )
        self.file_storage = file_storage or FileStorage(
            base_dir=ETLConfig().filings_dir
        )
        self.filing_processor = filing_processor or ParallelFilingProcessor(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            file_storage=self.file_storage,
            max_workers=max_workers
        )
        self.sec_downloader = sec_downloader or SECFilingsDownloader(
            file_storage=self.file_storage
        )

        # Initialize filing chunker and parallel embedding generator
        self.filing_chunker = FilingChunker(
            max_chunk_size=1500  # Maximum tokens per chunk
        )
        self.embedding_generator = ParallelEmbeddingGenerator(
            model=ETLConfig().embedding_model,
            max_workers=max_workers,
            rate_limit=rate_limit
        )
        
        # Configuration
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        logger.info(f"Initialized parallel ETL pipeline with {max_workers} workers")

    def process_companies(
        self,
        tickers: List[str],
        filing_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Process multiple companies in parallel.

        Args:
            tickers: List of company ticker symbols
            filing_types: List of filing types to process
            start_date: Start date for filing range
            end_date: End date for filing range

        Returns:
            Dictionary with results for each company
        """
        results = {
            "completed": [],
            "failed": [],
            "no_filings": []
        }
        
        if not tickers:
            return results
            
        logger.info(f"Processing {len(tickers)} companies in parallel")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(tickers))) as executor:
            # Submit tasks
            future_to_ticker = {
                executor.submit(
                    self.process_company, 
                    ticker, 
                    filing_types, 
                    start_date, 
                    end_date
                ): ticker for ticker in tickers
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result["status"] == "completed":
                        results["completed"].append(ticker)
                        logger.info(f"Successfully processed {ticker} with {result['filings_processed']} filings")
                    elif result["status"] == "no_filings":
                        results["no_filings"].append(ticker)
                        logger.info(f"No filings found for {ticker}")
                    else:
                        results["failed"].append(ticker)
                        logger.error(f"Failed to process {ticker}: {result['error']}")
                except Exception as e:
                    results["failed"].append(ticker)
                    logger.error(f"Exception processing {ticker}: {str(e)}")
        
        return results

    def process_company(
        self,
        ticker: str,
        filing_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process all filings for a company.

        Args:
            ticker: Company ticker symbol
            filing_types: List of filing types to process
            start_date: Start date for filing range
            end_date: End date for filing range
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "ticker": ticker,
            "status": "failed",
            "filings_processed": 0,
            "error": None
        }
        
        try:
            # Download filings
            downloaded_filings = self.sec_downloader.download_company_filings(
                ticker=ticker,
                filing_types=filing_types or ETLConfig().filing_types,
                start_date=start_date,
                end_date=end_date
            )
            
            if not downloaded_filings:
                logger.warning(f"No filings found for {ticker} in the specified date range and filing types")
                result["status"] = "no_filings"
                return result
            
            logger.info(f"Found {len(downloaded_filings)} filings for {ticker}")
            
            # Process filings in parallel
            processed_filings = []
            for filing_data in downloaded_filings:
                try:
                    processed_filing = self.process_filing_data(filing_data)
                    if processed_filing:
                        processed_filings.append(processed_filing)
                except Exception as e:
                    logger.error(f"Error processing filing {filing_data.get('accession_number', 'unknown')}: {e}")
            
            result["status"] = "completed"
            result["filings_processed"] = len(processed_filings)
            return result
            
        except Exception as e:
            error_msg = f"Error processing company {ticker}: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
            return result

    def process_filing_data(self, filing_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process filing data.

        Args:
            filing_data: The filing data to process
            
        Returns:
            Processed filing data or None if processing failed
        """
        try:
            # Get the accession number from the filing data
            accession_number = filing_data.get('accession_number')
            if not accession_number:
                logger.error(f"Missing accession_number in filing data: {filing_data}")
                return None
                
            # Get filing content
            filing_content = self.file_storage.load_raw_filing(accession_number)
            if not filing_content:
                logger.error(f"Could not load content for filing {accession_number}")
                return None

            # Get HTML content if available
            if filing_data.get('has_html'):
                try:
                    html_data = self.file_storage.load_html_filing(accession_number)
                    if html_data:
                        filing_content['html_content'] = html_data.get('html_content', '')
                except Exception as e:
                    logger.warning(f"Could not load HTML content for filing {accession_number}: {e}")

            # Process filing with chunker
            processed_data = self.filing_chunker.process_filing(filing_data, filing_content)

            # Extract chunk texts
            chunk_texts = processed_data.get('chunk_texts', [])

            # Generate embeddings for chunks in parallel
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks")
            chunk_embeddings = self.embedding_generator.generate_embeddings(
                chunk_texts, 
                batch_size=self.batch_size
            )

            # Add embeddings to processed data
            processed_data['chunk_embeddings'] = chunk_embeddings

            # Only generate full text embedding for small documents (optional)
            text = processed_data['text']
            try:
                # Check if text is small enough for embedding
                token_count = self.filing_chunker._count_tokens(text)
                if token_count <= 8000:  # Safe limit for embedding models
                    logger.info(f"Generating embedding for full text ({token_count} tokens)")
                    full_text_embedding = self.embedding_generator.generate_embeddings([text])[0]
                    processed_data['embedding'] = full_text_embedding
                else:
                    logger.info(f"Skipping full text embedding due to token count ({token_count} tokens)")
                    # Use the average of chunk embeddings as a fallback
                    if chunk_embeddings:
                        import numpy as np
                        avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                        processed_data['embedding'] = avg_embedding
                        logger.info("Using average of chunk embeddings for document embedding")
            except Exception as e:
                logger.warning(f"Error generating full text embedding: {e}")
                # Continue without full text embedding

            # Process filing with embeddings
            self.filing_processor.process_filing(processed_data)

            # Save processed filing to disk
            self.file_storage.save_processed_filing(
                filing_id=accession_number,
                processed_data=processed_data,
                metadata=filing_data
            )

            # Cache filing for quick access
            self.file_storage.cache_filing(
                filing_id=accession_number,
                filing_data={
                    "metadata": filing_data,
                    "processed_data": processed_data
                }
            )
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing filing: {str(e)}")
            return None
