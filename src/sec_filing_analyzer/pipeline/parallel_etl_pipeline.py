"""
Parallel SEC Filing ETL Pipeline

This module provides a parallelized ETL pipeline for processing SEC filings.
"""

import concurrent.futures
import logging
from typing import Any, Dict, List, Optional, Union

from sec_filing_analyzer.config import ETLConfig, StorageConfig
from sec_filing_analyzer.data_processing.chunking import FilingChunker
from sec_filing_analyzer.data_retrieval import SECFilingsDownloader
from sec_filing_analyzer.data_retrieval.file_storage import FileStorage
from sec_filing_analyzer.data_retrieval.parallel_filing_processor import (
    ParallelFilingProcessor,
)
from sec_filing_analyzer.pipeline.quantitative_pipeline import QuantitativeETLPipeline
from sec_filing_analyzer.semantic.embeddings.robust_embedding_generator import (
    RobustEmbeddingGenerator,
)
from sec_filing_analyzer.storage import (
    GraphStore,
    LlamaIndexVectorStore,
    OptimizedVectorStore,
)

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
        vector_store: Optional[Union[LlamaIndexVectorStore, OptimizedVectorStore]] = None,
        filing_processor: Optional[ParallelFilingProcessor] = None,
        file_storage: Optional[FileStorage] = None,
        sec_downloader: Optional[SECFilingsDownloader] = None,
        max_workers: int = 4,
        batch_size: int = 100,
        rate_limit: float = 0.1,
        use_optimized_vector_store: bool = True,
        process_semantic: bool = True,
        process_quantitative: bool = True,
        db_path: Optional[str] = None,
    ):
        """Initialize the parallel ETL pipeline."""
        self.graph_store = graph_store or GraphStore()

        # Use optimized vector store by default
        if vector_store is None:
            if use_optimized_vector_store:
                self.vector_store = OptimizedVectorStore(store_path=StorageConfig().vector_store_path)
            else:
                self.vector_store = LlamaIndexVectorStore(
                    store_path=StorageConfig().vector_store_path,
                    lazy_load=True,  # Use lazy loading to avoid rebuilding the index on startup
                )
        else:
            self.vector_store = vector_store
        self.file_storage = file_storage or FileStorage(base_dir=ETLConfig().filings_dir)
        self.filing_processor = filing_processor or ParallelFilingProcessor(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            file_storage=self.file_storage,
            max_workers=max_workers,
        )
        self.sec_downloader = sec_downloader or SECFilingsDownloader(file_storage=self.file_storage)

        # Initialize filing chunker and parallel embedding generator
        self.filing_chunker = FilingChunker(
            max_chunk_size=1500  # Maximum tokens per chunk
        )
        self.embedding_generator = RobustEmbeddingGenerator(
            model=ETLConfig().embedding_model,
            max_tokens_per_chunk=8000,  # Safe limit below the 8192 max
            rate_limit=rate_limit,
            batch_size=batch_size,  # Use the batch size from constructor
            max_retries=5,  # Increased retry logic
        )

        # Try to set up enhanced logging
        try:
            from ..utils.logging_utils import setup_logging

            setup_logging()
        except (ImportError, Exception) as e:
            logger.warning(f"Could not set up enhanced logging: {e}")

        # Configuration
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.process_semantic = process_semantic
        self.process_quantitative = process_quantitative
        self.db_path = db_path

        # Initialize quantitative pipeline if needed
        if self.process_quantitative:
            self.quantitative_pipeline = QuantitativeETLPipeline(downloader=self.sec_downloader, db_path=self.db_path)

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
        results = {"completed": [], "failed": [], "no_filings": []}

        if not tickers:
            return results

        logger.info(f"Processing {len(tickers)} companies in parallel")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(tickers))) as executor:
            # Submit tasks
            future_to_ticker = {
                executor.submit(self.process_company, ticker, filing_types, start_date, end_date): ticker
                for ticker in tickers
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
        force_rebuild_index: bool = False,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Process all filings for a company.

        Args:
            ticker: Company ticker symbol
            filing_types: List of filing types to process
            start_date: Start date for filing range
            end_date: End date for filing range
            force_rebuild_index: Whether to force rebuild the FAISS index
            force_download: Whether to force download filings even if they exist in cache

        Returns:
            Dictionary with processing results
        """
        result = {
            "ticker": ticker,
            "status": "failed",
            "filings_processed": 0,
            "error": None,
        }

        try:
            # Download filings
            downloaded_filings = self.sec_downloader.download_company_filings(
                ticker=ticker,
                filing_types=filing_types or ETLConfig().filing_types,
                start_date=start_date,
                end_date=end_date,
                force_download=force_download,
            )

            if not downloaded_filings:
                logger.warning(f"No filings found for {ticker} in the specified date range and filing types")
                result["status"] = "no_filings"
                return result

            logger.info(f"Found {len(downloaded_filings)} filings for {ticker}")

            # If using optimized vector store, load or rebuild the index
            if hasattr(self.vector_store, "_load_faiss_index_for_companies"):
                # This is an OptimizedVectorStore
                self.vector_store._load_faiss_index_for_companies([ticker], force_rebuild=force_rebuild_index)

            # Process filings in parallel
            processed_filings = []

            # Process using semantic pipeline if enabled
            if self.process_semantic:
                for filing_data in downloaded_filings:
                    try:
                        # Add accession_number if missing but id is present
                        if "id" in filing_data and not filing_data.get("accession_number"):
                            filing_data["accession_number"] = filing_data["id"]

                        processed_filing = self.process_filing_data(filing_data)
                        if processed_filing:
                            processed_filings.append(processed_filing)
                    except Exception as e:
                        logger.error(
                            f"Error processing filing {filing_data.get('accession_number', filing_data.get('id', 'unknown'))}: {e}"
                        )

            # Process using quantitative pipeline if enabled
            if self.process_quantitative:
                # Process each filing with the quantitative pipeline
                for filing_data in downloaded_filings:
                    try:
                        # Ensure both id and accession_number fields are present
                        if "id" in filing_data and not filing_data.get("accession_number"):
                            filing_data["accession_number"] = filing_data["id"]
                            logger.info(f"Using id as accession_number: {filing_data['id']}")
                        elif "accession_number" in filing_data and not filing_data.get("id"):
                            filing_data["id"] = filing_data["accession_number"]
                            logger.info(f"Using accession_number as id: {filing_data['accession_number']}")

                        accession_number = filing_data.get("accession_number")

                        # Handle filing_type/form field consistency
                        if "filing_type" not in filing_data and "form" in filing_data:
                            filing_data["filing_type"] = filing_data["form"]
                        elif "form" not in filing_data and "filing_type" in filing_data:
                            filing_data["form"] = filing_data["filing_type"]

                        filing_type = filing_data.get("filing_type") or filing_data.get("form")
                        filing_date = filing_data.get("filing_date")

                        if accession_number and filing_type:
                            self.quantitative_pipeline.process_filing(
                                ticker=ticker,
                                filing_type=filing_type,
                                filing_date=filing_date,
                                accession_number=accession_number,
                            )
                    except Exception as e:
                        # Get a reliable identifier for the filing
                        filing_id = filing_data.get("accession_number") or filing_data.get("id") or "unknown"
                        logger.error(f"Error processing XBRL data for filing {filing_id}: {e}")

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
            accession_number = filing_data.get("accession_number")

            # If accession_number is missing but id is present, use id as accession_number
            if not accession_number and "id" in filing_data:
                accession_number = filing_data["id"]
                filing_data["accession_number"] = accession_number
                logger.info(f"Using id as accession_number: {accession_number}")
            # If id is missing but accession_number is present, use accession_number as id
            elif not filing_data.get("id") and accession_number:
                filing_data["id"] = accession_number
                logger.info(f"Using accession_number as id: {accession_number}")

            if not accession_number:
                logger.error(f"Missing both accession_number and id in filing data: {filing_data}")
                return None

            # Get filing content
            filing_content = self.file_storage.load_raw_filing(accession_number)
            if not filing_content:
                logger.error(f"Could not load content for filing {accession_number}")
                return None

            # Get HTML content if available
            if filing_data.get("has_html"):
                try:
                    html_data = self.file_storage.load_html_filing(accession_number)
                    if html_data:
                        filing_content["html_content"] = html_data.get("html_content", "")
                except Exception as e:
                    logger.warning(f"Could not load HTML content for filing {accession_number}: {e}")

            # Process filing with chunker
            processed_data = self.filing_chunker.process_filing(filing_data, filing_content)

            # Extract chunk texts
            chunk_texts = processed_data.get("chunk_texts", [])

            # Create a filing-specific embedding generator with metadata
            filing_embedding_generator = RobustEmbeddingGenerator(
                model=ETLConfig().embedding_model,
                max_tokens_per_chunk=8000,  # Safe limit below the 8192 max
                rate_limit=self.embedding_generator.rate_limit,
                filing_metadata=filing_data,
                batch_size=self.batch_size,
                max_retries=5,  # Increased retry logic
            )

            # Generate embeddings for chunks in parallel
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks")
            chunk_embeddings_result = filing_embedding_generator.generate_embeddings(
                chunk_texts, batch_size=self.batch_size
            )

            # Handle both tuple return (embeddings, metadata) and direct embeddings return
            if isinstance(chunk_embeddings_result, tuple) and len(chunk_embeddings_result) == 2:
                chunk_embeddings, embedding_metadata = chunk_embeddings_result
            else:
                chunk_embeddings = chunk_embeddings_result
                embedding_metadata = {"any_fallbacks": False, "fallback_count": 0}

            # Add embeddings and metadata to processed data
            processed_data["chunk_embeddings"] = chunk_embeddings
            processed_data["embedding_metadata"] = {
                "chunk_embedding_stats": embedding_metadata,
                "has_fallbacks": embedding_metadata.get("any_fallbacks", False),
                "fallback_count": embedding_metadata.get("fallback_count", 0),
                "token_usage": embedding_metadata.get("token_usage", {}),
            }

            # Only generate full text embedding for small documents (optional)
            text = processed_data["text"]
            try:
                # Check if text is small enough for embedding
                token_count = self.filing_chunker._count_tokens(text)
                if token_count <= 8000:  # Safe limit for embedding models
                    logger.info(f"Generating embedding for full text ({token_count} tokens)")
                    full_text_result = filing_embedding_generator.generate_embeddings([text])

                    # Handle both tuple return (embeddings, metadata) and direct embeddings return
                    if isinstance(full_text_result, tuple) and len(full_text_result) == 2:
                        full_text_embedding, full_text_metadata = full_text_result
                    else:
                        full_text_embedding = full_text_result
                        full_text_metadata = {
                            "any_fallbacks": False,
                            "fallback_count": 0,
                        }
                    processed_data["embedding"] = full_text_embedding[0]

                    # Add full text embedding metadata
                    processed_data["embedding_metadata"]["full_text_embedding_stats"] = full_text_metadata
                    processed_data["embedding_metadata"]["full_text_has_fallback"] = full_text_metadata.get(
                        "any_fallbacks", False
                    )
                else:
                    logger.info(f"Skipping full text embedding due to token count ({token_count} tokens)")
                    # Use the average of chunk embeddings as a fallback
                    if chunk_embeddings:
                        import numpy as np

                        # Filter out fallback (zero) vectors if possible
                        if "fallback_flags" in embedding_metadata and not all(embedding_metadata["fallback_flags"]):
                            # Get indices of non-fallback chunks
                            valid_indices = [
                                i
                                for i, is_fallback in enumerate(embedding_metadata["fallback_flags"])
                                if not is_fallback
                            ]
                            if valid_indices:  # If we have any valid embeddings
                                valid_embeddings = [chunk_embeddings[i] for i in valid_indices]
                                avg_embedding = np.mean(valid_embeddings, axis=0).tolist()
                                processed_data["embedding"] = avg_embedding
                                logger.info(
                                    f"Using average of {len(valid_indices)} valid chunk embeddings for document embedding"
                                )
                                processed_data["embedding_metadata"]["full_text_has_fallback"] = False
                                return

                        # If all chunks are fallbacks or we couldn't filter, use all chunks
                        avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                        processed_data["embedding"] = avg_embedding
                        logger.info("Using average of all chunk embeddings for document embedding")
                        processed_data["embedding_metadata"]["full_text_has_fallback"] = True
            except Exception as e:
                logger.warning(f"Error generating full text embedding: {e}")
                # Continue without full text embedding

            # Process filing with embeddings
            self.filing_processor.process_filing(processed_data)

            # Save processed filing to disk
            self.file_storage.save_processed_filing(
                filing_id=accession_number,
                processed_data=processed_data,
                metadata=filing_data,
            )

            # Cache filing for quick access
            self.file_storage.cache_filing(
                filing_id=accession_number,
                filing_data={"metadata": filing_data, "processed_data": processed_data},
            )

            return processed_data

        except Exception as e:
            logger.error(f"Error processing filing: {str(e)}")
            return None
