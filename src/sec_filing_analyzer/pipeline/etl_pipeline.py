"""
SEC Filing ETL Pipeline

This module provides the main ETL pipeline for processing SEC filings.
"""

import logging
import concurrent.futures
from typing import Dict, List, Any, Optional
from pathlib import Path

from llama_index.embeddings.openai import OpenAIEmbedding

from ..config import ETLConfig, StorageConfig, VectorStoreConfig
from ..storage import GraphStore, LlamaIndexVectorStore, OptimizedVectorStore
from ..data_retrieval import SECFilingsDownloader, FilingProcessor
from ..data_retrieval.file_storage import FileStorage
from ..data_processing.chunking import FilingChunker
from ..semantic.embeddings.embedding_generator import EmbeddingGenerator
from ..semantic.embeddings.parallel_embeddings import ParallelEmbeddingGenerator
from ..semantic.embeddings.robust_embedding_generator import RobustEmbeddingGenerator
from ..data_retrieval.parallel_filing_processor import ParallelFilingProcessor

# Import the semantic and quantitative pipelines
from .semantic_pipeline import SemanticETLPipeline
from .quantitative_pipeline import QuantitativeETLPipeline
from .etl_pipeline_extension import ETLPipelineExtension

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
        max_workers: int = 4,
        batch_size: int = 100,
        rate_limit: float = 0.1,
        use_parallel: bool = True,
        process_semantic: bool = True,
        process_quantitative: bool = True,
        db_path: Optional[str] = None
    ):
        """Initialize the ETL pipeline.

        Args:
            graph_store: Graph store for storing filing relationships
            vector_store: Vector store for storing embeddings
            filing_processor: Processor for filings
            file_storage: Storage for filing files
            sec_downloader: Downloader for SEC filings
            max_workers: Maximum number of worker threads for parallel processing
            batch_size: Batch size for embedding generation
            rate_limit: Minimum time between API requests in seconds
            use_parallel: Whether to use parallel processing (default: True)
        """
        self.graph_store = graph_store or GraphStore()

        # Use OptimizedVectorStore by default
        if vector_store is None:
            vector_store_config = VectorStoreConfig()
            if vector_store_config.type == "optimized":
                self.vector_store = OptimizedVectorStore(
                    store_path=vector_store_config.path,
                    index_type=vector_store_config.index_type,
                    use_gpu=vector_store_config.use_gpu
                )
            else:
                self.vector_store = LlamaIndexVectorStore(
                    store_path=StorageConfig().vector_store_path,
                    lazy_load=True  # Use lazy loading to avoid rebuilding the index on startup
                )
        else:
            self.vector_store = vector_store
        self.file_storage = file_storage or FileStorage(
            base_dir=ETLConfig().filings_dir
        )
        self.sec_downloader = sec_downloader or SECFilingsDownloader(
            file_storage=self.file_storage
        )

        # Configuration
        self.max_workers = max_workers
        # Use a smaller batch size to avoid rate limiting issues
        self.batch_size = min(batch_size, 50)  # Limit to 50 chunks per batch
        # Increase rate limit to avoid 429 errors
        self.rate_limit = max(rate_limit, 0.5)  # At least 0.5 seconds between requests
        self.use_parallel = use_parallel
        self.process_semantic = process_semantic
        self.process_quantitative = process_quantitative
        self.db_path = db_path

        # Initialize filing processor based on parallel preference
        if use_parallel and not filing_processor:
            self.filing_processor = ParallelFilingProcessor(
                graph_store=self.graph_store,
                vector_store=self.vector_store,
                file_storage=self.file_storage,
                max_workers=max_workers
            )
        else:
            self.filing_processor = filing_processor or FilingProcessor(
                graph_store=self.graph_store,
                vector_store=self.vector_store,
                file_storage=self.file_storage
            )

        # Initialize filing chunker
        self.filing_chunker = FilingChunker()

        # Initialize embedding generator based on parallel preference
        if use_parallel:
            # Use the robust embedding generator for better error handling and resilience
            self.embedding_generator = RobustEmbeddingGenerator(
                model=ETLConfig().embedding_model,
                max_tokens_per_chunk=8000,  # Safe limit below the 8192 max
                rate_limit=rate_limit,
                batch_size=50,  # Default batch size
                max_retries=5  # Increased retry logic
            )
        else:
            # Use the standard embedding generator for simpler cases
            self.embedding_generator = EmbeddingGenerator(
                model=ETLConfig().embedding_model
            )

        # Get ETL config for database settings
        etl_config = ETLConfig()

        # Initialize the semantic and quantitative pipelines
        if self.process_semantic:
            self.semantic_pipeline = SemanticETLPipeline(
                downloader=self.sec_downloader
            )

        if self.process_quantitative:
            # Use read-write mode for the quantitative pipeline
            self.quantitative_pipeline = QuantitativeETLPipeline(
                downloader=self.sec_downloader,
                db_path=self.db_path,
                read_only=False  # Always use read-write mode for ETL operations
            )

        # Initialize ETL pipeline extension
        self.extension = ETLPipelineExtension(
            db_path=self.db_path,
            read_only=False  # Always use read-write mode for ETL operations
        )

        logger.info(f"Initialized ETL pipeline with parallel processing: {use_parallel}")

    def process_company(
        self,
        ticker: str,
        filing_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_download: bool = False,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process all filings for a company.

        Args:
            ticker: Company ticker symbol
            filing_types: List of filing types to process
            start_date: Start date for filing range
            end_date: End date for filing range
            force_download: Whether to force download filings even if they exist in cache
            limit: Maximum number of filings to process (None for no limit)

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
                end_date=end_date,
                force_download=force_download,
                limit=limit
            )

            if not downloaded_filings:
                logger.warning(f"No filings found for {ticker} in the specified date range and filing types")
                result["status"] = "no_filings"
                return result

            logger.info(f"Found {len(downloaded_filings)} filings for {ticker}")

            # Process filings (parallel or sequential)
            processed_count = 0
            if self.use_parallel and len(downloaded_filings) > 1:
                # Process filings in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(downloaded_filings))) as executor:
                    # Submit tasks
                    future_to_filing = {
                        executor.submit(self.process_filing_data, filing): filing
                        for filing in downloaded_filings
                    }

                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_filing):
                        filing = future_to_filing[future]
                        filing_id = filing.get('accession_number', 'unknown')
                        try:
                            future.result()
                            processed_count += 1
                            logger.info(f"Successfully processed filing {filing_id}")
                        except Exception as e:
                            logger.error(f"Error processing filing {filing_id}: {str(e)}")
            else:
                # Process filings sequentially
                for filing_data in downloaded_filings:
                    try:
                        self.process_filing_data(filing_data)
                        processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing filing {filing_data.get('accession_number', 'unknown')}: {e}")

            result["status"] = "completed"
            result["filings_processed"] = processed_count
            return result

        except Exception as e:
            error_msg = f"Error processing company {ticker}: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
            return result

    def process_company_filings(
        self,
        ticker: str,
        filing_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        force_download: bool = False
    ) -> Dict[str, Any]:
        """
        Process multiple filings for a company through the ETL pipeline.
        This method is used by the Streamlit app.

        Args:
            ticker: Company ticker symbol
            filing_types: List of filing types to process (e.g., ['10-K', '10-Q'])
            start_date: Start date for filings (optional)
            end_date: End date for filings (optional)
            limit: Maximum number of filings to process (None for no limit)
            force_download: Whether to force download even if cached

        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing filings for {ticker} from {start_date} to {end_date}")

            # Call the process_company method with the same parameters
            result = self.process_company(
                ticker=ticker,
                filing_types=filing_types,
                start_date=start_date,
                end_date=end_date,
                force_download=force_download,
                limit=limit
            )

            # Format the result to match what the Streamlit app expects
            return {
                "status": "success" if result["status"] == "completed" else result["status"],
                "message": f"Processed {result['filings_processed']} filings for {ticker}",
                "filings_processed": result["filings_processed"],
                "error": result.get("error")
            }

        except Exception as e:
            logger.error(f"Error processing filings for {ticker}: {e}")
            return {"error": str(e)}

    def process_filing(self, ticker: str, filing_type: str, filing_date: Optional[str] = None, accession_number: Optional[str] = None, force_download: bool = False) -> Dict[str, Any]:
        """Process a single filing.

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
            results = {}

            # Process using the legacy pipeline
            # Step 1: Get the filing object
            filings = self.sec_downloader.get_filings(
                ticker=ticker,
                filing_types=[filing_type],
                start_date=filing_date,
                end_date=filing_date,
                limit=1
            )

            if not filings:
                logger.error(f"Failed to find {filing_type} filing for {ticker}")
                return {"error": f"Failed to find {filing_type} filing for {ticker}"}

            # Get the first filing
            filing = filings[0]

            # Download the filing
            filing_data = self.sec_downloader.download_filing(filing, ticker)

            if not filing_data:
                logger.error(f"Failed to download {filing_type} filing for {ticker}")
                return {"error": f"Failed to download {filing_type} filing for {ticker}"}

            # Extract key information
            accession_number = filing_data.get("accession_number")
            filing_date = filing_data.get("filing_date")
            document_url = filing_data.get("document_url")
            file_path = filing_data.get("file_path")

            # Register filing in DuckDB
            self.extension.register_filing(
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                accession_number=accession_number,
                file_path=file_path,
                processing_status="downloaded",
                document_url=document_url
            )

            # Step 2: Process the filing
            legacy_result = self.filing_processor.process_filing(filing_data)
            results["legacy"] = legacy_result

            # Process using the semantic pipeline if enabled
            if self.process_semantic:
                semantic_result = self.semantic_pipeline.process_filing(
                    ticker=ticker,
                    filing_type=filing_type,
                    filing_date=filing_date,
                    accession_number=accession_number,
                    force_download=force_download
                )
                results["semantic"] = semantic_result

                # Update filing status in DuckDB
                self.extension.update_filing_status(
                    accession_number=accession_number,
                    processing_status="embedded"
                )

            # Process using the quantitative pipeline if enabled
            if self.process_quantitative:
                quantitative_result = self.quantitative_pipeline.process_filing(
                    ticker=ticker,
                    filing_type=filing_type,
                    filing_date=filing_date,
                    accession_number=accession_number,
                    force_download=force_download
                )
                results["quantitative"] = quantitative_result

                # Update filing status in DuckDB
                self.extension.update_filing_status(
                    accession_number=accession_number,
                    processing_status="xbrl_processed"
                )

            # Determine overall status
            status = "success"
            for _, result_data in results.items():
                if "error" in result_data:
                    status = "partial"
                    break

            return {
                "status": status,
                "ticker": ticker,
                "filing_type": filing_type,
                "results": results
            }

        except Exception as e:
            logger.error(f"Error processing {filing_type} filing for {ticker}: {e}")

            # Update filing status in DuckDB if accession_number is available
            if 'accession_number' in locals() and accession_number:
                self.extension.update_filing_status(
                    accession_number=accession_number,
                    processing_status="error"
                )

            return {"error": str(e)}

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

            # Generate embeddings for chunks
            logger.info(f"Generating embeddings for {len(chunk_texts)} chunks")
            if self.use_parallel:
                # Use parallel embedding generator with batch processing
                chunk_embeddings_result = self.embedding_generator.generate_embeddings(
                    chunk_texts,
                    batch_size=self.batch_size
                )

                # Handle both tuple return (embeddings, metadata) and direct embeddings return
                if isinstance(chunk_embeddings_result, tuple) and len(chunk_embeddings_result) == 2:
                    chunk_embeddings, embedding_metadata = chunk_embeddings_result
                    # Store metadata in processed data
                    processed_data['embedding_metadata'] = embedding_metadata
                else:
                    chunk_embeddings = chunk_embeddings_result
            else:
                # Use standard embedding generator
                chunk_embeddings_result = self.embedding_generator.generate_embeddings(chunk_texts)

                # Handle both tuple return (embeddings, metadata) and direct embeddings return
                if isinstance(chunk_embeddings_result, tuple) and len(chunk_embeddings_result) == 2:
                    chunk_embeddings, embedding_metadata = chunk_embeddings_result
                    # Store metadata in processed data
                    processed_data['embedding_metadata'] = embedding_metadata
                else:
                    chunk_embeddings = chunk_embeddings_result

            # Add embeddings to processed data
            processed_data['chunk_embeddings'] = chunk_embeddings

            # Only generate full text embedding for small documents (optional)
            text = processed_data['text']
            try:
                # Check if text is small enough for embedding
                token_count = self.filing_chunker._count_tokens(text)
                if token_count <= 8000:  # Safe limit for embedding models
                    logger.info(f"Generating embedding for full text ({token_count} tokens)")
                    full_text_result = self.embedding_generator.generate_embeddings([text])

                    # Handle both tuple return (embeddings, metadata) and direct embeddings return
                    if isinstance(full_text_result, tuple) and len(full_text_result) == 2:
                        full_text_embeddings, full_text_metadata = full_text_result
                        # Store metadata in processed data
                        if 'embedding_metadata' not in processed_data:
                            processed_data['embedding_metadata'] = {}
                        processed_data['embedding_metadata']['full_text'] = full_text_metadata
                        processed_data['embedding'] = full_text_embeddings[0]
                    else:
                        processed_data['embedding'] = full_text_result[0]
                else:
                    logger.info(f"Skipping full text embedding due to token count ({token_count} tokens)")
                    # Use the average of chunk embeddings as a fallback
                    if chunk_embeddings:
                        try:
                            import numpy as np
                            # Ensure all embeddings have the same shape
                            valid_embeddings = []
                            for emb in chunk_embeddings:
                                if isinstance(emb, list) and len(emb) == self.embedding_generator.get_embedding_dimensions():
                                    valid_embeddings.append(emb)
                                else:
                                    logger.warning(f"Skipping embedding with invalid shape: {len(emb) if isinstance(emb, list) else type(emb)}")

                            if valid_embeddings:
                                # Convert to numpy array and calculate mean
                                valid_embeddings_array = np.array(valid_embeddings)
                                avg_embedding = np.mean(valid_embeddings_array, axis=0).tolist()
                                processed_data['embedding'] = avg_embedding
                                logger.info("Using average of chunk embeddings for document embedding")
                            else:
                                logger.warning("No valid embeddings found for averaging")
                        except Exception as e:
                            logger.warning(f"Error calculating average embedding: {e}")
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

            # Update filing status in DuckDB
            self.extension.update_filing_status(
                accession_number=accession_number,
                processing_status="embedded",
                file_path=self.file_storage.get_processed_filing_path(accession_number)
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