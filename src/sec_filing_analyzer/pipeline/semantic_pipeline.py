"""
Semantic ETL Pipeline Module

This module provides a pipeline for extracting, transforming, and loading
semantic data from SEC filings.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..data_processing.chunking import FilingChunker
from ..data_retrieval.sec_downloader import SECFilingsDownloader
from ..semantic.embeddings.embedding_generator import EmbeddingGenerator
from ..semantic.embeddings.robust_embedding_generator import RobustEmbeddingGenerator
from ..semantic.storage.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticETLPipeline:
    """
    A pipeline for extracting, transforming, and loading semantic data from SEC filings.
    """

    def __init__(
        self,
        downloader: Optional[SECFilingsDownloader] = None,
        chunker: Optional[FilingChunker] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStore] = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
    ):
        """
        Initialize the semantic ETL pipeline.

        Args:
            downloader: SEC filings downloader
            chunker: Document chunker (FilingChunker instance)
            embedding_generator: Embedding generator
            vector_store: Vector store
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.downloader = downloader or SECFilingsDownloader()
        self.chunker = chunker or FilingChunker(max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_generator = embedding_generator or RobustEmbeddingGenerator()
        self.vector_store = vector_store or VectorStore()

        logger.info("Initialized semantic ETL pipeline")

    def process_filing(
        self,
        ticker: str,
        filing_type: str,
        filing_date: Optional[str] = None,
        accession_number: Optional[str] = None,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a single filing through the semantic ETL pipeline.

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
            logger.info(f"Processing {filing_type} filing for {ticker}")

            # Step 1: Download the filing
            # Get the filing object
            filings = self.downloader.get_filings(
                ticker=ticker, filing_types=[filing_type], start_date=filing_date, end_date=filing_date, limit=1
            )

            if not filings:
                logger.error(f"Failed to find {filing_type} filing for {ticker}")
                return {"error": f"Failed to find {filing_type} filing for {ticker}"}

            # Get the first filing
            filing = filings[0]

            # Download the filing
            filing_data = self.downloader.download_filing(filing, ticker)

            if not filing_data:
                logger.error(f"Failed to download {filing_type} filing for {ticker}")
                return {"error": f"Failed to download {filing_type} filing for {ticker}"}

            filing_id = filing_data.get("id")
            filing_text = filing_data.get("text", "")

            # Step 2: Chunk the document
            chunks = self.chunker.chunk_document(filing_text)

            if not chunks:
                logger.error(f"Failed to chunk {filing_type} filing for {ticker}")
                return {"error": f"Failed to chunk {filing_type} filing for {ticker}"}

            # Step 3: Generate embeddings
            # Extract text from chunks based on the format
            chunk_texts = []
            for chunk in chunks:
                if isinstance(chunk, dict) and "text" in chunk:
                    # FilingChunker format
                    chunk_texts.append(chunk["text"])
                elif hasattr(chunk, "text"):
                    # Old DocumentChunker format
                    chunk_texts.append(chunk.text)
                else:
                    logger.warning(f"Unexpected chunk format: {type(chunk)}")
                    continue

            # Generate embeddings with the robust generator
            embeddings, embedding_metadata = self.embedding_generator.generate_embeddings(chunk_texts)

            # Log any fallbacks
            if embedding_metadata.get("any_fallbacks", False):
                logger.warning(
                    f"Some embeddings used fallbacks: {embedding_metadata.get('fallback_count', 0)}/{len(chunk_texts)}"
                )

            if not embeddings:
                logger.error(f"Failed to generate embeddings for {filing_type} filing for {ticker}")
                return {"error": f"Failed to generate embeddings for {filing_type} filing for {ticker}"}

            # Step 4: Store in vector store
            metadata_list = []
            for i, chunk in enumerate(chunks):
                # Extract metadata based on the format
                if isinstance(chunk, dict) and "text" in chunk:
                    # FilingChunker format
                    chunk_text = chunk["text"]
                    chunk_metadata = chunk.get("metadata", {})
                    section = chunk_metadata.get("section", "")
                    page = chunk_metadata.get("page", 0)
                elif hasattr(chunk, "text") and hasattr(chunk, "metadata"):
                    # Old DocumentChunker format
                    chunk_text = chunk.text
                    section = chunk.metadata.get("section", "")
                    page = chunk.metadata.get("page", 0)
                else:
                    logger.warning(f"Skipping chunk with unexpected format: {type(chunk)}")
                    continue

                metadata = {
                    "filing_id": filing_id,
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "chunk_id": i,
                    "chunk_text": chunk_text,
                    "section": section,
                    "page": page,
                }
                metadata_list.append(metadata)

            self.vector_store.add_embeddings(embeddings, metadata_list)

            logger.info(f"Successfully processed {filing_type} filing for {ticker}")

            return {
                "status": "success",
                "filing_id": filing_id,
                "ticker": ticker,
                "filing_type": filing_type,
                "num_chunks": len(chunks),
                "num_embeddings": len(embeddings),
            }

        except Exception as e:
            logger.error(f"Error processing {filing_type} filing for {ticker}: {e}")
            return {"error": str(e)}

    def process_company_filings(
        self,
        ticker: str,
        filing_types: List[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Process multiple filings for a company through the semantic ETL pipeline.

        Args:
            ticker: Company ticker symbol
            filing_types: List of filing types to process (e.g., ['10-K', '10-Q'])
            start_date: Start date for filings (optional)
            end_date: End date for filings (optional)
            limit: Maximum number of filings to process
            force_download: Whether to force download even if cached

        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing filings for {ticker}")

            # Default to 10-K and 10-Q filings if not specified
            if filing_types is None:
                filing_types = ["10-K", "10-Q"]

            # Step 1: Get the list of filings
            filings = self.downloader.get_filings(
                ticker=ticker, filing_types=filing_types, start_date=start_date, end_date=end_date, limit=limit
            )

            if not filings:
                logger.error(f"No filings found for {ticker}")
                return {"error": f"No filings found for {ticker}"}

            # Step 2: Process each filing
            results = []
            for filing in filings:
                result = self.process_filing(
                    ticker=ticker,
                    filing_type=filing.get("filing_type"),
                    accession_number=filing.get("accession_number"),
                    force_download=force_download,
                )
                results.append(result)

            logger.info(f"Successfully processed {len(results)} filings for {ticker}")

            return {"status": "success", "ticker": ticker, "num_filings": len(results), "results": results}

        except Exception as e:
            logger.error(f"Error processing filings for {ticker}: {e}")
            return {"error": str(e)}

    def process_multiple_companies(
        self,
        tickers: List[str],
        filing_types: List[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit_per_company: int = 5,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Process filings for multiple companies through the semantic ETL pipeline.

        Args:
            tickers: List of company ticker symbols
            filing_types: List of filing types to process (e.g., ['10-K', '10-Q'])
            start_date: Start date for filings (optional)
            end_date: End date for filings (optional)
            limit_per_company: Maximum number of filings to process per company
            force_download: Whether to force download even if cached

        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing filings for {len(tickers)} companies")

            # Process each company
            results = {}
            for ticker in tickers:
                result = self.process_company_filings(
                    ticker=ticker,
                    filing_types=filing_types,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit_per_company,
                    force_download=force_download,
                )
                results[ticker] = result

            logger.info(f"Successfully processed filings for {len(tickers)} companies")

            return {"status": "success", "num_companies": len(tickers), "results": results}

        except Exception as e:
            logger.error(f"Error processing multiple companies: {e}")
            return {"error": str(e)}
