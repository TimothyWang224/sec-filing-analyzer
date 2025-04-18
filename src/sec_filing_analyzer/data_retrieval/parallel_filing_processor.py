"""
Parallel SEC Filing Processor

This module provides functionality for processing SEC filings in parallel.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
import concurrent.futures
from pathlib import Path

from ..storage import GraphStore, LlamaIndexVectorStore
from .file_storage import FileStorage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelFilingProcessor:
    """
    Parallel processor for SEC filings.
    """

    def __init__(
        self,
        graph_store: Optional[GraphStore] = None,
        vector_store: Optional[LlamaIndexVectorStore] = None,
        file_storage: Optional[FileStorage] = None,
        max_workers: int = 4
    ):
        """Initialize the parallel filing processor."""
        self.graph_store = graph_store or GraphStore()
        self.vector_store = vector_store or LlamaIndexVectorStore()
        self.file_storage = file_storage or FileStorage()
        self.max_workers = max_workers

        logger.info(f"Initialized parallel filing processor with {max_workers} workers")

    def _ensure_list_format(self, embedding: Union[np.ndarray, List[float], Any]) -> List[float]:
        """Ensure embedding is in list format.

        Args:
            embedding: The embedding to convert

        Returns:
            List of floats representing the embedding
        """
        if embedding is None:
            # Return a zero vector if embedding is None
            logger.warning("Embedding is None, returning zero vector")
            return [0.0] * 1536

        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        elif isinstance(embedding, list):
            # Check if this is a list of lists
            if len(embedding) > 0 and isinstance(embedding[0], list):
                # This is a list of lists, use the first embedding
                logger.warning("Embedding is a list of lists, using the first embedding")
                return embedding[0]
            return embedding
        else:
            try:
                return list(embedding)
            except Exception as e:
                logger.error(f"Could not convert embedding to list: {e}")
                # Return a zero vector as fallback
                return [0.0] * 1536

    def process_filings_parallel(
        self,
        filings_data: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Process multiple filings in parallel.

        Args:
            filings_data: List of filing data dictionaries

        Returns:
            Dictionary with completed and failed filing IDs
        """
        results = {
            "completed": [],
            "failed": []
        }

        if not filings_data:
            return results

        logger.info(f"Processing {len(filings_data)} filings in parallel")

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(filings_data))) as executor:
            # Submit tasks
            future_to_filing = {
                executor.submit(self.process_filing, filing): filing.get("id", "unknown")
                for filing in filings_data
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_filing):
                filing_id = future_to_filing[future]
                try:
                    result = future.result()
                    if result:
                        results["completed"].append(filing_id)
                        logger.info(f"Successfully processed filing {filing_id}")
                    else:
                        results["failed"].append(filing_id)
                        logger.error(f"Failed to process filing {filing_id}")
                except Exception as e:
                    results["failed"].append(filing_id)
                    logger.error(f"Error processing filing {filing_id}: {str(e)}")

        return results

    def process_filing(self, filing_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a filing and store it in the graph and vector stores.

        Args:
            filing_data: Dictionary containing filing data

        Returns:
            Dict containing processed filing data
        """
        filing_id = filing_data["id"]
        text = filing_data["text"]
        embedding = filing_data["embedding"]
        metadata = filing_data["metadata"]
        chunks = filing_data.get("chunks")
        chunk_embeddings = filing_data.get("chunk_embeddings", [])
        chunk_texts = filing_data.get("chunk_texts", [])

        # Check if filing is already processed
        cached_data = self.file_storage.load_cached_filing(filing_id)
        if cached_data:
            logger.info(f"Using cached data for filing {filing_id}")
            # Even if we have cached data, we still want to add it to the vector store
            # to ensure it's available for search
            processed_data = cached_data["processed_data"]

            # Process the filing directly instead of calling _add_to_vector_store
            # to avoid recursion
            try:
                filing_id = processed_data["id"]
                text = processed_data["text"]
                embedding = processed_data["embedding"]
                metadata = processed_data["metadata"]
                chunks = processed_data.get("chunks")
                chunk_embeddings = processed_data.get("chunk_embeddings", [])
                chunk_texts = processed_data.get("chunk_texts", [])

                # Ensure embedding is a list of floats
                embedding_list = self._ensure_list_format(embedding)

                # Add full document to vector store with text
                self.vector_store.upsert_vectors(
                    vectors=[embedding_list],
                    ids=[filing_id],
                    metadata=[metadata],
                    texts=[text]  # Store the full text
                )

                # Add chunks to vector store if available
                if chunks and chunk_embeddings and chunk_texts:
                    # Check if chunk_embeddings is a dict (from embedding_metadata)
                    if isinstance(chunk_embeddings, dict):
                        logger.warning(f"chunk_embeddings is a dict, not a list of embeddings. Skipping chunk processing.")
                        return processed_data

                    # Generate unique IDs for each chunk
                    chunk_ids = []
                    chunk_metadata = []
                    chunk_texts_to_store = []
                    chunk_embeddings_to_store = []

                    for i, (chunk, chunk_embedding, chunk_text) in enumerate(zip(chunks, chunk_embeddings, chunk_texts)):
                        chunk_id = f"{filing_id}_chunk_{i}"

                        # Create metadata for chunk
                        chunk_meta = metadata.copy()
                        chunk_meta.update({
                            "chunk_id": chunk_id,
                            "chunk_index": i,
                            "parent_id": filing_id,
                            "item": chunk.get("item", ""),
                            "is_table": chunk.get("is_table", False),
                            "is_signature": chunk.get("is_signature", False)
                        })

                        # Ensure chunk embedding is a list of floats
                        chunk_embedding_list = self._ensure_list_format(chunk_embedding)

                        # Add to our lists
                        chunk_ids.append(chunk_id)
                        chunk_metadata.append(chunk_meta)
                        chunk_texts_to_store.append(chunk_text)
                        chunk_embeddings_to_store.append(chunk_embedding_list)

                    # Add chunks to vector store with their text
                    self.vector_store.upsert_vectors(
                        vectors=chunk_embeddings_to_store,
                        ids=chunk_ids,
                        metadata=chunk_metadata,
                        texts=chunk_texts_to_store  # Store the chunk texts
                    )

                # Store in graph store
                self.graph_store.add_filing(filing_id, text, metadata, chunks)
            except Exception as e:
                logger.error(f"Error adding cached filing to vector store: {str(e)}")

            return processed_data

        # Process filing
        try:
            # Ensure embedding is a list of floats
            embedding_list = self._ensure_list_format(embedding)

            # Add full document to vector store with text
            self.vector_store.upsert_vectors(
                vectors=[embedding_list],
                ids=[filing_id],
                metadata=[metadata],
                texts=[text]  # Store the full text
            )

            # Add chunks to vector store if available
            if chunks and chunk_embeddings and chunk_texts:
                # Check if chunk_embeddings is a dict (from embedding_metadata)
                if isinstance(chunk_embeddings, dict):
                    logger.warning(f"chunk_embeddings is a dict, not a list of embeddings. Skipping chunk processing.")
                    # Continue without chunk processing
                    pass
                else:
                    # Generate unique IDs for each chunk, including split chunks
                    chunk_ids = []
                    chunk_metadata = []
                    chunk_texts_to_store = []
                    chunk_embeddings_to_store = []

                    for i, (chunk, chunk_embedding, chunk_text) in enumerate(zip(chunks, chunk_embeddings, chunk_texts)):
                        chunk_id = f"{filing_id}_chunk_{i}"

                        # Create metadata for chunk
                        chunk_meta = metadata.copy()
                        chunk_meta.update({
                            "chunk_id": chunk_id,
                            "chunk_index": i,
                            "parent_id": filing_id,
                            "item": chunk.get("item", ""),
                            "is_table": chunk.get("is_table", False),
                            "is_signature": chunk.get("is_signature", False)
                        })

                        # Ensure chunk embedding is a list of floats
                        chunk_embedding_list = self._ensure_list_format(chunk_embedding)

                        # Add to our lists
                        chunk_ids.append(chunk_id)
                        chunk_metadata.append(chunk_meta)
                        chunk_texts_to_store.append(chunk_text)
                        chunk_embeddings_to_store.append(chunk_embedding_list)

                # Add chunks to vector store with their text
                self.vector_store.upsert_vectors(
                    vectors=chunk_embeddings_to_store,
                    ids=chunk_ids,
                    metadata=chunk_metadata,
                    texts=chunk_texts_to_store  # Store the chunk texts
                )

            # Store in graph store
            self.graph_store.add_filing(filing_id, text, metadata, chunks)

            # Cache the processed data
            processed_data = {
                "id": filing_id,
                "text": text,
                "embedding": embedding_list,
                "metadata": metadata,
                "chunks": chunks,
                "chunk_embeddings": chunk_embeddings,
                "chunk_texts": chunk_texts
            }

            self.file_storage.cache_filing(filing_id, {
                "metadata": filing_data,
                "processed_data": processed_data
            })

            return processed_data

        except Exception as e:
            logger.error(f"Error processing filing {filing_id}: {str(e)}")
            return None

    def get_filing(self, filing_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a filing by ID.

        Args:
            filing_id: The filing ID

        Returns:
            Dict containing filing data if found, None otherwise
        """
        # Try to get from cache first
        cached_data = self.file_storage.load_cached_filing(filing_id)
        if cached_data:
            return cached_data

        # Try to get from processed files
        processed_data = self.file_storage.load_processed_filing(filing_id)
        if processed_data:
            return processed_data

        # Try to get from raw files
        raw_data = self.file_storage.load_raw_filing(filing_id)
        if raw_data:
            return raw_data

        return None

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
