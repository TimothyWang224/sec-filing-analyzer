"""
SEC Filing Processor

This module provides functionality for processing SEC filings.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..storage import GraphStore, LlamaIndexVectorStore
from .file_storage import FileStorage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilingProcessor:
    """
    Processor for SEC filings.
    """

    def __init__(
        self,
        graph_store: Optional[GraphStore] = None,
        vector_store: Optional[LlamaIndexVectorStore] = None,
        file_storage: Optional[FileStorage] = None,
    ):
        """Initialize the filing processor."""
        self.graph_store = graph_store or GraphStore()
        self.vector_store = vector_store or LlamaIndexVectorStore()
        self.file_storage = file_storage or FileStorage()

    def _ensure_list_format(
        self, embedding: Union[np.ndarray, List[float], Any]
    ) -> List[float]:
        """Ensure embedding is in list format.

        Args:
            embedding: The embedding to convert

        Returns:
            List of floats representing the embedding
        """
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        elif isinstance(embedding, list):
            return embedding
        else:
            return list(embedding)

    def process_filing(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
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
                    texts=[text],  # Store the full text
                )

                # Add chunks to vector store if available
                if chunks and chunk_embeddings and chunk_texts:
                    # Generate unique IDs for each chunk, including split chunks
                    chunk_ids = []
                    chunk_metadata = []
                    chunk_texts_to_store = []
                    chunk_embeddings_to_store = []

                    for i, (chunk, chunk_text, chunk_embedding) in enumerate(
                        zip(chunks, chunk_texts, chunk_embeddings)
                    ):
                        # Check if this is a split chunk
                        is_split_chunk = False
                        original_order = None

                        if isinstance(chunk, dict):
                            is_split_chunk = chunk.get("is_split_chunk", False)
                            original_order = chunk.get("original_order")

                        if is_split_chunk and original_order is not None:
                            # This is a split chunk, use a more specific ID
                            chunk_id = f"{filing_id}_chunk_{original_order}_split_{chunk.get('split_chunk_index', 0)}"
                        else:
                            # This is a regular chunk
                            chunk_id = f"{filing_id}_chunk_{i}"

                        # Create metadata for the chunk
                        chunk_meta = metadata.copy()
                        chunk_meta.update(
                            {
                                "chunk_id": i,
                                "parent_filing": filing_id,
                                "chunk_metadata": chunk
                                if isinstance(chunk, dict)
                                else {},
                            }
                        )

                        # Add relationship information for split chunks
                        if is_split_chunk:
                            chunk_meta.update(
                                {
                                    "is_split_chunk": True,
                                    "original_chunk_order": original_order,
                                    "split_chunk_index": chunk.get(
                                        "split_chunk_index", 0
                                    )
                                    if isinstance(chunk, dict)
                                    else 0,
                                    "split_chunk_count": chunk.get(
                                        "split_chunk_count", 1
                                    )
                                    if isinstance(chunk, dict)
                                    else 1,
                                    "prev_chunk_id": f"{filing_id}_chunk_{original_order}_split_{chunk.get('split_chunk_index', 0) - 1}"
                                    if chunk.get("prev_chunk_order")
                                    else None,
                                    "next_chunk_id": f"{filing_id}_chunk_{original_order}_split_{chunk.get('split_chunk_index', 0) + 1}"
                                    if chunk.get("next_chunk_order")
                                    else None,
                                }
                            )

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
                        texts=chunk_texts_to_store,  # Store the chunk texts
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
                texts=[text],  # Store the full text
            )

            # Add chunks to vector store if available
            if chunks and chunk_embeddings and chunk_texts:
                # Generate unique IDs for each chunk, including split chunks
                chunk_ids = []
                chunk_metadata = []
                chunk_texts_to_store = []
                chunk_embeddings_to_store = []

                for i, (chunk, chunk_text, chunk_embedding) in enumerate(
                    zip(chunks, chunk_texts, chunk_embeddings)
                ):
                    # Check if this is a split chunk
                    is_split_chunk = False
                    original_order = None

                    if isinstance(chunk, dict):
                        is_split_chunk = chunk.get("is_split_chunk", False)
                        original_order = chunk.get("original_order")

                    if is_split_chunk and original_order is not None:
                        # This is a split chunk, use a more specific ID
                        chunk_id = f"{filing_id}_chunk_{original_order}_split_{chunk.get('split_chunk_index', 0)}"
                    else:
                        # This is a regular chunk
                        chunk_id = f"{filing_id}_chunk_{i}"

                    # Create metadata for the chunk
                    chunk_meta = metadata.copy()
                    chunk_meta.update(
                        {
                            "chunk_id": i,
                            "parent_filing": filing_id,
                            "chunk_metadata": chunk if isinstance(chunk, dict) else {},
                        }
                    )

                    # Add relationship information for split chunks
                    if is_split_chunk:
                        chunk_meta.update(
                            {
                                "is_split_chunk": True,
                                "original_chunk_order": original_order,
                                "split_chunk_index": chunk.get("split_chunk_index", 0)
                                if isinstance(chunk, dict)
                                else 0,
                                "split_chunk_count": chunk.get("split_chunk_count", 1)
                                if isinstance(chunk, dict)
                                else 1,
                                "prev_chunk_id": f"{filing_id}_chunk_{original_order}_split_{chunk.get('split_chunk_index', 0) - 1}"
                                if chunk.get("prev_chunk_order")
                                else None,
                                "next_chunk_id": f"{filing_id}_chunk_{original_order}_split_{chunk.get('split_chunk_index', 0) + 1}"
                                if chunk.get("next_chunk_order")
                                else None,
                            }
                        )

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
                    texts=chunk_texts_to_store,  # Store the chunk texts
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
                "chunk_texts": chunk_texts,
            }

            self.file_storage.cache_filing(
                filing_id, {"metadata": filing_data, "processed_data": processed_data}
            )

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
        filing_type: Optional[str] = None,
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
            ticker=ticker, year=year, filing_type=filing_type
        )
