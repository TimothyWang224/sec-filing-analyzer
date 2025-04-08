"""
Optimized Vector Store Implementation

This module provides an optimized implementation of vector storage operations
using NumPy binary storage and FAISS for efficient similarity search.
"""

import os
import logging
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from pathlib import Path
from collections import defaultdict

from llama_index.core import Document
from llama_index.core.schema import NodeWithScore, TextNode

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedVectorStore:
    """
    Optimized vector store implementation using NumPy binary storage and FAISS.
    Loads embeddings on-demand and filters by company for efficient querying.
    """

    def __init__(self, store_path: Optional[str] = None):
        """Initialize the vector store.

        Args:
            store_path: Optional path to store the vector store data
        """
        self.store_path = Path(store_path) if store_path else Path("data/vector_store")
        self.store_path.mkdir(parents=True, exist_ok=True)

        # Create directories for persistent storage
        self.metadata_dir = self.store_path / "metadata"
        self.text_dir = self.store_path / "text"
        self.embeddings_dir = self.store_path / "embeddings"
        self.index_dir = self.store_path / "index"

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Create company-specific directories
        self.company_dir = self.store_path / "by_company"
        self.company_dir.mkdir(parents=True, exist_ok=True)

        # Initialize in-memory storage for metadata
        self.metadata_store = self._load_metadata_store()
        
        # Create company to document mapping
        self.company_to_docs = self._build_company_to_docs_mapping()
        
        # Initialize FAISS index (but don't load any vectors yet)
        self.faiss_index = None
        self.faiss_id_map = {}  # Maps FAISS index positions to document IDs
        self.loaded_companies = set()  # Track which companies are loaded
        
        logger.info(f"Initialized optimized vector store at {self.store_path}")

    def _load_metadata_store(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata for all documents from disk.

        Returns:
            Dictionary mapping document IDs to metadata
        """
        metadata_store = {}
        
        if not self.metadata_dir.exists():
            return metadata_store
            
        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                doc_id = metadata_file.stem
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                metadata_store[doc_id] = metadata
            except Exception as e:
                logger.warning(f"Error loading metadata from {metadata_file}: {e}")
                
        logger.info(f"Loaded metadata for {len(metadata_store)} documents")
        return metadata_store

    def _build_company_to_docs_mapping(self) -> Dict[str, Set[str]]:
        """Build a mapping from companies to document IDs.

        Returns:
            Dictionary mapping company tickers to sets of document IDs
        """
        company_to_docs = defaultdict(set)
        
        for doc_id, metadata in self.metadata_store.items():
            ticker = metadata.get("ticker", "unknown")
            company_to_docs[ticker].add(doc_id)
            
            # Also add to "all" category for queries that don't specify a company
            company_to_docs["all"].add(doc_id)
            
        # Save the mapping to disk for future reference
        mapping_file = self.store_path / "company_doc_mapping.json"
        serializable_mapping = {k: list(v) for k, v in company_to_docs.items()}
        
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(serializable_mapping, f, indent=2)
            
        logger.info(f"Built company-to-documents mapping for {len(company_to_docs)} companies")
        return company_to_docs

    def _save_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Save metadata to disk.

        Args:
            doc_id: Document ID
            metadata: Metadata dictionary
        """
        try:
            # Create a safe filename by replacing invalid characters
            safe_id = self._safe_filename(doc_id)

            # Save to disk
            metadata_path = self.metadata_dir / f"{safe_id}.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            # Update in-memory store
            self.metadata_store[doc_id] = metadata
            
            # Update company mapping if ticker is present
            if "ticker" in metadata:
                ticker = metadata["ticker"]
                self.company_to_docs[ticker].add(doc_id)
                self.company_to_docs["all"].add(doc_id)

            logger.debug(f"Saved metadata for {doc_id} to {metadata_path}")
        except Exception as e:
            logger.warning(f"Error saving metadata for {doc_id}: {e}")

    def _save_text(self, doc_id: str, text: str) -> None:
        """Save text to disk.

        Args:
            doc_id: Document ID
            text: Text content
        """
        try:
            # Create a safe filename by replacing invalid characters
            safe_id = self._safe_filename(doc_id)

            # Save to disk
            text_path = self.text_dir / f"{safe_id}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

            logger.debug(f"Saved text for {doc_id} to {text_path}")
        except Exception as e:
            logger.warning(f"Error saving text for {doc_id}: {e}")

    def _save_embedding(self, doc_id: str, embedding: List[float]) -> None:
        """Save an embedding to disk as NumPy binary file.

        Args:
            doc_id: Document ID
            embedding: Embedding vector
        """
        try:
            # Create a safe filename by replacing invalid characters
            safe_id = self._safe_filename(doc_id)

            # Convert to numpy array
            embedding_array = np.array(embedding, dtype=np.float32)
            
            # Save to disk as NumPy binary
            embedding_path = self.embeddings_dir / f"{safe_id}.npy"
            np.save(embedding_path, embedding_array)

            # If we have company information, also save by company
            metadata = self.metadata_store.get(doc_id, {})
            if "ticker" in metadata:
                ticker = metadata["ticker"]
                company_dir = self.company_dir / ticker
                company_dir.mkdir(parents=True, exist_ok=True)
                company_embedding_path = company_dir / f"{safe_id}.npy"
                np.save(company_embedding_path, embedding_array)

            logger.debug(f"Saved embedding for {doc_id} to {embedding_path}")
        except Exception as e:
            logger.warning(f"Error saving embedding for {doc_id}: {e}")

    def _load_text(self, doc_id: str) -> Optional[str]:
        """Load text from disk.

        Args:
            doc_id: Document ID

        Returns:
            Text content or None if not found
        """
        # Create a safe filename by replacing invalid characters
        safe_id = self._safe_filename(doc_id)

        # Try to load from disk
        try:
            text_path = self.text_dir / f"{safe_id}.txt"
            if text_path.exists():
                with open(text_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Error loading text for {doc_id}: {e}")

        return None

    def _load_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Load embedding from disk as NumPy array.

        Args:
            doc_id: Document ID

        Returns:
            Embedding vector as NumPy array or None if not found
        """
        # Create a safe filename by replacing invalid characters
        safe_id = self._safe_filename(doc_id)

        # Try to load from disk
        try:
            embedding_path = self.embeddings_dir / f"{safe_id}.npy"
            if embedding_path.exists():
                return np.load(embedding_path)
        except Exception as e:
            logger.warning(f"Error loading embedding for {doc_id}: {e}")

        return None

    def _safe_filename(self, doc_id: str) -> str:
        """Create a safe filename by replacing invalid characters.

        Args:
            doc_id: Document ID

        Returns:
            Safe filename
        """
        return doc_id.replace('/', '_').replace('\\', '_').replace(':', '_')

    def upsert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        texts: Optional[List[str]] = None
    ) -> None:
        """Add vectors and their associated text to the store.

        Args:
            vectors: List of embedding vectors
            ids: List of IDs for the vectors
            metadata: Optional list of metadata dictionaries
            texts: Optional list of text chunks associated with the vectors
        """
        try:
            logger.info(f"Upserting {len(vectors)} vectors with IDs: {ids}")

            for i, (vector, id_) in enumerate(zip(vectors, ids)):
                meta = metadata[i] if metadata and i < len(metadata) else {}
                text = texts[i] if texts and i < len(texts) else ""

                # Save metadata, text, and embedding
                self._save_metadata(id_, meta)
                if text:
                    self._save_text(id_, text)
                self._save_embedding(id_, vector)

                # If this document belongs to a company that's currently loaded,
                # we need to update the FAISS index
                if self.faiss_index is not None:
                    ticker = meta.get("ticker", "unknown")
                    if ticker in self.loaded_companies:
                        self._update_faiss_index_for_doc(id_, vector)

            logger.info(f"Successfully upserted {len(vectors)} vectors")

        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise

    def _update_faiss_index_for_doc(self, doc_id: str, vector: List[float]) -> None:
        """Update the FAISS index for a single document.

        Args:
            doc_id: Document ID
            vector: Embedding vector
        """
        if self.faiss_index is None:
            return

        # Check if document is already in the index
        if doc_id in self.faiss_id_map.values():
            # Find the index position
            idx = list(self.faiss_id_map.values()).index(doc_id)
            # Remove and re-add (FAISS doesn't support direct updates)
            self.faiss_index.remove_ids(np.array([idx], dtype=np.int64))
            
        # Add the vector to the index
        vector_array = np.array([vector], dtype=np.float32)
        idx = self.faiss_index.ntotal  # Get the next available index
        self.faiss_index.add(vector_array)
        self.faiss_id_map[idx] = doc_id

    def _load_faiss_index_for_companies(self, companies: List[str]) -> None:
        """Load embeddings for specified companies into a FAISS index.

        Args:
            companies: List of company tickers to load
        """
        # If no companies specified, use all
        if not companies:
            companies = ["all"]

        # Check if we already have these companies loaded
        if set(companies).issubset(self.loaded_companies) and self.faiss_index is not None:
            logger.info(f"Companies {companies} already loaded in FAISS index")
            return

        # Get document IDs for the specified companies
        doc_ids = set()
        for company in companies:
            doc_ids.update(self.company_to_docs.get(company, set()))

        logger.info(f"Loading {len(doc_ids)} documents for companies {companies} into FAISS index")

        # Load embeddings for these documents
        vectors = []
        valid_doc_ids = []

        for doc_id in doc_ids:
            embedding = self._load_embedding(doc_id)
            if embedding is not None:
                vectors.append(embedding)
                valid_doc_ids.append(doc_id)

        if not vectors:
            logger.warning(f"No embeddings found for companies {companies}")
            self.faiss_index = None
            self.faiss_id_map = {}
            self.loaded_companies = set()
            return

        # Create a new FAISS index
        vector_dim = vectors[0].shape[0]
        self.faiss_index = faiss.IndexFlatL2(vector_dim)
        
        # Add vectors to the index
        vectors_array = np.vstack(vectors).astype(np.float32)
        self.faiss_index.add(vectors_array)
        
        # Create mapping from FAISS index positions to document IDs
        self.faiss_id_map = {i: doc_id for i, doc_id in enumerate(valid_doc_ids)}
        
        # Update loaded companies
        self.loaded_companies = set(companies)
        
        logger.info(f"Created FAISS index with {len(vectors)} vectors of dimension {vector_dim}")

    def search_vectors(
        self,
        query_text: str,
        companies: Optional[List[str]] = None,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors and return their associated text.

        Args:
            query_text: Query text to search for
            companies: Optional list of company tickers to search within
            top_k: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            List of dictionaries containing search results with text
        """
        try:
            logger.info(f"Searching for documents matching: {query_text}")
            
            # Generate embedding for the query text
            from sec_filing_analyzer.embeddings import EmbeddingGenerator
            embedding_generator = EmbeddingGenerator()
            query_embedding = embedding_generator.generate_embeddings([query_text])[0]
            
            # Load FAISS index for the specified companies
            self._load_faiss_index_for_companies(companies)
            
            if self.faiss_index is None or self.faiss_index.ntotal == 0:
                logger.warning("No vectors available for search")
                return []
            
            # Convert query embedding to numpy array
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Search the FAISS index
            distances, indices = self.faiss_index.search(query_array, top_k * 2)  # Get more results for filtering
            
            # Process results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(self.faiss_id_map):
                    continue  # Skip invalid indices
                    
                doc_id = self.faiss_id_map[idx]
                metadata = self.metadata_store.get(doc_id, {})
                
                # Apply metadata filter if provided
                if metadata_filter and not self._matches_filter(metadata, metadata_filter):
                    continue
                
                # Load text
                text = self._load_text(doc_id)
                if text is None:
                    continue
                
                # Add to results
                results.append({
                    "id": doc_id,
                    "score": float(1.0 / (1.0 + distance)),  # Convert distance to similarity score
                    "metadata": metadata,
                    "text": text
                })
                
                # Stop once we have enough results
                if len(results) >= top_k:
                    break
            
            logger.info(f"Found {len(results)} matching documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter.

        Args:
            metadata: Document metadata
            filter_dict: Filter dictionary

        Returns:
            True if metadata matches the filter, False otherwise
        """
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a document.

        Args:
            doc_id: Document ID

        Returns:
            Metadata dictionary or None if not found
        """
        return self.metadata_store.get(doc_id)

    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Get text for a document.

        Args:
            doc_id: Document ID

        Returns:
            Text content or None if not found
        """
        return self._load_text(doc_id)

    def get_document_embedding(self, doc_id: str) -> Optional[List[float]]:
        """Get embedding for a document.

        Args:
            doc_id: Document ID

        Returns:
            Embedding vector or None if not found
        """
        embedding = self._load_embedding(doc_id)
        if embedding is not None:
            return embedding.tolist()
        return None

    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from the store.

        Args:
            ids: List of IDs to delete
        """
        try:
            for id_ in ids:
                # Remove metadata
                safe_id = self._safe_filename(id_)
                metadata_path = self.metadata_dir / f"{safe_id}.json"
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Remove text
                text_path = self.text_dir / f"{safe_id}.txt"
                if text_path.exists():
                    text_path.unlink()
                
                # Remove embedding
                embedding_path = self.embeddings_dir / f"{safe_id}.npy"
                if embedding_path.exists():
                    embedding_path.unlink()
                
                # Remove from company-specific directory
                metadata = self.metadata_store.get(id_, {})
                if "ticker" in metadata:
                    ticker = metadata["ticker"]
                    company_dir = self.company_dir / ticker
                    company_embedding_path = company_dir / f"{safe_id}.npy"
                    if company_embedding_path.exists():
                        company_embedding_path.unlink()
                
                # Remove from in-memory stores
                if id_ in self.metadata_store:
                    del self.metadata_store[id_]
                
                # Update company mapping
                for company_docs in self.company_to_docs.values():
                    if id_ in company_docs:
                        company_docs.remove(id_)
                
            # Reset FAISS index to force reload
            self.faiss_index = None
            self.faiss_id_map = {}
            self.loaded_companies = set()
            
            logger.info(f"Deleted {len(ids)} vectors from store")

        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")

    def list_documents(self, company: Optional[str] = None) -> List[str]:
        """List all document IDs in the store.

        Args:
            company: Optional company ticker to filter by

        Returns:
            List of document IDs
        """
        if company:
            return list(self.company_to_docs.get(company, set()))
        return list(self.metadata_store.keys())

    def list_companies(self) -> List[str]:
        """List all companies in the store.

        Returns:
            List of company tickers
        """
        # Exclude the "all" category
        return [company for company in self.company_to_docs.keys() if company != "all"]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_documents": len(self.metadata_store),
            "total_companies": len(self.list_companies()),
            "documents_by_company": {company: len(docs) for company, docs in self.company_to_docs.items() if company != "all"},
            "storage_size_mb": 0
        }
        
        # Calculate storage size
        total_size = 0
        for dir_path in [self.metadata_dir, self.text_dir, self.embeddings_dir, self.company_dir]:
            for file_path in dir_path.glob("**/*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        stats["storage_size_mb"] = total_size / (1024 * 1024)
        
        return stats
