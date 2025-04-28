"""
Vector Store Implementation

This module provides a focused implementation of vector storage operations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# Define a simple interface for vector stores
class VectorStoreInterface:
    """Interface for vector stores."""

    def add_embeddings(self, embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]) -> List[str]:
        """Add embeddings to the vector store."""
        raise NotImplementedError

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        raise NotImplementedError


from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import SimpleVectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore(VectorStoreInterface):
    """
    A class for storing and retrieving vector embeddings.
    """

    def __init__(self, store_path: Optional[str] = None):
        """
        Initialize the vector store.

        Args:
            store_path: Path to the vector store directory
        """
        self.store_path = store_path or "data/vector_store"
        self.embeddings_path = os.path.join(self.store_path, "embeddings")
        self.metadata_path = os.path.join(self.store_path, "metadata")

        # Create directories if they don't exist
        os.makedirs(self.embeddings_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)

        # Initialize in-memory storage
        self.embeddings = {}
        self.metadata = {}

        logger.info(f"Initialized vector store at {self.store_path}")

    def add_embeddings(self, embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]) -> List[str]:
        """
        Add embeddings to the vector store.

        Args:
            embeddings: List of embedding vectors
            metadata_list: List of metadata dictionaries

        Returns:
            List of IDs for the added embeddings
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")

        # Generate IDs for the embeddings
        ids = [self._generate_id(metadata) for metadata in metadata_list]

        # Store embeddings and metadata
        for i, (embedding_id, embedding, metadata) in enumerate(zip(ids, embeddings, metadata_list)):
            # Store in memory
            self.embeddings[embedding_id] = embedding
            self.metadata[embedding_id] = metadata

            # Store on disk
            self._save_embedding(embedding_id, embedding)
            self._save_metadata(embedding_id, metadata)

            if i % 100 == 0 and i > 0:
                logger.info(f"Stored {i} embeddings")

        logger.info(f"Added {len(embeddings)} embeddings to vector store")
        return ids

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of dictionaries with embedding ID, similarity score, and metadata
        """
        # Load all embeddings if not already in memory
        if not self.embeddings:
            self._load_all_embeddings()

        # Convert query embedding to numpy array
        query_embedding_np = np.array(query_embedding)

        # Calculate cosine similarity for all embeddings
        similarities = {}
        for embedding_id, embedding in self.embeddings.items():
            embedding_np = np.array(embedding)
            similarity = self._cosine_similarity(query_embedding_np, embedding_np)
            similarities[embedding_id] = similarity

        # Sort by similarity score (descending)
        sorted_ids = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)

        # Get top k results
        results = []
        for embedding_id in sorted_ids[:top_k]:
            metadata = self.get_metadata(embedding_id)
            results.append(
                {
                    "id": embedding_id,
                    "score": float(similarities[embedding_id]),
                    "metadata": metadata,
                }
            )

        return results

    def _generate_id(self, metadata: Dict[str, Any]) -> str:
        """
        Generate a unique ID for an embedding based on its metadata.

        Args:
            metadata: Metadata dictionary

        Returns:
            Unique ID string
        """
        # Use filing_id and chunk_id if available
        if "filing_id" in metadata and "chunk_id" in metadata:
            return f"{metadata['filing_id']}_{metadata['chunk_id']}"

        # Use a combination of metadata values
        metadata_str = json.dumps(metadata, sort_keys=True)
        import hashlib

        return hashlib.md5(metadata_str.encode()).hexdigest()

    def _save_embedding(self, embedding_id: str, embedding: List[float]) -> None:
        """
        Save an embedding to disk.

        Args:
            embedding_id: ID of the embedding
            embedding: Embedding vector
        """
        embedding_path = os.path.join(self.embeddings_path, f"{embedding_id}.json")
        with open(embedding_path, "w") as f:
            json.dump(embedding, f)

    def _save_metadata(self, embedding_id: str, metadata: Dict[str, Any]) -> None:
        """
        Save metadata to disk.

        Args:
            embedding_id: ID of the embedding
            metadata: Metadata dictionary
        """
        metadata_path = os.path.join(self.metadata_path, f"{embedding_id}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def get_metadata(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata by embedding ID.

        Args:
            embedding_id: ID of the embedding

        Returns:
            Metadata dictionary or None if not found
        """
        # Check in-memory cache first
        if embedding_id in self.metadata:
            return self.metadata[embedding_id]

        # Try to load from disk
        metadata_path = os.path.join(self.metadata_path, f"{embedding_id}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Cache in memory
            self.metadata[embedding_id] = metadata
            return metadata

        return None

    def _load_all_embeddings(self) -> None:
        """
        Load all embeddings from disk into memory.
        """
        logger.info("Loading all embeddings into memory")

        # Load embeddings
        for file in os.listdir(self.embeddings_path):
            if file.endswith(".json"):
                embedding_id = file[:-5]  # Remove .json extension
                embedding_path = os.path.join(self.embeddings_path, file)

                with open(embedding_path, "r") as f:
                    embedding = json.load(f)

                self.embeddings[embedding_id] = embedding

        # Load metadata
        for file in os.listdir(self.metadata_path):
            if file.endswith(".json"):
                embedding_id = file[:-5]  # Remove .json extension
                metadata_path = os.path.join(self.metadata_path, file)

                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                self.metadata[embedding_id] = metadata

        logger.info(f"Loaded {len(self.embeddings)} embeddings into memory")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity score
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class PineconeVectorStore(VectorStoreInterface):
    """Pinecone implementation of vector store."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: str = "gcp-starter",
        index_name: str = "sec-filings",
    ):
        """Initialize Pinecone vector store."""
        try:
            import pinecone

            self.api_key = api_key or os.getenv("PINECONE_API_KEY")
            if not self.api_key:
                raise ValueError("Pinecone API key not found")

            pinecone.init(api_key=self.api_key, environment=environment)

            if index_name not in pinecone.list_indexes():
                pinecone.create_index(name=index_name, dimension=1536, metric="cosine")

            self.index = pinecone.Index(index_name)
            logger.info(f"Initialized Pinecone vector store with index: {index_name}")

        except ImportError:
            logger.error("Pinecone package not installed")
            raise
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise

    def upsert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Upsert vectors to Pinecone."""
        try:
            vectors_to_upsert = []
            for i, (vector, id_) in enumerate(zip(vectors, ids)):
                vector_data = {
                    "id": id_,
                    "values": vector,
                    "metadata": metadata[i] if metadata else {},
                }
                vectors_to_upsert.append(vector_data)

            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i : i + batch_size]
                self.index.upsert(vectors=batch)

            logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
            return True

        except Exception as e:
            logger.error(f"Error upserting vectors to Pinecone: {str(e)}")
            return False

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone."""
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_metadata,
            )

            return [{"id": match.id, "score": match.score, "metadata": match.metadata} for match in results.matches]

        except Exception as e:
            logger.error(f"Error searching vectors in Pinecone: {str(e)}")
            return []

    def get_vector(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get a vector by ID from Pinecone."""
        try:
            result = self.index.fetch(ids=[vector_id])
            if vector_id in result.vectors:
                vector = result.vectors[vector_id]
                return {
                    "id": vector.id,
                    "values": vector.values,
                    "metadata": vector.metadata,
                }
            return None
        except Exception as e:
            logger.error(f"Error getting vector from Pinecone: {str(e)}")
            return None

    def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector by ID from Pinecone."""
        try:
            self.index.delete(ids=[vector_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting vector from Pinecone: {str(e)}")
            return False


class LlamaIndexVectorStore:
    """
    Vector store implementation using LlamaIndex's SimpleVectorStore.
    Stores both embeddings and text chunks for full LlamaIndex functionality.
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

        # Initialize in-memory storage for metadata and text
        self.metadata_store = self._load_metadata_store()
        self.text_store = self._load_text_store()
        self.embedding_store = {}

        # Initialize SimpleVectorStore
        self.vector_store = SimpleVectorStore(stores_text=True)

        # Create a storage context
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Create an empty index - we'll add documents directly to the vector store
        self.index = None

        # Try to load existing index
        self._load_or_create_index()

        logger.info(f"Initialized LlamaIndex vector store at {self.store_path}")

    def upsert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        texts: Optional[List[str]] = None,
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

            # Create nodes with text and embeddings
            nodes = []
            for i, (vector, id_) in enumerate(zip(vectors, ids)):
                meta = metadata[i] if metadata else {}
                text = texts[i] if texts else ""

                logger.debug(f"Creating node for ID {id_} with metadata: {meta}")

                # Add original document ID to metadata
                meta["original_doc_id"] = id_

                # Limit metadata size to avoid LlamaIndex errors
                limited_meta = {}
                for key, value in meta.items():
                    if key in [
                        "ticker",
                        "form",
                        "filing_date",
                        "company",
                        "cik",
                        "item",
                        "original_doc_id",
                    ]:
                        limited_meta[key] = value

                # Add chunk metadata if it's a chunk
                if "chunk_metadata" in meta and isinstance(meta["chunk_metadata"], dict):
                    chunk_meta = meta["chunk_metadata"]
                    if "item" in chunk_meta:
                        limited_meta["item"] = chunk_meta["item"]
                    if "is_table" in chunk_meta:
                        limited_meta["is_table"] = chunk_meta["is_table"]

                # Create node
                # Ensure embedding is a flat list of floats, not a list of lists
                if vector and isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                    # This is a list of lists, we need to flatten it
                    logger.warning(f"Embedding for {id_} is a list of lists, flattening it")

                    # Use the first embedding in the list
                    vector = vector[0]

                # Ensure vector is a list of floats
                if not isinstance(vector, list):
                    try:
                        vector = list(vector)
                    except Exception as e:
                        logger.error(f"Could not convert embedding to list: {e}")
                        # Use a zero vector as fallback
                        vector = [0.0] * 1536

                node = Document(
                    text=text,
                    embedding=vector,
                    metadata=limited_meta,
                    doc_id=id_,
                    id_=id_,  # Use the original ID as the node ID
                )
                nodes.append(node)

            # Add nodes to vector store
            for node in nodes:
                logger.info(f"Adding node {node.doc_id} to vector store with embedding shape: {len(node.embedding)}")

                # Add node to vector store
                self.vector_store.add(nodes=[node])

                # Store metadata and text in our in-memory storage and on disk
                if node.metadata:
                    self.metadata_store[node.doc_id] = node.metadata
                    self._save_metadata(node.doc_id, node.metadata)
                    logger.info(f"Stored metadata for {node.doc_id}: {node.metadata}")
                if node.text:
                    self.text_store[node.doc_id] = node.text
                    self._save_text(node.doc_id, node.text)
                    logger.info(f"Stored text for {node.doc_id} (length: {len(node.text)})")

                # Store embedding for exploration
                self._save_embedding(node.doc_id, node.embedding)

            logger.info(f"Added {len(nodes)} vectors with text to store")

        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise

    def search_vectors(
        self,
        query_vector: str,  # This is now a query string, not a vector
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors and return their associated text.

        Args:
            query_vector: Query string (not an embedding vector anymore)
            top_k: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            List of dictionaries containing search results with text
        """
        try:
            logger.info(f"Searching for documents matching: {query_vector}")

            # Make sure we have an index
            if self.index is None:
                self._load_or_create_index()

            if self.index is None:
                logger.error("Failed to create or load index")
                return []

            # Create a retriever with the specified parameters
            retriever = self.index.as_retriever(similarity_top_k=top_k)

            # Apply metadata filters if provided
            if metadata_filter:
                retriever = self.index.as_retriever(similarity_top_k=top_k, filters=metadata_filter)

            # Create a query engine from the retriever
            query_engine = RetrieverQueryEngine.from_args(retriever)

            # Execute the query
            logger.info(f"Executing query: {query_vector}")
            results = query_engine.query(query_vector)

            # Get source nodes from the response
            source_nodes = results.source_nodes if hasattr(results, "source_nodes") else []

            logger.info(f"Found {len(source_nodes)} results")

            # Format results
            formatted_results = []
            for node in source_nodes:
                try:
                    # LlamaIndex 0.9+ uses NodeWithScore objects
                    if hasattr(node, "node"):
                        # This is a NodeWithScore object
                        doc_id = node.node.node_id
                        score = node.score
                    else:
                        # Fallback for older versions or different node types
                        doc_id = (
                            node.doc_id
                            if hasattr(node, "doc_id")
                            else node.node_id
                            if hasattr(node, "node_id")
                            else "unknown"
                        )
                        score = node.score if hasattr(node, "score") else 0.0

                    # Try to get the original document ID from the node's metadata
                    original_doc_id = None
                    if hasattr(node, "node") and hasattr(node.node, "metadata"):
                        original_doc_id = node.node.metadata.get("original_doc_id")

                    # Use the original document ID if available
                    if original_doc_id:
                        doc_id = original_doc_id

                    # Get metadata and text directly from the node if possible
                    if hasattr(node, "node"):
                        node_metadata = node.node.metadata
                        node_text = node.node.text

                        # Use node metadata and text if available
                        metadata = node_metadata
                        text = node_text
                    else:
                        # Fall back to in-memory storage
                        metadata = self.metadata_store.get(doc_id, {})
                        text = self.text_store.get(doc_id, "")

                    # If we still don't have metadata or text, try to load from disk
                    if not metadata:
                        metadata = self._load_metadata(doc_id) or {}
                    if not text:
                        text = self._load_text(doc_id) or ""

                    logger.debug(f"Processing result {doc_id}")
                    logger.debug(f"Retrieved metadata: {metadata}")
                    logger.debug(f"Retrieved text length: {len(text)}")

                    # Apply metadata filter if provided
                    if metadata_filter:
                        if not all(metadata.get(k) == v for k, v in metadata_filter.items()):
                            logger.debug(f"Skipping {doc_id} due to metadata filter")
                            continue

                    formatted_results.append(
                        {
                            "id": doc_id,
                            "score": score,
                            "metadata": metadata,
                            "text": text,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error formatting node: {e}")
                    # Skip this node
                    continue

            logger.info(f"Returning {len(formatted_results)} formatted results")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            return []

    def _load_metadata_store(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from disk.

        Returns:
            Dictionary mapping document IDs to metadata
        """
        metadata_store = {}
        try:
            if self.metadata_dir.exists():
                for file_path in self.metadata_dir.glob("*.json"):
                    try:
                        doc_id = file_path.stem
                        with open(file_path, "r", encoding="utf-8") as f:
                            import json

                            metadata = json.load(f)
                            metadata_store[doc_id] = metadata
                    except Exception as e:
                        logger.warning(f"Error loading metadata from {file_path}: {e}")
            logger.info(f"Loaded metadata for {len(metadata_store)} documents")
        except Exception as e:
            logger.error(f"Error loading metadata store: {e}")
        return metadata_store

    def _load_text_store(self) -> Dict[str, str]:
        """Load text from disk.

        Returns:
            Dictionary mapping document IDs to text
        """
        text_store = {}
        try:
            if self.text_dir.exists():
                for file_path in self.text_dir.glob("*.txt"):
                    try:
                        doc_id = file_path.stem
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                            text_store[doc_id] = text
                    except Exception as e:
                        logger.warning(f"Error loading text from {file_path}: {e}")
            logger.info(f"Loaded text for {len(text_store)} documents")
        except Exception as e:
            logger.error(f"Error loading text store: {e}")
        return text_store

    def _save_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Save metadata to disk.

        Args:
            doc_id: Document ID
            metadata: Metadata dictionary
        """
        try:
            # Create a safe filename by replacing invalid characters
            safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")

            # Save to disk
            metadata_path = self.metadata_dir / f"{safe_id}.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                import json

                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.debug(f"Saved metadata for {doc_id} to {metadata_path}")
        except Exception as e:
            logger.warning(f"Error saving metadata for {doc_id}: {e}")

    def _save_text(self, doc_id: str, text: str) -> None:
        """Save text to disk.

        Args:
            doc_id: Document ID
            text: Document text
        """
        try:
            # Create a safe filename by replacing invalid characters
            safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")

            # Save to disk
            text_path = self.text_dir / f"{safe_id}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

            logger.debug(f"Saved text for {doc_id} to {text_path}")
        except Exception as e:
            logger.warning(f"Error saving text for {doc_id}: {e}")

    def _save_embedding(self, doc_id: str, embedding: List[float]) -> None:
        """Save an embedding to disk for exploration.

        Args:
            doc_id: Document ID
            embedding: Embedding vector
        """
        try:
            # Store in memory
            self.embedding_store[doc_id] = embedding

            # Create a safe filename by replacing invalid characters
            safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")

            # Save to disk
            embedding_path = self.embeddings_dir / f"{safe_id}.json"
            with open(embedding_path, "w") as f:
                import json

                json.dump(embedding, f)

            logger.debug(f"Saved embedding for {doc_id} to {embedding_path}")
        except Exception as e:
            logger.warning(f"Error saving embedding for {doc_id}: {e}")

    def list_documents(self) -> List[str]:
        """List all document IDs in the vector store.

        Returns:
            List of document IDs
        """
        # Get IDs from in-memory storage
        in_memory_ids = set(self.metadata_store.keys())

        # Get IDs from on-disk storage
        on_disk_ids = set()
        if self.metadata_dir.exists():
            # Get all JSON files in the metadata directory
            for file_path in self.metadata_dir.glob("*.json"):
                # Get the original ID from the metadata file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        import json

                        metadata = json.load(f)
                        if "accession_number" in metadata:
                            on_disk_ids.add(metadata["accession_number"])
                        else:
                            # Fallback to using the filename
                            on_disk_ids.add(file_path.stem)
                except Exception as e:
                    logger.warning(f"Error reading metadata file {file_path}: {e}")
                    # Fallback to using the filename
                    on_disk_ids.add(file_path.stem)

        # Combine and return as a sorted list
        all_ids = list(in_memory_ids.union(on_disk_ids))
        all_ids.sort()
        return all_ids

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a document.

        Args:
            doc_id: Document ID

        Returns:
            Document metadata, or None if not found
        """
        # Check in-memory storage first
        metadata = self.metadata_store.get(doc_id)
        if metadata is not None:
            return metadata

        # Create a safe filename by replacing invalid characters
        safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")

        # Try to load from disk
        try:
            metadata_path = self.metadata_dir / f"{safe_id}.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    import json

                    metadata = json.load(f)
                    # Cache in memory for future use
                    self.metadata_store[doc_id] = metadata
                    return metadata
        except Exception as e:
            logger.warning(f"Error loading metadata for {doc_id}: {e}")

        return None

    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Get text for a document.

        Args:
            doc_id: Document ID

        Returns:
            Document text, or None if not found
        """
        # Check in-memory storage first
        text = self.text_store.get(doc_id)
        if text is not None:
            return text

        # Create a safe filename by replacing invalid characters
        safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")

        # Try to load from disk
        try:
            text_path = self.text_dir / f"{safe_id}.txt"
            if text_path.exists():
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    # Cache in memory for future use
                    self.text_store[doc_id] = text
                    return text
        except Exception as e:
            logger.warning(f"Error loading text for {doc_id}: {e}")

        return None

    def get_document_embedding(self, doc_id: str) -> Optional[List[float]]:
        """Get embedding for a document.

        Args:
            doc_id: Document ID

        Returns:
            Document embedding, or None if not found
        """
        # Try to get from memory first
        embedding = self.embedding_store.get(doc_id)
        if embedding is not None:
            return embedding

        # Create a safe filename by replacing invalid characters
        safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")

        # Try to load from disk
        try:
            embedding_path = self.embeddings_dir / f"{safe_id}.json"
            if embedding_path.exists():
                with open(embedding_path, "r") as f:
                    import json

                    embedding = json.load(f)
                    # Cache in memory for future use
                    self.embedding_store[doc_id] = embedding
                    return embedding
        except Exception as e:
            logger.warning(f"Error loading embedding for {doc_id}: {e}")

        return None

    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors from the store.

        Args:
            ids: List of IDs to delete
        """
        try:
            for id_ in ids:
                self.vector_store.delete(id_)
            logger.info(f"Deleted {len(ids)} vectors from store")

        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")

    def _load_or_create_index(self) -> None:
        """Load existing index or create a new one."""
        try:
            # Check if index exists
            index_path = self.index_dir / "index"
            if index_path.exists():
                logger.info(f"Loading existing index from {index_path}")
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store, storage_context=self.storage_context
                )
                logger.info("Successfully loaded existing index")
            else:
                # Create a new index with the documents we have
                logger.info("Creating new index from existing documents")
                documents = []

                # Load documents from disk
                for doc_id in self.metadata_store.keys():
                    try:
                        text = self.text_store.get(doc_id, "")
                        metadata = self.metadata_store.get(doc_id, {})
                        embedding = self._load_embedding(doc_id)

                        if text and embedding:
                            # Limit metadata size to avoid LlamaIndex errors
                            limited_metadata = {}
                            for key, value in metadata.items():
                                if key in [
                                    "ticker",
                                    "form",
                                    "filing_date",
                                    "company",
                                    "cik",
                                    "item",
                                ]:
                                    limited_metadata[key] = value

                            # Add chunk metadata if it's a chunk
                            if "chunk_metadata" in metadata and isinstance(metadata["chunk_metadata"], dict):
                                chunk_meta = metadata["chunk_metadata"]
                                if "item" in chunk_meta:
                                    limited_metadata["item"] = chunk_meta["item"]
                                if "is_table" in chunk_meta:
                                    limited_metadata["is_table"] = chunk_meta["is_table"]

                            doc = Document(
                                text=text,
                                metadata=limited_metadata,
                                embedding=embedding,
                                doc_id=doc_id,
                            )
                            documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Error loading document {doc_id}: {e}")

                if documents:
                    logger.info(f"Creating index with {len(documents)} documents")
                    self.index = VectorStoreIndex.from_documents(
                        documents=documents, storage_context=self.storage_context
                    )
                    # Persist the index
                    self.index.storage_context.persist(persist_dir=str(self.index_dir))
                    logger.info(f"Index created and persisted to {self.index_dir}")
                else:
                    logger.info("No documents found, creating empty index")
                    # Create an empty index
                    self.index = VectorStoreIndex.from_documents(
                        documents=[Document(text="", doc_id="temp")],
                        storage_context=self.storage_context,
                    )
                    # Remove temporary document
                    self.vector_store.delete("temp")
        except Exception as e:
            logger.error(f"Error loading or creating index: {e}")
            # Create an empty index as fallback
            self.index = None

    def _load_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Load metadata from disk.

        Args:
            doc_id: Document ID

        Returns:
            Metadata dictionary or None if not found
        """
        # Create a safe filename by replacing invalid characters
        safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")

        # Try to load from disk
        try:
            metadata_path = self.metadata_dir / f"{safe_id}.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    return metadata
        except Exception as e:
            logger.warning(f"Error loading metadata for {doc_id}: {e}")

        return None

    def _load_text(self, doc_id: str) -> Optional[str]:
        """Load text from disk.

        Args:
            doc_id: Document ID

        Returns:
            Document text or None if not found
        """
        # Create a safe filename by replacing invalid characters
        safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")

        # Try to load from disk
        try:
            text_path = self.text_dir / f"{safe_id}.txt"
            if text_path.exists():
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    return text
        except Exception as e:
            logger.warning(f"Error loading text for {doc_id}: {e}")

        return None

    def _load_embedding(self, doc_id: str) -> Optional[List[float]]:
        """Load embedding from disk.

        Args:
            doc_id: Document ID

        Returns:
            Embedding vector or None if not found
        """
        # Create a safe filename by replacing invalid characters
        safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")

        # Try to load from disk
        try:
            embedding_path = self.embeddings_dir / f"{safe_id}.json"
            if embedding_path.exists():
                with open(embedding_path, "r") as f:
                    embedding = json.load(f)
                    return embedding
        except Exception as e:
            logger.warning(f"Error loading embedding for {doc_id}: {e}")

        return None
