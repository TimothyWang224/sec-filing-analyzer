"""
Optimized Vector Store Implementation

This module provides an optimized implementation of vector storage operations
using NumPy binary storage and FAISS for efficient similarity search.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np

# Import storage utilities
from sec_filing_analyzer.storage.cache_utils import get_metadata, load_cached_mapping
from sec_filing_analyzer.storage.duckdb_metadata_store import DuckDBMetadataStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedVectorStore:
    """
    Optimized vector store implementation using NumPy binary storage and FAISS.
    Loads embeddings on-demand and filters by company for efficient querying.
    """

    def __init__(
        self,
        store_path: Optional[str] = None,
        index_type: str = "flat",
        use_gpu: bool = False,
        use_duckdb: bool = False,
        nlist: int = 100,  # For IVF indexes
        nprobe: int = 10,  # For IVF indexes
        m: int = 16,  # For HNSW indexes
        ef_construction: int = 200,  # For HNSW indexes
        ef_search: int = 128,  # For HNSW indexes
    ):
        """Initialize the vector store.

        Args:
            store_path: Optional path to store the vector store data
            index_type: Type of FAISS index to use ('flat', 'ivf', 'hnsw', 'ivfpq')
            use_gpu: Whether to use GPU acceleration if available
            use_duckdb: Whether to use DuckDB for metadata storage
            nlist: Number of clusters for IVF indexes
            nprobe: Number of clusters to visit during search for IVF indexes
            m: Number of connections per element for HNSW indexes
            ef_construction: Size of the dynamic list for HNSW construction
            ef_search: Size of the dynamic list for HNSW search
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

        # DuckDB configuration
        self.use_duckdb = use_duckdb
        self.db_path = self.store_path / "metadata.duckdb"
        self.db_store = None

        if self.use_duckdb:
            # Initialize DuckDB metadata store
            try:
                self.db_store = DuckDBMetadataStore(
                    self.db_path, read_only=True, create_if_missing=False
                )
                logger.info(f"Using DuckDB for metadata storage at {self.db_path}")
            except FileNotFoundError:
                logger.warning(
                    f"DuckDB metadata store not found at {self.db_path}, falling back to file-based storage"
                )
                self.use_duckdb = False
            except Exception as e:
                logger.warning(
                    f"Error initializing DuckDB metadata store: {e}, falling back to file-based storage"
                )
                self.use_duckdb = False

        # Initialize in-memory storage for metadata
        self.metadata_store = self._load_metadata_store()

        # Create company to document mapping
        self.company_to_docs = self._build_company_to_docs_mapping()

        # FAISS index configuration
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # Initialize FAISS index (but don't load any vectors yet)
        self.faiss_index = None
        self.faiss_id_map = {}  # Maps FAISS index positions to document IDs
        self.loaded_companies = set()  # Track which companies are loaded
        self.index_params = {}  # Store index parameters

        # Delta index for incremental updates
        self.delta_index = None
        self.delta_id_map = {}  # Maps delta index positions to document IDs
        self.delta_companies = set()  # Track which companies have delta updates

        logger.info(f"Initialized optimized vector store at {self.store_path}")

    def _load_metadata_store(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata for all documents from disk.

        Returns:
            Dictionary mapping document IDs to metadata
        """
        start_time = time.time()

        # If using DuckDB, we don't need to load all metadata into memory
        # We'll just return an empty dictionary and load metadata on-demand
        if self.use_duckdb and self.db_store:
            logger.info(
                "Using DuckDB for metadata storage, skipping full metadata load"
            )
            # Return an empty dictionary - metadata will be loaded on-demand
            return {}

        # Check if we have a consolidated metadata cache
        cache_dir = self.store_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        consolidated_path = cache_dir / "consolidated_metadata.json"

        if consolidated_path.exists():
            try:
                # Try to use orjson for faster parsing if available
                try:
                    import orjson

                    with open(consolidated_path, "rb") as f:
                        metadata_store = orjson.loads(f.read())
                except ImportError:
                    with open(consolidated_path, "r", encoding="utf-8") as f:
                        metadata_store = json.load(f)

                logger.info(
                    f"Loaded consolidated metadata for {len(metadata_store)} documents in {time.time() - start_time:.2f} seconds"
                )
                return metadata_store
            except Exception as e:
                logger.warning(
                    f"Error loading consolidated metadata: {e}, falling back to individual files"
                )

        # If no consolidated file exists or loading failed, load from individual files
        metadata_store = {}

        if not self.metadata_dir.exists():
            return metadata_store

        # Try to use orjson for faster parsing if available
        try:
            import orjson

            use_orjson = True
        except ImportError:
            use_orjson = False

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                doc_id = metadata_file.stem
                if use_orjson:
                    with open(metadata_file, "rb") as f:
                        metadata = orjson.loads(f.read())
                else:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                metadata_store[doc_id] = metadata
            except Exception as e:
                logger.warning(f"Error loading metadata from {metadata_file}: {e}")

        # Save consolidated metadata for next time
        try:
            # Try to use orjson for faster serialization if available
            try:
                import orjson

                with open(consolidated_path, "wb") as f:
                    f.write(orjson.dumps(metadata_store))
            except ImportError:
                with open(consolidated_path, "w", encoding="utf-8") as f:
                    json.dump(metadata_store, f)

            logger.info(
                f"Created consolidated metadata cache with {len(metadata_store)} documents"
            )
        except Exception as e:
            logger.warning(f"Error creating consolidated metadata cache: {e}")

        logger.info(
            f"Loaded metadata for {len(metadata_store)} documents in {time.time() - start_time:.2f} seconds"
        )
        return metadata_store

    def _build_company_to_docs_mapping(self) -> Dict[str, Set[str]]:
        """Build a mapping from companies to document IDs.

        Returns:
            Dictionary mapping company tickers to sets of document IDs
        """
        start_time = time.time()

        # If using DuckDB, use it to build the mapping
        if self.use_duckdb and self.db_store:
            try:
                company_to_docs = self.db_store.build_company_to_docs_mapping()
                logger.info(
                    f"Built company-to-documents mapping from DuckDB for {len(company_to_docs)} companies in {time.time() - start_time:.2f} seconds"
                )
                return company_to_docs
            except Exception as e:
                logger.warning(
                    f"Error building company-to-documents mapping from DuckDB: {e}, falling back to file-based mapping"
                )

        # Define the rebuild function for file-based mapping
        def rebuild_mapping():
            company_to_docs = defaultdict(set)

            for doc_id, metadata in self.metadata_store.items():
                ticker = metadata.get("ticker", "unknown")
                company_to_docs[ticker].add(doc_id)

                # Also add to "all" category for queries that don't specify a company
                company_to_docs["all"].add(doc_id)

            # Save the mapping to disk for future reference (legacy format)
            mapping_file = self.store_path / "company_doc_mapping.json"
            serializable_mapping = {k: list(v) for k, v in company_to_docs.items()}

            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(serializable_mapping, f, indent=2)

            logger.info(
                f"Built company-to-documents mapping for {len(company_to_docs)} companies in {time.time() - start_time:.2f} seconds"
            )
            return company_to_docs

        # Use the cached mapping loader
        cache_file = self.store_path / "cache" / "company_doc_mapping.json"
        return load_cached_mapping(
            cache_file=cache_file,
            metadata_dir=self.metadata_dir,
            rebuild_func=rebuild_mapping,
            force_rebuild=False,
        )

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
        return doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")

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

    def _create_faiss_index(self, vector_dim: int, num_vectors: int) -> faiss.Index:
        """Create a FAISS index based on the specified index type.

        Args:
            vector_dim: Dimension of the vectors
            num_vectors: Number of vectors to be added

        Returns:
            FAISS index
        """
        # Store index parameters for future reference
        self.index_params = {
            "type": self.index_type,
            "dimension": vector_dim,
            "num_vectors": num_vectors,
        }

        # Create the appropriate index based on type
        if self.index_type == "flat":
            # Simple flat index (exact search)
            index = faiss.IndexFlatL2(vector_dim)
            self.index_params["description"] = "Flat L2 index (exact search)"

        elif self.index_type == "ivf":
            # IVF index (approximate search with clustering)
            # Adjust nlist based on dataset size
            nlist = min(self.nlist, max(1, num_vectors // 10))
            quantizer = faiss.IndexFlatL2(vector_dim)
            index = faiss.IndexIVFFlat(quantizer, vector_dim, nlist)
            index.nprobe = self.nprobe  # Number of clusters to visit during search
            index.train_mode = True  # Enable training mode
            self.index_params.update(
                {
                    "nlist": nlist,
                    "nprobe": self.nprobe,
                    "description": f"IVF index with {nlist} clusters, visiting {self.nprobe} during search",
                }
            )

        elif self.index_type == "hnsw":
            # HNSW index (hierarchical navigable small world graph)
            index = faiss.IndexHNSWFlat(vector_dim, self.m)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            self.index_params.update(
                {
                    "m": self.m,
                    "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search,
                    "description": f"HNSW index with m={self.m}, efConstruction={self.ef_construction}, efSearch={self.ef_search}",
                }
            )

        elif self.index_type == "ivfpq":
            # IVF with Product Quantization (for memory efficiency)
            nlist = min(self.nlist, max(1, num_vectors // 10))
            m = 8  # Number of subquantizers
            bits = 8  # Bits per subquantizer
            quantizer = faiss.IndexFlatL2(vector_dim)
            index = faiss.IndexIVFPQ(quantizer, vector_dim, nlist, m, bits)
            index.nprobe = self.nprobe
            self.index_params.update(
                {
                    "nlist": nlist,
                    "nprobe": self.nprobe,
                    "m": m,
                    "bits": bits,
                    "description": f"IVFPQ index with {nlist} clusters, {m} subquantizers, {bits} bits",
                }
            )
        else:
            # Default to flat index if unknown type
            logger.warning(
                f"Unknown index type '{self.index_type}', defaulting to flat index"
            )
            index = faiss.IndexFlatL2(vector_dim)
            self.index_params["description"] = "Flat L2 index (exact search)"

        # Use GPU if requested and available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                self.index_params["gpu"] = True
                logger.info("Using GPU acceleration for FAISS index")
            except Exception as e:
                logger.warning(f"Failed to use GPU for FAISS: {e}. Using CPU instead.")
                self.index_params["gpu"] = False

        return index

    def _get_index_path(self, companies: List[str], is_delta: bool = False) -> Path:
        """Get the path for a FAISS index file based on company list.

        Args:
            companies: List of company tickers
            is_delta: Whether this is a delta index

        Returns:
            Path to the index file
        """
        # Sort companies for consistent naming
        sorted_companies = sorted(companies)
        company_str = "_".join(sorted_companies)

        # Create a filename with index type
        suffix = "delta" if is_delta else "main"
        filename = f"{company_str}_{self.index_type}_{suffix}.index"
        return self.index_dir / filename

    def _get_mapping_path(self, companies: List[str], is_delta: bool = False) -> Path:
        """Get the path for a FAISS ID mapping file based on company list.

        Args:
            companies: List of company tickers
            is_delta: Whether this is a delta index

        Returns:
            Path to the mapping file
        """
        # Sort companies for consistent naming
        sorted_companies = sorted(companies)
        company_str = "_".join(sorted_companies)

        # Create a filename
        suffix = "delta" if is_delta else "main"
        filename = f"{company_str}_{self.index_type}_{suffix}.mapping.json"
        return self.index_dir / filename

    def _save_faiss_index(self, companies: List[str], is_delta: bool = False) -> bool:
        """Save a FAISS index to disk.

        Args:
            companies: List of company tickers
            is_delta: Whether this is a delta index

        Returns:
            True if successful, False otherwise
        """
        # Determine which index to save
        if is_delta:
            index = self.delta_index
            id_map = self.delta_id_map
            if index is None:
                logger.warning("No delta index to save")
                return False
        else:
            index = self.faiss_index
            id_map = self.faiss_id_map
            if index is None:
                logger.warning("No main index to save")
                return False

        try:
            # Get paths
            index_path = self._get_index_path(companies, is_delta)
            mapping_path = self._get_mapping_path(companies, is_delta)

            # Save the index
            faiss.write_index(index, str(index_path))

            # Save the ID mapping
            with open(mapping_path, "w") as f:
                # Convert int keys to strings for JSON
                mapping_dict = {str(k): v for k, v in id_map.items()}
                json.dump(mapping_dict, f)

            # Save index parameters
            params_path = self.index_dir / f"{index_path.stem}.params.json"
            with open(params_path, "w") as f:
                json.dump(self.index_params, f)

            index_type = "delta" if is_delta else "main"
            logger.info(
                f"Saved FAISS {index_type} index for companies {companies} to {index_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def _load_faiss_index(self, companies: List[str], is_delta: bool = False) -> bool:
        """Load a FAISS index from disk if available.

        Args:
            companies: List of company tickers
            is_delta: Whether this is a delta index

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get paths
            index_path = self._get_index_path(companies, is_delta)
            mapping_path = self._get_mapping_path(companies, is_delta)

            if not index_path.exists() or not mapping_path.exists():
                index_type = "delta" if is_delta else "main"
                logger.info(
                    f"No saved {index_type} index found for companies {companies}"
                )
                return False

            # Load the index
            loaded_index = faiss.read_index(str(index_path))

            # Load the ID mapping
            with open(mapping_path, "r") as f:
                # JSON keys are strings, convert back to integers
                mapping_dict = json.load(f)
                loaded_id_map = {int(k): v for k, v in mapping_dict.items()}

            # Assign to the appropriate index and mapping
            if is_delta:
                self.delta_index = loaded_index
                self.delta_id_map = loaded_id_map
                self.delta_companies = set(companies)
            else:
                self.faiss_index = loaded_index
                self.faiss_id_map = loaded_id_map
                self.loaded_companies = set(companies)

            # Load index parameters if available
            params_path = self.index_dir / f"{index_path.stem}.params.json"
            if params_path.exists():
                with open(params_path, "r") as f:
                    self.index_params = json.load(f)

            index_type = "delta" if is_delta else "main"
            logger.info(
                f"Loaded FAISS {index_type} index for companies {companies} from {index_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def _load_faiss_index_for_companies(
        self, companies: List[str], force_rebuild: bool = False
    ) -> None:
        """Load embeddings for specified companies into a FAISS index.

        Args:
            companies: List of company tickers to load
            force_rebuild: Whether to force rebuilding the index even if it exists
        """
        # If no companies specified, use all
        if not companies:
            companies = ["all"]

        # Check if we already have these companies loaded
        if (
            set(companies).issubset(self.loaded_companies)
            and self.faiss_index is not None
        ):
            logger.info(f"Companies {companies} already loaded in FAISS index")
            return

        # Try to load from disk if not forcing rebuild
        if not force_rebuild and self._load_faiss_index(companies):
            # Also load delta index if it exists
            self._load_faiss_index(companies, is_delta=True)
            return

        # Get document IDs for the specified companies
        doc_ids = set()
        for company in companies:
            doc_ids.update(self.company_to_docs.get(company, set()))

        logger.info(
            f"Loading {len(doc_ids)} documents for companies {companies} into FAISS index"
        )

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

        # Create vectors array
        vectors_array = np.vstack(vectors).astype(np.float32)
        vector_dim = vectors[0].shape[0]
        num_vectors = len(vectors)

        # Create a new FAISS index
        self.faiss_index = self._create_faiss_index(vector_dim, num_vectors)

        # Train the index if needed (for IVF and IVFPQ)
        if self.index_type in ["ivf", "ivfpq"]:
            logger.info(f"Training {self.index_type} index with {num_vectors} vectors")
            self.faiss_index.train(vectors_array)

        # Add vectors to the index
        self.faiss_index.add(vectors_array)

        # Create mapping from FAISS index positions to document IDs
        self.faiss_id_map = {i: doc_id for i, doc_id in enumerate(valid_doc_ids)}

        # Update loaded companies
        self.loaded_companies = set(companies)

        logger.info(
            f"Created {self.index_type} FAISS index with {len(vectors)} vectors of dimension {vector_dim}"
        )

        # Save the index to disk
        self._save_faiss_index(companies)

        # Clear any delta index for these companies
        if self.delta_index is not None and self.delta_companies.intersection(
            set(companies)
        ):
            self.delta_index = None
            self.delta_id_map = {}
            self.delta_companies = set()

    def _save_document(
        self, doc_id: str, text: str, metadata: Dict[str, Any], embedding: np.ndarray
    ) -> None:
        """Save a document to disk.

        Args:
            doc_id: Document ID
            text: Document text
            metadata: Document metadata
            embedding: Document embedding
        """
        # Save text
        text_path = self.text_dir / f"{doc_id}.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Save metadata
        metadata_path = self.metadata_dir / f"{doc_id}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Save embedding
        embedding_path = self.embeddings_dir / f"{doc_id}.npy"
        np.save(embedding_path, embedding)

        # Save embedding by company if ticker is available
        ticker = metadata.get("ticker")
        if ticker:
            company_dir = self.company_dir / ticker
            company_dir.mkdir(parents=True, exist_ok=True)

            company_embedding_path = company_dir / f"{doc_id}.npy"
            np.save(company_embedding_path, embedding)

        # Update metadata store
        self.metadata_store[doc_id] = metadata

    def add_documents_to_index(
        self, documents: List[Dict[str, Any]], companies: Optional[List[str]] = None
    ) -> bool:
        """Add new documents to the delta index.

        Args:
            documents: List of documents with 'id', 'embedding', and 'metadata'
            companies: List of company tickers these documents belong to

        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return False

            # Extract companies from documents if not provided
            if not companies:
                companies = set()
                for doc in documents:
                    ticker = doc.get("metadata", {}).get("ticker")
                    if ticker:
                        companies.add(ticker)
                companies = list(companies)

            if not companies:
                logger.warning("No companies specified for documents")
                return False

            # Extract vectors and IDs
            vectors = []
            doc_ids = []

            for doc in documents:
                doc_id = doc.get("id")
                embedding = doc.get("embedding")

                if doc_id is None or embedding is None:
                    continue

                # Convert embedding to numpy array if it's a list
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)

                vectors.append(embedding)
                doc_ids.append(doc_id)

                # Save the document data
                self._save_document(
                    doc_id, doc.get("text", ""), doc.get("metadata", {}), embedding
                )

                # Update company-to-docs mapping
                ticker = doc.get("metadata", {}).get("ticker")
                if ticker:
                    if ticker not in self.company_to_docs:
                        self.company_to_docs[ticker] = set()
                    self.company_to_docs[ticker].add(doc_id)

            if not vectors:
                logger.warning("No valid vectors to add")
                return False

            # Create vectors array
            vectors_array = np.vstack(vectors).astype(np.float32)
            vector_dim = vectors[0].shape[0]
            num_vectors = len(vectors)

            # Create or load delta index
            if self.delta_index is None:
                # Try to load existing delta index
                if not self._load_faiss_index(companies, is_delta=True):
                    # Create new delta index
                    self.delta_index = self._create_faiss_index(vector_dim, num_vectors)
                    self.delta_id_map = {}
                    self.delta_companies = set(companies)

                    # Train if needed
                    if self.index_type in ["ivf", "ivfpq"]:
                        logger.info(f"Training delta index with {num_vectors} vectors")
                        self.delta_index.train(vectors_array)

            # Get the starting index for new vectors
            start_idx = self.delta_index.ntotal

            # Add vectors to delta index
            self.delta_index.add(vectors_array)

            # Update delta ID mapping
            for i, doc_id in enumerate(doc_ids):
                self.delta_id_map[start_idx + i] = doc_id

            # Update delta companies
            self.delta_companies.update(companies)

            # Save delta index
            self._save_faiss_index(list(self.delta_companies), is_delta=True)

            logger.info(
                f"Added {len(vectors)} vectors to delta index for companies {companies}"
            )
            return True
        except Exception as e:
            logger.error(f"Error adding documents to index: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def merge_delta_index(self) -> bool:
        """Merge the delta index into the main index.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.delta_index is None or self.delta_index.ntotal == 0:
                logger.info("No delta index to merge")
                return False

            if self.faiss_index is None:
                logger.warning("No main index to merge into")
                return False

            # Get all affected companies
            all_companies = list(self.loaded_companies.union(self.delta_companies))

            logger.info(
                f"Merging delta index ({self.delta_index.ntotal} vectors) into main index ({self.faiss_index.ntotal} vectors)"
            )

            # Extract vectors from delta index
            delta_vectors = []
            delta_doc_ids = []

            for i, doc_id in self.delta_id_map.items():
                # Get the vector from the delta index
                # This is a simplified approach - in a real implementation, you'd need to extract the actual vectors
                # from the FAISS index, which depends on the index type
                embedding = self._load_embedding(doc_id)
                if embedding is not None:
                    delta_vectors.append(embedding)
                    delta_doc_ids.append(doc_id)

            if not delta_vectors:
                logger.warning("No valid vectors in delta index")
                return False

            # Create vectors array
            delta_vectors_array = np.vstack(delta_vectors).astype(np.float32)

            # Get the starting index for new vectors
            start_idx = self.faiss_index.ntotal

            # Add delta vectors to main index
            self.faiss_index.add(delta_vectors_array)

            # Update main ID mapping
            for i, doc_id in enumerate(delta_doc_ids):
                self.faiss_id_map[start_idx + i] = doc_id

            # Update loaded companies
            self.loaded_companies.update(self.delta_companies)

            # Save updated main index
            self._save_faiss_index(all_companies)

            # Clear delta index
            self.delta_index = None
            self.delta_id_map = {}
            self.delta_companies = set()

            # Remove delta index files
            delta_index_path = self._get_index_path(all_companies, is_delta=True)
            delta_mapping_path = self._get_mapping_path(all_companies, is_delta=True)

            if delta_index_path.exists():
                delta_index_path.unlink()
            if delta_mapping_path.exists():
                delta_mapping_path.unlink()

            logger.info(
                f"Successfully merged delta index into main index. New size: {self.faiss_index.ntotal} vectors"
            )
            return True
        except Exception as e:
            logger.error(f"Error merging delta index: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def rebuild_index(self, companies: Optional[List[str]] = None) -> bool:
        """Force rebuild of the FAISS index for the specified companies.

        Args:
            companies: List of company tickers to rebuild index for, or None for all

        Returns:
            True if successful, False otherwise
        """
        try:
            # If no companies specified, use all
            if not companies:
                companies = list(self.company_to_docs.keys())
                if "unknown" in companies:
                    companies.remove("unknown")

            if not companies:
                logger.warning("No companies to rebuild index for")
                return False

            # Force rebuild
            start_time = time.time()
            self._load_faiss_index_for_companies(companies, force_rebuild=True)
            elapsed_time = time.time() - start_time

            logger.info(
                f"Rebuilt index for {len(companies)} companies in {elapsed_time:.2f} seconds"
            )
            return True
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def search_vectors(
        self,
        query_text: str,
        companies: Optional[List[str]] = None,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        date_range: Optional[Tuple[str, str]] = None,
        filing_types: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        keyword_match_type: str = "any",  # "any", "all", or "exact"
        hybrid_search_weight: float = 0.5,  # 0.0 = pure vector, 1.0 = pure keyword
        sort_by: str = "relevance",  # "relevance", "date", or "company"
        force_rebuild: bool = False,  # Whether to force rebuilding the index
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors and return their associated text.

        Args:
            query_text: Query text to search for
            companies: Optional list of company tickers to search within
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            date_range: Optional tuple of (start_date, end_date) in format 'YYYY-MM-DD'
            filing_types: Optional list of filing types to filter by (e.g., ['10-K', '10-Q'])
            sections: Optional list of document sections to filter by
            keywords: Optional list of keywords to search for
            keyword_match_type: Type of keyword matching ('any', 'all', or 'exact')
            hybrid_search_weight: Weight for hybrid search (0.0 = pure vector, 1.0 = pure keyword)
            sort_by: How to sort results ('relevance', 'date', or 'company')

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
            self._load_faiss_index_for_companies(companies, force_rebuild=force_rebuild)

            # Check if we have any indexes to search
            main_index_available = (
                self.faiss_index is not None and self.faiss_index.ntotal > 0
            )
            delta_index_available = (
                self.delta_index is not None and self.delta_index.ntotal > 0
            )

            if not main_index_available and not delta_index_available:
                logger.warning("No vectors available for search")
                return []

            # Convert query embedding to numpy array
            query_array = np.array([query_embedding], dtype=np.float32)

            # Search both main and delta indexes if available
            search_k = top_k * 10  # Get 10x more results to allow for filtering

            # Initialize results
            all_distances = []
            all_indices = []
            all_id_maps = []

            # Search main index if available
            if main_index_available:
                main_distances, main_indices = self.faiss_index.search(
                    query_array, min(search_k, self.faiss_index.ntotal)
                )
                all_distances.append(main_distances[0])
                all_indices.append(main_indices[0])
                all_id_maps.append(self.faiss_id_map)

            # Search delta index if available
            if delta_index_available:
                delta_distances, delta_indices = self.delta_index.search(
                    query_array, min(search_k, self.delta_index.ntotal)
                )
                all_distances.append(delta_distances[0])
                all_indices.append(delta_indices[0])
                all_id_maps.append(self.delta_id_map)

            # Process results
            candidates = []

            # Process all search results
            for distances, indices, id_map in zip(
                all_distances, all_indices, all_id_maps
            ):
                for distance, idx in zip(distances, indices):
                    if idx < 0 or idx not in id_map:
                        continue  # Skip invalid indices

                    doc_id = id_map[idx]
                    metadata = self.metadata_store.get(doc_id, {})

                    # Apply basic metadata filter if provided
                    if metadata_filter and not self._matches_filter(
                        metadata, metadata_filter
                    ):
                        continue

                    # Apply filing type filter
                    if filing_types and metadata.get("form") not in filing_types:
                        continue

                    # Apply date range filter
                    if date_range and not self._in_date_range(
                        metadata.get("filing_date"), date_range
                    ):
                        continue

                    # Apply section filter
                    if sections and metadata.get("section") not in sections:
                        continue

                    # Load text
                    text = self._load_text(doc_id)
                    if text is None:
                        continue

                    # Calculate vector similarity score
                    vector_score = float(
                        1.0 / (1.0 + distance)
                    )  # Convert distance to similarity score

                    # Calculate keyword match score if keywords provided
                    keyword_score = 0.0
                    if keywords:
                        keyword_score = self._calculate_keyword_score(
                            text, keywords, keyword_match_type
                        )

                    # Calculate hybrid score
                    if keywords and hybrid_search_weight > 0:
                        # Combine vector and keyword scores based on weight
                        score = (
                            1 - hybrid_search_weight
                        ) * vector_score + hybrid_search_weight * keyword_score
                    else:
                        score = vector_score

                    # Add to candidates
                    candidates.append(
                        {
                            "id": doc_id,
                            "score": score,
                            "vector_score": vector_score,
                            "keyword_score": keyword_score,
                            "metadata": metadata,
                            "text": text,
                        }
                    )

            # Sort results based on sort_by parameter
            if (
                sort_by == "date"
                and candidates
                and "filing_date" in candidates[0]["metadata"]
            ):
                candidates.sort(
                    key=lambda x: x["metadata"].get("filing_date", ""), reverse=True
                )
            elif (
                sort_by == "company"
                and candidates
                and "ticker" in candidates[0]["metadata"]
            ):
                candidates.sort(key=lambda x: x["metadata"].get("ticker", ""))
            else:  # Default to relevance
                candidates.sort(key=lambda x: x["score"], reverse=True)

            # Return top_k results
            results = candidates[:top_k]

            logger.info(f"Found {len(results)} matching documents")
            return results

        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def _matches_filter(
        self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches the filter.

        Args:
            metadata: Document metadata
            filter_dict: Filter dictionary

        Returns:
            True if metadata matches the filter, False otherwise
        """
        for key, value in filter_dict.items():
            if key not in metadata:
                return False

            # Handle list values (any match)
            if isinstance(value, list):
                if not isinstance(metadata[key], list) and metadata[key] not in value:
                    return False
                elif isinstance(metadata[key], list) and not any(
                    v in value for v in metadata[key]
                ):
                    return False
            # Handle exact match
            elif metadata[key] != value:
                return False

        return True

    def _in_date_range(
        self, date_str: Optional[str], date_range: Tuple[str, str]
    ) -> bool:
        """Check if a date is within a specified range.

        Args:
            date_str: Date string in format 'YYYY-MM-DD'
            date_range: Tuple of (start_date, end_date) in format 'YYYY-MM-DD'

        Returns:
            True if date is within range, False otherwise
        """
        if not date_str:
            return False

        try:
            from datetime import datetime

            # Parse dates
            date_format = "%Y-%m-%d"
            date = datetime.strptime(date_str, date_format)
            start_date, end_date = date_range

            # Handle empty start or end date
            if start_date:
                start = datetime.strptime(start_date, date_format)
                if date < start:
                    return False

            if end_date:
                end = datetime.strptime(end_date, date_format)
                if date > end:
                    return False

            return True
        except Exception as e:
            logger.warning(f"Error parsing date: {e}")
            return False

    def _calculate_keyword_score(
        self, text: str, keywords: List[str], match_type: str
    ) -> float:
        """Calculate keyword match score for text.

        Args:
            text: Text to search in
            keywords: List of keywords to search for
            match_type: Type of matching ('any', 'all', or 'exact')

        Returns:
            Score between 0.0 and 1.0
        """
        if not keywords or not text:
            return 0.0

        text_lower = text.lower()

        if match_type == "exact":
            # Search for the exact phrase
            phrase = " ".join(keywords).lower()
            count = text_lower.count(phrase)
            # Normalize by text length
            return min(1.0, count * len(phrase) / len(text_lower))

        elif match_type == "all":
            # All keywords must be present
            matches = [k.lower() in text_lower for k in keywords]
            if all(matches):
                # Count total occurrences
                total_count = sum(text_lower.count(k.lower()) for k in keywords)
                # Normalize by text length and keyword count
                return min(1.0, total_count / (len(text_lower) * len(keywords) * 0.01))
            return 0.0

        else:  # "any"
            # Any keyword can match
            matches = [k.lower() in text_lower for k in keywords]
            match_count = sum(matches)
            if match_count > 0:
                # Count total occurrences
                total_count = sum(text_lower.count(k.lower()) for k in keywords)
                # Normalize by text length and keyword count
                return min(1.0, total_count / (len(text_lower) * len(keywords) * 0.01))
            return 0.0

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a document.

        Args:
            doc_id: Document ID

        Returns:
            Metadata dictionary or None if not found
        """
        # Check in-memory cache first
        if doc_id in self.metadata_store:
            return self.metadata_store[doc_id]

        # If using DuckDB, get metadata from there
        if self.use_duckdb and self.db_store:
            try:
                metadata = self.db_store.get_metadata(doc_id)
                if metadata:
                    # Update in-memory store
                    self.metadata_store[doc_id] = metadata
                    return metadata
            except Exception as e:
                logger.warning(
                    f"Error getting metadata from DuckDB for {doc_id}: {e}, falling back to file-based storage"
                )

        # Use the cached metadata getter
        metadata = get_metadata(self.metadata_dir, doc_id)

        # Update in-memory store if found
        if metadata:
            self.metadata_store[doc_id] = metadata

        return metadata

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
        if self.use_duckdb and self.db_store:
            # Get document count from DuckDB
            doc_count = self.db_store.get_document_count()
            company_count = self.db_store.get_company_count()
        else:
            doc_count = len(self.metadata_store)
            company_count = len(self.list_companies())

        stats = {
            "total_documents": doc_count,
            "total_companies": company_count,
            "documents_by_company": {
                company: len(docs)
                for company, docs in self.company_to_docs.items()
                if company != "all"
            },
            "storage_size_mb": 0,
            "using_duckdb": self.use_duckdb and self.db_store is not None,
        }

        # Calculate storage size
        total_size = 0
        for dir_path in [
            self.metadata_dir,
            self.text_dir,
            self.embeddings_dir,
            self.company_dir,
        ]:
            for file_path in dir_path.glob("**/*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

        stats["storage_size_mb"] = total_size / (1024 * 1024)

        return stats

    def __del__(self):
        """Clean up resources when the object is deleted."""
        # Close DuckDB connection if it exists
        if hasattr(self, "db_store") and self.db_store:
            try:
                self.db_store.close()
                logger.debug("Closed DuckDB connection")
            except Exception as e:
                logger.warning(f"Error closing DuckDB connection: {e}")
