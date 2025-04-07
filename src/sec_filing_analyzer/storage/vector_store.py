"""
Vector Store Implementation

This module provides a focused implementation of vector storage operations.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from .interfaces import VectorStoreInterface

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeVectorStore(VectorStoreInterface):
    """Pinecone implementation of vector store."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: str = "gcp-starter",
        index_name: str = "sec-filings"
    ):
        """Initialize Pinecone vector store."""
        try:
            import pinecone
            self.api_key = api_key or os.getenv("PINECONE_API_KEY")
            if not self.api_key:
                raise ValueError("Pinecone API key not found")
            
            pinecone.init(api_key=self.api_key, environment=environment)
            
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine"
                )
            
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
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Upsert vectors to Pinecone."""
        try:
            vectors_to_upsert = []
            for i, (vector, id_) in enumerate(zip(vectors, ids)):
                vector_data = {
                    "id": id_,
                    "values": vector,
                    "metadata": metadata[i] if metadata else {}
                }
                vectors_to_upsert.append(vector_data)
            
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i+batch_size]
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
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone."""
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_metadata
            )
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                for match in results.matches
            ]
            
        except Exception as e:
            logger.error(f"Error searching vectors in Pinecone: {str(e)}")
            return []

class LlamaIndexVectorStore(VectorStoreInterface):
    """LlamaIndex implementation of vector store."""
    
    def __init__(
        self,
        store_dir: str = "cache/vector_store",
        index_name: str = "sec-filings"
    ):
        """Initialize LlamaIndex vector store."""
        try:
            from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
            from llama_index.vector_stores import SimpleVectorStore
            from llama_index.storage.storage_context import StorageContext
            from llama_index.embeddings import OpenAIEmbedding
            
            self.store_dir = store_dir
            self.index_name = index_name
            
            self.embedding_model = OpenAIEmbedding()
            self.vector_store = SimpleVectorStore()
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context,
                embed_model=self.embedding_model
            )
            
            logger.info(f"Initialized LlamaIndex vector store with index: {index_name}")
            
        except ImportError:
            logger.error("LlamaIndex package not installed")
            raise
        except Exception as e:
            logger.error(f"Error initializing LlamaIndex: {str(e)}")
            raise
    
    def upsert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Upsert vectors to LlamaIndex."""
        try:
            from llama_index import Document
            
            documents = []
            for i, (vector, id_) in enumerate(zip(vectors, ids)):
                doc_metadata = metadata[i] if metadata else {}
                doc_metadata["vector_id"] = id_
                
                doc = Document(
                    text="",  # Empty text as we only care about the embedding
                    embedding=vector,
                    metadata=doc_metadata
                )
                documents.append(doc)
            
            self.index.insert_nodes(documents)
            logger.info(f"Successfully upserted {len(vectors)} vectors to LlamaIndex")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting vectors to LlamaIndex: {str(e)}")
            return False
    
    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in LlamaIndex."""
        try:
            query = VectorStoreQuery(
                query_vector=query_vector,
                similarity_top_k=top_k,
                filters=filter_metadata
            )
            
            results = self.index.query(query)
            
            return [
                {
                    "id": node.metadata.get("vector_id"),
                    "score": score,
                    "metadata": node.metadata
                }
                for node, score in zip(results.nodes, results.scores)
            ]
            
        except Exception as e:
            logger.error(f"Error searching vectors in LlamaIndex: {str(e)}")
            return [] 