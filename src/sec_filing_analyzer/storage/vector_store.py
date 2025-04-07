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
    
    def get_vector(
        self,
        vector_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a vector by ID from Pinecone."""
        try:
            result = self.index.fetch(ids=[vector_id])
            if vector_id in result.vectors:
                vector = result.vectors[vector_id]
                return {
                    "id": vector.id,
                    "values": vector.values,
                    "metadata": vector.metadata
                }
            return None
        except Exception as e:
            logger.error(f"Error getting vector from Pinecone: {str(e)}")
            return None
    
    def delete_vector(
        self,
        vector_id: str
    ) -> bool:
        """Delete a vector by ID from Pinecone."""
        try:
            self.index.delete(ids=[vector_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting vector from Pinecone: {str(e)}")
            return False

class LlamaIndexVectorStore(VectorStoreInterface):
    """LlamaIndex implementation of vector store."""
    
    def __init__(
        self,
        store_dir: str = "cache/vector_store",
        index_name: str = "sec-filings"
    ):
        """Initialize LlamaIndex vector store."""
        try:
            from llama_index.core import VectorStoreIndex, Document
            from llama_index.core.vector_stores import SimpleVectorStore
            from llama_index.core.storage import StorageContext
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            self.store_dir = store_dir
            self.index_name = index_name
            
            self.embedding_model = OpenAIEmbedding()
            self.vector_store = SimpleVectorStore()
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Create an empty index first
            self.index = VectorStoreIndex(
                [],
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
        metadata: Optional[List[Dict[str, Any]]] = None,
        texts: Optional[List[str]] = None
    ) -> bool:
        """Upsert vectors to LlamaIndex."""
        try:
            from llama_index.core import Document
            
            documents = []
            for i, (vector, id_) in enumerate(zip(vectors, ids)):
                doc_metadata = metadata[i] if metadata else {}
                doc_metadata["vector_id"] = id_
                
                # Use the text if provided, otherwise use a placeholder
                text = texts[i] if texts and i < len(texts) else f"Document {id_}"
                
                doc = Document(
                    text=text,
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
            from llama_index.core.vector_stores.types import VectorStoreQuery
            
            query = VectorStoreQuery(
                query_embedding=query_vector,
                similarity_top_k=top_k,
                filters=filter_metadata
            )
            
            results = self.vector_store.query(query)
            
            return [
                {
                    "id": node.metadata.get("vector_id"),
                    "score": score,
                    "metadata": node.metadata,
                    "text": node.text
                }
                for node, score in zip(results.nodes, results.similarities)
            ]
        except Exception as e:
            logger.error(f"Error searching vectors in LlamaIndex: {str(e)}")
            return []
    
    def get_vector(
        self,
        vector_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a vector by ID from LlamaIndex."""
        try:
            from llama_index.core.vector_stores.types import VectorStoreQuery
            
            # This is a simplified implementation
            # You might want to implement a more efficient way to retrieve vectors
            query = VectorStoreQuery(
                query_embedding=[0] * 1536,  # Dummy vector
                similarity_top_k=1,
                filters={"vector_id": vector_id}
            )
            
            results = self.vector_store.query(query)
            if results.nodes:
                node = results.nodes[0]
                return {
                    "id": node.metadata.get("vector_id"),
                    "values": node.embedding,
                    "metadata": node.metadata,
                    "text": node.text
                }
            return None
        except Exception as e:
            logger.error(f"Error getting vector from LlamaIndex: {str(e)}")
            return None
    
    def delete_vector(
        self,
        vector_id: str
    ) -> bool:
        """Delete a vector by ID from LlamaIndex."""
        try:
            # This is a simplified implementation
            # You might want to implement a more efficient way to delete vectors
            self.index.delete_nodes([vector_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting vector from LlamaIndex: {str(e)}")
            return False 