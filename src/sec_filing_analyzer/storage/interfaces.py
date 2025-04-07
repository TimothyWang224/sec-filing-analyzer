"""
Storage Interfaces

This module defines the interfaces for graph and vector storage.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

class VectorStoreInterface(ABC):
    """Interface for vector storage."""
    
    @abstractmethod
    def upsert_vectors(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Upsert vectors to the store.
        
        Args:
            vectors: List of vector embeddings
            ids: List of vector IDs
            metadata: Optional list of metadata dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector embedding
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of dictionaries containing:
                - id: Vector ID
                - score: Similarity score
                - metadata: Vector metadata
        """
        pass

class GraphStoreInterface(ABC):
    """Interface for graph storage."""
    
    @abstractmethod
    def add_node(
        self,
        node_id: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a node to the graph.
        
        Args:
            node_id: Node ID
            properties: Optional node properties
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a relation to the graph.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relation
            properties: Optional relation properties
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_node(
        self,
        node_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a node from the graph.
        
        Args:
            node_id: Node ID
            
        Returns:
            Optional node properties
        """
        pass
    
    @abstractmethod
    def get_relations(
        self,
        node_id: str,
        relation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relations for a node.
        
        Args:
            node_id: Node ID
            relation_type: Optional relation type filter
            
        Returns:
            List of relation dictionaries containing:
                - source: Source node ID
                - target: Target node ID
                - type: Relation type
                - properties: Relation properties
        """
        pass
    
    @abstractmethod
    def query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a query on the graph.
        
        Args:
            query: Query string
            parameters: Optional query parameters
            
        Returns:
            List of result dictionaries
        """
        pass
    
    @abstractmethod
    def vector_similarity_search(
        self,
        query_embedding: List[float],
        similarity_top_k: int = 2,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query_embedding: Query vector embedding
            similarity_top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of dictionaries containing:
                - id: Node ID
                - score: Similarity score
                - metadata: Node metadata
        """
        pass 