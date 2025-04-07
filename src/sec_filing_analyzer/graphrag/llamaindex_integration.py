"""
LlamaIndex Integration for GraphRAG

This module provides integration between LlamaIndex and our graph store.
Note: This module is deprecated. Use the storage module instead.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from llama_index.core.graph_stores.types import LabelledNode, ChunkNode, EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery

from ..config import STORAGE_CONFIG
from ..storage import GraphStore as UnifiedGraphStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This class is deprecated. Use the UnifiedGraphStore from the storage module instead.
class LlamaIndexGraphStore:
    """
    LlamaIndex integration for our graph store.
    
    This class provides a bridge between LlamaIndex's graph store interface
    and our unified graph store implementation.
    
    DEPRECATED: Use the UnifiedGraphStore from the storage module instead.
    """
    
    def __init__(self, graph_store: Optional[UnifiedGraphStore] = None):
        """Initialize the LlamaIndex graph store integration."""
        self.graph_store = graph_store or UnifiedGraphStore()
        
        # Log deprecation warning
        logger.warning("LlamaIndexGraphStore from graphrag module is deprecated. Use UnifiedGraphStore from storage module instead.")
    
    def add_node(self, node: LabelledNode) -> None:
        """Add a node to the graph store."""
        self.graph_store.add_node(node.id, node.properties)
    
    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the graph store."""
        self.graph_store.add_relation(relation.source_id, relation.target_id, relation.properties)
    
    def get_node(self, node_id: str) -> Optional[LabelledNode]:
        """Get a node from the graph store."""
        node = self.graph_store.get_node(node_id)
        if node:
            return LabelledNode(id=node_id, properties=node)
        return None
    
    def get_relations(self, node_id: str) -> List[Relation]:
        """Get relations for a node from the graph store."""
        relations = self.graph_store.get_relations(node_id)
        return [
            Relation(source_id=r["source"], target_id=r["target"], properties=r["properties"])
            for r in relations
        ]
    
    def query(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Execute a query on the graph store."""
        return self.graph_store.query(query, **kwargs)
    
    def vector_similarity_search(
        self,
        query_embedding: List[float],
        similarity_top_k: int = 2,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        return self.graph_store.vector_similarity_search(
            query_embedding,
            similarity_top_k=similarity_top_k,
            **kwargs
        ) 