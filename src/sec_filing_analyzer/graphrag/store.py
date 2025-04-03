"""
Graph Store for GraphRAG

This module provides the main storage class for GraphRAG implementation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import time
import os
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np

from llama_index.core.graph_stores.types import LabelledNode, ChunkNode, EntityNode, Relation
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from ..config import NEO4J_CONFIG, GRAPH_STORE_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphStore(Neo4jPropertyGraphStore):
    """
    Enhanced graph store that incorporates community detection and summarization.
    
    This class extends Neo4j's graph store with community detection capabilities
    while preserving compatibility with our existing retrieval methods.
    """
    
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
        database: Optional[str] = None,
        max_cluster_size: Optional[int] = None,
    ):
        """Initialize the graph store."""
        # Use provided values or fall back to config
        username = username or NEO4J_CONFIG["username"]
        password = password or NEO4J_CONFIG["password"]
        url = url or NEO4J_CONFIG["url"]
        database = database or NEO4J_CONFIG["database"]
        max_cluster_size = max_cluster_size or GRAPH_STORE_CONFIG["max_cluster_size"]
        
        super().__init__(username=username, password=password, url=url, database=database)
        self.max_cluster_size = max_cluster_size
        self.community_summary = {}
        self.entity_info = None
    
    def add_node(self, node_id: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a node to the graph store.
        
        Args:
            node_id: Unique identifier for the node
            properties: Optional properties for the node
        """
        if properties is None:
            properties = {}
            
        # Create node with properties
        node = EntityNode(
            id=node_id,
            properties=properties
        )
        
        # Add to Neo4j
        self.upsert_nodes([node])
    
    def add_relationship(
        self,
        from_node: str,
        to_node: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relationship between nodes.
        
        Args:
            from_node: Source node ID
            to_node: Target node ID
            relationship_type: Type of relationship
            properties: Optional properties for the relationship
        """
        if properties is None:
            properties = {}
            
        # Create relationship
        relation = Relation(
            from_node=from_node,
            to_node=to_node,
            type=relationship_type,
            properties=properties
        )
        
        # Add to Neo4j
        self.upsert_relations([relation])
    
    def query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query on the graph store.
        
        Args:
            query: Cypher query string
            
        Returns:
            List of query results
        """
        try:
            result = self._driver.execute_query(query)
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
    
    def get_community_summaries(self) -> Dict[str, str]:
        """
        Get summaries of communities in the graph.
        
        Returns:
            Dict mapping community IDs to their summaries
        """
        return self.community_summary
    
    def get_rel_map(
        self,
        graph_nodes: List[str],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None
    ) -> Dict[str, List[List[str]]]:
        """
        Get relationship map for given nodes.
        
        Args:
            graph_nodes: List of node IDs
            depth: Maximum depth for traversal
            limit: Maximum number of relationships to return
            ignore_rels: Optional list of relationship types to ignore
            
        Returns:
            Dict mapping node IDs to their relationship paths
        """
        if ignore_rels is None:
            ignore_rels = []
            
        # Build Cypher query
        query = f"""
        MATCH path = (n)
        WHERE n.id IN $nodes
        AND ALL(r IN relationships(path) WHERE type(r) NOT IN $ignore_rels)
        AND length(path) <= $depth
        RETURN path
        LIMIT $limit
        """
        
        # Execute query
        results = self.query(query, {
            "nodes": graph_nodes,
            "ignore_rels": ignore_rels,
            "depth": depth,
            "limit": limit
        })
        
        # Process results
        rel_map = defaultdict(list)
        for record in results:
            path = record["path"]
            nodes = [node["id"] for node in path.nodes]
            for i, node_id in enumerate(nodes[:-1]):
                rel_map[node_id].append(nodes[i+1:])
                
        return dict(rel_map)
    
    def vector_query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        """
        Execute a vector similarity query.
        
        Args:
            query: Vector store query
            **kwargs: Additional query parameters
            
        Returns:
            Tuple of (nodes, scores)
        """
        # Convert query to Cypher
        cypher_query = """
        CALL db.index.vector.queryNodes('vector_index', $k, $query_vector) YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        """
        
        # Execute query
        results = self.query(cypher_query, {
            "k": query.similarity_top_k,
            "query_vector": query.query_vector
        })
        
        # Process results
        nodes = []
        scores = []
        for record in results:
            nodes.append(record["node"])
            scores.append(record["score"])
            
        return nodes, scores 