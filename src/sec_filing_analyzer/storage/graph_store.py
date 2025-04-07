"""
Graph Store Implementation

This module provides a unified graph storage implementation that combines
features from both the original graph store and GraphRAG implementations.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import networkx as nx
from rich.console import Console
import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from collections import defaultdict

from .interfaces import GraphStoreInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class GraphStore(GraphStoreInterface):
    """Unified graph database interface for SEC filing data."""
    
    def __init__(
        self,
        store_dir: str = "cache/graph_store",
        use_neo4j: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
        database: Optional[str] = None,
        max_cluster_size: Optional[int] = None
    ):
        """Initialize the graph store.
        
        Args:
            store_dir: Directory to store graph data
            use_neo4j: Whether to use Neo4j instead of in-memory storage
            username: Neo4j username
            password: Neo4j password
            url: Neo4j URL
            database: Neo4j database name
            max_cluster_size: Maximum size for community clusters
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.use_neo4j = use_neo4j
        self.max_cluster_size = max_cluster_size or 5
        self.community_summary = {}
        
        # Initialize storage
        if use_neo4j:
            self._init_neo4j(username, password, url, database)
        else:
            self._init_in_memory()
    
    def _init_neo4j(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
        database: Optional[str] = None
    ):
        """Initialize Neo4j connection and constraints."""
        # Get Neo4j credentials from environment variables or parameters
        uri = url or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = username or os.getenv("NEO4J_USER", "neo4j")
        pwd = password or os.getenv("NEO4J_PASSWORD", "password")
        db = database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self.database = db
            
            # Create constraints
            with self.driver.session(database=db) as session:
                # Company constraint
                session.run("""
                    CREATE CONSTRAINT company_id IF NOT EXISTS
                    FOR (c:Company)
                    REQUIRE c.ticker IS UNIQUE
                """)
                
                # Filing constraint
                session.run("""
                    CREATE CONSTRAINT filing_id IF NOT EXISTS
                    FOR (f:Filing)
                    REQUIRE f.accession_number IS UNIQUE
                """)
                
                # Chunk constraint
                session.run("""
                    CREATE CONSTRAINT chunk_id IF NOT EXISTS
                    FOR (c:Chunk)
                    REQUIRE c.chunkId IS UNIQUE
                """)
                
                # Topic constraint
                session.run("""
                    CREATE CONSTRAINT topic_id IF NOT EXISTS
                    FOR (t:Topic)
                    REQUIRE t.label IS UNIQUE
                """)
                
                # Create vector index for embeddings
                try:
                    session.run("""
                        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                        FOR (c:Chunk)
                        ON (c.embedding)
                        OPTIONS {indexConfig: {
                            `vector.dimensions`: 1536,
                            `vector.similarity_function`: 'cosine'
                        }}
                    """)
                    logger.info("Created vector index for chunk embeddings")
                except Exception as e:
                    logger.warning(f"Could not create vector index: {str(e)}")
            
            logger.info(f"Connected to Neo4j at {uri} and created constraints")
        except ServiceUnavailable:
            logger.error(f"Could not connect to Neo4j at {uri}. Using in-memory storage instead.")
            self.use_neo4j = False
            self._init_in_memory()
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {str(e)}. Using in-memory storage instead.")
            self.use_neo4j = False
            self._init_in_memory()
    
    def _init_in_memory(self):
        """Initialize in-memory graph storage."""
        self.graph = nx.DiGraph()
        logger.info("Initialized in-memory graph storage")
    
    def add_node(
        self,
        node_id: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a node to the graph store."""
        if properties is None:
            properties = {}
            
        if self.use_neo4j:
            try:
                with self.driver.session(database=self.database) as session:
                    # Convert properties to Neo4j format
                    props_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
                    query = f"""
                    MERGE (n {{id: $node_id}})
                    SET n += {{{props_str}}}
                    """
                    session.run(query, node_id=node_id, **properties)
            except Exception as e:
                logger.error(f"Error adding node to Neo4j: {str(e)}")
        else:
            self.graph.add_node(node_id, **properties)
    
    def add_relationship(
        self,
        from_node: str,
        to_node: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a relationship between nodes."""
        if properties is None:
            properties = {}
            
        if self.use_neo4j:
            try:
                with self.driver.session(database=self.database) as session:
                    # Convert properties to Neo4j format
                    props_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
                    query = f"""
                    MATCH (from {{id: $from_node}})
                    MATCH (to {{id: $to_node}})
                    MERGE (from)-[r:{relationship_type}]->(to)
                    SET r += {{{props_str}}}
                    """
                    session.run(query, from_node=from_node, to_node=to_node, **properties)
            except Exception as e:
                logger.error(f"Error adding relationship to Neo4j: {str(e)}")
        else:
            self.graph.add_edge(from_node, to_node, type=relationship_type, **properties)
    
    def query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query on the graph store."""
        if self.use_neo4j:
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(query)
                    return [dict(record) for record in result]
            except Exception as e:
                logger.error(f"Error executing Neo4j query: {str(e)}")
                return []
        else:
            # For in-memory graph, we'll need to implement a simple query parser
            # This is a basic implementation that only supports simple patterns
            try:
                # Parse the query to extract node and edge patterns
                # This is a simplified version - you might want to implement a proper query parser
                if "MATCH" in query.upper():
                    # Extract node patterns
                    node_patterns = query.split("MATCH")[1].split("WHERE")[0].strip()
                    # Extract conditions
                    conditions = query.split("WHERE")[1].strip() if "WHERE" in query else ""
                    
                    # Execute the query on the in-memory graph
                    results = []
                    for node in self.graph.nodes(data=True):
                        if self._match_node_pattern(node, node_patterns, conditions):
                            results.append({"node": dict(node[1])})
                    return results
                return []
            except Exception as e:
                logger.error(f"Error executing in-memory query: {str(e)}")
                return []
    
    def _match_node_pattern(
        self,
        node: Tuple[str, Dict[str, Any]],
        pattern: str,
        conditions: str
    ) -> bool:
        """Match a node against a pattern and conditions."""
        # This is a simplified implementation
        # You might want to implement a more sophisticated pattern matching
        try:
            # Check if node matches the pattern
            if "{" in pattern:
                # Extract property conditions
                props = pattern.split("{")[1].split("}")[0]
                for prop in props.split(","):
                    key, value = prop.split(":")
                    key = key.strip()
                    value = value.strip()
                    if node[1].get(key) != value:
                        return False
            
            # Check additional conditions
            if conditions:
                # This is a very basic condition parser
                # You might want to implement a more sophisticated one
                for condition in conditions.split("AND"):
                    if "=" in condition:
                        key, value = condition.split("=")
                        key = key.strip()
                        value = value.strip().strip("'")
                        if node[1].get(key) != value:
                            return False
            
            return True
        except Exception:
            return False
    
    def vector_query(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Execute a vector similarity query."""
        if self.use_neo4j:
            try:
                with self.driver.session(database=self.database) as session:
                    # Build filter condition
                    filter_condition = ""
                    if filter_metadata:
                        conditions = []
                        for key, value in filter_metadata.items():
                            conditions.append(f"n.{key} = ${key}")
                        filter_condition = "WHERE " + " AND ".join(conditions)
                    
                    query = f"""
                    CALL db.index.vector.queryNodes('chunk_embeddings', $k, $query_vector) YIELD node, score
                    {filter_condition}
                    RETURN node, score
                    ORDER BY score DESC
                    LIMIT $k
                    """
                    
                    result = session.run(
                        query,
                        k=top_k,
                        query_vector=query_vector,
                        **filter_metadata or {}
                    )
                    
                    nodes = []
                    scores = []
                    for record in result:
                        nodes.append(dict(record["node"]))
                        scores.append(record["score"])
                    
                    return nodes, scores
            except Exception as e:
                logger.error(f"Error executing vector query in Neo4j: {str(e)}")
                return [], []
        else:
            # For in-memory graph, we'll need to implement vector similarity search
            # This is a basic implementation using numpy
            try:
                nodes = []
                scores = []
                
                # Get all nodes with embeddings
                for node, data in self.graph.nodes(data=True):
                    if "embedding" in data:
                        # Calculate cosine similarity
                        similarity = np.dot(query_vector, data["embedding"]) / (
                            np.linalg.norm(query_vector) * np.linalg.norm(data["embedding"])
                        )
                        
                        # Apply filters if specified
                        if filter_metadata:
                            if not all(data.get(k) == v for k, v in filter_metadata.items()):
                                continue
                        
                        nodes.append(data)
                        scores.append(similarity)
                
                # Sort by similarity score
                sorted_indices = np.argsort(scores)[::-1][:top_k]
                return [nodes[i] for i in sorted_indices], [scores[i] for i in sorted_indices]
            except Exception as e:
                logger.error(f"Error executing vector query in memory: {str(e)}")
                return [], []
    
    def get_community_summaries(self) -> Dict[str, str]:
        """Get summaries of communities in the graph."""
        return self.community_summary
    
    def get_rel_map(
        self,
        graph_nodes: List[str],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None
    ) -> Dict[str, List[List[str]]]:
        """Get relationship map for given nodes."""
        if ignore_rels is None:
            ignore_rels = []
            
        if self.use_neo4j:
            try:
                with self.driver.session(database=self.database) as session:
                    query = f"""
                    MATCH path = (n)
                    WHERE n.id IN $nodes
                    AND ALL(r IN relationships(path) WHERE type(r) NOT IN $ignore_rels)
                    AND length(path) <= $depth
                    RETURN path
                    LIMIT $limit
                    """
                    
                    result = session.run(
                        query,
                        nodes=graph_nodes,
                        ignore_rels=ignore_rels,
                        depth=depth,
                        limit=limit
                    )
                    
                    rel_map = defaultdict(list)
                    for record in result:
                        path = record["path"]
                        nodes = [node["id"] for node in path.nodes]
                        for i, node_id in enumerate(nodes[:-1]):
                            rel_map[node_id].append(nodes[i+1:])
                    
                    return dict(rel_map)
            except Exception as e:
                logger.error(f"Error getting relationship map from Neo4j: {str(e)}")
                return {}
        else:
            # For in-memory graph
            try:
                rel_map = defaultdict(list)
                for node_id in graph_nodes:
                    if node_id in self.graph:
                        # Get all paths starting from this node
                        paths = []
                        for target in self.graph.nodes():
                            if target != node_id:
                                for path in nx.all_simple_paths(
                                    self.graph,
                                    source=node_id,
                                    target=target,
                                    cutoff=depth
                                ):
                                    # Filter out ignored relationships
                                    if not any(
                                        self.graph[path[i]][path[i+1]]["type"] in ignore_rels
                                        for i in range(len(path)-1)
                                    ):
                                        paths.append(path)
                        
                        # Add paths to relationship map
                        rel_map[node_id] = paths[:limit]
                
                return dict(rel_map)
            except Exception as e:
                logger.error(f"Error getting relationship map from memory: {str(e)}")
                return {} 