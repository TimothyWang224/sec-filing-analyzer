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
        use_neo4j: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
        database: Optional[str] = None,
        max_cluster_size: Optional[int] = None
    ):
        """Initialize the graph store."""
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
    ) -> bool:
        """Add a node to the graph store."""
        try:
            if properties is None:
                properties = {}

            if self.use_neo4j:
                with self.driver.session(database=self.database) as session:
                    # Convert properties to Neo4j format
                    props_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
                    query = f"""
                    MERGE (n {{id: $node_id}})
                    SET n += {{{props_str}}}
                    """
                    session.run(query, node_id=node_id, **properties)
            else:
                self.graph.add_node(node_id, **properties)
            return True
        except Exception as e:
            logger.error(f"Error adding node to graph store: {str(e)}")
            return False

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a relation to the graph."""
        try:
            if properties is None:
                properties = {}

            if self.use_neo4j:
                with self.driver.session(database=self.database) as session:
                    # Convert properties to Neo4j format
                    props_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
                    query = f"""
                    MATCH (from {{id: $source_id}})
                    MATCH (to {{id: $target_id}})
                    MERGE (from)-[r:{relation_type}]->(to)
                    SET r += {{{props_str}}}
                    """
                    session.run(query, source_id=source_id, target_id=target_id, **properties)
            else:
                self.graph.add_edge(source_id, target_id, type=relation_type, **properties)
            return True
        except Exception as e:
            logger.error(f"Error adding relation to graph store: {str(e)}")
            return False

    def get_node(
        self,
        node_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a node from the graph."""
        try:
            if self.use_neo4j:
                with self.driver.session(database=self.database) as session:
                    query = """
                    MATCH (n {id: $node_id})
                    RETURN n
                    """
                    result = session.run(query, node_id=node_id)
                    record = result.single()
                    return dict(record["n"]) if record else None
            else:
                return self.graph.nodes[node_id] if node_id in self.graph else None
        except Exception as e:
            logger.error(f"Error getting node from graph store: {str(e)}")
            return None

    def get_relations(
        self,
        node_id: str,
        relation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get relations for a node."""
        try:
            if self.use_neo4j:
                with self.driver.session(database=self.database) as session:
                    # Build relationship filter
                    rel_filter = f"[r:{relation_type}]" if relation_type else "[r]"

                    query = f"""
                    MATCH (n {{id: $node_id}})-{rel_filter}-(related)
                    RETURN type(r) as type, properties(r) as properties,
                           n.id as source, related.id as target
                    """

                    result = session.run(query, node_id=node_id)
                    return [
                        {
                            "source": record["source"],
                            "target": record["target"],
                            "type": record["type"],
                            "properties": record["properties"]
                        }
                        for record in result
                    ]
            else:
                relations = []
                for neighbor in self.graph.neighbors(node_id):
                    edge_data = self.graph.get_edge_data(node_id, neighbor)
                    if not relation_type or edge_data.get("type") == relation_type:
                        relations.append({
                            "source": node_id,
                            "target": neighbor,
                            "type": edge_data.get("type", ""),
                            "properties": {k: v for k, v in edge_data.items() if k != "type"}
                        })
                return relations
        except Exception as e:
            logger.error(f"Error getting relations from graph store: {str(e)}")
            return []

    def vector_similarity_search(
        self,
        query_embedding: List[float],
        similarity_top_k: int = 2,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        try:
            if self.use_neo4j:
                with self.driver.session(database=self.database) as session:
                    # Convert query embedding to string for Neo4j
                    query_embedding_str = json.dumps(query_embedding)

                    # Build filter conditions
                    filter_conditions = ""
                    if filter_metadata:
                        conditions = []
                        for key, value in filter_metadata.items():
                            conditions.append(f"n.{key} = ${key}")
                        if conditions:
                            filter_conditions = "WHERE " + " AND ".join(conditions)

                    # Query for similar vectors
                    query = f"""
                    MATCH (n)
                    {filter_conditions}
                    WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS similarity
                    ORDER BY similarity DESC
                    LIMIT $top_k
                    RETURN n, similarity
                    """

                    result = session.run(
                        query,
                        query_embedding=query_embedding,
                        top_k=similarity_top_k,
                        **filter_metadata if filter_metadata else {}
                    )

                    return [
                        {
                            "id": record["n"]["id"],
                            "score": record["similarity"],
                            "metadata": dict(record["n"])
                        }
                        for record in result
                    ]
            else:
                # For in-memory graph, we'll need to implement vector similarity
                # This is a basic implementation using cosine similarity
                results = []
                for node_id, node_data in self.graph.nodes(data=True):
                    if "embedding" in node_data:
                        # Apply metadata filter if provided
                        if filter_metadata:
                            if not all(node_data.get(k) == v for k, v in filter_metadata.items()):
                                continue

                        # Convert embeddings to lists if they are numpy arrays
                        node_embedding = node_data["embedding"]
                        if hasattr(node_embedding, 'tolist'):
                            node_embedding = node_embedding.tolist()

                        query_emb = query_embedding
                        if hasattr(query_emb, 'tolist'):
                            query_emb = query_emb.tolist()

                        # Calculate cosine similarity
                        score = self._cosine_similarity(query_emb, node_embedding)
                        results.append({
                            "id": node_id,
                            "score": score,
                            "metadata": node_data
                        })

                # Sort by score and limit results
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:similarity_top_k]
        except Exception as e:
            logger.error(f"Error performing vector similarity search: {str(e)}")
            return []

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        return np.dot(v1_array, v2_array) / (np.linalg.norm(v1_array) * np.linalg.norm(v2_array))

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

    def add_filing(
        self,
        filing_id: str,
        text: str,
        metadata: Dict[str, Any],
        chunks: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Add a filing to the graph store."""
        try:
            # Extract key metadata for the filing node
            filing_type = metadata.get('form_type', '')
            filing_date = metadata.get('filing_date', '')
            company_name = metadata.get('company_name', '')
            ticker = metadata.get('ticker', '')
            fiscal_year = metadata.get('fiscal_year', '')
            fiscal_period = metadata.get('fiscal_period', '')

            # Add filing node with labels
            if self.use_neo4j:
                with self.driver.session(database=self.database) as session:
                    query = """
                    MERGE (f:Filing:Document {id: $filing_id})
                    SET f.type = 'filing',
                        f.filing_type = $filing_type,
                        f.filing_date = $filing_date,
                        f.company_name = $company_name,
                        f.ticker = $ticker,
                        f.fiscal_year = $fiscal_year,
                        f.fiscal_period = $fiscal_period,
                        f.accession_number = $filing_id
                    """
                    session.run(
                        query,
                        filing_id=filing_id,
                        filing_type=filing_type,
                        filing_date=filing_date,
                        company_name=company_name,
                        ticker=ticker,
                        fiscal_year=fiscal_year,
                        fiscal_period=fiscal_period
                    )

                    # Add company node and relationship to filing
                    if ticker:
                        query = """
                        MERGE (c:Company {ticker: $ticker})
                        SET c.name = $company_name
                        WITH c
                        MATCH (f:Filing {id: $filing_id})
                        MERGE (c)-[:FILED]->(f)
                        """
                        session.run(
                            query,
                            ticker=ticker,
                            company_name=company_name,
                            filing_id=filing_id
                        )
            else:
                # For in-memory storage
                self.add_node(
                    node_id=filing_id,
                    properties={
                        "type": "filing",
                        "text": text,
                        "filing_type": filing_type,
                        "filing_date": filing_date,
                        "company_name": company_name,
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "fiscal_period": fiscal_period,
                        "accession_number": filing_id,
                        **metadata
                    }
                )

                # Add company node and relationship to filing
                if ticker:
                    self.add_node(
                        node_id=f"company_{ticker}",
                        properties={
                            "type": "company",
                            "ticker": ticker,
                            "name": company_name
                        }
                    )
                    self.add_relation(
                        source_id=f"company_{ticker}",
                        target_id=filing_id,
                        relation_type="FILED",
                        properties={}
                    )

            # Add chunks if available
            if chunks:
                # First pass: Add all chunk nodes with improved properties
                for i, chunk in enumerate(chunks):
                    # Determine chunk ID based on whether it's a split chunk
                    is_split_chunk = chunk.get('is_split_chunk', False)
                    original_order = chunk.get('original_order')

                    if is_split_chunk and original_order is not None:
                        # This is a split chunk, use a more specific ID
                        chunk_id = f"{filing_id}_chunk_{original_order}_split_{chunk.get('split_chunk_index', 0)}"
                    else:
                        # This is a regular chunk
                        chunk_id = f"{filing_id}_chunk_{i}"

                    # Convert numpy arrays to lists in chunk metadata
                    clean_chunk = {}
                    for k, v in chunk.items():
                        if k == 'embedding':
                            # Skip embedding as it's stored separately
                            continue
                        if hasattr(v, 'tolist'):
                            clean_chunk[k] = v.tolist()
                        else:
                            clean_chunk[k] = v

                    # Extract section information
                    section = chunk.get('section', '')
                    section_title = chunk.get('section_title', '')
                    section_number = chunk.get('section_number', '')

                    # Add chunk node with improved properties
                    if self.use_neo4j:
                        with self.driver.session(database=self.database) as session:
                            query = """
                            MERGE (c:Chunk:Document {id: $chunk_id})
                            SET c.type = 'chunk',
                                c.text = $text,
                                c.order = $order,
                                c.parent_filing = $filing_id,
                                c.section = $section,
                                c.section_title = $section_title,
                                c.section_number = $section_number,
                                c.is_split_chunk = $is_split_chunk,
                                c.original_order = $original_order,
                                c.split_chunk_index = $split_chunk_index,
                                c.chunkId = $chunk_id,
                                c.item = $item,
                                c.is_table = $is_table
                            """
                            session.run(
                                query,
                                chunk_id=chunk_id,
                                text=chunk.get('text', ''),
                                order=i,
                                filing_id=filing_id,
                                section=section,
                                section_title=section_title,
                                section_number=section_number,
                                is_split_chunk=is_split_chunk,
                                original_order=original_order if original_order is not None else i,
                                split_chunk_index=chunk.get('split_chunk_index', 0),
                                item=chunk.get("item", ""),
                                is_table=chunk.get("is_table", False)
                            )

                            # Add relationship to filing
                            query = """
                            MATCH (f:Filing {id: $filing_id})
                            MATCH (c:Chunk {id: $chunk_id})
                            MERGE (f)-[r:CONTAINS]->(c)
                            SET r.order = $order,
                                r.item = $item,
                                r.is_table = $is_table
                            """
                            session.run(
                                query,
                                filing_id=filing_id,
                                chunk_id=chunk_id,
                                order=i,
                                item=chunk.get("item", ""),
                                is_table=chunk.get("is_table", False)
                            )
                    else:
                        # For in-memory storage
                        self.add_node(
                            node_id=chunk_id,
                            properties={
                                "type": "chunk",
                                "text": chunk.get("text", ""),
                                "chunkId": chunk_id,
                                "order": i,
                                "parent_filing": filing_id,
                                "section": section,
                                "section_title": section_title,
                                "section_number": section_number,
                                "is_split_chunk": is_split_chunk,
                                "original_order": original_order if original_order is not None else i,
                                "split_chunk_index": chunk.get('split_chunk_index', 0),
                                "item": chunk.get("item", ""),
                                "is_table": chunk.get("is_table", False),
                                **clean_chunk
                            }
                        )

                        # Add relationship from filing to chunk
                        self.add_relation(
                            source_id=filing_id,
                            target_id=chunk_id,
                            relation_type="CONTAINS",
                            properties={
                                "order": i,
                                "item": chunk.get("item", ""),
                                "is_table": chunk.get("is_table", False)
                            }
                        )

                # Second pass: Add relationships between chunks and create section nodes
                for i, chunk in enumerate(chunks):
                    # Determine chunk ID
                    is_split_chunk = chunk.get('is_split_chunk', False)
                    original_order = chunk.get('original_order')

                    if is_split_chunk and original_order is not None:
                        chunk_id = f"{filing_id}_chunk_{original_order}_split_{chunk.get('split_chunk_index', 0)}"
                    else:
                        chunk_id = f"{filing_id}_chunk_{i}"

                    # Add relationships between split chunks if applicable
                    if is_split_chunk:
                        # Add relationship to previous chunk if it exists
                        prev_chunk_id = f"{filing_id}_chunk_{original_order}_split_{chunk.get('split_chunk_index', 0)-1}"
                        if chunk.get('prev_chunk_order'):
                            # Check if previous chunk exists
                            prev_chunk_exists = False
                            if self.use_neo4j:
                                with self.driver.session(database=self.database) as session:
                                    query = "MATCH (c:Chunk {id: $prev_chunk_id}) RETURN count(c) as count"
                                    result = session.run(query, prev_chunk_id=prev_chunk_id)
                                    prev_chunk_exists = result.single()["count"] > 0
                            else:
                                prev_chunk_exists = prev_chunk_id in self.graph.nodes

                            if prev_chunk_exists:
                                if self.use_neo4j:
                                    with self.driver.session(database=self.database) as session:
                                        query = """
                                        MATCH (c1:Chunk {id: $chunk_id})
                                        MATCH (c2:Chunk {id: $prev_chunk_id})
                                        MERGE (c1)-[r:FOLLOWS]->(c2)
                                        SET r.split_group = $original_order
                                        """
                                        session.run(
                                            query,
                                            chunk_id=chunk_id,
                                            prev_chunk_id=prev_chunk_id,
                                            original_order=original_order
                                        )
                                else:
                                    self.add_relation(
                                        source_id=chunk_id,
                                        target_id=prev_chunk_id,
                                        relation_type="FOLLOWS",
                                        properties={
                                            "split_group": original_order
                                        }
                                    )

                        # Add relationship to next chunk if it exists
                        next_chunk_id = f"{filing_id}_chunk_{original_order}_split_{chunk.get('split_chunk_index', 0)+1}"
                        if chunk.get('next_chunk_order'):
                            # Check if next chunk exists
                            next_chunk_exists = False
                            if self.use_neo4j:
                                with self.driver.session(database=self.database) as session:
                                    query = "MATCH (c:Chunk {id: $next_chunk_id}) RETURN count(c) as count"
                                    result = session.run(query, next_chunk_id=next_chunk_id)
                                    next_chunk_exists = result.single()["count"] > 0
                            else:
                                next_chunk_exists = next_chunk_id in self.graph.nodes

                            if next_chunk_exists:
                                if self.use_neo4j:
                                    with self.driver.session(database=self.database) as session:
                                        query = """
                                        MATCH (c1:Chunk {id: $chunk_id})
                                        MATCH (c2:Chunk {id: $next_chunk_id})
                                        MERGE (c1)-[r:PRECEDES]->(c2)
                                        SET r.split_group = $original_order
                                        """
                                        session.run(
                                            query,
                                            chunk_id=chunk_id,
                                            next_chunk_id=next_chunk_id,
                                            original_order=original_order
                                        )
                                else:
                                    self.add_relation(
                                        source_id=chunk_id,
                                        target_id=next_chunk_id,
                                        relation_type="PRECEDES",
                                        properties={
                                            "split_group": original_order
                                        }
                                    )

                    # Create section nodes and relationships if section information is available
                    section = chunk.get('section', '')
                    section_title = chunk.get('section_title', '')

                    if section and section_title:
                        section_id = f"{filing_id}_section_{section}"

                        if self.use_neo4j:
                            with self.driver.session(database=self.database) as session:
                                # Create section node
                                query = """
                                MERGE (s:Section {id: $section_id})
                                SET s.name = $section,
                                    s.title = $section_title,
                                    s.parent_filing = $filing_id
                                """
                                session.run(
                                    query,
                                    section_id=section_id,
                                    section=section,
                                    section_title=section_title,
                                    filing_id=filing_id
                                )

                                # Link filing to section
                                query = """
                                MATCH (f:Filing {id: $filing_id})
                                MATCH (s:Section {id: $section_id})
                                MERGE (f)-[r:HAS_SECTION]->(s)
                                """
                                session.run(
                                    query,
                                    filing_id=filing_id,
                                    section_id=section_id
                                )

                                # Link section to chunk
                                query = """
                                MATCH (s:Section {id: $section_id})
                                MATCH (c:Chunk {id: $chunk_id})
                                MERGE (s)-[r:CONTAINS_CHUNK]->(c)
                                """
                                session.run(
                                    query,
                                    section_id=section_id,
                                    chunk_id=chunk_id
                                )
                        else:
                            # Add section node
                            self.add_node(
                                node_id=section_id,
                                properties={
                                    "type": "section",
                                    "name": section,
                                    "title": section_title,
                                    "parent_filing": filing_id
                                }
                            )

                            # Link filing to section
                            self.add_relation(
                                source_id=filing_id,
                                target_id=section_id,
                                relation_type="HAS_SECTION",
                                properties={}
                            )

                            # Link section to chunk
                            self.add_relation(
                                source_id=section_id,
                                target_id=chunk_id,
                                relation_type="CONTAINS_CHUNK",
                                properties={}
                            )

            return True

        except Exception as e:
            logger.error(f"Error adding filing to graph store: {str(e)}")
            return False

    def _get_form_description(self, form_type: str) -> str:
        """Get the description of a form type."""
        form_descriptions = {
            "10-K": "Annual report containing comprehensive business and financial information",
            "10-Q": "Quarterly report of financial performance and business updates",
            "8-K": "Current report of material events or corporate changes",
            "20-F": "Annual report for foreign private issuers",
            "6-K": "Report of foreign private issuer",
            "4": "Statement of changes in beneficial ownership",
            "13F": "Quarterly report of investment holdings",
            "13D": "Statement of beneficial ownership",
            "13G": "Statement of beneficial ownership by passive investors"
        }
        return form_descriptions.get(form_type, "SEC filing")

    def _get_topic_info(self, topic_label: str, form_type: str) -> Dict[str, str]:
        """Get topic information based on edgartools filing structure."""
        # Default category and description
        category = "Other"
        description = ""

        # Map form types to their structure classes
        form_structures = {
            "10-K": {"ITEM_1": {"Title": "Business"}},
            "10-Q": {"ITEM_1": {"Title": "Financial Statements"}},
            "8-K": {"ITEM_1": {"Title": "Registrant's Business and Operations"}}
        }

        # Get structure if available
        if form_type in form_structures:
            structure = form_structures[form_type]
            topic_label = topic_label.upper()

            # Search through structure for matching item
            for part, items in structure.items():
                if topic_label in items:
                    item_info = items[topic_label]
                    category = item_info.get("Title", category)
                    description = item_info.get("Description", description)
                    break

        return {
            "category": category,
            "description": description
        }

    def get_filing_nodes(self, filing_id: str) -> List[Dict[str, Any]]:
        """Get all nodes related to a filing.

        Args:
            filing_id: The filing ID

        Returns:
            List of node dictionaries
        """
        if self.use_neo4j:
            try:
                with self.driver.session(database=self.database) as session:
                    query = """
                    MATCH (f:Filing {accession_number: $filing_id})
                    OPTIONAL MATCH (f)-[r]-(related)
                    RETURN f, r, related
                    """
                    result = session.run(query, filing_id=filing_id)

                    nodes = []
                    for record in result:
                        nodes.append({
                            "filing": dict(record["f"]),
                            "relationship": dict(record["r"]),
                            "related": dict(record["related"])
                        })
                    return nodes
            except Exception as e:
                logger.error(f"Error getting filing nodes from Neo4j: {str(e)}")
                return []
        else:
            # For in-memory graph
            try:
                nodes = []
                if filing_id in self.graph:
                    for neighbor in self.graph.neighbors(filing_id):
                        edge_data = self.graph.get_edge_data(filing_id, neighbor)
                        nodes.append({
                            "filing": self.graph.nodes[filing_id],
                            "relationship": edge_data,
                            "related": self.graph.nodes[neighbor]
                        })
                return nodes
            except Exception as e:
                logger.error(f"Error getting filing nodes from memory: {str(e)}")
                return []

    def get_filing_relationships(self, filing_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a filing.

        Args:
            filing_id: The filing ID

        Returns:
            List of relationship dictionaries
        """
        if self.use_neo4j:
            try:
                with self.driver.session(database=self.database) as session:
                    query = """
                    MATCH (f:Filing {accession_number: $filing_id})-[r]-(related)
                    RETURN type(r) as type, properties(r) as properties,
                           f.accession_number as from_id, related.accession_number as to_id
                    """
                    result = session.run(query, filing_id=filing_id)

                    relationships = []
                    for record in result:
                        relationships.append({
                            "type": record["type"],
                            "properties": record["properties"],
                            "from_id": record["from_id"],
                            "to_id": record["to_id"]
                        })
                    return relationships
            except Exception as e:
                logger.error(f"Error getting filing relationships from Neo4j: {str(e)}")
                return []
        else:
            # For in-memory graph
            try:
                relationships = []
                if filing_id in self.graph:
                    for neighbor in self.graph.neighbors(filing_id):
                        edge_data = self.graph.get_edge_data(filing_id, neighbor)
                        relationships.append({
                            "type": edge_data.get("type", ""),
                            "properties": {k: v for k, v in edge_data.items() if k != "type"},
                            "from_id": filing_id,
                            "to_id": neighbor
                        })
                return relationships
            except Exception as e:
                logger.error(f"Error getting filing relationships from memory: {str(e)}")
                return []

    def add_entity(
        self,
        entity_name: str,
        entity_type: str,
        context: str,
        chunk_id: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an entity to the graph and create relationships.

        Args:
            entity_name: Name of the entity
            entity_type: Type of entity (Company, Person, Location, etc.)
            context: Context in which the entity appears
            chunk_id: ID of the chunk containing the entity
            properties: Additional entity properties
        """
        if properties is None:
            properties = {}

        # Create entity node
        entity_node = {
            "name": entity_name,
            "type": entity_type,
            "context": context,
            **properties
        }

        # Add entity node
        self.add_node("Entity", entity_node)

        # Add relationship between chunk and entity
        self.add_relation(
            "Chunk",
            {"chunkId": chunk_id},
            "CONTAINS_ENTITY",
            "Entity",
            {"name": entity_name},
            properties={
                "context": context,
                "relevance": properties.get("relevance", 1.0)
            }
        )

        # Add relationship between entity and filing if available
        if "_chunk_" in chunk_id:
            filing_id = chunk_id.split("_chunk_")[0]
            self.add_relation(
                "Entity",
                {"name": entity_name},
                "MENTIONED_IN",
                "Filing",
                {"accession_number": filing_id},
                properties={
                    "context": context,
                    "chunk_id": chunk_id
                }
            )

    def add_entity_relationship(
        self,
        entity1_name: str,
        entity2_name: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relationship between two entities.

        Args:
            entity1_name: Name of first entity
            entity2_name: Name of second entity
            relationship_type: Type of relationship
            properties: Additional relationship properties
        """
        if properties is None:
            properties = {}

        self.add_relation(
            "Entity",
            {"name": entity1_name},
            relationship_type,
            "Entity",
            {"name": entity2_name},
            properties=properties
        )

    def update_chunk_similarity(
        self,
        chunk1_id: str,
        chunk2_id: str,
        similarity_score: float,
        relationship_type: str = "semantic"
    ) -> None:
        """
        Update the similarity score between two chunks.

        Args:
            chunk1_id: ID of first chunk
            chunk2_id: ID of second chunk
            similarity_score: Similarity score between chunks
            relationship_type: Type of similarity relationship
        """
        self.add_relation(
            "Chunk",
            {"chunkId": chunk1_id},
            "SIMILAR_TO",
            "Chunk",
            {"chunkId": chunk2_id},
            properties={
                "similarity_score": similarity_score,
                "relationship_type": relationship_type
            }
        )

    def get_related_chunks(
        self,
        chunk_id: str,
        min_similarity: float = 0.7,
        max_chunks: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get chunks related to a given chunk through similarity or topic.

        Args:
            chunk_id: ID of the chunk
            min_similarity: Minimum similarity score
            max_chunks: Maximum number of chunks to return

        Returns:
            List of related chunks with their relationships
        """
        if self.use_neo4j:
            try:
                with self.driver.session(database=self.database) as session:
                    query = """
                    MATCH (c:Chunk {chunkId: $chunk_id})
                    OPTIONAL MATCH (c)-[r:SIMILAR_TO]-(related:Chunk)
                    WHERE r.similarity_score >= $min_similarity
                    OPTIONAL MATCH (c)-[:BELONGS_TO]->(t:Topic)<-[:BELONGS_TO]-(topic_related:Chunk)
                    WHERE topic_related.chunkId <> $chunk_id
                    RETURN DISTINCT related, r.similarity_score as score, 'similarity' as type
                    UNION
                    SELECT DISTINCT topic_related, 1.0 as score, 'topic' as type
                    ORDER BY score DESC
                    LIMIT $max_chunks
                    """

                    result = session.run(
                        query,
                        chunk_id=chunk_id,
                        min_similarity=min_similarity,
                        max_chunks=max_chunks
                    )

                    return [
                        {
                            "chunk": dict(record["related"]),
                            "score": record["score"],
                            "type": record["type"]
                        }
                        for record in result
                    ]
            except Exception as e:
                logger.error(f"Error getting related chunks from Neo4j: {str(e)}")
                return []
        else:
            # For in-memory graph
            try:
                related_chunks = []

                # Get similar chunks
                for neighbor in self.graph.neighbors(chunk_id):
                    edge_data = self.graph.get_edge_data(chunk_id, neighbor)
                    if (
                        edge_data.get("type") == "SIMILAR_TO"
                        and edge_data.get("similarity_score", 0) >= min_similarity
                    ):
                        related_chunks.append({
                            "chunk": self.graph.nodes[neighbor],
                            "score": edge_data["similarity_score"],
                            "type": "similarity"
                        })

                # Sort by score and limit
                related_chunks.sort(key=lambda x: x["score"], reverse=True)
                return related_chunks[:max_chunks]
            except Exception as e:
                logger.error(f"Error getting related chunks from memory: {str(e)}")
                return []

    def add_related_filing(
        self,
        filing_id: str,
        related_filing_id: str,
        relationship_type: str = "RELATED_TO",
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relationship between related filings.

        Args:
            filing_id: ID of the source filing
            related_filing_id: ID of the related filing
            relationship_type: Type of relationship
            properties: Additional relationship properties
        """
        if properties is None:
            properties = {}

        self.add_relation(
            "Filing",
            {"accession_number": filing_id},
            relationship_type,
            "Filing",
            {"accession_number": related_filing_id},
            properties=properties
        )

    def add_xbrl_fact(
        self,
        filing_id: str,
        element_id: str,
        context_id: str,
        value: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an XBRL fact to the graph.

        Args:
            filing_id: ID of the filing containing the fact
            element_id: XBRL element ID
            context_id: XBRL context ID
            value: Fact value
            properties: Additional fact properties
        """
        if properties is None:
            properties = {}

        # Create fact node
        fact_id = f"{filing_id}_{element_id}_{context_id}"
        fact_node = {
            "factId": fact_id,
            "elementId": element_id,
            "contextId": context_id,
            "value": value,
            **properties
        }

        # Add fact node
        self.add_node("XbrlFact", fact_node)

        # Add relationship to filing
        self.add_relation(
            "Filing",
            {"accession_number": filing_id},
            "CONTAINS_FACT",
            "XbrlFact",
            {"factId": fact_id}
        )

        # Add relationship to element if available
        if "element" in properties:
            self.add_relation(
                "XbrlFact",
                {"factId": fact_id},
                "USES_ELEMENT",
                "XbrlElement",
                {"elementId": element_id}
            )

        # Add relationship to context if available
        if "context" in properties:
            self.add_relation(
                "XbrlFact",
                {"factId": fact_id},
                "HAS_CONTEXT",
                "XbrlContext",
                {"contextId": context_id}
            )

    def add_xbrl_element(
        self,
        element_id: str,
        name: str,
        data_type: str,
        period_type: str,
        balance: Optional[str] = None,
        abstract: bool = False,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add an XBRL element to the graph.

        Args:
            element_id: XBRL element ID
            name: Element name
            data_type: Element data type
            period_type: Element period type
            balance: Element balance (debit/credit)
            abstract: Whether element is abstract
            labels: Element labels
        """
        if labels is None:
            labels = {}

        # Create element node
        element_node = {
            "elementId": element_id,
            "name": name,
            "dataType": data_type,
            "periodType": period_type,
            "balance": balance,
            "abstract": abstract,
            "labels": labels
        }

        # Add element node
        self.add_node("XbrlElement", element_node)

    def add_xbrl_context(
        self,
        context_id: str,
        entity: Dict[str, str],
        period: Dict[str, Any],
        dimensions: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add an XBRL context to the graph.

        Args:
            context_id: XBRL context ID
            entity: Entity information
            period: Period information
            dimensions: Context dimensions
        """
        if dimensions is None:
            dimensions = {}

        # Create context node
        context_node = {
            "contextId": context_id,
            "entity": entity,
            "period": period,
            "dimensions": dimensions
        }

        # Add context node
        self.add_node("XbrlContext", context_node)

        # Add relationship to entity if available
        if "identifier" in entity:
            self.add_relation(
                "XbrlContext",
                {"contextId": context_id},
                "FOR_ENTITY",
                "Company",
                {"cik": entity["identifier"]}
            )

    def get_related_filings(
        self,
        filing_id: str,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get filings related to a given filing.

        Args:
            filing_id: ID of the filing
            relationship_type: Optional relationship type filter

        Returns:
            List of related filings with their relationships
        """
        if self.use_neo4j:
            try:
                with self.driver.session(database=self.database) as session:
                    # Build relationship filter
                    rel_filter = f"[r:{relationship_type}]" if relationship_type else "[r]"

                    query = f"""
                    MATCH (f:Filing {{accession_number: $filing_id}})-{rel_filter}-(related:Filing)
                    RETURN related, type(r) as type, properties(r) as properties
                    ORDER BY related.filing_date DESC
                    """

                    result = session.run(query, filing_id=filing_id)

                    return [
                        {
                            "filing": dict(record["related"]),
                            "type": record["type"],
                            "properties": record["properties"]
                        }
                        for record in result
                    ]
            except Exception as e:
                logger.error(f"Error getting related filings from Neo4j: {str(e)}")
                return []
        else:
            # For in-memory graph
            try:
                related_filings = []

                # Get related filings
                for neighbor in self.graph.neighbors(filing_id):
                    edge_data = self.graph.get_edge_data(filing_id, neighbor)

                    # Apply relationship type filter if specified
                    if relationship_type and edge_data.get("type") != relationship_type:
                        continue

                    related_filings.append({
                        "filing": self.graph.nodes[neighbor],
                        "type": edge_data.get("type", ""),
                        "properties": {k: v for k, v in edge_data.items() if k != "type"}
                    })

                # Sort by filing date
                related_filings.sort(
                    key=lambda x: x["filing"].get("filing_date", ""),
                    reverse=True
                )

                return related_filings
            except Exception as e:
                logger.error(f"Error getting related filings from memory: {str(e)}")
                return []