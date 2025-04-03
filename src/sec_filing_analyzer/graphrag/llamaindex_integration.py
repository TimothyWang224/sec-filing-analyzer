"""
LlamaIndex Integration for GraphRAG

This module provides integration with LlamaIndex for enhanced GraphRAG capabilities.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import time
import os
from pathlib import Path
from collections import defaultdict
import re

import networkx as nx
import numpy as np

# LlamaIndex core imports
from llama_index.core import Document, ServiceContext
from llama_index.core.llms import LLM
from llama_index.core.indices import PropertyGraphIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever, KGRetriever, HybridRetriever
from llama_index.core.response_synthesizers import ResponseSynthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.schema import MetadataMode, RelatedNodeInfo, NodeRelationship, TextNode

# Neo4j graph store
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# Import our SEC document processing utilities
from .sec_structure import SECStructure
from .sec_entities import SECEntities
from .store import GraphStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaIndexIntegration:
    """
    Integration class for LlamaIndex that handles document processing and querying.
    """
    
    def __init__(
        self,
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_url: str = "bolt://localhost:7687",
        cache_dir: Optional[str] = None,
        llm: Optional[LLM] = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        """Initialize the LlamaIndex integration."""
        self.llm = llm
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        
        # Initialize graph store
        self.graph_store = GraphStore(
            username=neo4j_username,
            password=neo4j_password,
            url=neo4j_url
        )
        
        # Initialize document processors
        self.sec_structure = SECStructure()
        self.sec_entities = SECEntities()
        
        # Initialize service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=embedding_model
        )
        
        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=20
        )
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> PropertyGraphIndex:
        """
        Process SEC documents and create a property graph index.
        
        Args:
            documents: List of SEC documents to process
            
        Returns:
            PropertyGraphIndex: The created index
        """
        # Create index
        index = PropertyGraphIndex.from_documents(
            documents,
            service_context=self.service_context,
            graph_store=self.graph_store,
            node_parser=self.node_parser
        )
        
        # Extract structure and entities
        for doc in documents:
            # Extract structure
            structure = self.sec_structure.parse_filing_structure(doc["text"])
            self.sec_structure.extract_sections(doc["text"])
            
            # Extract entities
            entities = self.sec_entities.extract_entities(doc["text"])
            relationships = self.sec_entities.identify_relationships(entities)
            
            # Add to graph store
            for entity in entities:
                self.graph_store.add_node(
                    entity["id"],
                    properties=entity
                )
            
            for rel in relationships:
                self.graph_store.add_relationship(
                    rel["from_node"],
                    rel["to_node"],
                    rel["type"],
                    properties=rel.get("properties", {})
                )
        
        return index
    
    def query(self, query_text: str) -> str:
        """
        Query the document index.
        
        Args:
            query_text: The query text
            
        Returns:
            str: The response
        """
        # Create query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=VectorIndexRetriever(
                index=self.index,
                similarity_top_k=5
            ),
            response_synthesizer=ResponseSynthesizer.from_args(
                service_context=self.service_context
            )
        )
        
        # Execute query
        response = query_engine.query(query_text)
        return str(response) 