"""
SEC Filing Processor

This module provides functionality for processing SEC filings.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..storage import GraphStore, LlamaIndexVectorStore
from .file_storage import FileStorage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FilingProcessor:
    """
    Processor for SEC filings.
    """
    
    def __init__(
        self,
        graph_store: Optional[GraphStore] = None,
        vector_store: Optional[LlamaIndexVectorStore] = None,
        file_storage: Optional[FileStorage] = None,
    ):
        """Initialize the filing processor."""
        self.graph_store = graph_store or GraphStore()
        self.vector_store = vector_store or LlamaIndexVectorStore()
        self.file_storage = file_storage or FileStorage()
    
    def process_filing(self, filing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a filing and store it in the graph and vector stores.
        
        Args:
            filing_data: Dictionary containing filing data
            
        Returns:
            Dict containing processed filing data
        """
        filing_id = filing_data["id"]
        text = filing_data["text"]
        embedding = filing_data["embedding"]
        metadata = filing_data["metadata"]
        
        # Check if filing is already processed
        cached_data = self.file_storage.load_cached_filing(filing_id)
        if cached_data:
            logger.info(f"Using cached data for filing {filing_id}")
            return cached_data["processed_data"]
        
        # Process filing
        try:
            # Add to graph store with all metadata
            self.graph_store.add_filing(
                filing_id=filing_id,
                text=text,
                metadata=metadata
            )
            
            # Add to vector store with all metadata
            self.vector_store.upsert_vectors(
                vectors=[(filing_id, embedding)],
                metadata=[metadata]
            )
            
            # Create processed data with all metadata
            processed_data = {
                "filing_id": filing_id,
                "text": text,
                "embedding": embedding,
                "metadata": metadata,
                "graph_nodes": self.graph_store.get_filing_nodes(filing_id),
                "graph_relationships": self.graph_store.get_filing_relationships(filing_id)
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing filing {filing_id}: {e}")
            raise
    
    def get_filing(self, filing_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a filing by ID.
        
        Args:
            filing_id: The filing ID
            
        Returns:
            Dict containing filing data if found, None otherwise
        """
        # Try to get from cache first
        cached_data = self.file_storage.load_cached_filing(filing_id)
        if cached_data:
            return cached_data
        
        # Try to get from processed files
        processed_data = self.file_storage.load_processed_filing(filing_id)
        if processed_data:
            return processed_data
        
        # Try to get from raw files
        raw_data = self.file_storage.load_raw_filing(filing_id)
        if raw_data:
            return raw_data
        
        return None
    
    def list_filings(
        self,
        ticker: Optional[str] = None,
        year: Optional[str] = None,
        filing_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available filings.
        
        Args:
            ticker: Filter by company ticker
            year: Filter by filing year
            filing_type: Filter by filing type
            
        Returns:
            List of filing metadata
        """
        return self.file_storage.list_filings(
            ticker=ticker,
            year=year,
            filing_type=filing_type
        ) 