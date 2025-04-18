"""
SEC Filing Chunking Module

This module provides a backward-compatible wrapper for the FilingChunker class.
It maintains the same interface as the original DocumentChunker class but uses
the enhanced FilingChunker internally.
"""

import logging
from typing import Dict, List, Any

from ...data_processing.chunking import FilingChunker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    A backward-compatible wrapper for the FilingChunker class.
    This class maintains the same interface as the original DocumentChunker
    but uses the enhanced FilingChunker internally.

    Note: For new code, it's recommended to use FilingChunker directly.
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        """
        Initialize the DocumentChunker.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
        """
        # Use FilingChunker internally
        self.chunker = FilingChunker(max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info("DocumentChunker is now a wrapper around FilingChunker. Consider using FilingChunker directly for new code.")

    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces for embedding.

        Args:
            text: The document text

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Delegate to FilingChunker
        return self.chunker.chunk_document(text)
