"""
SEC Filing Chunking Module

This module provides functionality for chunking documents into semantically meaningful segments
while respecting token limits for embedding models.

This is the original implementation of the DocumentChunker class, which has been replaced by
a wrapper around the enhanced FilingChunker class. This file is kept for reference only.
"""

import logging
from typing import Dict, List, Any
import tiktoken

from llama_index.core.text_splitter import TokenTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    A class for chunking documents into smaller pieces for embedding.
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        """
        Initialize the DocumentChunker.

        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding

        # Initialize token splitter
        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces for embedding.

        Args:
            text: The document text

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []

        # Split the document into chunks
        chunk_texts = self.token_splitter.split_text(text)

        # Create chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk = {
                "text": chunk_text,
                "metadata": {
                    "chunk_index": i,
                    "total_chunks": len(chunk_texts),
                    "token_count": len(self.tokenizer.encode(chunk_text))
                }
            }
            chunks.append(chunk)

        return chunks
