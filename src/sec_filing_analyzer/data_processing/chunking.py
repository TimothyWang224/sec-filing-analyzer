"""
SEC Filing Chunking Module

This module provides functionality for chunking SEC filings into semantically meaningful segments
while respecting token limits for embedding models.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

from edgar.files.htmltools import ChunkedDocument, chunks2df, detect_decimal_items, adjust_for_empty_items
from llama_index.text_splitter import TokenTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FilingChunker:
    """
    Handles chunking of SEC filings into semantically meaningful segments.
    Leverages edgartools' understanding of filing structure while respecting token limits.
    """
    
    def __init__(self, max_tokens: int = 4000, chunk_overlap: int = 200):
        """
        Initialize the FilingChunker.
        
        Args:
            max_tokens: Maximum tokens per chunk for embedding models
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.text_splitter = TokenTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_html_filing(self, html_content: str) -> Tuple[List[Dict[str, Any]], List[str], List[List[float]], Dict[str, Any]]:
        """
        Chunk an HTML filing using edgartools' semantic understanding.
        
        Args:
            html_content: The HTML content of the filing
            
        Returns:
            Tuple containing:
            - List of chunk metadata dictionaries
            - List of chunk texts
            - List of chunk embeddings
            - Dictionary with overall chunk statistics
        """
        # Create chunked document using edgartools
        chunked_doc = ChunkedDocument(html_content)
        
        # Get chunks as a dataframe
        chunks_df = chunked_doc.as_dataframe()
        
        # Process chunks
        chunk_embeddings = []
        chunk_texts = []
        chunk_metadata = []
        
        for _, chunk in chunks_df.iterrows():
            chunk_text = chunk['text']
            chunk_meta = {
                'item': chunk.get('Item', ''),
                'part': chunk.get('Part', ''),
                'is_table': chunk.get('Table', False),
                'chars': chunk.get('Chars', 0)
            }
            
            # Only split if the chunk is too large for the embedding model
            if len(chunk_text.split()) > self.text_splitter.chunk_size:
                sub_chunks = self.text_splitter.split_text(chunk_text)
                for sub_chunk in sub_chunks:
                    chunk_texts.append(sub_chunk)
                    # Add parent chunk metadata to sub-chunks
                    sub_chunk_meta = chunk_meta.copy()
                    sub_chunk_meta['is_sub_chunk'] = True
                    sub_chunk_meta['parent_chunk_text'] = chunk_text
                    chunk_metadata.append(sub_chunk_meta)
            else:
                chunk_texts.append(chunk_text)
                chunk_metadata.append(chunk_meta)
        
        # Generate embeddings for all chunks
        for chunk_text in chunk_texts:
            embedding = self.embedding_model.get_text_embedding(chunk_text)
            chunk_embeddings.append(embedding)
        
        # Compile chunk statistics
        chunk_stats = {
            "count": len(chunks_df),
            "items": chunked_doc.list_items(),
            "avg_size": chunked_doc.average_chunk_size(),
            "structure": {
                "parts": list(chunks_df['Part'].dropna().unique()),
                "items": list(chunks_df['Item'].dropna().unique())
            }
        }
        
        return chunk_metadata, chunk_texts, chunk_embeddings, chunk_stats
    
    def chunk_full_text(self, text: str) -> Tuple[List[Dict[str, Any]], List[str], List[List[float]]]:
        """
        Chunk full text content when HTML structure is not available.
        
        Args:
            text: The full text content of the filing
            
        Returns:
            Tuple containing:
            - List of chunk metadata dictionaries
            - List of chunk texts
            - List of chunk embeddings
        """
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text)
        
        # Generate embeddings for chunks
        chunk_embeddings = []
        for chunk in text_chunks:
            embedding = self.embedding_model.get_text_embedding(chunk)
            chunk_embeddings.append(embedding)
        
        # Create simple metadata for each chunk
        chunk_metadata = [{"source": "full_text", "chunk_index": i} for i in range(len(text_chunks))]
        
        return chunk_metadata, text_chunks, chunk_embeddings
    
    def process_filing(self, filing_data: Dict[str, Any], filing_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a filing and return chunked data with embeddings.
        
        Args:
            filing_data: Metadata about the filing
            filing_content: The content of the filing
            
        Returns:
            Dictionary with processed filing data including chunks and embeddings
        """
        # Initialize result dictionary
        result = {
            "id": filing_data['accession_number'],
            "text": filing_content['content'],
            "metadata": filing_data,
            "embeddings": [],
            "chunk_texts": [],
            "chunk_metadata": []
        }
        
        # Process HTML content if available
        if filing_data.get('has_html'):
            try:
                html_content = filing_content.get('html')
                if html_content:
                    logger.info(f"Processing HTML content for filing {filing_data['accession_number']}")
                    
                    # Chunk HTML content
                    chunk_metadata, chunk_texts, chunk_embeddings, chunk_stats = self.chunk_html_filing(html_content)
                    
                    # Update result
                    result["chunk_metadata"] = chunk_metadata
                    result["chunk_texts"] = chunk_texts
                    result["embeddings"] = chunk_embeddings
                    result["chunks"] = chunk_stats
                    
            except Exception as e:
                logger.warning(f"Could not process HTML content for filing {filing_data['accession_number']}: {e}")
        
        # If no HTML content or HTML processing failed, use full text
        if not result["chunk_texts"]:
            logger.info(f"Using full text for filing {filing_data['accession_number']}")
            
            # Chunk full text
            chunk_metadata, chunk_texts, chunk_embeddings = self.chunk_full_text(filing_content['content'])
            
            # Update result
            result["chunk_metadata"] = chunk_metadata
            result["chunk_texts"] = chunk_texts
            result["embeddings"] = chunk_embeddings
        
        return result 