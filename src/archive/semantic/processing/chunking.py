"""
SEC Filing Chunking Module

This module provides functionality for chunking SEC filings into semantically meaningful segments
while respecting token limits for embedding models.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tiktoken
from bs4 import BeautifulSoup
from edgar.files.htmltools import ChunkedDocument
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilingChunker:
    """
    Handles chunking of SEC filings into semantically meaningful segments.
    Primarily uses edgartools' understanding of filing structure, with token-based
    splitting only as a fallback for chunks that exceed the embedding model's context window.
    """


# Add a DocumentChunker class that's more focused on chunking documents for embedding
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
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
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
                    "token_count": len(self.tokenizer.encode(chunk_text)),
                },
            }
            chunks.append(chunk)

        return chunks

    def __init__(self, max_chunk_size: int = 1500):
        """
        Initialize the FilingChunker.

        Args:
            max_chunk_size: Maximum number of tokens per chunk (default: 1500)
                            Smaller chunks provide better retrieval precision
        """
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding

        # Initialize token splitter for handling large chunks
        self.token_splitter = TokenTextSplitter(
            chunk_size=max_chunk_size, chunk_overlap=200
        )

        # Initialize embedding model
        self.embedding_model = OpenAIEmbedding()

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def _split_large_chunk(
        self, chunk_text: str, chunk_meta: Dict[str, Any]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Split a chunk that's too large for the embedding model.

        Args:
            chunk_text: The text of the chunk
            chunk_meta: Metadata about the chunk

        Returns:
            Tuple containing:
            - List of sub-chunk texts
            - List of sub-chunk metadata dictionaries
        """
        # Split the chunk into smaller pieces
        sub_chunks = self.token_splitter.split_text(chunk_text)

        # Create metadata for each sub-chunk
        sub_chunk_metadata = []
        for i, sub_chunk in enumerate(sub_chunks):
            sub_chunk_meta = chunk_meta.copy()
            sub_chunk_meta.update(
                {
                    "is_sub_chunk": True,
                    "sub_chunk_index": i,
                    "parent_chunk_text": chunk_text,
                }
            )
            sub_chunk_metadata.append(sub_chunk_meta)

        return sub_chunks, sub_chunk_metadata

    def _preprocess_html(self, html_content: str) -> str:
        """
        Preprocess HTML content to handle potential issues before chunking.
        """
        try:
            # Parse HTML and clean it up
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text()

            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = " ".join(chunk for chunk in chunks if chunk)

            return text
        except Exception as e:
            logger.warning(f"Error preprocessing HTML: {e}")
            return html_content

    def chunk_html_filing(self, html_content: str) -> Dict[str, Any]:
        """
        Chunk an HTML filing using EDGAR's ChunkedDocument class.
        Returns a dictionary containing chunk statistics and metadata.
        """
        try:
            logger.info("Creating ChunkedDocument from HTML content")
            chunked_doc = ChunkedDocument(html_content)

            logger.info("Converting to DataFrame")
            df = chunked_doc.as_dataframe()

            # Convert DataFrame values to basic Python types
            def convert_value(val):
                if pd.isna(val) or val is None:
                    return ""
                if isinstance(val, (bool, np.bool_)):
                    return bool(val)
                if isinstance(val, (int, np.integer)):
                    return int(val)
                if isinstance(val, (float, np.floating)):
                    return float(val)
                if isinstance(val, dict):
                    logger.warning(f"Found dictionary value: {val}")
                    return str(val)  # Convert dictionaries to strings
                return str(val).strip()

            # Process items safely
            logger.info("Processing items from DataFrame")
            items = set()
            for idx, item in enumerate(df["Item"].dropna()):
                try:
                    logger.debug(f"Processing item at index {idx}: {item}")
                    item_str = convert_value(item)
                    if item_str:  # Only add non-empty items
                        items.add(item_str)
                except Exception as e:
                    logger.warning(
                        f"Could not convert item at index {idx} to string: {e}"
                    )
                    continue

            # Calculate statistics
            logger.info("Calculating chunk statistics")
            stats = {
                "total_chunks": len(chunked_doc),
                "avg_chunk_size": chunked_doc.average_chunk_size(),
                "items": sorted(
                    list(items)
                ),  # Convert set to sorted list for consistency
                "tables": sum(1 for _ in chunked_doc.tables()),
            }

            # Process chunks and create metadata
            logger.info("Processing chunks and creating metadata")
            chunks = []
            for i, row in df.iterrows():
                try:
                    chunk_text = convert_value(row.get("Text", ""))
                    chunk_meta = {
                        "text": chunk_text,
                        "item": convert_value(row.get("Item", "")),
                        "is_table": bool(row.get("Table", False)),
                        "chars": int(row.get("Chars", 0)),
                        "is_signature": bool(row.get("Signature", False)),
                        "is_toc": bool(
                            row.get("TocLink", False) or row.get("Toc", False)
                        ),
                        "is_empty": bool(row.get("Empty", True)),
                        "order": i,
                    }

                    # Check if chunk exceeds token limit
                    token_count = self._count_tokens(chunk_text)
                    if token_count > self.max_chunk_size:
                        logger.info(
                            f"Chunk {i} exceeds token limit ({token_count} > {self.max_chunk_size}). Splitting into sub-chunks."
                        )
                        sub_chunks, sub_chunk_metadata = self._split_large_chunk(
                            chunk_text, chunk_meta
                        )

                        # Add sub-chunks to the chunks list
                        for j, (sub_chunk_text, sub_chunk_meta) in enumerate(
                            zip(sub_chunks, sub_chunk_metadata)
                        ):
                            # Create a unique order for each sub-chunk
                            sub_chunk_meta["order"] = f"{i}.{j}"
                            sub_chunk_meta["original_order"] = (
                                i  # Keep track of the original chunk order
                            )
                            sub_chunk_meta["token_count"] = self._count_tokens(
                                sub_chunk_text
                            )
                            chunks.append(sub_chunk_meta)
                    else:
                        # Add the original chunk
                        chunk_meta["token_count"] = token_count
                        chunks.append(chunk_meta)
                except Exception as e:
                    logger.warning(f"Error processing chunk {i}: {e}")
                    continue

            stats["chunks"] = chunks
            return stats

        except Exception as e:
            logger.error(f"Error in chunk_html_filing: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fall back to simple text chunking
            return self._fallback_text_chunking(html_content)

    def _fallback_text_chunking(self, content: str) -> Dict[str, Any]:
        """
        Fallback method for simple text-based chunking when HTML processing fails.
        """
        logger.info("Using fallback text chunking method")
        try:
            # Use token splitter to chunk the content
            chunks = self.token_splitter.split_text(content)

            # Create simple metadata for chunks
            chunk_data = []
            for i, chunk in enumerate(chunks):
                # Check if chunk exceeds token limit
                token_count = self._count_tokens(chunk)
                if token_count > self.max_chunk_size:
                    logger.info(
                        f"Fallback chunk {i} exceeds token limit ({token_count} > {self.max_chunk_size}). Splitting into sub-chunks."
                    )
                    # Create base metadata
                    base_meta = {
                        "text": chunk,
                        "item": "",  # No item information in fallback mode
                        "is_table": False,
                        "chars": len(chunk),
                        "is_signature": False,
                        "is_toc": False,
                        "is_empty": len(chunk.strip()) == 0,
                        "order": i,
                    }

                    # Split the chunk
                    sub_chunks, sub_chunk_metadata = self._split_large_chunk(
                        chunk, base_meta
                    )

                    # Add sub-chunks to the chunk data
                    for j, (sub_chunk_text, sub_chunk_meta) in enumerate(
                        zip(sub_chunks, sub_chunk_metadata)
                    ):
                        # Create a unique order for each sub-chunk
                        sub_chunk_meta["order"] = f"{i}.{j}"
                        sub_chunk_meta["original_order"] = (
                            i  # Keep track of the original chunk order
                        )
                        sub_chunk_meta["token_count"] = self._count_tokens(
                            sub_chunk_text
                        )
                        chunk_data.append(sub_chunk_meta)
                else:
                    # Add the original chunk
                    chunk_meta = {
                        "text": chunk,
                        "item": "",  # No item information in fallback mode
                        "is_table": False,
                        "chars": len(chunk),
                        "is_signature": False,
                        "is_toc": False,
                        "is_empty": len(chunk.strip()) == 0,
                        "order": i,
                        "token_count": token_count,
                    }
                    chunk_data.append(chunk_meta)

            # Return stats in the same format as HTML chunking
            return {
                "total_chunks": len(chunk_data),
                "avg_chunk_size": sum(len(c["text"]) for c in chunk_data)
                / len(chunk_data)
                if chunk_data
                else 0,
                "items": [],  # No item information in fallback mode
                "tables": 0,  # No table detection in fallback mode
                "chunks": chunk_data,
            }

        except Exception as e:
            logger.error(f"Error in fallback text chunking: {str(e)}")
            raise

    def chunk_full_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text to split into chunks

        Returns:
            List of text chunks
        """
        try:
            if not text:
                return []

            # Split into paragraphs first
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            chunks = []
            current_chunk = []
            current_token_count = 0

            for paragraph in paragraphs:
                paragraph_tokens = self._count_tokens(paragraph)

                # If a single paragraph is too large, split it into sentences
                if paragraph_tokens > self.max_chunk_size:
                    sentences = [s.strip() for s in paragraph.split(". ") if s.strip()]
                    for sentence in sentences:
                        sentence_tokens = self._count_tokens(sentence)
                        if current_token_count + sentence_tokens > self.max_chunk_size:
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                                current_chunk = []
                                current_token_count = 0
                        current_chunk.append(sentence)
                        current_token_count += sentence_tokens
                else:
                    if current_token_count + paragraph_tokens > self.max_chunk_size:
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                            current_chunk = []
                            current_token_count = 0
                    current_chunk.append(paragraph)
                    current_token_count += paragraph_tokens

            # Add the last chunk if there is one
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            return [text]  # Return original text as single chunk on error

    def process_filing(
        self, filing_data: Dict[str, Any], filing_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a filing and return chunked data.

        Args:
            filing_data: Metadata about the filing
            filing_content: The content of the filing

        Returns:
            Dictionary with processed filing data including chunks
        """
        # Initialize result dictionary
        text = filing_content.get("content", "")
        result = {
            "id": filing_data["accession_number"],
            "text": text,
            "metadata": filing_data,
            "chunk_texts": [],
            "chunk_metadata": [],
            "chunks": None,
        }

        try:
            # Process HTML content if available
            if filing_data.get("has_html"):
                html_content = filing_content.get("html_content")
                if html_content:
                    logger.info(
                        f"Processing HTML content for filing {filing_data['accession_number']}"
                    )

                    # Chunk HTML content
                    chunk_stats = self.chunk_html_filing(html_content)

                    # Get chunk texts
                    chunk_texts = [
                        chunk["text"] for chunk in chunk_stats.get("chunks", [])
                    ]

                    # Update result
                    result["chunk_metadata"] = chunk_stats
                    result["chunk_texts"] = chunk_texts
                    result["chunks"] = chunk_stats.get("chunks", [])
                    result["chunk_stats"] = chunk_stats

                    # Add relationship information for split chunks
                    self._add_chunk_relationships(result["chunks"])

            # Fallback to full text if no HTML or if HTML processing failed
            if not result["chunk_texts"]:
                logger.info(
                    f"Using full text for filing {filing_data['accession_number']}"
                )
                text = filing_content.get("content", "")
                if text:
                    chunk_texts = self.chunk_full_text(text)

                    # Convert chunk texts to the format expected by the graph store
                    chunks = []
                    for i, text in enumerate(chunk_texts):
                        # Check if chunk exceeds token limit
                        token_count = self._count_tokens(text)
                        if token_count > self.max_chunk_size:
                            logger.info(
                                f"Full text chunk {i} exceeds token limit ({token_count} > {self.max_chunk_size}). Splitting into sub-chunks."
                            )
                            # Create base metadata
                            base_meta = {
                                "text": text,
                                "item": "",  # No item information in fallback mode
                                "is_table": False,
                                "chars": len(text),
                                "is_signature": False,
                                "is_toc": False,
                                "is_empty": len(text.strip()) == 0,
                                "order": i,
                            }

                            # Split the chunk
                            sub_chunks, sub_chunk_metadata = self._split_large_chunk(
                                text, base_meta
                            )

                            # Add sub-chunks to the chunks list
                            for j, (sub_chunk_text, sub_chunk_meta) in enumerate(
                                zip(sub_chunks, sub_chunk_metadata)
                            ):
                                # Create a unique order for each sub-chunk
                                sub_chunk_meta["order"] = f"{i}.{j}"
                                sub_chunk_meta["original_order"] = (
                                    i  # Keep track of the original chunk order
                                )
                                sub_chunk_meta["token_count"] = self._count_tokens(
                                    sub_chunk_text
                                )
                                chunks.append(sub_chunk_meta)
                        else:
                            # Add the original chunk
                            chunk = {
                                "text": text,
                                "item": "",  # No item information in fallback mode
                                "is_table": False,
                                "chars": len(text),
                                "is_signature": False,
                                "is_toc": False,
                                "is_empty": len(text.strip()) == 0,
                                "order": i,
                                "token_count": token_count,
                            }
                            chunks.append(chunk)

                    # Add relationship information for split chunks
                    self._add_chunk_relationships(chunks)

                    result["chunk_metadata"] = chunks
                    result["chunk_texts"] = [chunk["text"] for chunk in chunks]
                    result["chunks"] = chunks
                else:
                    logger.warning(
                        f"No content available for filing {filing_data['accession_number']}"
                    )

        except Exception as e:
            logger.error(
                f"Error processing filing {filing_data['accession_number']}: {str(e)}"
            )
            # Ensure we have at least some basic metadata even if processing fails
            result["error"] = str(e)

        return result

    def _add_chunk_relationships(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add relationship information for split chunks.

        Args:
            chunks: List of chunk metadata dictionaries
        """
        # Group chunks by original order
        original_chunks = {}
        for chunk in chunks:
            original_order = chunk.get("original_order")
            if original_order is not None:
                if original_order not in original_chunks:
                    original_chunks[original_order] = []
                original_chunks[original_order].append(chunk)

        # Add relationship information for each group of split chunks
        for original_order, split_chunks in original_chunks.items():
            if len(split_chunks) > 1:
                # Sort split chunks by sub-chunk index
                split_chunks.sort(key=lambda c: c.get("sub_chunk_index", 0))

                # Add relationship information
                for i, chunk in enumerate(split_chunks):
                    chunk["split_chunk_count"] = len(split_chunks)
                    chunk["split_chunk_index"] = i
                    chunk["is_split_chunk"] = True

                    # Add reference to previous and next chunks in the sequence
                    if i > 0:
                        chunk["prev_chunk_order"] = split_chunks[i - 1]["order"]
                    if i < len(split_chunks) - 1:
                        chunk["next_chunk_order"] = split_chunks[i + 1]["order"]
