"""
Robust Embeddings Module

This module provides a more robust implementation for generating vector embeddings
using OpenAI's API through LlamaIndex, with better error handling and token limit management.
"""

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tiktoken
from llama_index.embeddings.openai import OpenAIEmbedding

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustEmbeddingGenerator:
    """
    A robust implementation for generating vector embeddings using OpenAI's API.

    Features:
    - Proper handling of empty strings and None values
    - Automatic chunking of texts that exceed token limits
    - Adaptive rate limiting to avoid 429 errors
    - Comprehensive error handling with detailed logging
    - Fallback mechanisms for failed embeddings
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        max_tokens_per_chunk: int = 8000,  # Safe limit below the 8192 max
        rate_limit: float = 0.5,  # Seconds between API calls
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
        batch_size: int = 20,
        filing_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the robust embedding generator.

        Args:
            model: OpenAI embedding model to use
            max_tokens_per_chunk: Maximum tokens per chunk
            rate_limit: Minimum time between API requests in seconds
            max_retries: Maximum number of retries for failed API calls
            retry_base_delay: Base delay for exponential backoff
            batch_size: Number of texts to process in each batch
            filing_metadata: Optional metadata about the filing being processed
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")

        self.embed_model = OpenAIEmbedding(model=model, api_key=api_key)
        self.dimensions = 1536  # text-embedding-3-small has 1536 dimensions
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.batch_size = batch_size
        self.filing_metadata = filing_metadata or {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding

        # Stats tracking
        self.token_usage = {
            "total_tokens": 0,
            "requests": 0,
            "failed_requests": 0,
            "retried_requests": 0,
            "chunked_texts": 0,
        }

        logger.info(f"Initialized Robust OpenAI embedding generator with model: {model}")

    def _ensure_list_format(self, embedding: Union[np.ndarray, List[float], Any]) -> List[float]:
        """Ensure embedding is in list format.

        Args:
            embedding: The embedding to convert

        Returns:
            List of floats representing the embedding
        """
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        elif isinstance(embedding, list):
            return embedding
        else:
            try:
                return list(embedding)
            except:
                logger.warning(f"Could not convert embedding to list, returning zeros")
                return [0.0] * self.dimensions

    def _apply_rate_limit(self):
        """Apply rate limiting to avoid 429 errors."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks that don't exceed token limits.

        Args:
            text: Text to split into chunks

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Count tokens
        tokens = self.tokenizer.encode(text)

        # If under limit, return as is
        if len(tokens) <= self.max_tokens_per_chunk:
            return [text]

        # Split into chunks
        chunks = []
        for i in range(0, len(tokens), self.max_tokens_per_chunk):
            chunk_tokens = tokens[i : i + self.max_tokens_per_chunk]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

        self.token_usage["chunked_texts"] += 1
        logger.info(f"Split text with {len(tokens)} tokens into {len(chunks)} chunks")
        return chunks

    def _process_batch(self, batch: List[str]) -> Tuple[List[List[float]], bool]:
        """Process a batch of texts to generate embeddings.

        Args:
            batch: List of texts to embed

        Returns:
            Tuple of (embeddings, is_fallback)
        """
        retries = 0
        is_fallback = False

        # Estimate token count for logging
        estimated_tokens = sum(self._count_tokens(text) for text in batch)

        # Store current batch index for error logging
        self._current_batch_idx = getattr(self, "_current_batch_idx", 0)

        while retries <= self.max_retries:
            try:
                # Apply rate limiting
                self._apply_rate_limit()

                # Ensure batch is properly formatted for the OpenAI API
                # The API expects a list of strings, not None or other types
                sanitized_batch = []
                for text in batch:
                    if text is None or text == "":
                        # Use a space for empty strings to avoid API errors
                        sanitized_batch.append(" ")
                    else:
                        sanitized_batch.append(str(text))

                # Use the LlamaIndex embedding model to get embeddings
                batch_embeddings = self.embed_model.get_text_embedding_batch(sanitized_batch)

                # Update token usage stats
                self.token_usage["total_tokens"] += estimated_tokens
                self.token_usage["requests"] += 1

                # Convert to list format
                return [self._ensure_list_format(emb) for emb in batch_embeddings], False

            except Exception as e:
                retries += 1
                self.token_usage["failed_requests"] += 1

                if retries <= self.max_retries:
                    self.token_usage["retried_requests"] += 1
                    # Exponential backoff with jitter
                    delay = self.retry_base_delay * (2 ** (retries - 1)) + random.uniform(0, 0.5)

                    error_msg = f"Error generating embeddings (attempt {retries}/{self.max_retries}): {str(e)}. Retrying in {delay:.2f}s"
                    logger.warning(error_msg)
                    time.sleep(delay)
                else:
                    error_msg = f"Error generating embeddings after {self.max_retries} retries: {str(e)}"
                    logger.error(error_msg)

                    # Return zero vectors as fallback
                    is_fallback = True
                    return [[0.0] * self.dimensions for _ in range(len(batch))], True

        # Should never reach here, but just in case
        return [[0.0] * self.dimensions for _ in range(len(batch))], True

    def generate_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats
        """
        # Handle empty or None input
        if not text:
            logger.warning("Empty text provided for embedding, returning zero vector")
            return [0.0] * self.dimensions

        # Check if text exceeds token limit
        token_count = self._count_tokens(text)
        if token_count > self.max_tokens_per_chunk:
            logger.warning(
                f"Text exceeds token limit ({token_count} > {self.max_tokens_per_chunk}), chunking and averaging"
            )
            chunks = self._chunk_text(text)
            chunk_embeddings = self.generate_embeddings(chunks)[0]  # Get embeddings only

            # Average the embeddings
            if chunk_embeddings:
                try:
                    avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                    return avg_embedding
                except Exception as e:
                    logger.error(f"Error averaging chunk embeddings: {e}")
                    return [0.0] * self.dimensions
            else:
                return [0.0] * self.dimensions

        # Process single text
        embeddings, _ = self._process_batch([text])
        return embeddings[0]

    def generate_embeddings(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """Generate vector embeddings for a list of texts.

        Args:
            texts: List of text chunks to embed
            batch_size: Number of texts to process in each batch (defaults to self.batch_size)

        Returns:
            Tuple of (embedding vectors, metadata) where metadata includes fallback flags and token usage
        """
        try:
            # Use instance batch size if not specified
            if batch_size is None:
                batch_size = self.batch_size

            # Handle empty list case
            if not texts:
                return [[0.0] * self.dimensions], {"all_fallbacks": False, "token_usage": self.token_usage}

            # Preprocess texts: handle None values, empty strings, and large texts
            processed_texts = []
            for text in texts:
                if text is None or text == "":
                    # Use a space for empty strings to avoid API errors
                    processed_texts.append(" ")
                    continue

                # Convert to string if not already
                text = str(text)

                # Check if text exceeds token limit
                token_count = self._count_tokens(text)
                if token_count > self.max_tokens_per_chunk:
                    # Split into chunks and add each chunk
                    chunks = self._chunk_text(text)
                    processed_texts.extend(chunks)
                else:
                    processed_texts.append(text)

            # Create batches
            batches = []
            for i in range(0, len(processed_texts), batch_size):
                batches.append(processed_texts[i : i + batch_size])

            logger.info(f"Processing {len(processed_texts)} texts in {len(batches)} batches with size {batch_size}")

            # Track which chunks used fallback embeddings
            fallback_flags = [False] * len(processed_texts)

            # Process batches
            all_embeddings = []
            for batch_idx, batch in enumerate(batches):
                try:
                    # Store current batch index for error logging
                    self._current_batch_idx = batch_idx

                    # Process batch
                    batch_embeddings, is_fallback = self._process_batch(batch)
                    all_embeddings.extend(batch_embeddings)

                    # Update fallback flags
                    start_idx = batch_idx * batch_size
                    for i in range(len(batch)):
                        if start_idx + i < len(fallback_flags):
                            fallback_flags[start_idx + i] = is_fallback

                    logger.info(f"Completed batch {batch_idx + 1}/{len(batches)} {'(fallback)' if is_fallback else ''}")

                    # Add a small delay between batches to avoid rate limiting
                    if batch_idx < len(batches) - 1:
                        time.sleep(0.1)

                except Exception as e:
                    error_msg = f"Error processing batch {batch_idx}: {str(e)}"
                    logger.error(error_msg)

                    # Use fallback for this batch
                    batch_embeddings = [[0.0] * self.dimensions for _ in range(len(batch))]
                    all_embeddings.extend(batch_embeddings)

                    # Update fallback flags
                    start_idx = batch_idx * batch_size
                    for i in range(len(batch)):
                        if start_idx + i < len(fallback_flags):
                            fallback_flags[start_idx + i] = True

            # Create metadata
            metadata = {
                "fallback_flags": fallback_flags,
                "any_fallbacks": any(fallback_flags),
                "fallback_count": sum(fallback_flags),
                "token_usage": self.token_usage,
            }

            return all_embeddings, metadata

        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}"
            logger.error(error_msg)

            # Return zero vectors as fallback with metadata
            fallback_flags = [True] * len(texts)
            self.token_usage["failed_requests"] += 1

            metadata = {
                "fallback_flags": fallback_flags,
                "any_fallbacks": True,
                "fallback_count": len(texts),
                "token_usage": self.token_usage,
                "error": str(e),
            }

            return [[0.0] * self.dimensions for _ in range(len(texts))], metadata

    def get_embedding_dimensions(self) -> int:
        """Get the number of dimensions for the current embedding model.

        Returns:
            Number of embedding dimensions
        """
        return self.dimensions
