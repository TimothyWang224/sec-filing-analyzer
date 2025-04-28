"""
Parallel Embeddings Module

This module provides functionality for generating vector embeddings in parallel using OpenAI's API through LlamaIndex.
"""

import concurrent.futures
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from llama_index.embeddings.openai import OpenAIEmbedding

# Import logging utilities
try:
    from ..utils.logging_utils import log_embedding_error, setup_logging
except ImportError:
    # Fallback if logging utils not available
    def log_embedding_error(*args, **kwargs):
        pass

    def setup_logging(*args, **kwargs):
        pass


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelEmbeddingGenerator:
    """Handles generation of vector embeddings in parallel using OpenAI's API through LlamaIndex."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        max_workers: int = 4,
        rate_limit: float = 0.1,
        filing_metadata: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        batch_size: int = 50,
    ):
        """Initialize the parallel embedding generator.

        Args:
            model: OpenAI embedding model to use
            max_workers: Maximum number of worker threads
            rate_limit: Minimum time between API requests in seconds
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")

        self.embed_model = OpenAIEmbedding(model=model, api_key=api_key)
        self.dimensions = 1536  # text-embedding-3-small has 1536 dimensions
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.filing_metadata = filing_metadata or {}
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.batch_size = batch_size
        self.token_usage = {
            "total_tokens": 0,
            "requests": 0,
            "failed_requests": 0,
            "retried_requests": 0,
        }

        # Set up enhanced logging
        try:
            setup_logging()
        except Exception as e:
            logger.warning(f"Failed to set up enhanced logging: {e}")

        logger.info(f"Initialized Parallel OpenAI embedding generator with model: {model}, workers: {max_workers}")

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
            return list(embedding)

    def _apply_rate_limit(self):
        """Apply rate limiting to avoid API throttling."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit:
            time_to_wait = self.rate_limit - time_since_last
            time.sleep(time_to_wait)

        self.last_request_time = time.time()

    def _estimate_tokens(self, texts: List[str]) -> int:
        """Estimate the number of tokens in a list of texts.

        Args:
            texts: List of texts to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters for English text
        char_count = sum(len(text) for text in texts)
        return char_count // 4

    def _process_batch(self, batch: List[str]) -> Tuple[List[List[float]], bool]:
        """Process a batch of texts with retry logic.

        Args:
            batch: List of text chunks to embed

        Returns:
            Tuple of (embedding vectors, is_fallback_flag)
        """
        retries = 0
        is_fallback = False
        estimated_tokens = self._estimate_tokens(batch)

        while retries <= self.max_retries:
            try:
                # Apply rate limiting
                self._apply_rate_limit()

                # Get embeddings
                batch_embeddings = self.embed_model.get_text_embedding_batch(batch)

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

                    # Log detailed error information
                    if self.filing_metadata:
                        batch_idx = getattr(self, "_current_batch_idx", None)
                        log_embedding_error(
                            error=e,
                            filing_id=self.filing_metadata.get("accession_number", "unknown"),
                            company=self.filing_metadata.get("ticker", "unknown"),
                            filing_type=self.filing_metadata.get("form", "unknown"),
                            batch_index=batch_idx,
                            chunk_count=len(batch) if batch else 0,
                        )

                    # Return zero vectors as fallback
                    is_fallback = True
                    return [[0.0] * self.dimensions for _ in range(len(batch))], True

    def generate_embeddings(
        self, texts: List[str], batch_size: Optional[int] = None
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """Generate vector embeddings for a list of texts in parallel.

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
                return [[0.0] * self.dimensions], {
                    "all_fallbacks": False,
                    "token_usage": self.token_usage,
                }

            # Ensure all texts are strings
            processed_texts = [str(text) if text is not None else "" for text in texts]

            # Create batches with smaller size for better rate limit handling
            batches = []
            for i in range(0, len(processed_texts), batch_size):
                batches.append(processed_texts[i : i + batch_size])

            logger.info(f"Processing {len(processed_texts)} texts in {len(batches)} batches with size {batch_size}")

            # Track which chunks used fallback embeddings
            fallback_flags = [False] * len(processed_texts)

            # Process batches in parallel
            all_embeddings = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
                # Submit batch tasks
                future_to_batch_idx = {
                    executor.submit(self._process_batch, batch): i for i, batch in enumerate(batches)
                }

                # Collect results
                batch_results = [None] * len(batches)
                batch_fallbacks = [False] * len(batches)

                for future in concurrent.futures.as_completed(future_to_batch_idx):
                    batch_idx = future_to_batch_idx[future]
                    try:
                        # Store current batch index for error logging
                        self._current_batch_idx = batch_idx
                        batch_embeddings, is_fallback = future.result()
                        batch_results[batch_idx] = batch_embeddings
                        batch_fallbacks[batch_idx] = is_fallback

                        # Update fallback flags for this batch
                        if is_fallback:
                            start_idx = batch_idx * batch_size
                            end_idx = min(start_idx + batch_size, len(fallback_flags))
                            for i in range(start_idx, end_idx):
                                fallback_flags[i] = True

                        logger.info(
                            f"Completed batch {batch_idx + 1}/{len(batches)} {'(fallback)' if is_fallback else ''}"
                        )
                    except Exception as e:
                        error_msg = f"Error processing batch {batch_idx}: {str(e)}"
                        logger.error(error_msg)

                        # Log detailed error information
                        if self.filing_metadata:
                            log_embedding_error(
                                error=e,
                                filing_id=self.filing_metadata.get("accession_number", "unknown"),
                                company=self.filing_metadata.get("ticker", "unknown"),
                                filing_type=self.filing_metadata.get("form", "unknown"),
                                batch_index=batch_idx,
                                chunk_count=len(batches[batch_idx]) if batch_idx < len(batches) else 0,
                            )

                        # Use zero vectors as fallback
                        batch_results[batch_idx] = [[0.0] * self.dimensions for _ in range(len(batches[batch_idx]))]
                        batch_fallbacks[batch_idx] = True

                        # Update fallback flags for this batch
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(fallback_flags))
                        for i in range(start_idx, end_idx):
                            fallback_flags[i] = True

            # Flatten results
            for batch_embedding in batch_results:
                if batch_embedding:
                    all_embeddings.extend(batch_embedding)

            # Prepare metadata
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

            # Log detailed error information
            if self.filing_metadata:
                log_embedding_error(
                    error=e,
                    filing_id=self.filing_metadata.get("accession_number", "unknown"),
                    company=self.filing_metadata.get("ticker", "unknown"),
                    filing_type=self.filing_metadata.get("form", "unknown"),
                    chunk_count=len(texts) if texts else 0,
                )

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
