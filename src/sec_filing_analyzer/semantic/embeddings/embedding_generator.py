"""
Embeddings Module

This module provides functionality for generating vector embeddings using OpenAI's API through LlamaIndex.
"""

import logging
import os
from typing import Any, List, Union

import numpy as np
from llama_index.embeddings.openai import OpenAIEmbedding

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles generation of vector embeddings using OpenAI's API through LlamaIndex."""

    def __init__(self, model: str = "text-embedding-3-small"):
        """Initialize the embedding generator.

        Args:
            model: OpenAI embedding model to use
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")

        self.embed_model = OpenAIEmbedding(model=model, api_key=api_key)
        self.dimensions = 1536  # text-embedding-3-small has 1536 dimensions
        logger.info(f"Initialized LlamaIndex OpenAI embedding generator with model: {model}")

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

    def generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate vector embeddings for a list of texts.

        Args:
            texts: List of text chunks to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors, where each vector is a list of floats
        """
        try:
            # Handle empty list case
            if not texts:
                return [[0.0] * self.dimensions]

            # Ensure all texts are strings
            processed_texts = [str(text) if text is not None else "" for text in texts]

            all_embeddings = []

            # Process in batches to avoid rate limits
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i : i + batch_size]
                # Ensure batch is properly formatted for the OpenAI API
                sanitized_batch = [str(text) if text is not None else "" for text in batch]
                batch_embeddings = self.embed_model.get_text_embedding_batch(sanitized_batch)

                # Convert all embeddings to list format
                for emb in batch_embeddings:
                    all_embeddings.append(self._ensure_list_format(emb))

            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * self.dimensions for _ in range(len(texts))]

    def get_embedding_dimensions(self) -> int:
        """Get the number of dimensions for the current embedding model.

        Returns:
            Number of embedding dimensions
        """
        return self.dimensions
