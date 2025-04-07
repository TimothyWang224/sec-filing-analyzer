"""
Embeddings Module

This module provides functionality for generating vector embeddings using OpenAI's API through LlamaIndex.
"""

import os
import logging
import numpy as np
from typing import List, Optional
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
        
        self.embed_model = OpenAIEmbedding(
            model=model,
            api_key=api_key
        )
        self.dimensions = 1536  # text-embedding-3-small has 1536 dimensions
        logger.info(f"Initialized LlamaIndex OpenAI embedding generator with model: {model}")
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> np.ndarray:
        """Generate vector embeddings for a list of texts.
        
        Args:
            texts: List of text chunks to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        try:
            # Handle empty list case
            if not texts:
                return np.zeros((0, self.dimensions))
                
            all_embeddings = []
            
            # Process in batches to avoid rate limits
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = self.embed_model.get_text_embedding_batch(batch)
                all_embeddings.extend(batch_embeddings)
            
            return np.array(all_embeddings)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.dimensions))
    
    def get_embedding_dimensions(self) -> int:
        """Get the number of dimensions for the current embedding model.
        
        Returns:
            Number of embedding dimensions
        """
        return self.dimensions 