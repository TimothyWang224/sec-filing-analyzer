"""
Parallel Embeddings Module

This module provides functionality for generating vector embeddings in parallel using OpenAI's API through LlamaIndex.
"""

import os
import logging
import numpy as np
import concurrent.futures
import time
from typing import List, Optional, Union, Dict, Any, Tuple
from llama_index.embeddings.openai import OpenAIEmbedding

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelEmbeddingGenerator:
    """Handles generation of vector embeddings in parallel using OpenAI's API through LlamaIndex."""
    
    def __init__(
        self, 
        model: str = "text-embedding-3-small",
        max_workers: int = 4,
        rate_limit: float = 0.1
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
        
        self.embed_model = OpenAIEmbedding(
            model=model,
            api_key=api_key
        )
        self.dimensions = 1536  # text-embedding-3-small has 1536 dimensions
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
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
    
    def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """Process a batch of texts.
        
        Args:
            batch: List of text chunks to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Apply rate limiting
            self._apply_rate_limit()
            
            # Get embeddings
            batch_embeddings = self.embed_model.get_text_embedding_batch(batch)
            
            # Convert to list format
            return [self._ensure_list_format(emb) for emb in batch_embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings for batch: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * self.dimensions for _ in range(len(batch))]
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate vector embeddings for a list of texts in parallel.
        
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
            
            # Create batches
            batches = []
            for i in range(0, len(processed_texts), batch_size):
                batches.append(processed_texts[i:i+batch_size])
            
            logger.info(f"Processing {len(processed_texts)} texts in {len(batches)} batches")
            
            # Process batches in parallel
            all_embeddings = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
                # Submit batch tasks
                future_to_batch_idx = {
                    executor.submit(self._process_batch, batch): i 
                    for i, batch in enumerate(batches)
                }
                
                # Collect results
                batch_results = [None] * len(batches)
                for future in concurrent.futures.as_completed(future_to_batch_idx):
                    batch_idx = future_to_batch_idx[future]
                    try:
                        batch_embeddings = future.result()
                        batch_results[batch_idx] = batch_embeddings
                        logger.info(f"Completed batch {batch_idx+1}/{len(batches)}")
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                        # Use zero vectors as fallback
                        batch_results[batch_idx] = [[0.0] * self.dimensions for _ in range(len(batches[batch_idx]))]
            
            # Flatten results
            for batch_embedding in batch_results:
                if batch_embedding:
                    all_embeddings.extend(batch_embedding)
            
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
