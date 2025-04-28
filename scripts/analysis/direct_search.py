"""
Direct search script for the vector store.
"""

import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from sec_filing_analyzer.embeddings import EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()

    # Generate embedding for the query
    query = "revenue concentration"
    logger.info(f"Generating embedding for query: {query}")
    query_embedding = embedding_generator.generate_embeddings([query])[0]

    # Load embeddings from disk
    store_path = Path("data/vector_store")
    embeddings_dir = store_path / "embeddings"
    metadata_dir = store_path / "metadata"
    text_dir = store_path / "text"

    logger.info(f"Loading embeddings from {embeddings_dir}")

    # Load all embeddings
    embeddings = {}
    for file_path in embeddings_dir.glob("*.json"):
        try:
            doc_id = file_path.stem
            with open(file_path, "r") as f:
                embedding = json.load(f)
                embeddings[doc_id] = embedding
        except Exception as e:
            logger.warning(f"Error loading embedding from {file_path}: {e}")

    logger.info(f"Loaded {len(embeddings)} embeddings")

    # Calculate similarity scores
    scores = {}
    for doc_id, embedding in embeddings.items():
        try:
            score = cosine_similarity(query_embedding, embedding)
            scores[doc_id] = score
        except Exception as e:
            logger.warning(f"Error calculating similarity for {doc_id}: {e}")

    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Get top 5 results
    top_k = 5
    top_results = sorted_scores[:top_k]

    logger.info(f"Top {top_k} results:")
    for i, (doc_id, score) in enumerate(top_results):
        logger.info(f"Result {i + 1}:")
        logger.info(f"  ID: {doc_id}")
        logger.info(f"  Score: {score:.4f}")

        # Load metadata
        try:
            safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            metadata_path = metadata_dir / f"{safe_id}.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    logger.info(f"  Metadata: {metadata}")
        except Exception as e:
            logger.warning(f"Error loading metadata for {doc_id}: {e}")

        # Load text
        try:
            safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")
            text_path = text_dir / f"{safe_id}.txt"
            if text_path.exists():
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    logger.info(f"  Text: {text[:100]}...")
        except Exception as e:
            logger.warning(f"Error loading text for {doc_id}: {e}")


if __name__ == "__main__":
    main()
