"""
Test script for the robust embedding generator.
"""

import json
import logging

from dotenv import load_dotenv

from src.sec_filing_analyzer.semantic.embeddings.robust_embedding_generator import (
    RobustEmbeddingGenerator,
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_robust_embedding_generator():
    """Test the robust embedding generator with various inputs."""
    # Initialize generator
    generator = RobustEmbeddingGenerator(
        model="text-embedding-3-small",
        max_tokens_per_chunk=4000,
        rate_limit=0.5,
        batch_size=10,
    )

    # Test cases
    test_cases = [
        {"name": "normal_text", "input": "This is a normal text."},
        {"name": "empty_string", "input": ""},
        {"name": "none_value", "input": None},
        {"name": "very_long", "input": "This is a test. " * 2000},  # Should be chunked
        {
            "name": "batch_processing",
            "input": ["Text " + str(i) for i in range(25)],
        },  # Should be processed in batches
    ]

    results = {}

    # Test single text embedding
    for test_case in test_cases:
        name = test_case["name"]
        input_text = test_case["input"]

        if name != "batch_processing":
            logger.info(f"Testing single text embedding: {name}")
            try:
                embedding = generator.generate_embedding(input_text)
                results[name] = {
                    "success": True,
                    "embedding_length": len(embedding),
                    "token_usage": generator.token_usage.copy(),
                }
            except Exception as e:
                results[name] = {
                    "success": False,
                    "error": str(e),
                    "token_usage": generator.token_usage.copy(),
                }

    # Test batch processing
    logger.info("Testing batch processing")
    try:
        batch_texts = test_cases[-1]["input"]
        embeddings, metadata = generator.generate_embeddings(batch_texts)
        results["batch_processing"] = {
            "success": True,
            "num_embeddings": len(embeddings),
            "metadata": metadata,
            "token_usage": generator.token_usage.copy(),
        }
    except Exception as e:
        results["batch_processing"] = {
            "success": False,
            "error": str(e),
            "token_usage": generator.token_usage.copy(),
        }

    # Save results to file
    with open("robust_embedding_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Tests completed. Results saved to robust_embedding_test_results.json")


if __name__ == "__main__":
    test_robust_embedding_generator()
