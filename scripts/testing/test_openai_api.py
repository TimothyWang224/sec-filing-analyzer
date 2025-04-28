"""
Script to test the OpenAI API directly with AAPL filing chunks.

This script sends chunks from the AAPL filing directly to the OpenAI API
to identify any issues with the API calls themselves.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import openai
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")


def load_aapl_filing() -> Dict[str, Any]:
    """Load the AAPL filing from the cache.

    Returns:
        The filing data as a dictionary
    """
    filing_path = Path("data/filings/cache/0000320193-23-000106.json")

    if not filing_path.exists():
        logger.error(f"Filing not found: {filing_path}")
        return {}

    try:
        with open(filing_path, "r", encoding="utf-8") as f:
            filing_data = json.load(f)
        return filing_data
    except Exception as e:
        logger.error(f"Error loading filing: {e}")
        return {}


def test_openai_embedding(text: str) -> Dict[str, Any]:
    """Test OpenAI embedding generation for a text.

    Args:
        text: Text to generate embedding for

    Returns:
        Dictionary with test results
    """
    result = {"success": False, "error": None, "embedding": None, "response_time": None}

    try:
        start_time = time.time()

        # Call OpenAI API directly
        response = openai.Embedding.create(model="text-embedding-3-small", input=text)

        end_time = time.time()
        result["response_time"] = end_time - start_time

        # Extract embedding
        embedding = response["data"][0]["embedding"]
        result["embedding"] = embedding[:10]  # Store just the first 10 values
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    """Main function to test OpenAI API with AAPL filing chunks."""
    logger.info("Loading AAPL filing...")
    filing_data = load_aapl_filing()

    if not filing_data:
        logger.error("Failed to load AAPL filing")
        return

    # Extract text chunks
    if "processed_data" not in filing_data:
        logger.error("No processed data found in filing")
        return

    processed_data = filing_data["processed_data"]

    if "chunk_texts" not in processed_data:
        logger.error("No chunk texts found in processed data")
        return

    chunks = processed_data["chunk_texts"]
    logger.info(f"Found {len(chunks)} text chunks in AAPL filing")

    # Test a sample of chunks
    sample_size = min(10, len(chunks))
    sample_indices = [0]  # Always include the first chunk

    # Add some chunks from the middle and end
    if len(chunks) > 1:
        sample_indices.extend(
            [
                len(chunks) // 4,
                len(chunks) // 2,
                (3 * len(chunks)) // 4,
                len(chunks) - 1,
            ]
        )

    # Add some random chunks
    import random

    random.seed(42)  # For reproducibility
    while len(sample_indices) < sample_size:
        idx = random.randint(0, len(chunks) - 1)
        if idx not in sample_indices:
            sample_indices.append(idx)

    # Sort indices
    sample_indices.sort()

    # Test each sample chunk
    results = []
    for idx in sample_indices:
        chunk = chunks[idx]
        if not chunk or chunk.strip() == "":
            logger.info(f"Skipping empty chunk at index {idx}")
            continue

        logger.info(f"Testing chunk {idx} with OpenAI API...")
        test_result = test_openai_embedding(chunk)

        results.append(
            {
                "index": idx,
                "success": test_result["success"],
                "error": test_result["error"],
                "response_time": test_result["response_time"],
                "embedding_preview": test_result["embedding"],
                "text_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk,
            }
        )

        # Add a delay to avoid rate limiting
        time.sleep(0.5)

    # Print results
    print("\nOpenAI API Test Results:")
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Tested {len(results)} chunks")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed Chunks:")
        for failure in failed:
            print(f"  Chunk {failure['index']}: {failure['error']}")
            print(f"  Preview: {failure['text_preview']}")
            print()

    # Save results to file
    results_path = Path("data/logs/openai_api_test_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_tested": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"Test results saved to {results_path}")


if __name__ == "__main__":
    main()
