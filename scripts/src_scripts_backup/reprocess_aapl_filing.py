"""
Script to reprocess the AAPL 10-K filing with improved logging.

This script specifically targets the AAPL 10-K filing with accession number 0000320193-23-000106
that had zero vectors in the embedding.
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sec_filing_analyzer.config import ETLConfig
from sec_filing_analyzer.data_retrieval.file_storage import FileStorage
from sec_filing_analyzer.pipeline.parallel_etl_pipeline import (
    ParallelSECFilingETLPipeline,
)
from sec_filing_analyzer.utils.logging_utils import (
    generate_embedding_error_report,
    setup_logging,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def reprocess_aapl_filing():
    """Reprocess the AAPL 10-K filing with improved logging."""
    # Set up enhanced logging
    setup_logging()

    # Initialize the ETL pipeline
    pipeline = ParallelSECFilingETLPipeline(
        max_workers=4,
        batch_size=50,  # Smaller batch size for better error tracking
        rate_limit=0.2,  # Slightly higher rate limit to avoid API throttling
    )

    # Load the filing data from cache
    file_storage = FileStorage(base_dir=ETLConfig().filings_dir)
    filing_cache_path = Path("data/filings/cache/0000320193-23-000106.json")

    if not filing_cache_path.exists():
        logger.error(f"Filing cache file not found: {filing_cache_path}")
        return

    logger.info(f"Loading filing data from cache: {filing_cache_path}")

    try:
        # Load the cached filing data
        filing_data = file_storage.load_cached_filing("0000320193-23-000106")

        if not filing_data or "metadata" not in filing_data:
            logger.error("Invalid or missing filing data in cache")
            return

        # Extract metadata
        metadata = filing_data["metadata"]

        logger.info(f"Reprocessing AAPL filing: {metadata.get('form')} from {metadata.get('filing_date')}")

        # Process the filing
        processed_data = pipeline.process_filing_data(metadata)

        if processed_data:
            logger.info("Successfully reprocessed AAPL filing")

            # Check if embeddings are still zero vectors
            embedding = processed_data.get("embedding", [])
            if embedding and all(v == 0.0 for v in embedding[:10]):
                logger.warning("Embedding is still a zero vector after reprocessing")
            else:
                logger.info("Successfully generated non-zero embeddings")
        else:
            logger.error("Failed to reprocess AAPL filing")

        # Generate and print error report
        error_report = generate_embedding_error_report()
        print("\nEmbedding Error Report:")
        print(error_report)

    except Exception as e:
        logger.error(f"Error reprocessing AAPL filing: {e}")
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    reprocess_aapl_filing()
