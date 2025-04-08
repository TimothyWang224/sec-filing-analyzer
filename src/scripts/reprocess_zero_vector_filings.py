"""
Script to reprocess filings with zero vectors.

This script identifies filings with zero vectors and reprocesses them with improved
embedding generation settings to fix the issue.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sec_filing_analyzer.pipeline.parallel_etl_pipeline import ParallelSECFilingETLPipeline
from sec_filing_analyzer.utils.logging_utils import setup_logging, generate_embedding_error_report
from sec_filing_analyzer.config import ETLConfig
from sec_filing_analyzer.data_retrieval.file_storage import FileStorage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_zero_vector(vector: List[float]) -> bool:
    """Check if a vector contains only zeros.
    
    Args:
        vector: The vector to check
        
    Returns:
        True if the vector contains only zeros, False otherwise
    """
    # Check the first 10 elements to determine if it's a zero vector
    return all(v == 0.0 for v in vector[:10])

def find_zero_vector_filings() -> List[Dict[str, Any]]:
    """Find all filings with zero vectors in their embeddings.
    
    Returns:
        List of dictionaries containing information about filings with zero vectors
    """
    cache_dir = Path(ETLConfig().filings_dir) / "cache"
    
    if not cache_dir.exists():
        logger.error(f"Cache directory not found: {cache_dir}")
        return []
    
    zero_vector_filings = []
    
    # Scan all JSON files in the cache directory
    for file_path in cache_dir.glob("*.json"):
        try:
            with open(file_path, "r") as f:
                filing_data = json.load(f)
            
            # Check if the filing has metadata
            if "metadata" not in filing_data:
                logger.warning(f"Missing metadata in filing: {file_path.name}")
                continue
            
            metadata = filing_data["metadata"]
            
            # Check if the filing has processed data
            if "processed_data" not in filing_data:
                logger.warning(f"Missing processed data in filing: {file_path.name}")
                continue
            
            processed_data = filing_data["processed_data"]
            
            # Check if the filing has an embedding
            if "embedding" not in processed_data:
                logger.warning(f"Missing embedding in filing: {file_path.name}")
                continue
            
            embedding = processed_data["embedding"]
            
            # Check if the embedding is a zero vector
            if is_zero_vector(embedding):
                zero_vector_filings.append({
                    "filing_id": file_path.stem,
                    "company": metadata.get("ticker", "Unknown"),
                    "form": metadata.get("form", "Unknown"),
                    "filing_date": metadata.get("filing_date", "Unknown"),
                    "file_path": str(file_path),
                    "metadata": metadata
                })
                logger.info(f"Found zero vector in filing: {file_path.name}")
        
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
    
    return zero_vector_filings

def reprocess_filing(filing_info: Dict[str, Any], pipeline: ParallelSECFilingETLPipeline) -> bool:
    """Reprocess a filing with zero vectors.
    
    Args:
        filing_info: Information about the filing to reprocess
        pipeline: The ETL pipeline to use for reprocessing
        
    Returns:
        True if reprocessing was successful, False otherwise
    """
    try:
        logger.info(f"Reprocessing filing {filing_info['filing_id']} for {filing_info['company']}")
        
        # Process the filing with improved settings
        processed_data = pipeline.process_filing_data(filing_info["metadata"])
        
        if not processed_data:
            logger.error(f"Failed to reprocess filing {filing_info['filing_id']}")
            return False
        
        # Check if the embedding is still a zero vector
        embedding = processed_data.get("embedding", [])
        if embedding and is_zero_vector(embedding):
            logger.warning(f"Embedding is still a zero vector after reprocessing for {filing_info['filing_id']}")
            return False
        
        logger.info(f"Successfully reprocessed filing {filing_info['filing_id']} with non-zero embeddings")
        
        # Check embedding metadata
        embedding_metadata = processed_data.get("embedding_metadata", {})
        has_fallbacks = embedding_metadata.get("has_fallbacks", False)
        fallback_count = embedding_metadata.get("fallback_count", 0)
        
        if has_fallbacks:
            logger.warning(f"Reprocessed filing has {fallback_count} fallback chunks")
        
        return True
    
    except Exception as e:
        logger.error(f"Error reprocessing filing {filing_info['filing_id']}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main(max_filings: Optional[int] = None, batch_size: int = 20, rate_limit: float = 0.2):
    """Main function to reprocess filings with zero vectors.
    
    Args:
        max_filings: Maximum number of filings to reprocess (None for all)
        batch_size: Batch size for embedding generation
        rate_limit: Rate limit for API requests in seconds
    """
    # Set up enhanced logging
    setup_logging()
    
    # Find filings with zero vectors
    logger.info("Finding filings with zero vectors...")
    zero_vector_filings = find_zero_vector_filings()
    
    if not zero_vector_filings:
        logger.info("No filings with zero vectors found.")
        return
    
    logger.info(f"Found {len(zero_vector_filings)} filings with zero vectors")
    
    # Limit the number of filings to reprocess if specified
    if max_filings is not None and max_filings > 0:
        zero_vector_filings = zero_vector_filings[:max_filings]
        logger.info(f"Limiting reprocessing to {max_filings} filings")
    
    # Initialize the ETL pipeline with improved settings
    pipeline = ParallelSECFilingETLPipeline(
        max_workers=2,  # Use fewer workers to avoid rate limiting
        batch_size=batch_size,  # Smaller batch size
        rate_limit=rate_limit  # Higher rate limit
    )
    
    # Reprocess each filing
    successful = 0
    failed = 0
    
    for i, filing_info in enumerate(zero_vector_filings):
        logger.info(f"Processing filing {i+1}/{len(zero_vector_filings)}: {filing_info['company']} {filing_info['form']}")
        
        if reprocess_filing(filing_info, pipeline):
            successful += 1
        else:
            failed += 1
        
        # Add a delay between filings to avoid rate limiting
        if i < len(zero_vector_filings) - 1:
            time.sleep(1.0)
    
    # Generate and print error report
    logger.info(f"Reprocessing complete. Successful: {successful}, Failed: {failed}")
    
    error_report = generate_embedding_error_report()
    print("\nEmbedding Error Report:")
    print(error_report)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reprocess filings with zero vectors")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of filings to reprocess")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for embedding generation")
    parser.add_argument("--rate-limit", type=float, default=0.2, help="Rate limit for API requests in seconds")
    
    args = parser.parse_args()
    
    main(max_filings=args.max, batch_size=args.batch_size, rate_limit=args.rate_limit)
