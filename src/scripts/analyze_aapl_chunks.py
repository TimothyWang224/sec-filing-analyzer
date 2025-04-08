"""
Script to analyze the AAPL filing chunks and identify potential issues.

This script examines the text chunks from the AAPL filing to identify any unusual
characters, formatting, or other issues that might cause embedding generation to fail.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import unicodedata
import statistics
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sec_filing_analyzer.embeddings.parallel_embeddings import ParallelEmbeddingGenerator
from sec_filing_analyzer.config import ETLConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def analyze_text_chunks(chunks: List[str]) -> Dict[str, Any]:
    """Analyze text chunks for potential issues.
    
    Args:
        chunks: List of text chunks to analyze
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        "total_chunks": len(chunks),
        "empty_chunks": 0,
        "chunks_with_unusual_chars": 0,
        "chunks_with_control_chars": 0,
        "chunks_with_long_lines": 0,
        "chunks_with_excessive_whitespace": 0,
        "chunk_length_stats": {},
        "unusual_chars": set(),
        "problematic_chunks": []
    }
    
    chunk_lengths = []
    
    for i, chunk in enumerate(chunks):
        # Skip empty chunks
        if not chunk or chunk.strip() == "":
            results["empty_chunks"] += 1
            continue
        
        chunk_lengths.append(len(chunk))
        issues = []
        
        # Check for unusual characters
        unusual_chars = set()
        control_chars = set()
        for char in chunk:
            if ord(char) > 127:
                category = unicodedata.category(char)
                name = unicodedata.name(char, "Unknown")
                unusual_chars.add(f"{char} (U+{ord(char):04X}, {category}, {name})")
            elif unicodedata.category(char).startswith('C') and char not in '\n\t\r':
                control_chars.add(f"{char} (U+{ord(char):04X})")
        
        if unusual_chars:
            results["chunks_with_unusual_chars"] += 1
            results["unusual_chars"].update(unusual_chars)
            issues.append(f"Contains {len(unusual_chars)} unusual characters")
        
        if control_chars:
            results["chunks_with_control_chars"] += 1
            issues.append(f"Contains {len(control_chars)} control characters: {', '.join(control_chars)}")
        
        # Check for long lines
        lines = chunk.split('\n')
        long_lines = [line for line in lines if len(line) > 1000]
        if long_lines:
            results["chunks_with_long_lines"] += 1
            issues.append(f"Contains {len(long_lines)} lines longer than 1000 characters")
        
        # Check for excessive whitespace
        whitespace_ratio = len(re.findall(r'\s', chunk)) / max(1, len(chunk))
        if whitespace_ratio > 0.5:
            results["chunks_with_excessive_whitespace"] += 1
            issues.append(f"Excessive whitespace ({whitespace_ratio:.2f} ratio)")
        
        # If any issues were found, add to problematic chunks
        if issues:
            results["problematic_chunks"].append({
                "index": i,
                "issues": issues,
                "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
    
    # Calculate statistics on chunk lengths
    if chunk_lengths:
        results["chunk_length_stats"] = {
            "min": min(chunk_lengths),
            "max": max(chunk_lengths),
            "mean": statistics.mean(chunk_lengths),
            "median": statistics.median(chunk_lengths),
            "std_dev": statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0
        }
    
    return results

def test_embedding_generation(chunks: List[str]) -> Tuple[List[bool], List[str]]:
    """Test embedding generation for each chunk.
    
    Args:
        chunks: List of text chunks to test
        
    Returns:
        Tuple of (success_flags, error_messages)
    """
    embedding_generator = ParallelEmbeddingGenerator(
        model=ETLConfig().embedding_model,
        max_workers=1,  # Use single worker for testing
        rate_limit=0.2  # Slightly higher rate limit
    )
    
    success_flags = []
    error_messages = []
    
    for i, chunk in enumerate(chunks):
        if not chunk or chunk.strip() == "":
            success_flags.append(True)  # Empty chunks are not a problem
            error_messages.append("")
            continue
        
        try:
            logger.info(f"Testing embedding generation for chunk {i+1}/{len(chunks)}")
            
            # Generate embedding for the chunk
            embedding = embedding_generator._process_batch([chunk])
            
            # Check if it's a zero vector
            if all(v == 0.0 for v in embedding[0][:10]):
                success_flags.append(False)
                error_messages.append("Generated zero vector")
            else:
                success_flags.append(True)
                error_messages.append("")
        except Exception as e:
            success_flags.append(False)
            error_messages.append(str(e))
    
    return success_flags, error_messages

def main():
    """Main function to analyze AAPL filing chunks."""
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
    
    # Analyze chunks
    logger.info("Analyzing text chunks...")
    analysis_results = analyze_text_chunks(chunks)
    
    # Test embedding generation for each chunk
    logger.info("Testing embedding generation for each chunk...")
    success_flags, error_messages = test_embedding_generation(chunks)
    
    # Add embedding test results to analysis
    analysis_results["embedding_test_results"] = {
        "successful_chunks": sum(success_flags),
        "failed_chunks": len(success_flags) - sum(success_flags),
        "failures": [
            {
                "index": i,
                "error": error_messages[i],
                "preview": chunks[i][:100] + "..." if len(chunks[i]) > 100 else chunks[i]
            }
            for i in range(len(chunks))
            if not success_flags[i]
        ]
    }
    
    # Print summary
    print("\nAnalysis Results:")
    print(f"Total chunks: {analysis_results['total_chunks']}")
    print(f"Empty chunks: {analysis_results['empty_chunks']}")
    print(f"Chunks with unusual characters: {analysis_results['chunks_with_unusual_chars']}")
    print(f"Chunks with control characters: {analysis_results['chunks_with_control_chars']}")
    print(f"Chunks with long lines: {analysis_results['chunks_with_long_lines']}")
    print(f"Chunks with excessive whitespace: {analysis_results['chunks_with_excessive_whitespace']}")
    
    print("\nChunk Length Statistics:")
    for key, value in analysis_results["chunk_length_stats"].items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\nEmbedding Test Results:")
    print(f"Successful chunks: {analysis_results['embedding_test_results']['successful_chunks']}")
    print(f"Failed chunks: {analysis_results['embedding_test_results']['failed_chunks']}")
    
    if analysis_results["embedding_test_results"]["failures"]:
        print("\nFailed Chunks:")
        for failure in analysis_results["embedding_test_results"]["failures"][:5]:  # Show first 5 failures
            print(f"  Chunk {failure['index']}: {failure['error']}")
            print(f"  Preview: {failure['preview']}")
            print()
    
    # Save detailed results to file
    results_path = Path("data/logs/aapl_chunk_analysis.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w", encoding="utf-8") as f:
        # Convert sets to lists for JSON serialization
        analysis_results["unusual_chars"] = list(analysis_results["unusual_chars"])
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Detailed analysis results saved to {results_path}")

if __name__ == "__main__":
    main()
