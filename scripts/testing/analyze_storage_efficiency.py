"""
Script to analyze storage efficiency of NumPy vs JSON for embeddings.

This script compares the storage efficiency and performance of NumPy binary format
versus JSON for storing embedding vectors.
"""

import os
import sys
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_embedding(dimensions: int = 1536) -> np.ndarray:
    """Generate a test embedding vector.
    
    Args:
        dimensions: Number of dimensions for the embedding
        
    Returns:
        NumPy array with random values
    """
    return np.random.random(dimensions).astype(np.float32)

def measure_storage_size(embedding: np.ndarray) -> Tuple[int, int]:
    """Measure storage size of an embedding in NumPy and JSON formats.
    
    Args:
        embedding: NumPy array with embedding vector
        
    Returns:
        Tuple of (numpy_size, json_size) in bytes
    """
    # Create temporary files
    numpy_path = Path("temp_embedding.npy")
    json_path = Path("temp_embedding.json")
    
    # Save as NumPy
    np.save(numpy_path, embedding)
    numpy_size = numpy_path.stat().st_size
    
    # Save as JSON
    with open(json_path, "w") as f:
        json.dump(embedding.tolist(), f)
    json_size = json_path.stat().st_size
    
    # Clean up
    numpy_path.unlink()
    json_path.unlink()
    
    return numpy_size, json_size

def measure_load_time(embedding: np.ndarray, iterations: int = 100) -> Tuple[float, float]:
    """Measure time to load an embedding from NumPy and JSON formats.
    
    Args:
        embedding: NumPy array with embedding vector
        iterations: Number of iterations for timing
        
    Returns:
        Tuple of (numpy_time, json_time) in seconds
    """
    # Create temporary files
    numpy_path = Path("temp_embedding.npy")
    json_path = Path("temp_embedding.json")
    
    # Save files
    np.save(numpy_path, embedding)
    with open(json_path, "w") as f:
        json.dump(embedding.tolist(), f)
    
    # Measure NumPy load time
    start_time = time.time()
    for _ in range(iterations):
        loaded = np.load(numpy_path)
    numpy_time = (time.time() - start_time) / iterations
    
    # Measure JSON load time
    start_time = time.time()
    for _ in range(iterations):
        with open(json_path, "r") as f:
            loaded = json.load(f)
    json_time = (time.time() - start_time) / iterations
    
    # Clean up
    numpy_path.unlink()
    json_path.unlink()
    
    return numpy_time, json_time

def plot_comparison(dimensions: List[int], numpy_sizes: List[int], json_sizes: List[int], 
                   numpy_times: List[float], json_times: List[float]) -> None:
    """Plot comparison of NumPy vs JSON.
    
    Args:
        dimensions: List of dimensions tested
        numpy_sizes: List of NumPy storage sizes
        json_sizes: List of JSON storage sizes
        numpy_times: List of NumPy load times
        json_times: List of JSON load times
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot storage size comparison
    ax1.plot(dimensions, numpy_sizes, 'b-', label='NumPy')
    ax1.plot(dimensions, json_sizes, 'r-', label='JSON')
    ax1.set_xlabel('Dimensions')
    ax1.set_ylabel('Size (bytes)')
    ax1.set_title('Storage Size Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot load time comparison
    ax2.plot(dimensions, numpy_times, 'b-', label='NumPy')
    ax2.plot(dimensions, json_times, 'r-', label='JSON')
    ax2.set_xlabel('Dimensions')
    ax2.set_ylabel('Load Time (seconds)')
    ax2.set_title('Load Time Comparison')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('numpy_vs_json_comparison.png')
    logger.info("Saved comparison plot to numpy_vs_json_comparison.png")

def main():
    """Main function to analyze storage efficiency."""
    logger.info("Analyzing storage efficiency of NumPy vs JSON for embeddings")
    
    # Test different dimensions
    dimensions = [64, 128, 256, 512, 768, 1024, 1536, 2048]
    
    # Initialize result lists
    numpy_sizes = []
    json_sizes = []
    numpy_times = []
    json_times = []
    
    # Test each dimension
    for dim in dimensions:
        logger.info(f"Testing dimension: {dim}")
        
        # Generate test embedding
        embedding = generate_test_embedding(dim)
        
        # Measure storage size
        numpy_size, json_size = measure_storage_size(embedding)
        numpy_sizes.append(numpy_size)
        json_sizes.append(json_size)
        
        logger.info(f"  NumPy size: {numpy_size} bytes")
        logger.info(f"  JSON size: {json_size} bytes")
        logger.info(f"  Size ratio (JSON/NumPy): {json_size/numpy_size:.2f}x")
        
        # Measure load time
        numpy_time, json_time = measure_load_time(embedding)
        numpy_times.append(numpy_time)
        json_times.append(json_time)
        
        logger.info(f"  NumPy load time: {numpy_time*1000:.2f} ms")
        logger.info(f"  JSON load time: {json_time*1000:.2f} ms")
        logger.info(f"  Time ratio (JSON/NumPy): {json_time/numpy_time:.2f}x")
    
    # Plot comparison
    try:
        plot_comparison(dimensions, numpy_sizes, json_sizes, numpy_times, json_times)
    except Exception as e:
        logger.error(f"Error creating plot: {e}")
    
    # Print summary
    avg_size_ratio = sum(j/n for j, n in zip(json_sizes, numpy_sizes)) / len(dimensions)
    avg_time_ratio = sum(j/n for j, n in zip(json_times, numpy_times)) / len(dimensions)
    
    logger.info("\nSummary:")
    logger.info(f"Average size ratio (JSON/NumPy): {avg_size_ratio:.2f}x")
    logger.info(f"Average load time ratio (JSON/NumPy): {avg_time_ratio:.2f}x")
    
    # Estimate storage savings for real-world scenario
    logger.info("\nEstimated storage for 50 companies with 100 filings each:")
    total_vectors = 50 * 100 * 50  # 50 companies, 100 filings, 50 chunks per filing
    
    json_total = (json_sizes[-2] * total_vectors) / (1024 * 1024)  # Use 1536 dimension result
    numpy_total = (numpy_sizes[-2] * total_vectors) / (1024 * 1024)
    
    logger.info(f"  JSON storage: {json_total:.2f} MB")
    logger.info(f"  NumPy storage: {numpy_total:.2f} MB")
    logger.info(f"  Storage savings: {json_total - numpy_total:.2f} MB")

if __name__ == "__main__":
    main()
