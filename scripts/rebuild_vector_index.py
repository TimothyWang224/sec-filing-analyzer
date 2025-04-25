"""
Rebuild the FAISS vector index using the NumPy binary files.

This script rebuilds the FAISS index for the vector store using the NumPy binary files
that were created by the migration script. It also updates the company_doc_mapping.json file.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np

from sec_filing_analyzer.config import VectorStoreConfig

# Import from the project
from sec_filing_analyzer.storage import OptimizedVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def rebuild_index(vector_store_path: str = "data/vector_store", use_gpu: bool = True):
    """
    Rebuild the FAISS index using the NumPy binary files.

    Args:
        vector_store_path: Path to the vector store directory
        use_gpu: Whether to use GPU acceleration
    """
    # Initialize the OptimizedVectorStore
    vector_store = OptimizedVectorStore(store_path=vector_store_path, use_gpu=use_gpu)

    # Get all NumPy files in the embeddings directory
    embeddings_dir = Path(vector_store_path) / "embeddings"
    npy_files = list(embeddings_dir.glob("*.npy"))

    if not npy_files:
        logger.error("No NumPy embedding files found. Run the migration script first.")
        return

    logger.info(f"Found {len(npy_files)} NumPy embedding files")

    # Build company mapping
    company_mapping = {}
    company_dir = Path(vector_store_path) / "companies"

    if company_dir.exists():
        for ticker_dir in company_dir.iterdir():
            if ticker_dir.is_dir():
                ticker = ticker_dir.name
                doc_ids = [f.stem for f in ticker_dir.glob("*.npy")]
                company_mapping[ticker] = doc_ids
                logger.info(f"Found {len(doc_ids)} documents for company {ticker}")

    # Save company mapping
    company_mapping_path = Path(vector_store_path) / "company_doc_mapping.json"
    with open(company_mapping_path, "w") as f:
        json.dump(company_mapping, f, indent=2)

    logger.info(f"Updated company_doc_mapping.json with {len(company_mapping)} companies")

    # Force rebuild the index for all companies
    companies = list(company_mapping.keys())
    if companies:
        logger.info(f"Rebuilding index for companies: {companies}")
        vector_store._load_faiss_index_for_companies(companies, force_rebuild=True)
    else:
        logger.info("No companies found. Rebuilding global index.")
        # Load all embeddings
        embeddings = []
        doc_ids = []

        for npy_file in npy_files:
            try:
                embedding = np.load(npy_file)
                embeddings.append(embedding)
                doc_ids.append(npy_file.stem)
            except Exception as e:
                logger.error(f"Error loading embedding from {npy_file}: {e}")

        if not embeddings:
            logger.error("No valid embeddings found.")
            return

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)

        # Use GPU if available and requested
        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info(f"Using GPU acceleration with {faiss.get_num_gpus()} GPUs")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        # Add embeddings to index
        index.add(embeddings_array)

        # Save index
        index_dir = Path(vector_store_path) / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "global_index.faiss"

        # Convert back to CPU for saving if using GPU
        if use_gpu and faiss.get_num_gpus() > 0:
            index = faiss.index_gpu_to_cpu(index)

        faiss.write_index(index, str(index_path))

        # Save doc_ids
        doc_ids_path = index_dir / "global_doc_ids.json"
        with open(doc_ids_path, "w") as f:
            json.dump(doc_ids, f)

        logger.info(f"Rebuilt global index with {len(embeddings)} embeddings")

    logger.info("Index rebuild complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild the FAISS vector index")
    parser.add_argument(
        "--vector-store-path", type=str, default="data/vector_store", help="Path to the vector store directory"
    )
    parser.add_argument("--no-gpu", action="store_true", help="Don't use GPU acceleration")

    args = parser.parse_args()

    rebuild_index(vector_store_path=args.vector_store_path, use_gpu=not args.no_gpu)
