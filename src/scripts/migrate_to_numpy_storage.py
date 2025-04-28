"""
Migration Script for Vector Store

This script migrates the existing JSON-based vector store to the optimized NumPy binary format.
It preserves all existing data while converting embeddings to a more efficient storage format.
"""

import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sec_filing_analyzer.config import StorageConfig
from sec_filing_analyzer.storage.optimized_vector_store import OptimizedVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def backup_existing_store(vector_store_path: Path) -> Path:
    """Create a backup of the existing vector store.

    Args:
        vector_store_path: Path to the existing vector store

    Returns:
        Path to the backup directory
    """
    if not vector_store_path.exists():
        logger.warning(f"Vector store path {vector_store_path} does not exist, nothing to backup")
        return vector_store_path

    # Create backup directory
    backup_path = vector_store_path.parent / f"{vector_store_path.name}_backup_{int(time.time())}"
    logger.info(f"Creating backup of vector store at {backup_path}")

    # Copy files to backup directory
    shutil.copytree(vector_store_path, backup_path)
    logger.info(f"Backup created at {backup_path}")

    return backup_path


def migrate_json_to_numpy(vector_store_path: Path) -> None:
    """Migrate JSON embeddings to NumPy binary format.

    Args:
        vector_store_path: Path to the vector store
    """
    # Initialize paths
    embeddings_dir = vector_store_path / "embeddings"
    metadata_dir = vector_store_path / "metadata"
    company_dir = vector_store_path / "by_company"

    if not embeddings_dir.exists():
        logger.warning(f"Embeddings directory {embeddings_dir} does not exist, nothing to migrate")
        return

    # Create company directory if it doesn't exist
    company_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON embedding files
    json_files = list(embeddings_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON embedding files to migrate")

    # Create company to document mapping
    company_to_docs = {}

    # Process each JSON file
    for json_file in tqdm(json_files, desc="Migrating embeddings"):
        try:
            # Get document ID from filename
            doc_id = json_file.stem

            # Load JSON embedding
            with open(json_file, "r") as f:
                embedding = json.load(f)

            # Convert to NumPy array
            embedding_array = np.array(embedding, dtype=np.float32)

            # Save as NumPy binary
            npy_file = embeddings_dir / f"{doc_id}.npy"
            np.save(npy_file, embedding_array)

            # Get company information from metadata
            metadata_file = metadata_dir / f"{doc_id}.json"
            if metadata_file.exists():
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # Save by company if ticker is available
                if "ticker" in metadata:
                    ticker = metadata["ticker"]

                    # Add to company mapping
                    if ticker not in company_to_docs:
                        company_to_docs[ticker] = []
                    company_to_docs[ticker].append(doc_id)

                    # Save embedding by company
                    company_ticker_dir = company_dir / ticker
                    company_ticker_dir.mkdir(parents=True, exist_ok=True)
                    company_npy_file = company_ticker_dir / f"{doc_id}.npy"
                    np.save(company_npy_file, embedding_array)

            # Remove the original JSON file
            json_file.unlink()

        except Exception as e:
            logger.error(f"Error migrating embedding {json_file}: {e}")

    # Save company to document mapping
    mapping_file = vector_store_path / "company_doc_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(company_to_docs, f, indent=2)

    logger.info(f"Migration complete. Converted {len(json_files)} embeddings to NumPy format")
    logger.info(f"Created company mapping for {len(company_to_docs)} companies")


def main():
    """Main function to migrate the vector store."""

    # Get vector store path from config
    vector_store_path = Path(StorageConfig().vector_store_path)
    logger.info(f"Migrating vector store at {vector_store_path}")

    # Create backup
    backup_path = backup_existing_store(vector_store_path)

    # Migrate JSON to NumPy
    migrate_json_to_numpy(vector_store_path)

    # Initialize the optimized vector store to verify migration
    optimized_store = OptimizedVectorStore(store_path=str(vector_store_path))
    stats = optimized_store.get_stats()

    logger.info("Migration successful!")
    logger.info(f"Vector store statistics: {stats}")
    logger.info(f"Backup of original store available at: {backup_path}")


if __name__ == "__main__":
    main()
