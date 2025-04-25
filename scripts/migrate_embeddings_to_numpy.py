"""
Migrate embeddings from JSON to NumPy binary format.

This script converts all embedding files in the vector store from JSON to NumPy binary format
for faster loading and processing during semantic search.
"""

import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def migrate_embeddings(vector_store_path: str = "data/vector_store", backup: bool = True):
    """
    Migrate embeddings from JSON to NumPy binary format.

    Args:
        vector_store_path: Path to the vector store directory
        backup: Whether to create a backup of the original JSON files
    """
    # Set up paths
    vector_store_dir = Path(vector_store_path)
    embeddings_dir = vector_store_dir / "embeddings"
    backup_dir = vector_store_dir / "embeddings_json_backup"

    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        return

    # Create backup directory if needed
    if backup and not backup_dir.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON embedding files
    json_files = list(embeddings_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON embedding files to migrate")

    if not json_files:
        logger.warning("No JSON embedding files found. Nothing to migrate.")
        return

    # Create company directories structure
    company_dir = vector_store_dir / "companies"
    company_dir.mkdir(parents=True, exist_ok=True)

    # Load company mapping
    company_mapping = {}
    company_mapping_path = vector_store_dir / "company_doc_mapping.json"
    if company_mapping_path.exists():
        try:
            with open(company_mapping_path, "r") as f:
                company_mapping = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Could not parse company_doc_mapping.json, creating a new one")
            company_mapping = {}

    # Process each JSON file
    migrated_count = 0
    error_count = 0

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

            # If we have company information, also save by company
            # Extract ticker from doc_id if possible (format: TICKER_ACCESSION)
            ticker = None
            if "_" in doc_id:
                ticker_candidate = doc_id.split("_")[0]
                if len(ticker_candidate) <= 5 and ticker_candidate.isalpha():
                    ticker = ticker_candidate

            # Also check company mapping
            for company, docs in company_mapping.items():
                if doc_id in docs:
                    ticker = company
                    break

            if ticker:
                ticker_dir = company_dir / ticker
                ticker_dir.mkdir(parents=True, exist_ok=True)
                company_npy_file = ticker_dir / f"{doc_id}.npy"
                np.save(company_npy_file, embedding_array)

            # Backup original JSON if requested
            if backup:
                backup_file = backup_dir / json_file.name
                shutil.copy2(json_file, backup_file)

            migrated_count += 1

        except Exception as e:
            logger.error(f"Error migrating {json_file}: {e}")
            error_count += 1

    logger.info(f"Migration complete: {migrated_count} files migrated, {error_count} errors")

    # Update company mapping if it was empty
    if not company_mapping and company_dir.exists():
        new_mapping = {}
        for ticker_dir in company_dir.iterdir():
            if ticker_dir.is_dir():
                ticker = ticker_dir.name
                doc_ids = [f.stem for f in ticker_dir.glob("*.npy")]
                new_mapping[ticker] = doc_ids

        with open(company_mapping_path, "w") as f:
            json.dump(new_mapping, f, indent=2)

        logger.info(f"Updated company_doc_mapping.json with {len(new_mapping)} companies")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate embeddings from JSON to NumPy binary format")
    parser.add_argument(
        "--vector-store-path", type=str, default="data/vector_store", help="Path to the vector store directory"
    )
    parser.add_argument("--no-backup", action="store_true", help="Don't create a backup of the original JSON files")

    args = parser.parse_args()

    migrate_embeddings(vector_store_path=args.vector_store_path, backup=not args.no_backup)
