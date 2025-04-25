"""
Migration script to move files from the old directory structure to the new one.

This script:
1. Creates the new directory structure
2. Moves files from the old structure to the new one
3. Removes the old directories
"""

import logging
import os
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define old and new directory structures
OLD_STRUCTURE = {
    "data/cache/filings/cache": "data/filings/cache",
    "data/cache/filings/raw": "data/filings/raw",
    "data/cache/filings/processed": "data/filings/processed",
    "data/cache/filings/html": "data/filings/html",
    "data/cache/filings/xml": "data/filings/xml",
    "data/filings/cache": "data/filings/cache",
    "data/filings/raw": "data/filings/raw",
    "data/filings/processed": "data/filings/processed",
    "data/filings/html": "data/filings/html",
    "data/filings/xml": "data/filings/xml",
}

# Directories to remove after migration
DIRS_TO_REMOVE = [
    "data/cache/filings",
    "data/cache/sec_filings",
]


def create_new_structure():
    """Create the new directory structure."""
    logger.info("Creating new directory structure...")

    # Create new directories
    for _, new_dir in OLD_STRUCTURE.items():
        os.makedirs(new_dir, exist_ok=True)
        logger.info(f"Created directory: {new_dir}")


def move_files():
    """Move files from old structure to new structure."""
    logger.info("Moving files from old structure to new structure...")

    # Move files
    for old_dir, new_dir in OLD_STRUCTURE.items():
        old_path = Path(old_dir)
        new_path = Path(new_dir)

        # Skip if old directory doesn't exist
        if not old_path.exists():
            logger.info(f"Skipping non-existent directory: {old_dir}")
            continue

        # Move files
        for file_path in old_path.glob("*"):
            if file_path.is_file():
                # Create target directory for ticker-specific files
                if (
                    "/raw/" in str(file_path)
                    or "/processed/" in str(file_path)
                    or "/html/" in str(file_path)
                    or "/xml/" in str(file_path)
                ):
                    # Extract ticker from path if it exists
                    parts = file_path.parts
                    if len(parts) > 1 and parts[-2].isupper():  # Assume ticker is uppercase
                        ticker = parts[-2]
                        ticker_dir = new_path / ticker
                        os.makedirs(ticker_dir, exist_ok=True)
                        target_path = ticker_dir / file_path.name
                    else:
                        target_path = new_path / file_path.name
                else:
                    target_path = new_path / file_path.name

                # Skip if target file already exists
                if target_path.exists():
                    logger.info(f"Skipping existing file: {target_path}")
                    continue

                # Copy file
                try:
                    shutil.copy2(file_path, target_path)
                    logger.info(f"Copied: {file_path} -> {target_path}")
                except Exception as e:
                    logger.error(f"Error copying {file_path}: {e}")
            elif file_path.is_dir():
                # Handle ticker directories
                ticker = file_path.name
                if ticker.isupper():  # Assume ticker is uppercase
                    ticker_dir = new_path / ticker
                    os.makedirs(ticker_dir, exist_ok=True)

                    # Copy all files in ticker directory
                    for ticker_file in file_path.glob("*"):
                        if ticker_file.is_file():
                            target_path = ticker_dir / ticker_file.name
                            if not target_path.exists():
                                try:
                                    shutil.copy2(ticker_file, target_path)
                                    logger.info(f"Copied: {ticker_file} -> {target_path}")
                                except Exception as e:
                                    logger.error(f"Error copying {ticker_file}: {e}")


def remove_old_structure():
    """Remove old directories."""
    logger.info("Removing old directories...")

    for dir_path in DIRS_TO_REMOVE:
        path = Path(dir_path)
        if path.exists():
            try:
                shutil.rmtree(path)
                logger.info(f"Removed directory: {dir_path}")
            except Exception as e:
                logger.error(f"Error removing {dir_path}: {e}")


def main():
    """Run the migration."""
    logger.info("Starting data structure migration...")

    # Create new structure
    create_new_structure()

    # Move files
    move_files()

    # Remove old structure
    remove_old_structure()

    logger.info("Migration completed successfully!")


if __name__ == "__main__":
    main()
