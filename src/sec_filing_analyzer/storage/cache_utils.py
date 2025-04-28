"""
Cache utilities for optimizing storage operations.

This module provides caching utilities to improve performance of metadata loading
and other storage operations that involve many small files.
"""

import hashlib
import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_directory_digest(directory: Path, pattern: str = "*.json") -> str:
    """Calculate a digest of directory contents based on file modification times.

    Args:
        directory: Directory path to calculate digest for
        pattern: File pattern to match

    Returns:
        String digest representing the state of the directory
    """
    if not directory.exists():
        return "empty"

    h = hashlib.md5()
    for f in sorted(directory.glob(pattern)):
        h.update(str(f.stat().st_mtime_ns).encode())
    return h.hexdigest()


def load_cached_mapping(
    cache_file: Path,
    metadata_dir: Path,
    rebuild_func: Callable,
    force_rebuild: bool = False,
) -> Any:
    """Load a cached mapping, rebuilding if necessary.

    Args:
        cache_file: Path to the cache file
        metadata_dir: Directory containing metadata files
        rebuild_func: Function to call to rebuild the cache
        force_rebuild: Whether to force rebuilding the cache

    Returns:
        The cached mapping
    """
    start_time = time.time()

    # Force rebuild if requested
    if force_rebuild:
        logger.info("Forced rebuild of cache requested")
        mapping = rebuild_func()
        try:
            # Calculate directory digest
            digest = calculate_directory_digest(metadata_dir)
            # Save cache with digest
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert sets to lists for JSON serialization
            serializable_mapping = {}
            for key, value in mapping.items():
                if isinstance(value, set):
                    serializable_mapping[key] = list(value)
                else:
                    serializable_mapping[key] = value

            # Save as JSON
            cache_data = {"digest": digest, "mapping": serializable_mapping}

            # Try to use orjson for faster serialization if available
            try:
                import orjson

                with open(cache_file, "wb") as f:
                    f.write(orjson.dumps(cache_data))
            except ImportError:
                import json

                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f)

            logger.info(f"Cache rebuilt and saved to {cache_file}")
        except Exception as e:
            logger.warning(f"Error saving cache to {cache_file}: {e}")

        logger.info(f"Cache rebuilt in {time.time() - start_time:.2f} seconds")
        return mapping

    # Try to load from cache
    if cache_file.exists():
        try:
            # Try to use orjson for faster parsing if available
            try:
                import orjson

                with open(cache_file, "rb") as f:
                    cache_data = orjson.loads(f.read())
            except ImportError:
                import json

                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

            cached_digest = cache_data["digest"]
            serialized_mapping = cache_data["mapping"]

            # Convert lists back to sets
            mapping = {}
            for key, value in serialized_mapping.items():
                if isinstance(value, list):
                    mapping[key] = set(value)
                else:
                    mapping[key] = value

            # Check if cache is still valid
            current_digest = calculate_directory_digest(metadata_dir)
            if cached_digest == current_digest:
                logger.info(f"Cache hit! Loaded mapping from {cache_file} in {time.time() - start_time:.2f} seconds")
                return mapping
            else:
                logger.info("Cache digest mismatch, rebuilding...")
        except Exception as e:
            logger.warning(f"Error loading cache from {cache_file}: {e}")

    # Rebuild cache
    mapping = rebuild_func()

    try:
        # Calculate directory digest
        digest = calculate_directory_digest(metadata_dir)
        # Save cache with digest
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert sets to lists for JSON serialization
        serializable_mapping = {}
        for key, value in mapping.items():
            if isinstance(value, set):
                serializable_mapping[key] = list(value)
            else:
                serializable_mapping[key] = value

        # Save as JSON
        cache_data = {"digest": digest, "mapping": serializable_mapping}

        # Try to use orjson for faster serialization if available
        try:
            import orjson

            with open(cache_file, "wb") as f:
                f.write(orjson.dumps(cache_data))
        except ImportError:
            import json

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)

        logger.info(f"Cache rebuilt and saved to {cache_file}")
    except Exception as e:
        logger.warning(f"Error saving cache to {cache_file}: {e}")

    logger.info(f"Cache rebuilt in {time.time() - start_time:.2f} seconds")
    return mapping


@lru_cache(maxsize=2000)
def get_metadata(metadata_dir: Path, doc_id: str) -> Optional[Dict[str, Any]]:
    """Load metadata for a document with caching.

    Args:
        metadata_dir: Directory containing metadata files
        doc_id: Document ID

    Returns:
        Metadata dictionary or None if not found
    """
    try:
        # Create a safe filename
        safe_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        metadata_path = metadata_dir / f"{safe_id}.json"

        if metadata_path.exists():
            # Try to use orjson for faster parsing if available
            try:
                import orjson

                with open(metadata_path, "rb") as f:
                    return orjson.loads(f.read())
            except ImportError:
                # Fall back to standard json
                with open(metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading metadata for {doc_id}: {e}")

    return None
