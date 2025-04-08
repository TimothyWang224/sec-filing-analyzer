"""
Script to check the structure of a cached filing.
"""

import os
import json
from pathlib import Path

def main():
    """Check the structure of a cached filing."""
    # Define the path to the cache directory
    cache_dir = Path("data/cache/filings/raw")

    # Check if the directory exists
    if not cache_dir.exists():
        print(f"Cache directory {cache_dir} does not exist")
        return

    # List all files in the cache directory
    print(f"Files in {cache_dir}:")
    for item in os.listdir(cache_dir):
        print(f"  {item}")

    # Check for NVDA directory
    nvda_dir = cache_dir / "NVDA"
    if not nvda_dir.exists():
        print(f"NVDA directory {nvda_dir} does not exist")
        return

    # List all files in the NVDA directory
    print(f"\nFiles in {nvda_dir}:")
    for item in os.listdir(nvda_dir):
        print(f"  {item}")

    # Check for 2023 directory
    year_dir = nvda_dir / "2023"
    if not year_dir.exists():
        print(f"2023 directory {year_dir} does not exist")
        return

    # List all files in the 2023 directory
    print(f"\nFiles in {year_dir}:")
    for item in os.listdir(year_dir):
        print(f"  {item}")

    # Check for the specific filing
    filing_id = "0001045810-23-000017"
    metadata_file = year_dir / f"{filing_id}_metadata.json"

    if not metadata_file.exists():
        print(f"Metadata file {metadata_file} does not exist")
        return

    # Read the metadata file
    print(f"\nReading metadata file {metadata_file}:")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Print the metadata
    print("Metadata keys:", list(metadata.keys()))
    for key, value in metadata.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
