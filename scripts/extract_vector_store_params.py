"""
Extract Vector Store Parameters

This script extracts vector store index parameters from the params.json files
and updates the unified configuration file.
"""

import json
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    # Find all vector store index parameter files
    vector_store_dir = Path("data/vector_store/index")
    if not vector_store_dir.exists():
        print(f"Vector store directory not found: {vector_store_dir}")
        sys.exit(1)

    param_files = list(vector_store_dir.glob("*.params.json"))
    if not param_files:
        print(f"No vector store parameter files found in {vector_store_dir}")
        sys.exit(1)

    print(f"Found {len(param_files)} vector store parameter files")

    # Extract parameters from the first file (they should all have the same structure)
    with open(param_files[0], "r") as f:
        params = json.load(f)

    print(f"Extracted parameters from {param_files[0]}: {params}")

    # Load the unified configuration file
    config_path = Path("data/config/etl_config.json")
    if not config_path.exists():
        print(f"Unified configuration file not found: {config_path}")
        print("Creating a new configuration file...")
        config = {}
    else:
        with open(config_path, "r") as f:
            config = json.load(f)

    # Update the configuration with the vector store parameters
    if "vector_store" not in config:
        config["vector_store"] = {}

    # Add the parameters
    config["vector_store"]["dimension"] = params.get("dimension", 1536)
    config["vector_store"]["description"] = params.get("description", "")

    # Get a list of all companies from the parameter files
    companies = set()
    for param_file in param_files:
        # Extract company names from the filename (e.g., AAPL_flat.params.json -> AAPL)
        filename = param_file.stem.replace(".params", "")
        if "_" in filename:
            # Handle multi-company files (e.g., AAPL_MSFT_flat)
            parts = filename.split("_")
            if parts[-1] in ["flat", "hnsw", "ivf"]:
                # The last part is the index type, not a company
                for company in parts[:-1]:
                    if company not in ["flat", "hnsw", "ivf", "main"]:
                        companies.add(company)
            else:
                # All parts are companies
                for company in parts:
                    if company not in ["flat", "hnsw", "ivf", "main"]:
                        companies.add(company)
        else:
            # Single company file
            companies.add(filename)

    # Update the companies list
    if "companies" not in config["vector_store"]:
        config["vector_store"]["companies"] = list(companies)
    else:
        # Merge with existing companies
        existing_companies = set(config["vector_store"]["companies"])
        config["vector_store"]["companies"] = list(existing_companies.union(companies))

    # Save the updated configuration
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated unified configuration file: {config_path}")
    print(f"Added companies: {', '.join(companies)}")

except Exception as e:
    print(f"Error extracting vector store parameters: {str(e)}")
    import traceback

    traceback.print_exc()
