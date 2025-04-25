"""
Check Vector Store

This script checks if a Vector Store is valid and can be opened.
"""

import os
import sys
from pathlib import Path


def check_vector_store(store_path):
    """Check if a Vector Store is valid and can be opened."""
    print(f"Checking Vector Store: {store_path}")

    # Check if the directory exists
    if not os.path.exists(store_path):
        print(f"Error: Vector Store directory not found: {store_path}")
        return False

    # Check if the directory is empty
    files = list(Path(store_path).glob("*"))
    if not files:
        print(f"Warning: Vector Store directory is empty: {store_path}")
        return False

    # Print the files in the directory
    print(f"Found {len(files)} files in Vector Store directory:")
    for file in files:
        print(f"  {file.name}")

    # Try to import the necessary modules
    try:
        import faiss

        print("Successfully imported faiss")
    except ImportError as e:
        print(f"Error importing faiss: {e}")
        return False

    # Try to import the LlamaIndexVectorStore
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sec_filing_analyzer.storage import LlamaIndexVectorStore

        print("Successfully imported LlamaIndexVectorStore")
    except ImportError as e:
        print(f"Error importing LlamaIndexVectorStore: {e}")
        return False

    # Try to open the Vector Store
    try:
        vector_store = LlamaIndexVectorStore(store_path=store_path)
        print("Successfully opened Vector Store")
    except Exception as e:
        print(f"Error opening Vector Store: {e}")
        return False

    return True


def main():
    """Main function."""
    # Get the Vector Store path from the command line
    if len(sys.argv) < 2:
        store_path = "data/vector_store"
        print(f"No Vector Store path provided, using default: {store_path}")
    else:
        store_path = sys.argv[1]

    # Check the Vector Store
    success = check_vector_store(store_path)

    if success:
        print("\nVector Store check completed successfully")
    else:
        print("\nVector Store check failed")


if __name__ == "__main__":
    main()
