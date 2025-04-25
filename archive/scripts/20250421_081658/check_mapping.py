"""
Check the company_doc_mapping.json file.
"""

import json
from pathlib import Path


def check_mapping():
    """Check the company_doc_mapping.json file."""
    try:
        # Check if the file exists
        mapping_file = Path("data/vector_store/company_doc_mapping.json")
        if not mapping_file.exists():
            print("company_doc_mapping.json not found")
            return

        # Load the mapping
        with open(mapping_file, "r") as f:
            data = json.load(f)

        # Print the companies in the mapping
        print(f"Companies in mapping: {list(data.keys())}")

        # Print the total number of documents
        total_docs = sum(len(docs) for docs in data.values())
        print(f"Total documents: {total_docs}")

        # Print the number of documents for each company
        for company, docs in data.items():
            print(f"{company}: {len(docs)} documents")
    except Exception as e:
        print(f"Error checking mapping: {e}")


if __name__ == "__main__":
    check_mapping()
