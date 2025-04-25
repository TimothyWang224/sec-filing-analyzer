"""
Script to add NVDA to the database and sync its filings.
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.sec_filing_analyzer.storage.sync_manager import StorageSyncManager


def main():
    """Run the script."""
    # Create a sync manager with the correct database path
    sync_manager = StorageSyncManager(
        db_path="data/db_backup/improved_financial_data.duckdb",
        vector_store_path="data/vector_store",
        filings_dir="data/filings",
        graph_store_dir="data/graph_store",
        read_only=False,  # Need write access to fix mismatches
    )

    # Add NVDA to the database
    print("Adding NVDA to the database...")

    # Check if NVDA already exists
    nvda_exists = sync_manager.conn.execute("SELECT 1 FROM companies WHERE ticker = 'NVDA'").fetchone() is not None

    if not nvda_exists:
        # Add NVDA to the companies table
        company_id = sync_manager._get_company_id("NVDA")
        print(f"Added NVDA with company_id {company_id}")
    else:
        print("NVDA already exists in the database")

    # Sync all storage systems
    print("\nSynchronizing all storage systems...")
    results = sync_manager.sync_all()

    # Print the results
    print("\nSync results:")
    print(json.dumps(results, indent=2))

    # Close the connection
    sync_manager.close()


if __name__ == "__main__":
    main()
