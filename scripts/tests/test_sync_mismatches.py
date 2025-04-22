"""
Test script for the new mismatch detection and fixing functionality.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.sec_filing_analyzer.storage.sync_manager import StorageSyncManager

def main():
    """Run the test."""
    # Create a sync manager with the correct database path
    sync_manager = StorageSyncManager(
        db_path="data/db_backup/improved_financial_data.duckdb",
        vector_store_path="data/vector_store",
        filings_dir="data/filings",
        graph_store_dir="data/graph_store",
        read_only=False  # Need write access to fix mismatches
    )
    
    # First, just detect mismatches without fixing them
    print("Detecting mismatches...")
    results = sync_manager.detect_and_fix_mismatches(auto_fix=False)
    
    # Print the results
    print("\nMismatches detected:")
    print(json.dumps(results, indent=2))
    
    # Ask if we should fix the mismatches
    fix = input("\nDo you want to fix these mismatches? (y/n): ")
    
    if fix.lower() == 'y':
        # Fix the mismatches
        print("\nFixing mismatches...")
        fix_results = sync_manager.detect_and_fix_mismatches(auto_fix=True)
        
        # Print the fix results
        print("\nFix results:")
        print(json.dumps(fix_results, indent=2))
    
    # Close the connection
    sync_manager.close()

if __name__ == "__main__":
    main()
