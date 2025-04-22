"""
Test script to verify sync_manager.py works after fixing the database schema.
"""

try:
    from src.sec_filing_analyzer.storage.sync_manager import StorageSyncManager
    
    # Create an instance of the sync manager
    sync_manager = StorageSyncManager(
        db_path="data/db_backup/improved_financial_data.duckdb",
        vector_store_path="data/vector_store",
        filings_dir="data/filings",
        graph_store_dir="data/graph_store",
        read_only=False  # Set to False to allow updates
    )
    
    # Test the sync_all function
    print("Testing sync_all function...")
    results = sync_manager.sync_all()
    print(f"sync_all results: {results}")
    
    # Test the update_filing_paths function
    print("\nTesting update_filing_paths function...")
    results = sync_manager.update_filing_paths()
    print(f"update_filing_paths results: {results}")
    
    # Test the update_processing_status function
    print("\nTesting update_processing_status function...")
    results = sync_manager.update_processing_status()
    print(f"update_processing_status results: {results}")
    
    # Test the get_inventory_summary function
    print("\nTesting get_inventory_summary function...")
    results = sync_manager.get_inventory_summary()
    print(f"get_inventory_summary results: {results}")
    
    # Close the connection
    sync_manager.close()
    
    print("\nAll tests completed successfully!")
    
except Exception as e:
    print(f"Error testing StorageSyncManager: {e}")
    import traceback
    traceback.print_exc()
