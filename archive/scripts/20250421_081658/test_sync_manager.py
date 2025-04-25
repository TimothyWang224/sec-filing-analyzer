"""
Test script to verify sync_manager.py imports correctly.
"""

try:
    from src.sec_filing_analyzer.storage.sync_manager import StorageSyncManager

    print("Successfully imported StorageSyncManager")
except Exception as e:
    print(f"Error importing StorageSyncManager: {e}")
