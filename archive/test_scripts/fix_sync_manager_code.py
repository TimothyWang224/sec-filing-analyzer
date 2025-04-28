import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_sync_manager_code():
    """Fix the sync manager code to handle both schema versions."""
    # Path to the sync manager file
    sync_manager_path = Path("src/sec_filing_analyzer/storage/sync_manager.py")

    if not sync_manager_path.exists():
        logger.error(f"Sync manager file not found at {sync_manager_path}")
        return

    try:
        # Read the file
        with open(sync_manager_path, "r") as f:
            content = f.read()

        # Fix the update_processing_status method
        old_query = """SELECT filing_id, accession_number FROM filings WHERE fiscal_period IS NULL"""
        new_query = """
                    SELECT filing_id, accession_number 
                    FROM filings 
                    WHERE (
                        (fiscal_period IS NULL) OR 
                        (fiscal_period IS NULL AND fiscal_quarter IS NOT NULL)
                    )
                    """

        # Replace the query
        content = content.replace(old_query, new_query)

        # Fix the UPDATE statement
        old_update = """
                    UPDATE filings
                    SET fiscal_period = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE filing_id = ?
                    """
        new_update = """
                    UPDATE filings
                    SET fiscal_period = ?, 
                        updated_at = CASE WHEN updated_at IS NOT NULL THEN CURRENT_TIMESTAMP ELSE created_at END
                    WHERE filing_id = ?
                    """

        # Replace the update statement
        content = content.replace(old_update, new_update)

        # Write the updated file
        with open(sync_manager_path, "w") as f:
            f.write(content)

        logger.info(f"Updated sync manager code at {sync_manager_path}")

    except Exception as e:
        logger.error(f"Error updating sync manager code: {e}")


if __name__ == "__main__":
    fix_sync_manager_code()
