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

        # Fix the update_filing_paths method
        old_query = """SELECT id, ticker, accession_number FROM filings WHERE local_file_path IS NULL"""
        new_query = """
                SELECT 
                    CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'id') 
                        THEN id 
                        ELSE filing_id 
                    END as id,
                    CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'ticker') 
                        THEN ticker 
                        ELSE (SELECT c.ticker FROM companies c WHERE c.company_id = filings.company_id) 
                    END as ticker,
                    accession_number 
                FROM filings 
                WHERE (
                    (EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'local_file_path') AND local_file_path IS NULL) OR
                    (NOT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'local_file_path'))
                )
                """

        # Replace the query
        content = content.replace(old_query, new_query)

        # Fix the update_processing_status method
        old_query2 = """
                SELECT filing_id, accession_number
                FROM filings
                WHERE (
                    (fiscal_period IS NULL) OR
                    (fiscal_period IS NULL AND fiscal_quarter IS NOT NULL)
                )
                """
        new_query2 = """
                SELECT 
                    CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'filing_id') 
                        THEN filing_id 
                        ELSE id 
                    END as filing_id,
                    accession_number
                FROM filings
                WHERE (
                    (EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'fiscal_period') AND fiscal_period IS NULL) OR
                    (EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'fiscal_period') AND fiscal_period IS NULL AND fiscal_quarter IS NOT NULL) OR
                    (NOT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'fiscal_period'))
                )
                """

        # Replace the query
        content = content.replace(old_query2, new_query2)

        # Fix the company_info query
        old_query3 = """
                    SELECT c.ticker
                    FROM filings f
                    JOIN companies c ON f.company_id = c.company_id
                    WHERE f.filing_id = ?
                    """
        new_query3 = """
                    SELECT c.ticker
                    FROM filings f
                    JOIN companies c ON 
                        CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'company_id') 
                            THEN f.company_id = c.company_id
                            ELSE f.ticker = c.ticker
                        END
                    WHERE 
                        CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'filing_id') 
                            THEN f.filing_id = ?
                            ELSE f.id = ?
                        END
                    """

        # Replace the query
        content = content.replace(old_query3, new_query3)

        # Fix the UPDATE statement for fiscal_period
        old_update = """
                    UPDATE filings
                    SET fiscal_period = ?, 
                        updated_at = CASE WHEN updated_at IS NOT NULL THEN CURRENT_TIMESTAMP ELSE created_at END
                    WHERE filing_id = ?
                    """
        new_update = """
                    UPDATE filings
                    SET 
                        fiscal_period = ?,
                        updated_at = CASE 
                            WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'updated_at') 
                                THEN CASE WHEN updated_at IS NOT NULL THEN CURRENT_TIMESTAMP ELSE created_at END
                            ELSE NULL
                        END
                    WHERE 
                        CASE WHEN EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'filing_id') 
                            THEN filing_id = ?
                            ELSE id = ?
                        END
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
