"""
Script to fix the sync_manager.py file to work with the new database schema.
"""

import os
from typing import Dict


def fix_sync_manager() -> bool:
    """
    Fix the sync_manager.py file to work with the new database schema.
    
    Returns:
        True if the file was fixed successfully, False otherwise.
    """
    # Get the path to the sync_manager.py file
    file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            "src", "sec_filing_analyzer", "utils", "sync_manager.py")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # Read the file content
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Fix 1: Update the imports
    content = content.replace(
        "from ..storage import DuckDBStore",
        "from ..storage import OptimizedDuckDBStore"
    )
    
    # Fix 2: Update the class definition
    content = content.replace(
        "class SyncManager(DuckDBStore):",
        "class SyncManager(OptimizedDuckDBStore):"
    )
    
    # Fix 3: Update the column names in the sync_filings method
    content = content.replace(
        """                                    id, company_id, accession_number, filing_type, filing_date,""",
        """                                    filing_id, company_id, accession_number, filing_type, filing_date,"""
    )
    
    # Fix 4: Update the update_filing_paths method
    content = content.replace(
        """    def update_filing_paths(self) -> Dict[str, int]:
        """
        Update file paths for filings in DuckDB.

        Returns:
            Dictionary with update results
        """
        results = {}
        results["updated"] = 0
        results["errors"] = 0
        results["not_found"] = 0

        try:
            # Get all filings without a local file path
            filings = self.conn.execute(
                """
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
            ).fetchdf()""",
        """    def update_filing_paths(self) -> Dict[str, int]:
        """
        Update file paths for filings in DuckDB.

        Returns:
            Dictionary with update results
        """
        results = {}
        results["updated"] = 0
        results["errors"] = 0
        results["not_found"] = 0

        try:
            # Get all filings without a local file path
            filings = self.conn.execute(
                """
                SELECT
                    filing_id,
                    (SELECT c.ticker FROM companies c WHERE c.company_id = filings.company_id) as ticker,
                    accession_number
                FROM filings
                WHERE local_file_path IS NULL
                """
            ).fetchdf()"""
    )
    
    # Fix 5: Update the update_processing_status method
    content = content.replace(
        """    def update_processing_status(self) -> Dict[str, int]:
        """
        Update fiscal period for filings in DuckDB (equivalent to processing status in old schema).

        Returns:
            Dictionary with update results
        """
        results = {}
        results["updated"] = 0
        results["errors"] = 0

        try:
            # Get all filings with null fiscal period
            filings = self.conn.execute(
                """
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
            ).fetchdf()""",
        """    def update_processing_status(self) -> Dict[str, int]:
        """
        Update fiscal period for filings in DuckDB (equivalent to processing status in old schema).

        Returns:
            Dictionary with update results
        """
        results = {}
        results["updated"] = 0
        results["errors"] = 0

        try:
            # Get all filings with null fiscal period
            filings = self.conn.execute(
                """
                SELECT
                    filing_id,
                    accession_number
                FROM filings
                WHERE fiscal_period IS NULL
                """
            ).fetchdf()"""
    )
    
    # Fix 6: Update the company_info query
    content = content.replace(
        """                # Get company ticker from filing
                company_info = self.conn.execute(
                    """
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
                    """,
                    [filing['filing_id'], filing['filing_id']]
                ).fetchone()""",
        """                # Get company ticker from filing
                company_info = self.conn.execute(
                    """
                    SELECT c.ticker
                    FROM filings f
                    JOIN companies c ON f.company_id = c.company_id
                    WHERE f.filing_id = ?
                    """,
                    [filing['filing_id']]
                ).fetchone()"""
    )
    
    # Fix 7: Update the UPDATE statement
    content = content.replace(
        """                # Update the filing
                self.conn.execute(
                    """
                    UPDATE filings
                    SET fiscal_period = ?,
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
                    """,
                    [fiscal_period, filing['filing_id'], filing['filing_id']]
                )""",
        """                # Update the filing
                self.conn.execute(
                    """
                    UPDATE filings
                    SET fiscal_period = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE filing_id = ?
                    """,
                    [fiscal_period, filing['filing_id']]
                )"""
    )
    
    # Write the updated content back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Successfully fixed {file_path}")
    return True

if __name__ == "__main__":
    fix_sync_manager()
