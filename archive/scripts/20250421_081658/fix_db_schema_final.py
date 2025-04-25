"""
Fix database schema for the SEC Filing Analyzer.

This script fixes the database schema to match what the sync_manager.py file expects.
"""

import os
from pathlib import Path

import duckdb


def fix_database_schema(db_path="data/db_backup/improved_financial_data.duckdb"):
    """Fix the database schema to match what the sync_manager.py file expects."""
    print(f"Fixing database schema for {db_path}...")

    # Check if the database file exists
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return False

    # Connect to the database
    conn = duckdb.connect(db_path)

    try:
        # Check if the filings table exists
        filings_exists = (
            conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'filings'").fetchone()[0]
            > 0
        )

        if filings_exists:
            # Check if both id and filing_id columns exist in the filings table
            id_exists = (
                conn.execute(
                    "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'id'"
                ).fetchone()[0]
                > 0
            )
            filing_id_exists = (
                conn.execute(
                    "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'filing_id'"
                ).fetchone()[0]
                > 0
            )

            if id_exists and filing_id_exists:
                print("Both id and filing_id columns exist in filings table. Updating filing_id values...")

                # Drop the filings_new table if it exists
                conn.execute("""
                DROP TABLE IF EXISTS filings_new
                """)

                # Create a new filings table with auto-incrementing filing_id
                conn.execute("""
                CREATE TABLE filings_new (
                    filing_id INTEGER PRIMARY KEY,
                    company_id INTEGER,
                    accession_number VARCHAR,
                    filing_type VARCHAR,
                    filing_date DATE,
                    fiscal_year INTEGER,
                    fiscal_period VARCHAR,
                    fiscal_period_end_date DATE,
                    document_url VARCHAR,
                    local_file_path VARCHAR,
                    has_xbrl BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

                # Get the maximum filing_id value
                max_id = conn.execute(
                    "SELECT COALESCE(MAX(filing_id), 0) FROM filings WHERE filing_id IS NOT NULL"
                ).fetchone()[0]

                # Insert data from the old table to the new table with auto-incrementing filing_id
                conn.execute(
                    """
                INSERT INTO filings_new (
                    filing_id, company_id, accession_number, filing_type, filing_date,
                    fiscal_year, fiscal_period, fiscal_period_end_date, document_url,
                    local_file_path, has_xbrl, created_at, updated_at
                )
                SELECT
                    CASE WHEN filing_id IS NOT NULL THEN filing_id ELSE ROW_NUMBER() OVER (ORDER BY filing_date) + ? END,
                    company_id, accession_number, filing_type, filing_date,
                    fiscal_year, fiscal_period, fiscal_period_end_date, document_url,
                    local_file_path, has_xbrl, created_at, updated_at
                FROM filings
                """,
                    [max_id],
                )

                print("Created new filings table with updated filing_id values")

                # Try to drop the id column from the filings table
                try:
                    conn.execute("""
                    ALTER TABLE filings
                    DROP COLUMN id
                    """)

                    print("Dropped id column from filings table")
                except Exception as e:
                    print(f"Could not drop id column: {e}")

                    # Try to drop the old table and rename the new one
                    try:
                        conn.execute("DROP TABLE filings")
                        conn.execute("ALTER TABLE filings_new RENAME TO filings")
                        print("Replaced filings table with new version")
                    except Exception as e:
                        print(f"Could not replace filings table: {e}")

                        # If all else fails, just keep the new table for reference
                        print("Keeping filings_new table for reference")

                print("Removed id column from filings table")

        # Check if the companies table exists
        companies_exists = (
            conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'companies'").fetchone()[0]
            > 0
        )

        if companies_exists:
            # Check if the updated_at column exists in the companies table
            updated_at_exists = (
                conn.execute(
                    "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'updated_at'"
                ).fetchone()[0]
                > 0
            )

            if not updated_at_exists:
                print("Adding updated_at column to companies table...")
                # Add updated_at column to companies table
                conn.execute("ALTER TABLE companies ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

                print("Added updated_at column to companies table")

        print("Database schema fixed successfully!")
        return True

    except Exception as e:
        print(f"Error fixing database schema: {e}")
        return False

    finally:
        # Close the connection
        conn.close()


if __name__ == "__main__":
    # Fix the database schema
    fix_database_schema()
