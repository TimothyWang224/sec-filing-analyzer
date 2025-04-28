"""
Fix database schema for the SEC Filing Analyzer.

This script fixes the database schema to match what the sync_manager.py file expects.
"""

import os

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
        # Check if the companies table exists
        companies_exists = (
            conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'companies'"
            ).fetchone()[0]
            > 0
        )

        if companies_exists:
            # Check if the company_id column exists in the companies table
            company_id_exists = (
                conn.execute(
                    "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'companies' AND column_name = 'company_id'"
                ).fetchone()[0]
                > 0
            )

            if not company_id_exists:
                print("Adding company_id column to companies table...")
                # Add company_id column to companies table
                conn.execute("ALTER TABLE companies ADD COLUMN company_id INTEGER")

                # Set company_id values based on row number
                conn.execute("""
                UPDATE companies
                SET company_id = (SELECT row_number() OVER (ORDER BY ticker) FROM companies c2 WHERE c2.ticker = companies.ticker)
                """)

                print("Added company_id column to companies table")

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
                conn.execute(
                    "ALTER TABLE companies ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                )

                print("Added updated_at column to companies table")
        else:
            print("Creating companies table...")
            # Create companies table
            conn.execute("""
            CREATE TABLE companies (
                company_id INTEGER PRIMARY KEY,
                ticker VARCHAR,
                name VARCHAR,
                exchange VARCHAR,
                sector VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            print("Created companies table")

        # Check if the filings table exists
        filings_exists = (
            conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'filings'"
            ).fetchone()[0]
            > 0
        )

        if filings_exists:
            # Check if the company_id column exists in the filings table
            company_id_exists = (
                conn.execute(
                    "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'company_id'"
                ).fetchone()[0]
                > 0
            )

            if not company_id_exists:
                print("Adding company_id column to filings table...")
                # Add company_id column to filings table
                conn.execute("ALTER TABLE filings ADD COLUMN company_id INTEGER")

                # Check if ticker column exists in filings table
                ticker_exists = (
                    conn.execute(
                        "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'ticker'"
                    ).fetchone()[0]
                    > 0
                )

                if ticker_exists:
                    # Set company_id values based on ticker
                    conn.execute("""
                    UPDATE filings
                    SET company_id = (SELECT company_id FROM companies WHERE companies.ticker = filings.ticker)
                    """)

                print("Added company_id column to filings table")

            # Check if the id column exists in the filings table
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

            if id_exists and not filing_id_exists:
                print("Renaming id column to filing_id in filings table...")
                # Create a new filings table with the correct schema
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

                # Copy data from the old table to the new table
                conn.execute("""
                INSERT INTO filings_new (
                    filing_id, company_id, accession_number, filing_type, filing_date,
                    fiscal_year, fiscal_period, fiscal_period_end_date, document_url,
                    local_file_path, has_xbrl, created_at, updated_at
                )
                SELECT
                    id, company_id, accession_number, filing_type, filing_date,
                    fiscal_year, fiscal_period, fiscal_period_end_date, document_url,
                    local_file_path, has_xbrl, created_at, CURRENT_TIMESTAMP
                FROM filings
                """)

                # Drop the old table
                conn.execute("DROP TABLE filings")

                # Rename the new table to filings
                conn.execute("ALTER TABLE filings_new RENAME TO filings")

                print("Renamed id column to filing_id in filings table")
        else:
            print("Creating filings table...")
            # Create filings table
            conn.execute("""
            CREATE TABLE filings (
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
            print("Created filings table")

        # Fix the CASE expression type mismatch issues
        print("Fixing CASE expression type mismatch issues...")

        # Check if the fiscal_period column exists in the filings table
        fiscal_period_exists = (
            conn.execute(
                "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'fiscal_period'"
            ).fetchone()[0]
            > 0
        )

        if fiscal_period_exists:
            # Check the data type of the fiscal_period column
            fiscal_period_type = conn.execute(
                "SELECT data_type FROM information_schema.columns WHERE table_name = 'filings' AND column_name = 'fiscal_period'"
            ).fetchone()[0]

            if fiscal_period_type != "VARCHAR":
                print(
                    f"Converting fiscal_period column from {fiscal_period_type} to VARCHAR..."
                )
                # Create a new filings table with the correct schema
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

                # Copy data from the old table to the new table, converting fiscal_period to VARCHAR
                conn.execute("""
                INSERT INTO filings_new (
                    filing_id, company_id, accession_number, filing_type, filing_date,
                    fiscal_year, fiscal_period, fiscal_period_end_date, document_url,
                    local_file_path, has_xbrl, created_at, updated_at
                )
                SELECT
                    filing_id, company_id, accession_number, filing_type, filing_date,
                    fiscal_year, CAST(fiscal_period AS VARCHAR), fiscal_period_end_date, document_url,
                    local_file_path, has_xbrl, created_at, updated_at
                FROM filings
                """)

                # Drop the old table
                conn.execute("DROP TABLE filings")

                # Rename the new table to filings
                conn.execute("ALTER TABLE filings_new RENAME TO filings")

                print("Converted fiscal_period column to VARCHAR")

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
