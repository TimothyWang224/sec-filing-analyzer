import logging
import os
from pathlib import Path

import duckdb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_database_schema():
    """Fix the database schema to match the expected columns in the sync manager."""
    # Path to the database
    db_path = "data/db_backup/improved_financial_data.duckdb"

    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {db_path}")
        return

    try:
        # Connect to the database
        conn = duckdb.connect(db_path, read_only=False)

        # Check if the filings table exists
        tables = conn.execute("SHOW TABLES").fetchdf()
        logger.info(f"Tables in database: {tables['name'].tolist()}")

        if "filings" in tables["name"].tolist():
            # Get schema of filings table
            schema = conn.execute("PRAGMA table_info(filings)").fetchdf()
            columns = schema["name"].tolist()
            logger.info(f"Columns in filings table: {columns}")

            # Add missing columns
            missing_columns = []

            # Check for filing_id column
            if "filing_id" not in columns:
                missing_columns.append(("filing_id", "INTEGER"))

            # Check for id column
            if "id" not in columns:
                missing_columns.append(("id", "INTEGER"))

            # Check for local_file_path column
            if "local_file_path" not in columns:
                missing_columns.append(("local_file_path", "VARCHAR"))

            # Check for document_url column
            if "document_url" not in columns:
                missing_columns.append(("document_url", "VARCHAR"))

            # Check for processing_status column
            if "processing_status" not in columns:
                missing_columns.append(("processing_status", "VARCHAR"))

            # Check for last_updated column
            if "last_updated" not in columns:
                missing_columns.append(("last_updated", "TIMESTAMP"))

            # Check for updated_at column
            if "updated_at" not in columns:
                missing_columns.append(("updated_at", "TIMESTAMP"))

            # Check for created_at column
            if "created_at" not in columns:
                missing_columns.append(("created_at", "TIMESTAMP"))

            # Add missing columns
            for column_name, column_type in missing_columns:
                logger.info(f"Adding {column_name} column to filings table")
                conn.execute(f"""
                    ALTER TABLE filings 
                    ADD COLUMN {column_name} {column_type}
                """)
                logger.info(f"Added {column_name} column to filings table")

            # If filing_id is missing but company_id and accession_number exist, populate filing_id
            if (
                "filing_id" in [col[0] for col in missing_columns]
                and "company_id" in columns
                and "accession_number" in columns
            ):
                logger.info("Populating filing_id column")
                conn.execute("""
                    UPDATE filings
                    SET filing_id = ROW_NUMBER() OVER (ORDER BY company_id, accession_number)
                """)
                logger.info("Populated filing_id column")

            # If id is missing but filing_id exists, copy filing_id to id
            if "id" in [col[0] for col in missing_columns] and "filing_id" in columns:
                logger.info("Copying filing_id to id column")
                conn.execute("""
                    UPDATE filings
                    SET id = filing_id
                """)
                logger.info("Copied filing_id to id column")

            # If updated_at is missing but created_at exists, copy created_at to updated_at
            if "updated_at" in [col[0] for col in missing_columns] and "created_at" in columns:
                logger.info("Copying created_at to updated_at column")
                conn.execute("""
                    UPDATE filings
                    SET updated_at = created_at
                """)
                logger.info("Copied created_at to updated_at column")

            # If created_at is missing, set it to current timestamp
            if "created_at" in [col[0] for col in missing_columns]:
                logger.info("Setting created_at to current timestamp")
                conn.execute("""
                    UPDATE filings
                    SET created_at = CURRENT_TIMESTAMP
                """)
                logger.info("Set created_at to current timestamp")

            logger.info("Database schema fixed successfully")
        else:
            logger.error("No filings table found in the database")

        # Close the connection
        conn.close()
        logger.info("Database connection closed")

    except Exception as e:
        logger.error(f"Error fixing database schema: {e}")


if __name__ == "__main__":
    fix_database_schema()
