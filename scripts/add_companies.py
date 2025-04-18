"""
Script to add companies to the DuckDB database
"""

import duckdb
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_companies(db_path="data/financial_data.duckdb"):
    """
    Add companies to the DuckDB database.

    Args:
        db_path: Path to the DuckDB database
    """
    try:
        # Connect to DuckDB
        logger.info(f"Connecting to DuckDB at {db_path}")
        conn = duckdb.connect(db_path)

        # Add companies
        companies = [
            ("AAPL", "Apple Inc."),
            ("MSFT", "Microsoft Corporation"),
            ("GOOGL", "Alphabet Inc."),
            ("AMZN", "Amazon.com, Inc."),
            ("META", "Meta Platforms, Inc."),
            ("TSLA", "Tesla, Inc."),
            ("NVDA", "NVIDIA Corporation"),
            ("JPM", "JPMorgan Chase & Co."),
            ("V", "Visa Inc."),
            ("JNJ", "Johnson & Johnson")
        ]

        # First, check the schema of the companies table
        schema = conn.execute("DESCRIBE companies").fetchdf()
        logger.info(f"Companies table schema: {schema}")

        # Create a temporary table for companies
        conn.execute("""
            CREATE TEMPORARY TABLE temp_companies (
                ticker VARCHAR,
                name VARCHAR
            )
        """)

        # Insert companies into temporary table
        for ticker, name in companies:
            conn.execute(
                "INSERT INTO temp_companies VALUES (?, ?)",
                [ticker, name]
            )

        # Insert companies that don't already exist
        result = conn.execute("""
            INSERT INTO companies (ticker, name)
            SELECT t.ticker, t.name
            FROM temp_companies t
            LEFT JOIN companies c ON t.ticker = c.ticker
            WHERE c.ticker IS NULL
        """)

        # Get count of added companies
        count = conn.execute("SELECT COUNT(*) FROM temp_companies").fetchone()[0]
        logger.info(f"Added {count} companies to database")

        # Drop temporary table
        conn.execute("DROP TABLE temp_companies")

        # Close connection
        conn.close()
        logger.info("Companies added successfully")

        return True

    except Exception as e:
        logger.error(f"Error adding companies: {e}")
        return False

if __name__ == "__main__":
    add_companies()
