"""
Add NVDA company to the database.
"""

import logging
import os
from pathlib import Path

import duckdb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_nvda_company(db_path="data/financial_data.duckdb"):
    """Add NVDA company to the database."""
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logger.error(f"Database not found at {db_path}")
            return False

        # Connect to DuckDB
        logger.info(f"Connecting to DuckDB at {db_path}")
        conn = duckdb.connect(db_path)

        # Check if NVDA already exists
        result = conn.execute("SELECT ticker FROM companies WHERE ticker = 'NVDA'").fetchone()
        if result:
            logger.info("NVDA already exists in the database")
            return True

        # Add NVDA company
        logger.info("Adding NVDA to the database")
        conn.execute("INSERT INTO companies (ticker, name) VALUES (?, ?)", ["NVDA", "NVIDIA Corporation"])

        # Verify company was added
        result = conn.execute("SELECT ticker FROM companies WHERE ticker = 'NVDA'").fetchone()
        if result:
            logger.info("NVDA successfully added to the database")
            return True
        else:
            logger.error("Failed to add NVDA to the database")
            return False

    except Exception as e:
        logger.error(f"Error adding NVDA to the database: {e}")
        return False


if __name__ == "__main__":
    add_nvda_company()
