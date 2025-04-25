"""
Test script to verify the database connection to the improved financial database.
"""

import logging
import sys
from pathlib import Path

import duckdb

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the ETL configuration
from src.sec_filing_analyzer.config import ConfigProvider, ETLConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_db_connection():
    """Test the connection to the DuckDB database."""
    # Initialize configuration
    ConfigProvider.initialize()
    etl_config = ConfigProvider.get_config(ETLConfig)

    # Get database path
    db_path = etl_config.db_path
    logger.info(f"Testing connection to database: {db_path}")

    try:
        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        logger.info(f"Successfully connected to database: {db_path}")

        # Get tables
        tables = conn.execute("SHOW TABLES").fetchall()
        logger.info(f"Tables in database: {len(tables)}")

        for table in tables:
            table_name = table[0]
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            logger.info(f"  - {table_name}: {count} rows")

            # Get schema
            schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
            logger.info(f"    Schema: {len(schema)} columns")
            for col in schema[:5]:  # Show first 5 columns only
                logger.info(f"      - {col[0]}: {col[1]}")
            if len(schema) > 5:
                logger.info(f"      - ... and {len(schema) - 5} more columns")

        # Close connection
        conn.close()
        logger.info("Database connection test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return False


if __name__ == "__main__":
    test_db_connection()
