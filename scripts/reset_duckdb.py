"""
Reset DuckDB Database

This script deletes the existing DuckDB database and recreates it with a consistent schema.
"""

import logging
import os
import sys
from pathlib import Path

import duckdb

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reset_database(db_path="data/financial_data.duckdb"):
    """Reset the DuckDB database."""
    try:
        # Delete the existing database file if it exists
        if os.path.exists(db_path):
            logger.info(f"Deleting existing database at {db_path}")
            os.remove(db_path)
            logger.info(f"Deleted database at {db_path}")

        # Create the database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Connect to the database
        logger.info(f"Creating new database at {db_path}")
        conn = duckdb.connect(db_path)

        # Create companies table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                ticker VARCHAR PRIMARY KEY,
                name VARCHAR,
                cik VARCHAR,
                sic VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                exchange VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create filings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                id VARCHAR PRIMARY KEY,
                ticker VARCHAR,
                accession_number VARCHAR,
                filing_type VARCHAR,
                filing_date DATE,
                document_url VARCHAR,
                local_file_path VARCHAR,
                processing_status VARCHAR DEFAULT 'unknown',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                has_xbrl BOOLEAN DEFAULT FALSE,
                has_html BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (ticker) REFERENCES companies(ticker)
            )
        """)

        # Create financial facts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_facts (
                id VARCHAR PRIMARY KEY,
                filing_id VARCHAR,
                xbrl_tag VARCHAR,
                metric_name VARCHAR,
                value DOUBLE,
                unit VARCHAR,
                period_type VARCHAR,
                start_date DATE,
                end_date DATE,
                segment VARCHAR,
                context_id VARCHAR,
                decimals INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (filing_id) REFERENCES filings(id)
            )
        """)

        # Create time series metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS time_series_metrics (
                ticker VARCHAR,
                metric_name VARCHAR,
                fiscal_year INTEGER,
                fiscal_quarter INTEGER,
                value DOUBLE,
                unit VARCHAR,
                filing_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, metric_name, fiscal_year, fiscal_quarter),
                FOREIGN KEY (ticker) REFERENCES companies(ticker),
                FOREIGN KEY (filing_id) REFERENCES filings(id)
            )
        """)

        # Create financial ratios table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS financial_ratios (
                ticker VARCHAR,
                fiscal_year INTEGER,
                fiscal_quarter INTEGER,
                ratio_name VARCHAR,
                value DOUBLE,
                filing_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, ratio_name, fiscal_year, fiscal_quarter),
                FOREIGN KEY (ticker) REFERENCES companies(ticker),
                FOREIGN KEY (filing_id) REFERENCES filings(id)
            )
        """)

        # Create XBRL tag mappings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS xbrl_tag_mappings (
                xbrl_tag VARCHAR PRIMARY KEY,
                standard_metric_name VARCHAR,
                category VARCHAR,
                description VARCHAR,
                is_custom BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Close the connection
        conn.close()

        logger.info(f"Successfully reset database at {db_path}")
        return True

    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return False


if __name__ == "__main__":
    reset_database()
