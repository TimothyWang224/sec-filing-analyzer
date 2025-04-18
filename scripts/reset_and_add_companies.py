"""
Reset the DuckDB database and add companies.
"""

import logging
import duckdb
import os
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_database(db_path="data/financial_data.duckdb"):
    """Reset the DuckDB database."""
    try:
        # Check if database exists
        if os.path.exists(db_path):
            logger.info(f"Removing existing database at {db_path}")
            os.remove(db_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect to DuckDB (this will create a new database)
        logger.info(f"Creating new database at {db_path}")
        conn = duckdb.connect(db_path)
        
        # Create companies table
        conn.execute("""
        CREATE TABLE companies (
            ticker VARCHAR PRIMARY KEY,
            name VARCHAR,
            sector VARCHAR,
            industry VARCHAR,
            description TEXT,
            cik VARCHAR,
            exchange VARCHAR,
            last_updated TIMESTAMP
        )
        """)
        
        # Create filings table with enhanced schema
        conn.execute("""
        CREATE TABLE filings (
            id VARCHAR PRIMARY KEY,
            ticker VARCHAR,
            accession_number VARCHAR UNIQUE,
            filing_type VARCHAR,
            filing_date DATE,
            document_url VARCHAR,
            local_file_path VARCHAR,
            processing_status VARCHAR,
            last_updated TIMESTAMP,
            has_xbrl BOOLEAN DEFAULT FALSE,
            has_html BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (ticker) REFERENCES companies(ticker)
        )
        """)
        
        # Close connection
        conn.close()
        
        logger.info("Database reset successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return False

def add_companies(db_path="data/financial_data.duckdb"):
    """Add companies to the database."""
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logger.error(f"Database not found at {db_path}")
            return False
        
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
        
        # Insert companies
        for ticker, name in companies:
            # Check if company already exists
            result = conn.execute(f"SELECT ticker FROM companies WHERE ticker = '{ticker}'").fetchone()
            if result:
                logger.info(f"Company {ticker} already exists in the database")
                continue
            
            # Add company
            conn.execute(
                "INSERT INTO companies (ticker, name) VALUES (?, ?)",
                [ticker, name]
            )
            logger.info(f"Added company {ticker} to the database")
        
        # Close connection
        conn.close()
        
        logger.info("Companies added successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error adding companies: {e}")
        return False

if __name__ == "__main__":
    # Reset database
    reset_database()
    
    # Add companies
    add_companies()
