"""
Script to enhance the DuckDB schema for better file tracking
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

def enhance_duckdb_schema(db_path="data/financial_data.duckdb"):
    """
    Enhance the DuckDB schema by adding columns for better file tracking.
    
    Args:
        db_path: Path to the DuckDB database
    """
    try:
        # Connect to DuckDB
        logger.info(f"Connecting to DuckDB at {db_path}")
        conn = duckdb.connect(db_path)
        
        # Check if the local_file_path column already exists
        result = conn.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'filings' AND column_name = 'local_file_path'
        """).fetchdf()
        
        if len(result) == 0:
            # Add local_file_path column to filings table
            logger.info("Adding local_file_path column to filings table")
            conn.execute("""
                ALTER TABLE filings 
                ADD COLUMN local_file_path VARCHAR
            """)
            logger.info("Added local_file_path column to filings table")
        else:
            logger.info("local_file_path column already exists in filings table")
        
        # Check if the processing_status column already exists
        result = conn.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'filings' AND column_name = 'processing_status'
        """).fetchdf()
        
        if len(result) == 0:
            # Add processing_status column to filings table
            logger.info("Adding processing_status column to filings table")
            conn.execute("""
                ALTER TABLE filings 
                ADD COLUMN processing_status VARCHAR DEFAULT 'unknown'
            """)
            logger.info("Added processing_status column to filings table")
        else:
            logger.info("processing_status column already exists in filings table")
        
        # Check if the last_updated column already exists
        result = conn.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'filings' AND column_name = 'last_updated'
        """).fetchdf()
        
        if len(result) == 0:
            # Add last_updated column to filings table
            logger.info("Adding last_updated column to filings table")
            conn.execute("""
                ALTER TABLE filings 
                ADD COLUMN last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """)
            logger.info("Added last_updated column to filings table")
        else:
            logger.info("last_updated column already exists in filings table")
        
        # Create an index on ticker and filing_type for faster queries
        logger.info("Creating index on ticker and filing_type")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_filings_ticker_type 
            ON filings (ticker, filing_type)
        """)
        
        # Create an index on filing_date for faster date range queries
        logger.info("Creating index on filing_date")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_filings_date 
            ON filings (filing_date)
        """)
        
        # Commit changes and close connection
        conn.close()
        logger.info("Schema enhancement completed successfully")
        
        return True
    
    except Exception as e:
        logger.error(f"Error enhancing DuckDB schema: {e}")
        return False

if __name__ == "__main__":
    enhance_duckdb_schema()
