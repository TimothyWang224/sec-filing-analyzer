import duckdb
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_sync_manager():
    """Fix the sync manager to handle both schema versions."""
    # Path to the database
    db_path = 'data/db_backup/improved_financial_data.duckdb'
    
    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {db_path}")
        return
    
    try:
        # Connect to the database
        conn = duckdb.connect(db_path, read_only=False)
        
        # Check if the filings table exists
        tables = conn.execute("SHOW TABLES").fetchdf()
        logger.info(f"Tables in database: {tables['name'].tolist()}")
        
        if 'filings' in tables['name'].tolist():
            # Check the schema of the filings table
            schema = conn.execute("PRAGMA table_info(filings)").fetchdf()
            columns = schema['name'].tolist()
            logger.info(f"Columns in filings table: {columns}")
            
            # Check if fiscal_period exists
            has_fiscal_period = 'fiscal_period' in columns
            # Check if fiscal_quarter exists
            has_fiscal_quarter = 'fiscal_quarter' in columns
            
            if not has_fiscal_period and has_fiscal_quarter:
                logger.info("Database has fiscal_quarter but not fiscal_period. Adding fiscal_period column...")
                
                # Add fiscal_period column
                conn.execute("""
                    ALTER TABLE filings 
                    ADD COLUMN fiscal_period VARCHAR
                """)
                
                # Update fiscal_period based on fiscal_quarter
                conn.execute("""
                    UPDATE filings
                    SET fiscal_period = 
                        CASE 
                            WHEN fiscal_quarter = 1 THEN 'Q1'
                            WHEN fiscal_quarter = 2 THEN 'Q2'
                            WHEN fiscal_quarter = 3 THEN 'Q3'
                            WHEN fiscal_quarter = 4 THEN 'Q4'
                            ELSE NULL
                        END
                """)
                
                logger.info("Added fiscal_period column and populated it based on fiscal_quarter")
            elif not has_fiscal_period and not has_fiscal_quarter:
                logger.info("Database has neither fiscal_period nor fiscal_quarter. Adding fiscal_period column...")
                
                # Add fiscal_period column
                conn.execute("""
                    ALTER TABLE filings 
                    ADD COLUMN fiscal_period VARCHAR
                """)
                
                logger.info("Added fiscal_period column")
            else:
                logger.info("Database already has fiscal_period column")
        else:
            logger.error("No filings table found in the database")
        
        # Close the connection
        conn.close()
        logger.info("Database connection closed")
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    fix_sync_manager()
