"""
Initialize a new DuckDB database with the improved schema.
"""

import os
import duckdb

def initialize_database(db_path):
    """Initialize a new DuckDB database with the improved schema."""
    print(f"Initializing database at {db_path}...")

    # Create the database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Connect to the database
    print("Connecting to database...")
    conn = duckdb.connect(db_path)

    # Initialize the schema
    print("Loading schema...")
    schema_path = "src/sec_filing_analyzer/storage/improved_financial_db_schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    print("Executing schema...")
    conn.execute(schema_sql)

    # Close the connection
    print("Closing connection...")
    conn.close()

    print(f"Initialized database at {db_path}")

def main():
    print("Starting database initialization...")

    # Define parameters
    db_path = "data/db_backup/improved_financial_data.duckdb"

    print(f"Using database: {db_path}")

    # Initialize the database
    initialize_database(db_path)

    print("Database initialization complete.")

if __name__ == "__main__":
    main()
