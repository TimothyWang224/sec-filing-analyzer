import duckdb
import os

# Check if the database file exists
db_path = 'data/db_backup/improved_financial_data.duckdb'
if not os.path.exists(db_path):
    print(f"Database file not found at {db_path}")
    exit(1)

try:
    # Connect to the database
    conn = duckdb.connect(db_path, read_only=True)
    
    # Check if the filings table exists
    tables = conn.execute("SHOW TABLES").fetchdf()
    print(f"Tables in database: {tables['name'].tolist()}")
    
    if 'filings' in tables['name'].tolist():
        # Get schema of filings table
        schema = conn.execute("PRAGMA table_info(filings)").fetchdf()
        print("\nSchema of filings table:")
        for _, row in schema.iterrows():
            print(f"  {row['name']} ({row['type']})")
    else:
        print("No filings table found in the database")
    
    # Close the connection
    conn.close()
except Exception as e:
    print(f"Error: {e}")
